from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

try:
    import jenkspy
    JENKSPY_AVAILABLE = True
except ImportError:
    JENKSPY_AVAILABLE = False

from murasa.core.normalize import classify_jenks
from murasa.core.analysis import calculate_zonal_stats
from murasa.core.data_source import DataRegistry, SourceType
from murasa.core.logger import logger
from murasa.core.base import PostProcessorPlugin, ParameterPlugin, register_plugin, plugin_registry


"""
This is a post-processor.

Currently, it is very specific for my use case.
"""


def _extract_time_metadata(config: Any) -> Tuple[Optional[int], Optional[int]]:
    if not config:
        return None, None

    for attr in ('plugin_parameters', 'parameters'):
        params = getattr(config, attr, None)
        if isinstance(params, dict):
            for sub in params.values():
                if isinstance(sub, dict) and ('year' in sub or 'month' in sub):
                    return sub.get('year'), sub.get('month')
            if 'year' in params or 'month' in params:
                return params.get('year'), params.get('month')

    factors = getattr(config, 'factors', None)
    if isinstance(factors, dict):
        for fcfg in factors.values():
            fparams = getattr(fcfg, 'parameters', {})
            if isinstance(fparams, dict) and ('year' in fparams or 'month' in fparams):
                return fparams.get('year'), fparams.get('month')

    return getattr(config, 'year', None), getattr(config, 'month', None)


@register_plugin("vector_explainability")
class VectorExplainabilityPlugin(PostProcessorPlugin):
    """
    Post-processor plugin for vector explainability of risk analysis outputs.

    Currently the defaults are specifically written for my use case, but can be generalized later.
    """


    _DEFAULT_THRESHOLD_SCENARIOS: Dict[str, Dict[str, float]] = {
        'Conservative': {
            'extreme_pct': 40, 'very_high_pct': 35, 'high_pct': 35,
            'extreme_score': 5.8, 'very_high_score': 5.3, 'high_score': 4.5,
        },
        'Moderate': {
            'extreme_pct': 35, 'very_high_pct': 30, 'high_pct': 30,
            'extreme_score': 5.5, 'very_high_score': 5.0, 'high_score': 4.2,
        },
        'Aggressive': {
            'extreme_pct': 25, 'very_high_pct': 20, 'high_pct': 25,
            'extreme_score': 5.0, 'very_high_score': 4.5, 'high_score': 3.8,
        },
    }

    _DEFAULT_EXPLANATIONS: Dict[str, Any] = {
        'elevation': {
            True: (
                "Elevasi Rendah ({score:.2f}): wilayah berada di dataran rendah "
                "sehingga rentan terhadap akumulasi air dan sulit untuk drainase alami"
            ),
            False: "Elevasi Tinggi ({score:.2f})",
        },
        'slope': {
            True: (
                "Kemiringan Datar ({score:.2f}): lereng yang landai memperlambat "
                "aliran air sehingga air cenderung menggenang lebih lama"
            ),
            False: "Kemiringan Lereng Curam ({score:.2f})",
        },
        'land_use': (
            "Tutupan Lahan Rentan ({score:.2f}): dominasi lahan kedap air atau "
            "permukiman padat mengurangi kemampuan tanah menyerap air hujan"
        ),
        'landuse': (
            "Tutupan Lahan Rentan ({score:.2f}): dominasi lahan kedap air atau "
            "permukiman padat mengurangi kemampuan tanah menyerap air hujan"
        ),
        'soil': "Jenis Tanah Peka/Liat/Vulkanik ({score:.2f})",
        'proximity': (
            "Dekat Sungai/Badan Air ({score:.2f}): kedekatan dengan sungai atau "
            "saluran air meningkatkan risiko luapan saat debit air meningkat"
        ),
        'prox': (
            "Dekat Sungai/Badan Air ({score:.2f}): kedekatan dengan sungai atau "
            "saluran air meningkatkan risiko luapan saat debit air meningkat"
        ),
        'curvature': "Kelengkungan Lereng ({score:.2f})",
        'aspect': "Orientasi Lereng Rawan ({score:.2f})",
        'twi': (
            "Akumulasi Air Tinggi/TWI ({score:.2f}): indeks topografi menunjukkan "
            "kecenderungan kuat untuk penumpukan air di wilayah ini"
        ),
        'rainfall': (
            "Curah Hujan Tinggi ({score:.2f}): intensitas hujan yang tinggi "
            "meningkatkan volume air permukaan dan memperbesar potensi genangan"
        ),
    }

    _LANDSLIDE_EXPLANATIONS: Dict[str, Any] = {
        'elevation': "Elevasi Tinggi ({score:.2f}): wilayah berada di zona perbukitan atau pegunungan tinggi",
        'slope': (
            "Kemiringan Lereng Curam ({score:.2f}): lereng yang terjal meningkatkan "
            "gaya gravitasi pendorong massa tanah"
        ),
        'land_use': (
            "Tutupan Lahan Rentan ({score:.2f}): dominasi lahan minim vegetasi berakar kuat "
            "atau terbangun mengurangi daya ikat tanah pada lereng"
        ),
        'landuse': (
            "Tutupan Lahan Rentan ({score:.2f}): dominasi lahan minim vegetasi berakar kuat "
            "atau terbangun mengurangi daya ikat tanah pada lereng"
        ),
        'soil': (
            "Jenis Tanah Peka/Liat/Vulkanik ({score:.2f}): karakteristik tanah mudah "
            "jenuh air dan rentan mengalami pergerakan"
        ),
        'rainfall': (
            "Curah Hujan Tinggi ({score:.2f}): intensitas hujan tinggi meningkatkan "
            "kejenuhan air pori tanah dan memicu ketidakstabilan lereng"
        ),
    }

    def __init__(
        self,
        name: str = "vector_explainability",
        labels: Optional[List[str]] = None,
        output_prefix: str = "flood_risk",
        title_label: str = "Tingkat Risiko",
        explain_threshold: float = 0.70,
        run_sensitivity: bool = True,
        explanations: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name)
        self.labels = labels or ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
        self.output_prefix = output_prefix
        self.title_label = title_label
        self.explain_threshold = explain_threshold
        self.run_sensitivity = run_sensitivity
        is_landslide = "landslide" in output_prefix.lower() or "longsor" in title_label.lower()
        base_explanations = dict(self._LANDSLIDE_EXPLANATIONS) if is_landslide else {}
        if explanations:
            base_explanations.update(explanations)
        self.explanations = base_explanations
        self._active_plugins: Dict[str, ParameterPlugin] = {}
        self._uniform_p_cols: Dict[str, float] = {}

    def _resolve_active_factors(
        self,
        registry: Optional[DataRegistry] = None,
        gdf: Optional[gpd.GeoDataFrame] = None,
        weights: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Discover active factors dynamically from DataRegistry, weights,
        or existing columns in the GeoDataFrame.
        """
        plugin_registry.auto_discover()
        known_plugins: Dict[str, ParameterPlugin] = {}

        for pname in plugin_registry.list_plugins(ParameterPlugin):
            try:
                inst = plugin_registry.create(pname)
                if not getattr(inst, 'is_exclusion', False):
                    known_plugins[pname] = inst
            except Exception:
                pass

        col_to_plugin: Dict[str, Tuple[str, Optional[ParameterPlugin]]] = {}
        for pname, inst in known_plugins.items():
            col_to_plugin[inst.column_key] = (pname, inst)
            col_to_plugin[pname] = (pname, inst)

        w_dict = {}
        if weights is not None:
            w_dict = weights.as_dict() if hasattr(weights, 'as_dict') else dict(weights)

        discovered_keys = []
        if registry is not None:
            for src_key in registry.list_sources():
                if src_key.startswith("risk_"):
                    fname = src_key[len("risk_"):]
                    if fname not in discovered_keys:
                        discovered_keys.append(fname)

        for fname in w_dict.keys():
            if fname not in discovered_keys:
                discovered_keys.append(fname)

        if gdf is not None:
            for col in gdf.columns:
                if col.startswith("p_"):
                    short = col[2:]
                    pname, _ = col_to_plugin.get(short, (short, None))
                    if pname not in discovered_keys:
                        discovered_keys.append(pname)

        factors_meta = []
        seen_p_cols = set()

        for fname in discovered_keys:
            _, inst = col_to_plugin.get(fname, (fname, known_plugins.get(fname)))
            short = inst.column_key if inst else fname
            p_col = f"p_{short}"
            if p_col in seen_p_cols:
                continue
            seen_p_cols.add(p_col)

            c_key = inst.contrib_key if inst else short
            c_col = f"c_{c_key}"
            label = inst.label if inst else fname.replace("_", " ").title()

            factors_meta.append({
                "name": fname,
                "short": short,
                "p_col": p_col,
                "c_col": c_col,
                "label": label,
                "plugin": inst,
                "weight": w_dict.get(fname, w_dict.get(short, 0.0)),
            })

        return factors_meta

    def post_process(
        self,
        registry: DataRegistry,
        risk_raster: np.ndarray,
        transform: Any,
        crs: str,
        output_dir: Path,
        config: Any = None
    ):
        logger.info(f"Running Post-Processor: {self.name}")

        study_area_key = (
            config.spatial.study_area_key
            if config and hasattr(config, 'spatial')
            else 'admin'
        )
        if not registry.has(study_area_key):
            logger.warning(f"   No study area boundaries ('{study_area_key}') found in registry. Skipping vector analysis.")
            return

        admin_source = registry.get(study_area_key)
        gdf_admin = admin_source.data.to_crs(crs).copy()

        if 'area_ha' not in gdf_admin.columns:
            metric_crs = config.spatial.metric_crs if config and hasattr(config, 'spatial') else "EPSG:3857"
            try:
                gdf_metric = gdf_admin.to_crs(metric_crs)
                gdf_admin['area_ha'] = (gdf_metric.geometry.area / 10000.0).round(2)
            except Exception as e:
                logger.warning(f"Failed to compute area_ha using metric CRS ({metric_crs}): {e}")
                gdf_admin['area_ha'] = 0.0

        n_classes = len(self.labels)

        if len(gdf_admin) == 1:
            valid_pixels = risk_raster[~np.isnan(risk_raster)]
            if len(valid_pixels) > 0:
                quantiles = np.unique(np.percentile(valid_pixels, np.linspace(0, 100, n_classes + 1)))
                if len(quantiles) - 1 < n_classes:
                    quantiles = np.linspace(valid_pixels.min(), valid_pixels.max(), n_classes + 1)
                classified_raster = np.digitize(risk_raster, bins=quantiles[1:-1]) + 1
                classified_raster[np.isnan(risk_raster)] = 0
            else:
                classified_raster, _ = self._classify_raster_pixels(risk_raster, n_classes=n_classes)
        else:
            classified_raster, _ = self._classify_raster_pixels(risk_raster, n_classes=n_classes)

        stats = calculate_zonal_stats(
            gdf_admin,
            classified_raster,
            stats=['count', 'mean', 'max'],
            categorical=True,
            nodata=0,
            transform=transform,
            crs=crs
        )

        year, month = _extract_time_metadata(config)

        results = []
        for row, stat in zip(gdf_admin.itertuples(), stats):
            if not stat or stat.get('count', 0) == 0:
                continue

            total_pixels = stat.get('count', 0)
            pcts = {
                f"pct_class_{cls}": (stat.get(cls, 0) / total_pixels * 100) if total_pixels > 0 else 0.0
                for cls in range(1, n_classes + 1)
            }

            data = {
                'geometry': row.geometry,
                'area_ha': getattr(row, 'area_ha', 0.0),
                'risk_score': stat.get('mean', 0),
                'risk_max': stat.get('max', 0),
                'pixel_count': total_pixels,
            }

            if n_classes >= 6:
                data['pct_extreme'] = pcts.get("pct_class_6", 0.0)
                data['pct_very_high'] = pcts.get("pct_class_5", 0.0)
                data['pct_high'] = pcts.get("pct_class_4", 0.0)
            else:
                data['pct_very_high'] = pcts.get(f"pct_class_{n_classes}", 0.0)
                data['pct_high'] = pcts.get(f"pct_class_{max(1, n_classes - 1)}", 0.0)
                data['pct_medium'] = pcts.get(f"pct_class_{max(1, n_classes - 2)}", 0.0)
                data['pct_low'] = pcts.get("pct_class_1", 0.0)

            for col in gdf_admin.columns:
                if col not in data and col != 'geometry':
                    data[col] = getattr(row, col)

            if year is not None:
                data['year'] = year
            if month is not None:
                data['month'] = month

            results.append(data)

        if not results:
            logger.warning("No results to export.")
            return

        out_gdf = gpd.GeoDataFrame(results, crs=crs)

        weights = config.weights if config and hasattr(config, 'weights') else None
        self._enrich_factors(out_gdf, registry, transform, crs, weights, config=config)

        if config and hasattr(config, 'classification') and getattr(config.classification, 'method', None):
            method = config.classification.method
        elif config and hasattr(config, 'analysis') and getattr(config.analysis, 'classification_method', None):
            method = config.analysis.classification_method
        else:
            method = 'jenks'
        out_gdf = self._classify_vector_legacy(out_gdf, method)

        self._add_explainability_metrics(out_gdf)
        out_gdf['reason'] = out_gdf.apply(self._generate_explanation, axis=1)
        self._add_goodness_and_stability(out_gdf, weights)
        admin_source.data = out_gdf.copy()

        if self.run_sensitivity:
            self._run_threshold_sensitivity(out_gdf, output_dir)
            if weights:
                self._run_weight_sensitivity(out_gdf, weights, output_dir)

        out_geojson = output_dir / f"{self.output_prefix}_vector_explained.geojson"
        out_csv = output_dir / f"{self.output_prefix}_explainability.csv"

        try:
            out_gdf.to_file(out_geojson, driver="GeoJSON")
            logger.info(f"   Saved explained vector output: {out_geojson.name}")
        except Exception as e:
            logger.error(f"Failed to save GeoJSON: {e}")

        df_export = pd.DataFrame(out_gdf.drop(columns='geometry'))
        df_export.to_csv(out_csv, index=False)
        logger.info(f"   Saved CSV output: {out_csv.name}")

    def _enrich_factors(self, gdf, registry, transform, crs, weights, config=None):
        logger.info("   Extracting contributing factors...")
        factors_meta = self._resolve_active_factors(registry=registry, gdf=gdf, weights=weights)

        params_cfg = getattr(config, 'parameters', {}) if config else {}

        for fmeta in factors_meta:
            key = fmeta['name']
            col_name = fmeta['p_col']
            risk_key = f"risk_{key}"
            source = None

            if fmeta['plugin'] is not None and isinstance(params_cfg, dict) and key in params_cfg:
                try:
                    fmeta['plugin'] = plugin_registry.create(
                        key, weight=fmeta['weight'], parameters=params_cfg[key]
                    )
                except Exception:
                    pass
            if fmeta['plugin'] is not None:
                self._active_plugins[col_name] = fmeta['plugin']

            if registry.has(risk_key):
                source = registry.get(risk_key)
            elif registry.has(key) and registry.get(key).source_type == SourceType.RASTER:
                source = registry.get(key)

            if source is None:
                gdf[col_name] = 0.0
                continue

            use_q3 = (
                key == "slope"
                and fmeta['plugin'] is not None
                and not getattr(fmeta['plugin'], 'inverse', True)
            )
            stat_key = 'percentile_75' if use_q3 else 'mean'
            stats = calculate_zonal_stats(
                gdf,
                source,
                stats=[stat_key],
                nodata=np.nan,
                transform=transform,
                crs=crs
            )
            gdf[col_name] = [
                float(s[stat_key]) if s and s.get(stat_key) is not None else 0.0
                for s in stats
            ]

        if weights:
            for fmeta in factors_meta:
                p_col = fmeta['p_col']
                c_col = fmeta['c_col']
                if p_col in gdf.columns:
                    gdf[c_col] = gdf[p_col] * fmeta['weight']

    def _classify_raster_pixels(self, risk_raster, n_classes: int = 6):
        logger.info(f"   Classifying Raster Pixels (1-{n_classes} Scale)...")
        return classify_jenks(risk_raster, n_classes=n_classes)

    def _classify_vector_legacy(self, gdf, method):
        if 'pct_extreme' in gdf.columns:
            gdf['composite_risk'] = (
                gdf['pct_extreme'] * 2.0 +
                gdf.get('pct_very_high', 0) * 1.5 +
                gdf.get('pct_high', 0) * 1.0 +
                gdf['risk_score'] * 0.5
            )
        else:
            gdf['composite_risk'] = (
                gdf.get('pct_very_high', 0) * 1.5 +
                gdf.get('pct_high', 0) * 1.0 +
                gdf.get('pct_medium', 0) * 0.5 +
                gdf['risk_score'] * 0.5
            )

        labels = self.labels
        n_classes = len(labels)
        unique_vals = gdf['composite_risk'].dropna().nunique()

        if len(gdf) <= 1 or unique_vals < n_classes:
            gdf['severity'] = labels[len(labels) // 2] if unique_vals > 0 else labels[0]
            gdf.attrs['classification_method'] = 'Single Unit Fallback'
            return gdf

        if method == 'jenks' and JENKSPY_AVAILABLE:
            try:
                breaks = jenkspy.jenks_breaks(gdf['composite_risk'], n_classes=n_classes)
                gdf['severity'] = pd.cut(gdf['composite_risk'], bins=breaks, labels=labels, include_lowest=True)
                gdf.attrs['jenks_breaks'] = breaks
                gdf.attrs['classification_method'] = 'Jenks Natural Breaks'
                return gdf
            except Exception as e:
                logger.warning(f"Jenks on vector failed: {e}")

        try:
            gdf['severity'] = pd.qcut(gdf['composite_risk'], q=n_classes, labels=labels, duplicates='drop')
            gdf.attrs['classification_method'] = 'Quantile'
        except ValueError:
            gdf['severity'] = labels[0]
            gdf.attrs['classification_method'] = 'Constant (Fallback)'

        return gdf

    def _classify_vector_expert(self, gdf):
        severity_list = []

        for _, row in gdf.iterrows():
            extreme_pct = row.get('pct_extreme', 0)
            very_high_pct = row.get('pct_very_high', 0)
            high_pct = row.get('pct_high', 0)
            risk_score = row.get('risk_score', 0)

            if extreme_pct > 35 or (extreme_pct > 25 and risk_score >= 5.5):
                severity = 'Extreme'
            elif very_high_pct > 30 or extreme_pct > 15 or risk_score >= 5.0:
                severity = 'Very High'
            elif high_pct > 30 or very_high_pct > 20 or risk_score >= 4.2:
                severity = 'High'
            elif risk_score >= 3.5:
                severity = 'Medium'
            elif risk_score >= 2.5:
                severity = 'Low'
            else:
                severity = 'Very Low'

            severity_list.append(severity)

        gdf['severity'] = severity_list
        gdf.attrs['classification_method'] = 'Expert Rules'
        return gdf

    def _classify_risk(self, gdf, method='expert'):
        if method == 'expert':
            return self._classify_vector_expert(gdf)
        if method == 'jenks':
            return self._classify_vector_legacy(gdf, 'jenks')
        return self._classify_vector_legacy(gdf, 'quantile')

    def _add_explainability_metrics(self, gdf):
        factors_meta = self._resolve_active_factors(gdf=gdf)
        active_c = [
            (
                fmeta['p_col'],
                fmeta['c_col'] if fmeta['c_col'] in gdf.columns else fmeta['p_col'],
                fmeta['label'],
            )
            for fmeta in factors_meta
            if fmeta['c_col'] in gdf.columns or fmeta['p_col'] in gdf.columns
        ]

        self._uniform_p_cols = {}
        if len(gdf) > 3:
            for p_col, _, _ in active_c:
                if p_col in gdf.columns:
                    col_iqr = float(gdf[p_col].quantile(0.75) - gdf[p_col].quantile(0.25))
                    if np.isfinite(col_iqr) and col_iqr < 0.05:
                        self._uniform_p_cols[p_col] = float(gdf[p_col].median())

        dominant = []
        confidence = []
        certainty = []

        for _, row in gdf.iterrows():
            vals = [float(row.get(c_col, 0) or 0) for _, c_col, _ in active_c]
            total = sum(vals)

            if total <= 0 or not active_c:
                dominant.append("None")
                confidence.append(0.0)
                certainty.append(0.0)
                continue

            eff_vals = []
            for (p_col, _, _), c_val in zip(active_c, vals):
                p_val = float(row.get(p_col, 0) or 0)
                if p_col in self._uniform_p_cols and p_val <= self._uniform_p_cols[p_col] + 0.05:
                    eff_vals.append(c_val * 0.4)
                else:
                    eff_vals.append(c_val)

            max_idx = int(np.argmax(eff_vals))
            dom_name = active_c[max_idx][2]
            conf = vals[max_idx] / total

            norm_vals = np.array(vals) / total
            cert = min(float(np.sqrt(np.var(norm_vals)) * 2.5), 1.0)

            dominant.append(dom_name)
            confidence.append(round(conf, 3))
            certainty.append(round(cert, 3))

        gdf['dominant'] = dominant
        gdf['conf'] = confidence
        gdf['cert'] = certainty

    def _generate_explanation(self, row) -> str:
        reasons = [f"{self.title_label}: {row.get('severity', 'Tidak Diketahui')} ({row['risk_score']:.2f})"]

        if 'pct_extreme' in row:
            pct_ext = row.get('pct_extreme', 0)
            pct_vh = row.get('pct_very_high', 0)
            pct_h = row.get('pct_high', 0)
            total_pct = pct_ext + pct_vh + pct_h
            if total_pct > 0:
                reasons.append(
                    f"Area Terdampak (Risiko Tinggi-Ekstrem): {total_pct:.1f}% "
                    f"(Ext: {pct_ext:.1f}%, VHigh: {pct_vh:.1f}%, High: {pct_h:.1f}%)"
                )
            else:
                reasons.append("Area Terdampak (Risiko Tinggi-Ekstrem): 0%")
        else:
            pct_vh = row.get('pct_very_high', 0)
            pct_h = row.get('pct_high', 0)
            total_pct = pct_vh + pct_h
            if total_pct > 0:
                reasons.append(
                    f"Area Terdampak (Tinggi-Sangat Tinggi): {total_pct:.1f}% "
                    f"(VHigh: {pct_vh:.1f}%, High: {pct_h:.1f}%)"
                )
            else:
                reasons.append("Area Terdampak (Tinggi-Sangat Tinggi): 0%")

        scored_texts = []
        for col in row.index:
            col_str = str(col)
            if not col_str.startswith("p_"):
                continue
            score = float(row.get(col, 0) or 0)
            plugin = self._active_plugins.get(col_str)
            if plugin is None:
                short = col_str[2:]
                for pname in plugin_registry.list_plugins(ParameterPlugin):
                    try:
                        cand = plugin_registry.create(pname)
                        if cand.column_key == short or cand.name == short:
                            plugin = cand
                            self._active_plugins[col_str] = cand
                            break
                    except Exception:
                        pass

            text = self._explain_factor(col_str, score, plugin)
            if text:
                is_local = not (
                    col_str in self._uniform_p_cols
                    and score <= self._uniform_p_cols[col_str] + 0.05
                )
                scored_texts.append((1 if is_local else 0, score, text))

        scored_texts.sort(key=lambda item: (item[0], item[1]), reverse=True)
        factor_texts = [item[2] for item in scored_texts]

        if factor_texts:
            reasons.append("Faktor Penyebab: " + " | ".join(factor_texts))
        else:
            dom = row.get('dominant', 'None')
            if dom and dom != 'None':
                reasons.append(f"Terutama dipengaruhi oleh faktor: {dom}")
            else:
                reasons.append("Tidak ada faktor dominan yang teridentifikasi secara signifikan")

        return " || ".join(reasons)

    def _explain_factor(
        self,
        col: str,
        score: float,
        plugin: Optional[ParameterPlugin] = None,
    ) -> Optional[str]:
        if not np.isfinite(score) or score <= self.explain_threshold:
            return None

        short = col[2:] if col.startswith("p_") else col
        pname = plugin.name if plugin is not None else short
        label = plugin.label if plugin is not None else short.replace("_", " ").title()

        template = (
            self.explanations.get(pname)
            or self.explanations.get(short)
            or (plugin.metadata.get("explanation") if plugin is not None else None)
            or self._DEFAULT_EXPLANATIONS.get(pname)
            or self._DEFAULT_EXPLANATIONS.get(short)
        )

        if isinstance(template, dict):
            inverse = getattr(plugin, "inverse", True) if plugin is not None else True
            template = template.get(bool(inverse))

        if template:
            try:
                return template.format(score=score, name=label)
            except Exception:
                return f"{template} ({score:.2f})"

        return f"{label} Tinggi ({score:.2f})"

    def _compute_gvf(self, values: np.ndarray, labels: np.ndarray) -> float:
        vals = np.asarray(values, dtype=float)
        labs = np.asarray(labels)

        mask = np.isfinite(vals)
        if mask.sum() < 2:
            return 0.0

        vals = vals[mask]
        labs = labs[mask]

        sdam = np.sum((vals - vals.mean()) ** 2)
        if sdam == 0:
            return 1.0

        sdcm = 0.0
        for lab in np.unique(labs):
            group = vals[labs == lab]
            if group.size > 0:
                sdcm += np.sum((group - group.mean()) ** 2)

        return float(max(0.0, min(1.0, (sdam - sdcm) / sdam)))

    def _compute_gvf_with_breaks(self, values: np.ndarray, breaks: List[float]) -> float:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size < 2:
            return 0.0

        sdam = np.sum((vals - vals.mean()) ** 2)
        if sdam == 0:
            return 1.0

        sdcm = 0.0
        for i in range(1, len(breaks)):
            if i == len(breaks) - 1:
                class_mask = vals >= breaks[i - 1]
            else:
                class_mask = (vals >= breaks[i - 1]) & (vals < breaks[i])

            class_values = vals[class_mask]
            if class_values.size > 0:
                sdcm += np.sum((class_values - class_values.mean()) ** 2)

        return float(max(0.0, min(1.0, 1.0 - (sdcm / sdam))))

    def _compute_spearman(self, a, b) -> float:
        try:
            sa = pd.Series(a, dtype=float).rank(method='average')
            sb = pd.Series(b, dtype=float).rank(method='average')
            df = pd.concat([sa, sb], axis=1).dropna()
            if df.shape[0] < 2:
                return 0.0
            rho = df.iloc[:, 0].corr(df.iloc[:, 1], method='pearson')
            return float(rho) if pd.notnull(rho) else 0.0
        except Exception:
            return 0.0

    def _add_goodness_and_stability(self, gdf, weights=None):
        if 'composite_risk' not in gdf.columns:
            return

        jenks_breaks = gdf.attrs.get('jenks_breaks')
        if jenks_breaks is not None:
            gvf = self._compute_gvf_with_breaks(gdf['composite_risk'].values, jenks_breaks)
        elif 'severity' in gdf.columns:
            codes = pd.Categorical(gdf['severity']).codes
            gvf = self._compute_gvf(gdf['composite_risk'].values, codes)
        else:
            try:
                q_labels = pd.qcut(
                    gdf['composite_risk'].rank(method='average'),
                    q=len(self.labels), labels=False, duplicates='drop'
                ).values
                gvf = self._compute_gvf(gdf['composite_risk'].values, q_labels)
            except Exception:
                gvf = 0.0

        spearman_risk = (
            self._compute_spearman(gdf['composite_risk'], gdf['risk_score'])
            if 'risk_score' in gdf.columns else 0.0
        )

        factor_cols = [c for c in gdf.columns if str(c).startswith("p_")]
        rho_per_factor = {
            f: self._compute_spearman(gdf['composite_risk'], gdf[f])
            for f in factor_cols
        }
        spearman_factors = (
            float(np.mean([abs(v) for v in rho_per_factor.values()]))
            if rho_per_factor else 0.0
        )

        gdf['gvf'] = round(float(gvf), 4)
        gdf['spearman_risk'] = round(float(spearman_risk), 4)
        gdf['spearman_factors'] = round(float(spearman_factors), 4)

    def _run_threshold_sensitivity(
        self,
        gdf,
        output_dir: Path,
        scenarios: Optional[Dict] = None
    ):
        scenarios = scenarios or self._DEFAULT_THRESHOLD_SCENARIOS
        total_area = gdf['area_ha'].sum() if 'area_ha' in gdf.columns else 0

        results = {}
        for name, thresh in scenarios.items():
            temp_severity = []
            for _, row in gdf.iterrows():
                ep = row.get('pct_extreme', 0)
                vhp = row.get('pct_very_high', 0)
                hp = row.get('pct_high', 0)
                rs = row.get('risk_score', 0)

                if ep > thresh['extreme_pct'] or (ep > 25 and rs >= thresh['extreme_score']):
                    sev = 'Extreme'
                elif vhp > thresh['very_high_pct'] or ep > 15 or rs >= thresh['very_high_score']:
                    sev = 'Very High'
                elif hp > thresh['high_pct'] or vhp > 20 or rs >= thresh['high_score']:
                    sev = 'High'
                elif rs >= 3.5:
                    sev = 'Medium'
                elif rs >= 2.5:
                    sev = 'Low'
                else:
                    sev = 'Very Low'
                temp_severity.append(sev)

            temp_gdf = gdf.copy()
            temp_gdf['temp_severity'] = temp_severity

            extreme_count = int((temp_gdf['temp_severity'] == 'Extreme').sum())
            high_risk_mask = temp_gdf['temp_severity'].isin(['Extreme', 'Very High', 'High'])
            high_risk_count = int(high_risk_mask.sum())
            high_risk_area = (
                float(temp_gdf.loc[high_risk_mask, 'area_ha'].sum())
                if 'area_ha' in temp_gdf.columns else 0.0
            )
            high_risk_pct = (high_risk_area / total_area * 100) if total_area > 0 else 0.0

            results[name] = {
                'extreme_count': extreme_count,
                'high_risk_count': high_risk_count,
                'high_risk_area_ha': round(high_risk_area, 1),
                'high_risk_area_pct': round(high_risk_pct, 1),
            }

        try:
            df_sens = pd.DataFrame.from_dict(results, orient='index')
            df_sens.index.name = 'scenario'
            df_sens.to_csv(output_dir / "sensitivity_thresholds.csv")
        except Exception as e:
            logger.debug(f"      Could not save threshold sensitivity CSV: {e}")

        gdf.attrs['sensitivity_analysis'] = results
        return results

    def _run_weight_sensitivity(
        self,
        gdf,
        weights,
        output_dir: Path,
        perturbation: float = 0.2
    ):
        factors_meta = self._resolve_active_factors(gdf=gdf, weights=weights)
        active_pairs = [
            (fmeta['name'], fmeta['p_col'], fmeta['weight'])
            for fmeta in factors_meta
            if fmeta['p_col'] in gdf.columns
        ]


        baseline_score = sum(
            gdf[col].fillna(0) * w for _, col, w in active_pairs
        )

        sensitivity_results = []
        for focus_key, focus_col, _ in active_pairs:
            perturbed = [
                (col, w * (1.0 + perturbation) if key == focus_key else w)
                for key, col, w in active_pairs
            ]
            total_w = sum(w for _, w in perturbed)
            new_score = sum(
                gdf[col].fillna(0) * (w / total_w if total_w > 0 else 0.0)
                for col, w in perturbed
            )

            rho = self._compute_spearman(baseline_score, new_score)
            sensitivity_results.append({
                'factor': focus_key.replace('_', ' ').title(),
                'stability_index': round(rho, 4),
                'impact': 'Low' if rho >= 0.95 else 'High',
            })

        df_sens = pd.DataFrame(sensitivity_results)
        try:
            df_sens.to_csv(output_dir / "sensitivity_weights.csv", index=False)
        except Exception as e:
            logger.debug(f"      Could not save weight sensitivity CSV: {e}")

        return df_sens


@register_plugin("landslide_explainability")
class LandslideExplainabilityPlugin(VectorExplainabilityPlugin):
    """
    Preset configuration of VectorExplainabilityPlugin (Currently specifically written for my use case)
    """

    def __init__(self, name: str = "landslide_explainability"):
        super().__init__(
            name=name,
            labels=['Low', 'Medium', 'High', 'Very High'],
            output_prefix="landslide_risk",
            title_label="Tingkat Kerentanan Longsor",
            explain_threshold=0.60,
            run_sensitivity=False,
        )
