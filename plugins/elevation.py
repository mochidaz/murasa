from murasa.core.base import RasterParameterPlugin, register_plugin


@register_plugin("elevation")
class ElevationPlugin(RasterParameterPlugin):
    """
    Elevation-based risk parameter.
    When inverse=True, lower elevation produces higher risk. Floods for example.
    When inverse=False, higher elevation produces higher risk. Landslides for example.
    """

    source_keys = ['dem', 'elevation']
    inverse = True
    short_name = "elevation"
    contrib_name = "elev"
    display_name = "Elevation"
