# VRAD-RTX Test Manifest
# Each entry defines a test case for the three-way validation harness.
#
# Keys:
#   MapName             - BSP filename (without extension) in bsp_unit_tests\
#   ExtraArgs           - Additional vrad arguments (array of strings)
#   TimeoutMultiplier   - GPU timeout = ControlTime * Multiplier (+ extension)
#   RefCtrlTolerance    - Max visual diff % for Reference vs Control
#   CtrlTestTolerance   - Max visual diff % for Control vs Test (GPU)
#   LightmapThreshold   - bsp_diff_lightmaps.py --threshold value
#   ArchiveSuffix       - Subfolder suffix in bsp_unit_test_logs\
#   Groups              - Array of group tags for -Group selection (optional)

@{
    full                   = @{
        MapName           = "validation"
        ExtraArgs         = @("-avx2")
        TimeoutMultiplier = 1.5
        RefCtrlTolerance  = 1.0
        CtrlTestTolerance = 15.0
        LightmapThreshold = 0.2
        ArchiveSuffix     = "full"
        Groups            = @("core")
    }
    quick                  = @{
        MapName           = "validation_vrad_quick"
        ExtraArgs         = @("-avx2")
        TimeoutMultiplier = 2.5
        RefCtrlTolerance  = 1.0
        CtrlTestTolerance = 8.0
        LightmapThreshold = 0.1
        ArchiveSuffix     = "quick"
        Groups            = @("core")
    }
    props                  = @{
        MapName           = "validation_props"
        ExtraArgs         = @("-largeDispSampleRadius", "-textureshadows", "-StaticPropPolys", "-StaticPropLighting", "-final", "-lights", "E:\lights_custom.rad")
        TimeoutMultiplier = 2.0
        RefCtrlTolerance  = 0.5
        CtrlTestTolerance = 0.5
        LightmapThreshold = 0.1
        ArchiveSuffix     = "props"
        Groups            = @("props")
    }
    radiosity              = @{
        MapName           = "validation_radiosity"
        ExtraArgs         = @()
        TimeoutMultiplier = 1.5
        RefCtrlTolerance  = 0.5
        CtrlTestTolerance = 0.5
        LightmapThreshold = 0.1
        ArchiveSuffix     = "radiosity"
        Groups            = @("core")
    }
    supersampling_indoors  = @{
        MapName           = "validation_vrad_supersampling_indoors"
        ExtraArgs         = @()
        TimeoutMultiplier = 3.0
        RefCtrlTolerance  = 1.0
        CtrlTestTolerance = 22.0
        LightmapThreshold = 0.1
        ArchiveSuffix     = "supersampling_indoors"
        Groups            = @("supersampling")
    }
    supersampling_outdoors = @{
        MapName           = "validation_vrad_supersampling_outdoors"
        ExtraArgs         = @()
        TimeoutMultiplier = 3.0
        RefCtrlTolerance  = 3
        CtrlTestTolerance = 5
        LightmapThreshold = 0.1
        ArchiveSuffix     = "supersampling_outdoors"
        Groups            = @("supersampling")
    }
    props_textureshadows = @{
        MapName           = "validation_vrad_proptextureshadows"
        ExtraArgs         = @("-textureshadows", "-StaticPropPolys", "-StaticPropLighting", "-lights", "E:\lights_custom.rad")
        TimeoutMultiplier = 3.0
        RefCtrlTolerance  = 3
        CtrlTestTolerance = 5
        LightmapThreshold = 0.1
        ArchiveSuffix     = "props_textureshadows"
        Groups            = @("props")
    }
    named_lights = @{
        MapName           = "validation_vrad_namedlights"
        ExtraArgs         = @()
        TimeoutMultiplier = 3.0
        RefCtrlTolerance  = 3
        CtrlTestTolerance = 5
        LightmapThreshold = 0.1
        ArchiveSuffix     = "named_lights"
        Groups            = @("props")
    }
}
