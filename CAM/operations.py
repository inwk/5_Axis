import NXOpen
import NXOpen.CAM


def create_cavity_milling(
        session,
        work_part, 
        tool_name, 
        path_type, 
        object_blank, 
        cut_face_list, 
        operation_list, 
        cycle_time_list,
        axial_depth_of_cut,
        tool_orientation=None
):
    """Creates and solves a Cavity Mill operation."""
    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
    method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
    tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
    
    operation = work_part.CAMSetup.CAMOperationCollection.Create(
        nCGroup, method, tool, object_blank, 
        "mill_contour", 
        "CAVITY_MILL", 
        NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue, 
        "CAVITY_MILL"
    )

    cavity_milling_builder = work_part.CAMSetup.CAMOperationCollection.CreateCavityMillingBuilder(operation)
    cavity_milling_builder.FeedsBuilder.FeedCutBuilder.Value = 1500.0
    cavity_milling_builder.FeedsBuilder.SpindleRpmBuilder.Value = 15000.0

    if path_type == "FollowPart":
        cavity_milling_builder.CutPattern.CutPattern = NXOpen.CAM.CutPatternBuilder.Types.FollowPart
    elif path_type == "FollowPeriphery":
        cavity_milling_builder.CutPattern.CutPattern = NXOpen.CAM.CutPatternBuilder.Types.FollowPeriphery
    elif path_type == "ZigZag":
        cavity_milling_builder.CutPattern.CutPattern = NXOpen.CAM.CutPatternBuilder.Types.ZigZag
        
    cavity_milling_builder.CutAreaGeometry.InitializeData(False)
    cavity_milling_builder.CutLevel.GlobalDepthPerCut.DistanceBuilder.Value = axial_depth_of_cut
    cavity_milling_builder.CutLevel.ApplyGlobalDepthPerCut()
    cavity_milling_builder.NonCuttingBuilder.EngageClosedAreaBuilder.MinRampLengthBuilder.Value = 10.0

    if tool_orientation is not None:
        vector = NXOpen.Vector3d(tool_orientation[0], tool_orientation[1], tool_orientation[2])
        origin = NXOpen.Point3d(0.0, 0.0, 0.0)
        direction = work_part.Directions.CreateDirection(origin, vector, NXOpen.SmartObject.UpdateOption.AfterModeling)
        cavity_milling_builder.ToolAxisFix.ToolAxisType = NXOpen.CAM.ToolAxisFixed.Types.Fixed
        cavity_milling_builder.ToolAxisFix.Vector = direction
    
    nXObject = cavity_milling_builder.Commit()
    operation_list.append(nXObject)

    objects = [NXOpen.CAM.CAMObject.Null]
    objects[0] = operation
    work_part.CAMSetup.GenerateToolPath(objects)
    work_part.CAMSetup.Show3dWorkpiece(objects)
    cycle_time_list.append(operation.GetToolpathTime())
    cavity_milling_builder.Destroy()


def create_surface_contour(
        session, 
        work_part, 
        tool_name, 
        object_blank, 
        cut_face_list, 
        operation_list, 
        cycle_time_list, 
        tool_orientation =None
):
    """Creates and solves a Surface Contour operation."""
    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
    method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
    tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
    operation = work_part.CAMSetup.CAMOperationCollection.Create(nCGroup, method, tool, object_blank, "mill_contour", "AREA_MILL", NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue, "AREA_MILL")

    surface_contour_builder = work_part.CAMSetup.CAMOperationCollection.CreateSurfaceContourBuilder(operation)
    surface_contour_builder.DmareaMillingBuilder.SteepCutting.DepthPerCut.StepoverType = NXOpen.CAM.StepoverBuilder.StepoverTypes.Constant
    surface_contour_builder.FeedsBuilder.SpindleRpmBuilder.Value = 15000.0
    surface_contour_builder.FeedsBuilder.FeedCutBuilder.Value = 1500.0
    surface_contour_builder.CutAreaGeometry.InitializeData(False)
    
    geometrySetList = surface_contour_builder.CutAreaGeometry.GeometryList
    geometrySet = geometrySetList.FindItem(0)
    scCollector = geometrySet.ScCollector

    selectionIntentRuleOptions = work_part.ScRuleFactory.CreateRuleOptions()
    selectionIntentRuleOptions.SetSelectedFromInactive(False)
    faceDumbRule = work_part.ScRuleFactory.CreateRuleFaceDumb(cut_face_list, selectionIntentRuleOptions)
    selectionIntentRuleOptions.Dispose()
    scCollector.ReplaceRules([faceDumbRule], False)
    
    if tool_orientation is not None:
        vector = NXOpen.Vector3d(tool_orientation[0], tool_orientation[1], tool_orientation[2])
        origin = NXOpen.Point3d(0.0, 0.0, 0.0)
        direction = work_part.Directions.CreateDirection(origin, vector, NXOpen.SmartObject.UpdateOption.AfterModeling)
        surface_contour_builder.ToolAxisFixed.Vector = direction
        tool_axis_fixed = surface_contour_builder.ToolAxisFixed
        tool_axis_fixed.ToolAxisType = NXOpen.CAM.ToolAxisFixed.Types.Fixed

        tool_axis_fixed.Vector = direction
            
    nXObject = surface_contour_builder.Commit()
    objects = [NXOpen.CAM.CAMObject.Null]
    objects[0] = operation
    work_part.CAMSetup.GenerateToolPath(objects)
    work_part.CAMSetup.Show3dWorkpiece(objects)
    surface_contour_builder.Destroy()
    operation_list.append(nXObject)
    cycle_time_list.append(operation.GetToolpathTime())


def create_3d_adaptive_roughing(
    session,
    work_part,
    tool_name,
    object_blank,
    operation_list,
    cycle_time_list,
    tool_orientation=None,
    feed_cut=1500.0,
    spindle_rpm=15000.0,
    bottom_up_cutting=True,
):
    """Creates and solves a 3D Adaptive Roughing operation."""
    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
    method  = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
    tool    = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)

    operation = work_part.CAMSetup.CAMOperationCollection.Create(
        nCGroup, method, tool, object_blank,
        "mill_contour",
        "3D_ADAPTIVE_ROUGHING",
        NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
        "3D_ADAPTIVE_ROUGHING"
    )

    planar_roughing_builder = work_part.CAMSetup.CAMOperationCollection.CreatePlanarRoughingBuilder(operation)

    planar_roughing_builder.FeedsBuilder.FeedCutBuilder.Value = float(feed_cut)
    planar_roughing_builder.FeedsBuilder.SpindleRpmBuilder.Value = float(spindle_rpm)
    bottom_up_builder = planar_roughing_builder.GetCustomizableItemBuilder("Bottom Up Cutting")
    bottom_up_builder.Value = 1 if bool(bottom_up_cutting) else 0

    if tool_orientation is not None:
        vector = NXOpen.Vector3d(float(tool_orientation[0]), float(tool_orientation[1]), float(tool_orientation[2]))
        origin = NXOpen.Point3d(0.0, 0.0, 0.0)
        direction = work_part.Directions.CreateDirection(origin, vector, NXOpen.SmartObject.UpdateOption.AfterModeling)

        tool_axis_fixed = planar_roughing_builder.GetCustomizableItemBuilder("Tool Axis")
        tool_axis_fixed.ToolAxisType = NXOpen.CAM.ToolAxisFixed.Types.Fixed
        tool_axis_fixed.Vector = direction

    nXObject = planar_roughing_builder.Commit()
    operation_list.append(nXObject)

    objects = [NXOpen.CAM.CAMObject.Null]
    objects[0] = operation
    work_part.CAMSetup.GenerateToolPath(objects)
    work_part.CAMSetup.Show3dWorkpiece(objects)
    cycle_time_list.append(operation.GetToolpathTime())

    planar_roughing_builder.Destroy()
    return nXObject


def create_swarf_milling(
    session,
    work_part,
    tool_name,
    object_blank,
    cut_face_list,
    operation_list,
    cycle_time_list,
    follow_wall_bottom=True,
    feed_cut=1500.0,
    spindle_rpm=15000.0,
):
    """Creates and solves a swarf (Contour Profile) multi-axis finishing operation."""
    if not cut_face_list:
        raise ValueError("create_swarf_milling requires at least one wall face.")

    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
    method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
    tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)

    operation = work_part.CAMSetup.CAMOperationCollection.CreateWithUserName(
        nCGroup,
        method,
        tool,
        object_blank,
        "mill_multi-axis",
        "CONTOUR_PROFILE",
        NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
        "CONTOUR_PROFILE",
        "Contour Profile",
    )

    swarf_builder = work_part.CAMSetup.CAMOperationCollection.CreateSurfaceContourBuilder(operation)
    swarf_builder.FeedsBuilder.FeedCutBuilder.Value = float(feed_cut)
    swarf_builder.FeedsBuilder.SpindleRpmBuilder.Value = float(spindle_rpm)
    swarf_builder.CutParameters.AcrossVoids.MotionType = NXOpen.CAM.AcrossVoids.MotionTypes.Cut
    # Enable holder-aware toolpath and IPW collision checks for safer swarf paths.
    swarf_builder.CutParameters.UseToolHolder = True
    swarf_builder.CutParameters.CheckIpwCollisions = True

    # Swarf uses wall geometry as the driving feature.
    swarf_builder.Geometry.AutoWallSelection = False
    swarf_builder.WallGeometry.InitializeData(False)
    wall_geometry_set_list = swarf_builder.WallGeometry.GeometryList
    wall_geometry_set = wall_geometry_set_list.FindItem(0)
    wall_sc_collector = wall_geometry_set.ScCollector

    selection_options = work_part.ScRuleFactory.CreateRuleOptions()
    selection_options.SetSelectedFromInactive(False)
    face_rule = work_part.ScRuleFactory.CreateRuleFaceDumb(cut_face_list, selection_options)
    selection_options.Dispose()
    wall_sc_collector.ReplaceRules([face_rule], False)

    swarf_builder.DmCmBuilder.FollowWallBottom = bool(follow_wall_bottom)
    swarf_builder.ToolAxisVariable.ToolAxisType = NXOpen.CAM.ToolAxisVariable.Types.SwarfBaseUV

    nXObject = swarf_builder.Commit()
    operation_list.append(nXObject)

    objects = [NXOpen.CAM.CAMObject.Null]
    objects[0] = operation
    work_part.CAMSetup.GenerateToolPath(objects)
    work_part.CAMSetup.Show3dWorkpiece(objects)
    cycle_time_list.append(operation.GetToolpathTime())
    swarf_builder.Destroy()
    return nXObject


def create_point_milling(
    session,
    work_part,
    tool_name,
    object_blank,
    cut_face_list,
    operation_list,
    cycle_time_list,
    scallop_height=0.01,
    feed_cut=1500.0,
    spindle_rpm=15000.0,
):
    """Creates and solves a 5-axis point milling (Variable Contour) operation."""
    if not cut_face_list:
        raise ValueError("create_point_milling requires at least one drive face.")

    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
    method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
    tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)

    operation = work_part.CAMSetup.CAMOperationCollection.CreateWithUserName(
        nCGroup,
        method,
        tool,
        object_blank,
        "mill_multi-axis",
        "VARIABLE_CONTOUR",
        NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
        "VARIABLE_CONTOUR_1",
        "Variable Contour 1",
    )

    point_builder = work_part.CAMSetup.CAMOperationCollection.CreateSurfaceContourBuilder(operation)
    point_builder.FeedsBuilder.FeedCutBuilder.Value = float(feed_cut)
    point_builder.FeedsBuilder.SpindleRpmBuilder.Value = float(spindle_rpm)
    point_builder.CutParameters.UseToolHolder = True
    point_builder.CutParameters.CheckIpwCollisions = True

    # Select cut area faces.
    point_builder.CutAreaGeometry.InitializeData(False)
    cut_set_list = point_builder.CutAreaGeometry.GeometryList
    cut_set = cut_set_list.FindItem(0)
    cut_collector = cut_set.ScCollector
    selection_options = work_part.ScRuleFactory.CreateRuleOptions()
    selection_options.SetSelectedFromInactive(False)
    cut_face_rule = work_part.ScRuleFactory.CreateRuleFaceDumb(cut_face_list, selection_options)
    selection_options.Dispose()
    cut_collector.ReplaceRules([cut_face_rule], False)

    # Journal-equivalent point milling setup.
    point_builder.ProjectionVector.DpmProjType = (
        NXOpen.CAM.ProjVecCiBuilder.DpmProjTypes.ProjVecNormToDrive
    )
    point_builder.DmSurfBuilder.StepoverBuilder.StepoverType = (
        NXOpen.CAM.StepoverBuilder.StepoverTypes.Scallop
    )
    point_builder.DmSurfBuilder.StepoverBuilder.ScallopBuilder.Value = float(scallop_height)

    # Use one representative drive face, consistent with the recorded journal.
    drive_geom = point_builder.DmSurfBuilder.DriveGeometry
    drive_set_list = drive_geom.GeometryList
    drive_set = drive_set_list.FindItem(0)
    drive_set.Surface = cut_face_list[0]
    drive_geom.Validate()
    drive_geom.Commit()

    nXObject = point_builder.Commit()
    operation_list.append(nXObject)

    objects = [NXOpen.CAM.CAMObject.Null]
    objects[0] = operation
    work_part.CAMSetup.GenerateToolPath(objects)
    work_part.CAMSetup.Show3dWorkpiece(objects)
    cycle_time_list.append(operation.GetToolpathTime())
    point_builder.Destroy()
    return nXObject


# Backward-compatible aliases for older scripts.
create_cavityMilling = create_cavity_milling
create_surfaceContour = create_surface_contour
create_3dAdaptiveRoughing = create_3d_adaptive_roughing
create_swarfMilling = create_swarf_milling
create_pointMilling = create_point_milling
