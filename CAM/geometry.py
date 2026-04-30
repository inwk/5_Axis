import NXOpen
import NXOpen.CAM


def _safe_destroy(builder):
    """Best-effort NX builder cleanup without masking the original failure."""
    if builder is None:
        return
    try:
        builder.Destroy()
    except Exception:
        pass


def create_geometry(session, work_part, input_file_dir, workpiece_name_list, origin_body, add_list=True, fixed_stock_size = False):
    """Creates or updates NX workpiece geometry for CAM operations."""
    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    if len(workpiece_name_list) == 0:
        featureGeometry = work_part.CAMSetup.CAMGroupCollection.FindObject("WORKPIECE")
        if add_list:
            workpiece_name_list.append("WORKPIECE")
    else:
        workpiece_name = f"workpiece_{len(workpiece_name_list)}"
        orientGeometry = work_part.CAMSetup.CAMGroupCollection.FindObject("MCS_MILL")
        featureGeometry = work_part.CAMSetup.CAMGroupCollection.CreateGeometry(orientGeometry, "mill_contour", "WORKPIECE", NXOpen.CAM.NCGroupCollection.UseDefaultName.TrueValue, workpiece_name)
        if add_list:
            workpiece_name_list.append(featureGeometry.Name)

    millGeomBuilder = work_part.CAMSetup.CAMGroupCollection.CreateMillGeomBuilder(featureGeometry)
    try:
        millGeomBuilder.PartGeometry.InitializeData(False)

        geometrySetList = millGeomBuilder.PartGeometry.GeometryList
        taggedObject = geometrySetList.FindItem(0)
        millGeomBuilder.BlankGeometry.ResetBlockSize()
        selectionIntentRuleOptions = work_part.ScRuleFactory.CreateRuleOptions()
        bodyList = [origin_body]
        try:
            bodyDumbRule = work_part.ScRuleFactory.CreateRuleBodyDumb(bodyList, True, selectionIntentRuleOptions)
        finally:
            selectionIntentRuleOptions.Dispose()
        scCollector = taggedObject.ScCollector
        scCollector.ReplaceRules([bodyDumbRule], False)
        nXObject = millGeomBuilder.Commit()
    finally:
        _safe_destroy(millGeomBuilder)

    featureGeometryBlank = nXObject
    millGeomBuilderBlank = work_part.CAMSetup.CAMGroupCollection.CreateMillGeomBuilder(featureGeometryBlank)
    try:
        origin = NXOpen.Point3d(0.0, 0.0, 0.0)
        xDirection = NXOpen.Vector3d(1.0, 0.0, 0.0)
        yDirection = NXOpen.Vector3d(0.0, 1.0, 0.0)
        cartesianCoordinateSystem = work_part.CoordinateSystems.CreateCoordinateSystem(origin, xDirection, yDirection)

        if len(workpiece_name_list) == 1:
            millGeomBuilderBlank.BlankGeometry.BlankDefinitionType = NXOpen.CAM.GeometryGroup.BlankDefinitionTypes.AutoBlock
            millGeomBuilderBlank.BlankGeometry.OrientationType = NXOpen.CAM.GeometryGroup.OrientationTypes.Specify
            millGeomBuilderBlank.BlankGeometry.Csys = cartesianCoordinateSystem
            millGeomBuilderBlank.BlankGeometry.BlankToggleValue = False
            millGeomBuilderBlank.BlankGeometry.IpwPositionType = NXOpen.CAM.GeometryGroup.PositionTypes.Coordinate
            millGeomBuilderBlank.BlankGeometry.IpwPositionCsys = cartesianCoordinateSystem
            if (fixed_stock_size):
                millGeomBuilderBlank.BlankGeometry.BlockLength = 150.0
                millGeomBuilderBlank.BlankGeometry.BlockWidth = 150.0
                millGeomBuilderBlank.BlankGeometry.BlockHeight = 150.0

        else:
            blankIpwSetList = millGeomBuilderBlank.BlankGeometry.BlankIpwMultipleSource.SetList
            blankIpwSet = blankIpwSetList.FindItem(0)
            millGeomBuilderBlank.BlankGeometry.BlankDefinitionType = NXOpen.CAM.GeometryGroup.BlankDefinitionTypes.Ipw
            if add_list:
                blankIpwSet.SetSource(input_file_dir, workpiece_name_list[-2])
            else:
                blankIpwSet.SetSource(input_file_dir, workpiece_name_list[-1])
            blankIpwSet.IpwPositionCsys = cartesianCoordinateSystem
            blankIpwSet.Update()
            millGeomBuilderBlank.BlankGeometry.IpwPositionType = NXOpen.CAM.GeometryGroup.PositionTypes.Coordinate
            millGeomBuilderBlank.BlankGeometry.IpwPositionCsys = cartesianCoordinateSystem
        nXObjectBlank = millGeomBuilderBlank.Commit()
        target_size = [millGeomBuilderBlank.BlankGeometry.BlockHeight,millGeomBuilderBlank.BlankGeometry.BlockLength,millGeomBuilderBlank.BlankGeometry.BlockWidth]
    finally:
        _safe_destroy(millGeomBuilderBlank)

    return nXObjectBlank, bodyList[0],target_size

def get_block_size(session, body):
    """Computes automatic block stock dimensions and body centroid."""
    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "start")

    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    featureGeometry = work_part.CAMSetup.CAMGroupCollection.FindObject("WORKPIECE")

    millGeomBuilder = work_part.CAMSetup.CAMGroupCollection.CreateMillGeomBuilder(featureGeometry)
    try:
        millGeomBuilder.PartGeometry.InitializeData(False)

        geometrySetList = millGeomBuilder.PartGeometry.GeometryList
        taggedObject = geometrySetList.FindItem(0)

        selectionIntentRuleOptions = work_part.ScRuleFactory.CreateRuleOptions()
        bodyList = [body]
        try:
            bodyDumbRule = work_part.ScRuleFactory.CreateRuleBodyDumb(bodyList, True, selectionIntentRuleOptions)
        finally:
            selectionIntentRuleOptions.Dispose()
        scCollector = taggedObject.ScCollector
        scCollector.ReplaceRules([bodyDumbRule], False)

        nXObject = millGeomBuilder.Commit()
    finally:
        _safe_destroy(millGeomBuilder)

    featureGeometryBlank = nXObject
    millGeomBuilderBlank = work_part.CAMSetup.CAMGroupCollection.CreateMillGeomBuilder(featureGeometryBlank)
    try:
        origin = NXOpen.Point3d(0.0, 0.0, 0.0)
        xDirection = NXOpen.Vector3d(1.0, 0.0, 0.0)
        yDirection = NXOpen.Vector3d(0.0, 1.0, 0.0)
        cartesianCoordinateSystem = work_part.CoordinateSystems.CreateCoordinateSystem(origin, xDirection, yDirection)

        millGeomBuilderBlank.BlankGeometry.BlankDefinitionType = NXOpen.CAM.GeometryGroup.BlankDefinitionTypes.AutoBlock
        millGeomBuilderBlank.BlankGeometry.OrientationType = NXOpen.CAM.GeometryGroup.OrientationTypes.Specify
        millGeomBuilderBlank.BlankGeometry.Csys = cartesianCoordinateSystem
        millGeomBuilderBlank.BlankGeometry.BlankToggleValue = False
        millGeomBuilderBlank.BlankGeometry.IpwPositionType = NXOpen.CAM.GeometryGroup.PositionTypes.Coordinate
        millGeomBuilderBlank.BlankGeometry.IpwPositionCsys = cartesianCoordinateSystem

        target_size = [millGeomBuilderBlank.BlankGeometry.BlockHeight,millGeomBuilderBlank.BlankGeometry.BlockLength,millGeomBuilderBlank.BlankGeometry.BlockWidth]
        nXObjectBlank = millGeomBuilderBlank.Commit()
        a = work_part.Bodies
        bodylist = [body for body in a]
        _, _, centroid1, _, _, _ = session.Measurement.GetBodyProperties([bodylist[1]], 0.98999999999999999, False)
        cen_x = centroid1.X
        cen_y = centroid1.Y
        cen_z = centroid1.Z
    finally:
        _safe_destroy(millGeomBuilderBlank)
        undo_error = None
        try:
            session.UndoToMark(markId,"init")
        except Exception as exc:
            undo_error = exc
        try:
            session.DeleteUndoMark(markId,"init")
        except Exception:
            pass
        if undo_error is not None:
            raise undo_error

    return target_size, [cen_x, cen_y, cen_z]


# Backward-compatible alias for older scripts.
getBlockSize = get_block_size
