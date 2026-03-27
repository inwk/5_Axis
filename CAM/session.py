import NXOpen

def create_session(input_file_dir=None):
    """Opens a part and initializes an NX CAM session."""
    session = NXOpen.Session.GetSession()
    work_part = session.Parts.Work
    basePart, partLoadStatus = session.Parts.OpenActiveDisplay(input_file_dir, NXOpen.DisplayPartOption.AllowAdditional)
    partLoadStatus.Dispose()
    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    session.CreateCamSession()
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)
    
    work_part = session.Parts.Work
    cAMSetup = work_part.CreateCamSetup("mill_contour")
    return session, work_part
