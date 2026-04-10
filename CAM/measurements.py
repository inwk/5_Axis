import NXOpen
import NXOpen.UF
import numpy as np
import math
# import pyransac3d as pyrsc
def calculate_distance_to_body(face, body):
    """Performs: calculate distance to body."""
    theSession = NXOpen.Session.GetSession()
    work_part = theSession.Parts.Work
    measure_manager = work_part.MeasureManager
    unit = theSession.Parts.Work.UnitCollection.FindObject("MilliMeter")
    measure_type = NXOpen.MeasureManager.MeasureType.Minimum
    distance = measure_manager.NewDistance(unit, measure_type, face, body)
    return float(distance.Value)


def create_line_with_point_and_vector(workPart, start_point, direction_vector, length=100.0):
    #start_point_xyz = NXOpen.Point3d(*start_point)  # start_point: (x, y, z)
    """Performs: create line with point and vector."""
    start_point_xyz = NXOpen.Point3d(
        start_point[0] - direction_vector.Vector.X * 0.1,
        start_point[1] - direction_vector.Vector.Y * 0.1,
        start_point[2] - direction_vector.Vector.Z * 0.1
    )   
    end_point_xyz = NXOpen.Point3d(
        start_point[0] + direction_vector.Vector.X * length,
        start_point[1] + direction_vector.Vector.Y * length,
        start_point[2] + direction_vector.Vector.Z * length
    )
    
    line = workPart.Curves.CreateLine(start_point_xyz, end_point_xyz)
    
    return line

def calculate_distance_to_body_from_point(point, line, body, norm_vecs):
    """Performs: calculate distance to body from point."""
    theSession = NXOpen.Session.GetSession()
    work_part = theSession.Parts.Work
    #point3d = point.Coordinates
    #nx_point = work_part.Points.CreatePoint(point3d)
    #measure_manager = work_part.MeasureManager
    distance1, pt1_1, pt2_1, isapproximate1 = theSession.Measurement.GetDistance(NXOpen.Measurement.AlternateDistance.Minimum, False, [line], [body])
    distance2, pt3_1, pt4_1, isapproximate2 = theSession.Measurement.GetDistance(NXOpen.Measurement.AlternateDistance.Minimum, False, [point], [body])
    pt = [point.Coordinates.X - norm_vecs.Vector.X * 0.1,point.Coordinates.Y- norm_vecs.Vector.Y * 0.1,point.Coordinates.Z - norm_vecs.Vector.Z * 0.1]  # small offset to stabilize roughing measurements
    vec2 = np.subtract([pt4_1.X,pt4_1.Y, pt4_1.Z],pt)
    vec1 = [norm_vecs.Vector.X,norm_vecs.Vector.Y,norm_vecs.Vector.Z]
    if np.dot(vec1, vec2)<0:
        dist = math.sqrt((point.Coordinates.X-pt1_1.X)**2 + (point.Coordinates.Y-pt1_1.Y)**2 + (point.Coordinates.Z-pt1_1.Z)**2)
    else:
        dist = distance2

    #unit = theSession.Parts.Work.UnitCollection.FindObject("MilliMeter")
    #measure_type = NXOpen.MeasureManager.MeasureType.Minimum
    
    #direction = work_part.Directions.CreateDirection(NXOpen.Point3d(0.0, 0.0, 0.0), NXOpen.Vector3d(float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2])), NXOpen.SmartObject.UpdateOption.DontUpdate)
    # dist = theSession.Measurement.GetProjectedDistanceProperties([point], [body], norm_vec, unit, False, 0)[0]
    # if dist<0:
    #     dist = theSession.Measurement.GetProjectedDistanceProperties([point], [body], norm_vec, unit, False, 5)[0]
    #distance = measure_manager.NewDistance(unit, measure_type, nx_point, body)
    return float(dist)

def calculate_distance_to_face(face, point):
    """Performs: calculate distance to face."""
    theSession = NXOpen.Session.GetSession()
    work_part = theSession.Parts.Work
    point3d = point.Coordinates
    point = work_part.Points.CreatePoint(point3d)
    measure_manager = work_part.MeasureManager
    unit = theSession.Parts.Work.UnitCollection.FindObject("MilliMeter")
    measure_type = NXOpen.MeasureManager.MeasureType.Minimum
    distance = measure_manager.NewDistance(unit, measure_type, face, point)
    return float(distance.Value)

def get_deviation_per_face(ipw, points_array, norm_vecs_array, lines_array):
    """Performs: get deviation per face."""
    deviation_per_face = []
    objects = convert_facet_to_body(ipw)
    for i in range(len(points_array)):
        dist_list = []
        points = points_array[i]
        norm_vecs = norm_vecs_array[i]
        lines = lines_array[i]
        for j in range(len(points)): 
            distance = calculate_distance_to_body_from_point(points[j], lines[j], objects[0],norm_vecs[j])
            dist_list.append(round(distance, 6))
            # if dist_max<distance:
            #     dist_max = distance
           # dist_sum += distance
            #cnt += 1
        num_ = min(30,len(dist_list))
        sorted_numbers = sorted(dist_list, reverse=True)
        top_10 = sorted_numbers[:num_]
        average = sum(top_10) / num_
        deviation_per_face.append(average)
    return deviation_per_face, objects[0]
        
        
        
    for face in origin_faces:
        if face.SolidFaceType.value==10:
            points,norm_vecs = generate_points_convergent_face(face)
        else:
            points,norm_vecs = generate_points(face)
        #dist_sum = 0.0
        #dist_max = 0.0
        dist_list = []
        #cnt = 0
        for i in range(len(points)): 
            distance = calculate_distance_to_body_from_point(points[i], objects[0], norm_vecs[i])
            dist_list.append(round(distance, 6))
            # if dist_max<distance:
            #     dist_max = distance
           # dist_sum += distance
            #cnt += 1
        num_ = min(30,len(dist_list))
        sorted_numbers = sorted(dist_list, reverse=True)
        top_10 = sorted_numbers[:num_]
        average = sum(top_10) / num_
        deviation_per_face.append(average)
    return deviation_per_face, objects[0]

def generate_points_convergent_face(face):
    """Performs: generate points convergent face."""
    session = NXOpen.Session.GetSession()
    work_part = session.Parts.Work
    num_points = 500
    num_facets = face.GetNumberOfFacets()
    points = []
    norm_vecs = []
    facet = face.GetFirstFacetOnFace()
    if num_facets > 1:
        for _ in range(num_facets):
            facet = face.GetNextFacet(facet)
            if facet == None:
                break
            vertices = facet.GetVertices()
            vertex = vertices[0] # Get only first point
            point = (vertex.X, vertex.Y, vertex.Z)  # Convert point to a tuple
            vec = NXOpen.ConvergentFacet.GetUnitNormal(facet)
            norm_vecs.append(work_part.Directions.CreateDirection(NXOpen.Point3d(0.0, 0.0, 0.0), NXOpen.Vector3d(float(vec.X),float(vec.Y),float(vec.Z)), NXOpen.SmartObject.UpdateOption.DontUpdate))
            points.append([vertex.X, vertex.Y, vertex.Z])  # Add point to list if unique
    else:
        vertices = facet.GetVertices()
        vertex = vertices[0] # Get only first point
        points.append([vertex.X, vertex.Y, vertex.Z])  # Add point to list if unique
        vec = NXOpen.ConvergentFacet.GetUnitNormal(facet)
        norm_vecs.append(work_part.Directions.CreateDirection(NXOpen.Point3d(0.0, 0.0, 0.0), NXOpen.Vector3d(float(vec.X),float(vec.Y),float(vec.Z)), NXOpen.SmartObject.UpdateOption.DontUpdate))

    norm_vecs = np.array(norm_vecs)
    points_array = np.array(points)
    
    n = points_array.shape[0]
    if n > num_points:
        indices = np.random.choice(n, num_points, replace=False)  # random sample to num_points
        sample_points = points_array[indices]
        sample_vecs = norm_vecs[indices]
    else:
        sample_points = points_array
        sample_vecs = norm_vecs
    points = []
    lines = []
    for i in range(len(sample_points)):
        point_3d = NXOpen.Point3d(float(sample_points[i][0]), float(sample_points[i][1]), float(sample_points[i][2]))
        point_feature = work_part.Points.CreatePoint(point_3d)
        point_feature.SetVisibility(NXOpen.SmartObject.VisibilityOption.Visible)
        points.append(point_feature)
        lines.append(create_line_with_point_and_vector(work_part, [float(sample_points[i][0]), float(sample_points[i][1]), float(sample_points[i][2])], norm_vecs[i], length=1000.0))     
    return points, sample_vecs, lines
    
def generate_points(face):
    """Performs: generate points."""
    session = NXOpen.Session.GetSession()
    work_part = session.Parts.Work
    theUfSession = NXOpen.UF.UFSession.GetUFSession()
    uvminmax=theUfSession.Modeling.AskFaceUvMinmax(face.Tag)
    
    theUfSession.Modeling.AskAdjacFaces(face.Tag)
    list_u = np.linspace(uvminmax[0], uvminmax[1], 7).tolist()
    list_v = np.linspace(uvminmax[2], uvminmax[3], 7).tolist()
    points=[]
    norm_vecs=[]
    lines = []
    for u in list_u:
        for v in list_v:
            point, _, _, _, _, unit_norm, _ = theUfSession.Modeling.AskFaceProps(face.Tag,[u,v])
            point_3d = NXOpen.Point3d(point[0], point[1], point[2])
            point_feature = work_part.Points.CreatePoint(point_3d)
            point_feature.SetVisibility(NXOpen.SmartObject.VisibilityOption.Visible)
            points.append(point_feature)
            direction = work_part.Directions.CreateDirection(NXOpen.Point3d(0.0, 0.0, 0.0), NXOpen.Vector3d(float(unit_norm[0]),float(unit_norm[1]),float(unit_norm[2])), NXOpen.SmartObject.UpdateOption.DontUpdate)
            norm_vecs.append(direction)
            lines.append(create_line_with_point_and_vector(work_part, point, direction, length=1000.0))     

    min_dist = 1000
    min_dist_pt = [0,0,0]
    min_dist_vec = [0,0,0]
    min_dist_line = None
    for i in range(len(points) - 1, -1, -1):
         distance = calculate_distance_to_face(face, points[i])
         if min_dist>distance:
             min_dist = distance
             min_dist_pt = points[i].Coordinates
             min_dist_vec = norm_vecs[i]
             min_dist_line = lines[i]
         if distance > 0.1:
             work_part.Points.DeletePoint(points[i])
             del points[i]
             del norm_vecs[i]
             del lines[i]
    if len(points)==0:
        point_3d = NXOpen.Point3d(min_dist_pt.X, min_dist_pt.Y, min_dist_pt.Z)
        point_feature = work_part.Points.CreatePoint(point_3d)
        point_feature.SetVisibility(NXOpen.SmartObject.VisibilityOption.Visible)
        points.append(point_feature)
        norm_vecs.append(min_dist_vec)
        lines.append(min_dist_line)
    return points, norm_vecs, lines 
def sampling(points, num_points):
    """Performs: sampling."""
    n = points.shape[0]
    if n <= num_points:
        return points  # already small enough
    else:
        indices = np.random.choice(n, num_points, replace=False)  # random sample to num_points
        return points[indices]
    
    
def convert_facet_to_body(body, option=2):
    """Performs: convert facet to body."""
    theSession = NXOpen.Session.GetSession()
    work_part = theSession.Parts.Work
    convertFacetBodyBuilder = work_part.FacetedBodies.FacetModelingCollection.CreateConvertFacetBodyBuilder()
    if option == 0:
        convertFacetBodyBuilder.OriginalBodyOption = NXOpen.Facet.ConvertFacetBodyBuilder.OriginalBodyOptions.Keep
    elif option == 1:
        convertFacetBodyBuilder.OriginalBodyOption = NXOpen.Facet.ConvertFacetBodyBuilder.OriginalBodyOptions.Hide
    else:       
        convertFacetBodyBuilder.OriginalBodyOption = NXOpen.Facet.ConvertFacetBodyBuilder.OriginalBodyOptions.Delete
    convertFacetBodyBuilder.OutputType = NXOpen.Facet.ConvertFacetBodyBuilder.OutputTypes.ConvergentBody
    convertFacetBodyBuilder.FacetedBodiesToConvert.Add(body)
    nXObject = convertFacetBodyBuilder.Commit()
    objects = convertFacetBodyBuilder.GetCommittedObjects()
    #convertFacetBodyBuilder.Destroy()
    return objects
# def classify_shape(points, plane_thresh=0.01, sphere_thresh=0.05, cylinder_thresh=0.05):

#     # Fit a plane and count inliers
#     plane_model = pyrsc.Plane()
#     _, plane_inliers = plane_model.fit(points, plane_thresh)
#     num_plane_inliers = len(plane_inliers)
    
#     if num_plane_inliers > points.shape[0] / 2:
#         return 1  # Planar
    

#     # Fit a cylinder and count inliers
#     cylinder_model = pyrsc.Cylinder()
#     _, _, _, cylinder_inliers = cylinder_model.fit(points, cylinder_thresh, maxIteration=100)
#     num_cylinder_inliers = len(cylinder_inliers)
    
#         return 2  # Cylindrical
    
    
#     # Fit a sphere and count inliers
#     sphere_model = pyrsc.Sphere()
#     _, _, sphere_inliers = sphere_model.fit(points, sphere_thresh, maxIteration=100)
#     num_sphere_inliers = len(sphere_inliers)
    
#         return 4  # Spherical

#     return 5
# def getConvergentFaceInfo(face):
#     num_points = 500
#     num_facets = face.GetNumberOfFacets()
#     points = []
#     unique_points = set()  # Set to track unique points as tuples
#     facet = face.GetFirstFacetOnFace()
#     vector_sum = np.array([0,0,0])
#     if num_facets > 1:
#         for _ in range(num_facets):
#             try:
#                 facet = face.GetNextFacet(facet)
#                 vertices = facet.GetVertices()
#                 vertex = vertices[0] # Get only first point
#                 point = (vertex.X, vertex.Y, vertex.Z)  # Convert point to a tuple
#                 vec = NXOpen.ConvergentFacet.GetUnitNormal(facet)
#                 vector_sum = vector_sum + np.array([vec.X,vec.Y,vec.Z])
#                 if point not in unique_points:  # Check for duplicates
#                     unique_points.add(point)   # Add point to the set
#                     points.append([vertex.X, vertex.Y, vertex.Z])  # Add point to list if unique
#             except:
#                 pass
#     else :
#         vertices = facet.GetVertices()
#         vertex = vertices[0] # Get only first point
#         point = (vertex.X, vertex.Y, vertex.Z)  # Convert point to a tuple
#         vec = NXOpen.ConvergentFacet.GetUnitNormal(facet)
#         vector_sum = vector_sum + np.array([vec.X,vec.Y,vec.Z])
#         points.append([vertex.X, vertex.Y, vertex.Z])  # Add point to list if unique

#     face_vector = vector_sum / np.linalg.norm(vector_sum)
#     points_array = np.array(points)

#     sample_points = sampling(points_array, num_points)
#     # s = time.time()
#     try:
#         face_type = classify_shape(sample_points)
#     except:

#     return face_vector, face_type
def getFaceVector(face_tag):
    """Performs: get face vector."""
    theUfSession = NXOpen.UF.UFSession.GetUFSession()
    
    uvminmax=theUfSession.Modeling.AskFaceUvMinmax(face_tag)
    
    theUfSession.Modeling.AskAdjacFaces(face_tag)
    list_u = np.linspace(uvminmax[0], uvminmax[1], 6).tolist()
    list_v = np.linspace(uvminmax[2], uvminmax[3], 6).tolist()
    accumulated_normals = [0.0, 0.0, 0.0]
    for u in list_u:
        for v in list_v:
            _, _, _, _, _, unit_norm, _ = theUfSession.Modeling.AskFaceProps(face_tag,[u,v])
            accumulated_normals[0] += unit_norm[0]
            accumulated_normals[1] += unit_norm[1]
            accumulated_normals[2] += unit_norm[2]
            
    magnitude = math.sqrt(accumulated_normals[0]**2 + accumulated_normals[1]**2 + accumulated_normals[2]**2)
    if magnitude != 0:
        normalized_normals = [accumulated_normals[0] / magnitude, accumulated_normals[1] / magnitude, accumulated_normals[2] / magnitude]
    else:
        normalized_normals = [0.0, 0.0, 0.0]
    return normalized_normals

def getFaceArea(face):
    """Performs: get face area."""
    the_session = NXOpen.Session.GetSession()
    area, _, _, _, _, _, _, _ = the_session.Measurement.GetFaceProperties([face], 0.98999999999999999, NXOpen.Measurement.AlternateFace.Radius, True)
    return area    

def getVolume(body):
    """Performs: get volume."""
    theUfSession = NXOpen.UF.UFSession.GetUFSession()
    theBodyTag = [body.Tag]
    (massProps, Stats) = theUfSession.Modeling.AskMassProps3d(theBodyTag, len(theBodyTag), 1, 3, .03, 1, [0.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    volume = massProps[1]
    return volume


def generate_points_v2(face):
    """Performs: generate points v2."""
    session = NXOpen.Session.GetSession()
    work_part = session.Parts.Work
    theUfSession = NXOpen.UF.UFSession.GetUFSession()
    uvminmax=theUfSession.Modeling.AskFaceUvMinmax(face.Tag)
    theUfSession.Modeling.AskAdjacFaces(face.Tag)
    list_u = np.linspace(uvminmax[0], uvminmax[1], 20).tolist()
    list_v = np.linspace(uvminmax[2], uvminmax[3], 20).tolist()
    points_array=[]
    for edge in face.GetEdges():
        for vertex in edge.GetVertices():
            points_array.append([vertex.X,vertex.Y,vertex.Z])
    for u in list_u:
        for v in list_v:
            point, _, _, _, _, unit_norm, _ = theUfSession.Modeling.AskFaceProps(face.Tag,[u,v])
            point_3d = NXOpen.Point3d(point[0], point[1], point[2])
            point_feature = work_part.Points.CreatePoint(point_3d)
            point_feature.SetVisibility(NXOpen.SmartObject.VisibilityOption.Visible)
            distance = calculate_distance_to_face(face, point_feature)
            if distance < 0.1:
                points_array.append(point)
    if len(points_array)==0:
        print("error")
    return points_array
def get_pointwise_deviation_per_face(ipw, points_array, norm_vecs_array, lines_array):
    """Returns per-face pointwise distances from sampled points to the current IPW body.

    This function reuses the existing geometric distance computation logic but keeps
    raw point-level distances instead of reducing them to a face-level scalar.
    """

    pointwise_per_face = []
    objects = convert_facet_to_body(ipw)
    ipw_body = objects[0]

    for face_points, face_norms, face_lines in zip(points_array, norm_vecs_array, lines_array):
        count = min(len(face_points), len(face_norms), len(face_lines))
        values = np.zeros((count,), dtype=np.float32)

        for idx in range(count):
            dist = calculate_distance_to_body_from_point(
                face_points[idx],
                face_lines[idx],
                ipw_body,
                face_norms[idx],
            )
            values[idx] = float(dist)

        pointwise_per_face.append(values)

    return pointwise_per_face, ipw_body


def get_body_volume(body):
    """Returns body volume in cubic centimeters."""
    return getVolume(body)


def sample_convergent_face_points(face):
    """Samples points, normals, and probe lines on a convergent face."""
    return generate_points_convergent_face(face)


def sample_face_points(face):
    """Samples points, normals, and probe lines on a regular face."""
    return generate_points(face)
