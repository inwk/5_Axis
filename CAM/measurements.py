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


# Adaptive octree occupancy sampling.
def _is_point_inside_body_uf(body, x: float, y: float, z: float) -> bool:
    """Returns True when the point (x, y, z) is inside or on the body surface.

    Uses ``NXOpen.UF.Modeling.AskPointContainment``:
        1 = point is inside the body
        2 = point is outside the body
        3 = point is on the body

    Occupancy is defined as inside or on-boundary material.
    """
    uf = NXOpen.UF.UFSession.GetUFSession()
    status = int(uf.Modeling.AskPointContainment([float(x), float(y), float(z)], body.Tag))
    if status not in (1, 2, 3):
        raise RuntimeError(f"Unexpected AskPointContainment status: {status}")
    return status in (1, 3)


def get_body_axis_aligned_bbox(body) -> "tuple[np.ndarray, np.ndarray]":
    """Returns an axis-aligned bounding box by sampling body face geometry."""
    uf = NXOpen.UF.UFSession.GetUFSession()
    points = []

    for face in body.GetFaces():
        try:
            for edge in face.GetEdges():
                for vertex in edge.GetVertices():
                    points.append([float(vertex.X), float(vertex.Y), float(vertex.Z)])
        except Exception:
            pass

        try:
            if hasattr(face, "GetNumberOfFacets"):
                facet = face.GetFirstFacetOnFace()
                for _ in range(int(face.GetNumberOfFacets())):
                    if facet is None:
                        break
                    for vertex in facet.GetVertices():
                        points.append([float(vertex.X), float(vertex.Y), float(vertex.Z)])
                    facet = face.GetNextFacet(facet)
        except Exception:
            pass

        try:
            uv = uf.Modeling.AskFaceUvMinmax(face.Tag)
            us = np.linspace(float(uv[0]), float(uv[1]), 4)
            vs = np.linspace(float(uv[2]), float(uv[3]), 4)
            for u in us:
                for v in vs:
                    point, _, _, _, _, _, _ = uf.Modeling.AskFaceProps(face.Tag, [float(u), float(v)])
                    points.append([float(point[0]), float(point[1]), float(point[2])])
        except Exception:
            pass

    if not points:
        raise RuntimeError("Could not sample any body points for bbox")

    arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    bbox_min = arr.min(axis=0)
    bbox_max = arr.max(axis=0)
    return bbox_min.astype(np.float32), bbox_max.astype(np.float32)


def sample_octree_occupancy(
    body,
    bbox_min: "np.ndarray",
    bbox_max: "np.ndarray",
    coarse_depth: int = 3,
    fine_depth: int = 5,
    max_nodes: int = 4096,
    rng: "np.random.Generator | None" = None,
    contains_points_fn=None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Samples adaptive octree leaf centers and occupancy labels from an NX body.

    The sampler builds a uniform coarse grid first, detects boundary cells from
    6-neighbor occupancy changes, and refines only those boundary cells to
    ``fine_depth``.  Returned centers are raw world coordinates in millimeters.

    Returns:
        centers: [K, 3] float32 leaf centers.
        depths:  [K] int16 octree depth for each leaf.
        labels:  [K] float32, 1.0 inside material, 0.0 outside.
    """
    if rng is None:
        rng = np.random.default_rng()

    coarse_depth = int(max(0, coarse_depth))
    fine_depth = int(max(coarse_depth, fine_depth))
    max_nodes = int(max(1, max_nodes))

    bbox_min_f = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max_f = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    extent = bbox_max_f - bbox_min_f
    max_extent = float(np.max(np.abs(extent)))
    if max_extent <= 1e-9:
        max_extent = 1.0
    extent = np.where(np.abs(extent) <= 1e-9, max_extent * 1e-6, extent)
    bbox_max_f = bbox_min_f + extent

    coarse_n = 2 ** coarse_depth
    coarse_indices = np.indices((coarse_n, coarse_n, coarse_n), dtype=np.int32)
    coarse_indices = coarse_indices.reshape(3, -1).T
    coarse_cell = extent / float(coarse_n)
    coarse_centers = bbox_min_f.reshape(1, 3) + (coarse_indices.astype(np.float64) + 0.5) * coarse_cell.reshape(1, 3)

    def label_points(points: "np.ndarray") -> "np.ndarray":
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if contains_points_fn is not None:
            return np.asarray(contains_points_fn(points), dtype=np.float32).reshape(-1)
        labels_out = np.zeros((points.shape[0],), dtype=np.float32)
        for point_i, point in enumerate(points):
            if _is_point_inside_body_uf(body, float(point[0]), float(point[1]), float(point[2])):
                labels_out[point_i] = 1.0
        return labels_out

    coarse_labels = label_points(coarse_centers)

    labels_grid = coarse_labels.reshape(coarse_n, coarse_n, coarse_n)
    boundary_grid = np.zeros_like(labels_grid, dtype=bool)
    for axis in range(3):
        diff = np.diff(labels_grid, axis=axis) != 0
        left = [slice(None), slice(None), slice(None)]
        right = [slice(None), slice(None), slice(None)]
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        boundary_grid[tuple(left)] |= diff
        boundary_grid[tuple(right)] |= diff

    boundary_flat = boundary_grid.reshape(-1)
    non_boundary = ~boundary_flat

    leaf_centers = []
    leaf_depths = []
    leaf_labels = []

    if np.any(non_boundary):
        leaf_centers.append(coarse_centers[non_boundary])
        leaf_depths.append(np.full(int(non_boundary.sum()), coarse_depth, dtype=np.int16))
        leaf_labels.append(coarse_labels[non_boundary])

    if np.any(boundary_flat) and fine_depth > coarse_depth:
        refine_factor = 2 ** (fine_depth - coarse_depth)
        fine_cell = extent / float(2 ** fine_depth)
        child_offsets = np.indices((refine_factor, refine_factor, refine_factor), dtype=np.int32)
        child_offsets = child_offsets.reshape(3, -1).T.astype(np.float64) + 0.5

        boundary_parent_indices = coarse_indices[boundary_flat]
        child_count = child_offsets.shape[0]
        fine_budget = max_nodes - int(non_boundary.sum())
        total_fine = int(boundary_parent_indices.shape[0] * child_count)
        if fine_budget > 0 and total_fine > fine_budget:
            sampled_child_ids = rng.choice(total_fine, size=int(fine_budget), replace=False)
            parent_ids = sampled_child_ids // child_count
            child_ids = sampled_child_ids % child_count
            parent_min = bbox_min_f.reshape(1, 3) + boundary_parent_indices[parent_ids].astype(np.float64) * coarse_cell.reshape(1, 3)
            fine_centers = parent_min + child_offsets[child_ids] * fine_cell.reshape(1, 3)
        else:
            fine_chunks = []
            for parent_index in boundary_parent_indices:
                parent_min = bbox_min_f + parent_index.astype(np.float64) * coarse_cell
                fine_chunks.append(parent_min.reshape(1, 3) + child_offsets * fine_cell.reshape(1, 3))
            fine_centers = np.vstack(fine_chunks) if fine_chunks else np.empty((0, 3), dtype=np.float64)
        fine_labels = label_points(fine_centers)
        leaf_centers.append(fine_centers)
        leaf_depths.append(np.full(fine_centers.shape[0], fine_depth, dtype=np.int16))
        leaf_labels.append(fine_labels)
    elif np.any(boundary_flat):
        leaf_centers.append(coarse_centers[boundary_flat])
        leaf_depths.append(np.full(int(boundary_flat.sum()), coarse_depth, dtype=np.int16))
        leaf_labels.append(coarse_labels[boundary_flat])

    centers = np.vstack(leaf_centers).astype(np.float32) if leaf_centers else coarse_centers.astype(np.float32)
    depths = np.concatenate(leaf_depths).astype(np.int16) if leaf_depths else np.full(coarse_centers.shape[0], coarse_depth, dtype=np.int16)
    labels = np.concatenate(leaf_labels).astype(np.float32) if leaf_labels else coarse_labels.astype(np.float32)

    if centers.shape[0] > max_nodes:
        fine_idx = np.flatnonzero(depths == fine_depth)
        coarse_idx = np.flatnonzero(depths != fine_depth)
        if fine_idx.size >= max_nodes:
            keep = rng.choice(fine_idx, size=max_nodes, replace=False)
        else:
            remaining = max_nodes - fine_idx.size
            extra = rng.choice(coarse_idx, size=min(remaining, coarse_idx.size), replace=False) if coarse_idx.size else np.asarray([], dtype=np.int64)
            keep = np.concatenate([fine_idx, extra])
        rng.shuffle(keep)
        centers = centers[keep]
        depths = depths[keep]
        labels = labels[keep]

    return centers.astype(np.float32), depths.astype(np.int16), labels.astype(np.float32)


def query_occupancy_at_positions(
    body,
    centers_xyz: "np.ndarray",
    contains_points_fn=None,
) -> "np.ndarray":
    """Returns occupancy labels at a fixed set of 3-D positions.

    Unlike :func:`sample_octree_occupancy`, this function does **not** perform
    adaptive sampling.  It simply evaluates containment at the provided centers,
    making it suitable for querying the *before-operation* body at exactly the
    same positions that were sampled from the *after-operation* body.

    Args:
        body:               NX body object.  Used only when *contains_points_fn*
                            is ``None`` (falls back to ``AskPointContainment``).
        centers_xyz:        ``[K, 3]`` float array of world-coordinate positions.
        contains_points_fn: Optional batched containment callable ``f(pts) → bool[K]``.
                            When provided (e.g. a trimesh proximity function),
                            *body* is not queried directly.

    Returns:
        ``[K]`` float32 — 1.0 = inside / on-surface material, 0.0 = outside.
    """
    centers = np.asarray(centers_xyz, dtype=np.float64).reshape(-1, 3)
    K = centers.shape[0]
    if K == 0:
        return np.zeros(0, dtype=np.float32)

    if contains_points_fn is not None:
        return np.asarray(contains_points_fn(centers), dtype=np.float32).reshape(-1)

    labels = np.zeros((K,), dtype=np.float32)
    for i in range(K):
        if _is_point_inside_body_uf(body, float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])):
            labels[i] = 1.0
    return labels
