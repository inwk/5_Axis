import functools
import trimesh
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import sys, subprocess

def visualize_stl(path, title="", block=False):
    """Performs: visualize stl."""
    mesh = pv.read(path)
    if block:
        p = pv.Plotter()
        p.add_mesh(mesh, show_edges=False)
        p.view_isometric()
        p.show_grid()
        p.show(title=title)  # blocking plot window
    else:
        # pip install pyvistaqt
        p = BackgroundPlotter(title=title, show=True)  # non-blocking interactive viewer
        p.add_mesh(mesh, show_edges=False)
        p.view_isometric()
        p.show_grid()
        return p  

def identify_exterior_faces(body, direction):
    """Performs: identify exterior faces."""
    session = NXOpen.Session.GetSession()
    work_part = session.Parts.Work
    uf_session = NXOpen.UF.UFSession.GetUFSession()
    xform_collection = work_part.Xforms

    origin = NXOpen.Point3d(0.0, 0.0, 0.0)
    x_direction = NXOpen.Vector3d(1.0, 0.0, 0.0)
    y_direction = NXOpen.Vector3d(0.0, 1.0, 0.0)
    update_option = NXOpen.SmartObject.UpdateOption.WithinModeling
    scale = 1.0

    xform = xform_collection.CreateXform(origin, x_direction, y_direction, update_option, scale)

    num_bodies = len([body])
    xforms = [xform]
    num_dirs = 1
    num_faces = 0

    chordal_tol = 0.01
    resolution = 1

    direction = [[-direction.X, -direction.Y, -direction.Z]]
    num_faces, face_tags, body_indices = uf_session.Modl.IdentifyExteriorUsingHl(
        num_bodies, [body.Tag], [xform.Tag], 1, direction, chordal_tol, resolution, num_faces
    )

    return face_tags

def scale_to_unit_sphere(mesh):
    """Performs: scale to unit sphere."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube(mesh):
    """Performs: scale to unit cube."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

# Use get_raster_points.cache_clear() to clear the cache
@functools.lru_cache(maxsize=4)
def get_raster_points(voxel_resolution):
    """Performs: get raster points."""
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points

def check_voxels(voxels):
    """Performs: check voxels."""
    block = voxels[:-1, :-1, :-1]
    d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
    d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
    d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

    max_distance = max(np.max(d1), np.max(d2), np.max(d3))
    return max_distance < 2.0 / voxels.shape[0] * 3**0.5 * 1.1

def sample_uniform_points_in_unit_sphere(amount):
    """Performs: sample uniform points in unit sphere."""
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]
import os
import NXOpen
import NXOpen.CAM
import networkx as nx
import pandas as pd
import sys
import json
import math
import numpy as np
try:
    from measurements import getFaceVector, getConvergentFaceInfo
    from measurements import getFaceArea
    from measurements import get_deviation_per_face, get_pointwise_deviation_per_face
    from measurements import getVolume
    from measurements import generate_points_v2, generate_points_convergent_face
except ImportError as e:
    from CAM.measurements import getFaceVector#,getConvergentFaceInfo
    from CAM.measurements import getFaceArea
    from CAM.measurements import get_deviation_per_face, get_pointwise_deviation_per_face
    from CAM.measurements import getVolume
    from CAM.measurements import generate_points_v2, generate_points_convergent_face
def create_tool(session, work_part, tool_diameter, tool_type, tool_list):
    """Performs: create tool."""
    tool_name = f"{tool_type}_{tool_diameter}PI"
    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("GENERIC_MACHINE")
    try:
        if (tool_name in tool_list):
            tool_list.append(tool_name)
            return
        if tool_type == "STD_DRILL":
            tool = work_part.CAMSetup.CAMGroupCollection.CreateTool(nCGroup, "hole_making", tool_type, NXOpen.CAM.NCGroupCollection.UseDefaultName.FalseValue, tool_name)
            ToolBuilder = work_part.CAMSetup.CAMGroupCollection.CreateDrillStdToolBuilder(tool)
        else:
            tool = work_part.CAMSetup.CAMGroupCollection.CreateTool(nCGroup, "mill_contour", tool_type, NXOpen.CAM.NCGroupCollection.UseDefaultName.FalseValue, tool_name)
            ToolBuilder = work_part.CAMSetup.CAMGroupCollection.CreateMillToolBuilder(tool)

        ToolBuilder.TlDiameterBuilder.Value = float(tool_diameter)
        ###
        if tool_diameter >= 12:

            ToolBuilder.TlHeightBuilder.Value = tool_diameter * 4
            ToolBuilder.TlFluteLnBuilder.Value = tool_diameter * 2

            # lower_diameter = 50.0
            # holder_length = 120.0
            # taper_angle = 0.0
            # _ = ToolBuilder.HolderSectionBuilder.Add(0, lower_diameter, holder_length, taper_angle, 0.0)
        else:
            ToolBuilder.TlHeightBuilder.Value = tool_diameter * 6
            ToolBuilder.TlFluteLnBuilder.Value = tool_diameter * 3

            # lower_diameter = 32.0
            # holder_length = 150.0
            # taper_angle = 1.5275
            # _ = ToolBuilder.HolderSectionBuilder.Add(0, lower_diameter, holder_length, taper_angle, 0.0)

        ###
        ToolBuilder.Commit()
        ToolBuilder.Destroy()
        tool_list.append(tool_name)
    except:
        pass

def rotate_vector(rot, vector):
    """Performs: rotate vector."""
    dx, dy, dz = vector
    
    if rot == 1:
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])
    elif rot == 2:
        rotation_matrix = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]])
    elif rot == 3:
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]])
    elif rot == 4:
        rotation_matrix = np.array([[0, 0, -1],
                                    [0, 1, 0],
                                    [1, 0, 0]])
    else:
        pass
    vector_np = np.array([dx, dy, dz])
    
    rotated_vector = np.dot(rotation_matrix, vector_np)
    
    return rotated_vector

def face_classify(session=None, work_part=None, target_size=None, operation_type=None, origin_faces=None, deviation_list=None, drill_orientation=None):
    """Performs: face classify."""
    if session is None:
        session = NXOpen.Session.GetSession()
    if work_part is None:
        work_part = session.Parts.Work

    theUfSession = NXOpen.UF.UFSession.GetUFSession()
    the_session = session
    # ----------------------------------------------
    #   Enumerator
    #   Rubber               Rubber face, no surface attached
    #   Planar               Planar face
    #   Cylindrical           Cylindrical face
    #   Conical               Conical face
    #   Spherical           Spherical face
    #   SurfaceOfRevolution   Face from surface of revolution
    #   Parametric           Parametric face
    #   Blending           Blending face
    #   Offset               Offset face
    #   Swept               Swept face
    #   Convergent           Convergent face
    #   Undefined           Undefined face type
    # ----------------------------------------------
    # block_height = target_size[0]
    # block_length = target_size[1]
    # block_width = target_size[2]

    face_idx_list = []
    diff_threshold = 0.02
    if (operation_type == "Cavity Mill"):
        for idx, face in enumerate(origin_faces):
            if (deviation_list[idx] > diff_threshold):  # skip nearly-finished faces
                _, point_data, direction_data, _, major_radius, _, _ = theUfSession.Modeling.AskFaceData(face.Tag)
                nx, ny, nz = direction_data
                #     face_idx_list.append(idx)
                #     face.Color = 20
                #     if (nz == 1 or nz == -1):
                #         if (major_radius <= 5) and (0 + major_radius <= point_data[0] <= block_length - major_radius) and (0 + major_radius <= point_data[1] <= block_width - major_radius):
                #             pass
                #         face_idx_list.append(idx)
                #         face.Color = 20
                # else:
                face_idx_list.append(idx)
                face.Color = 20
           
    if (operation_type == "Area Mill"):
        for idx, face in enumerate(origin_faces):
            if (deviation_list[idx] > diff_threshold):  # skip nearly-finished faces
                _, point_data, direction_data, _, major_radius, _, _ = theUfSession.Modeling.AskFaceData(face.Tag)
                nx, ny, nz = direction_data
                if face.SolidFaceType.value == 1:
                    if (nx == 0) and (ny == 0):
                        pass
                    elif (nz == 0):
                        pass
                    else:
                        face_idx_list.append(idx)
                        face.Color = 40
                # elif face.SolidFaceType.value == 2:
                #     if (nz != 1 and nz != -1):
                #         face_idx_list.append(idx)
                #         face.Color = 40
                else:
                    face_idx_list.append(idx)
                    face.Color = 40
                    
    if (operation_type == "Drill"):
        for idx, face in enumerate(origin_faces):
            _, center, direction_data, _, major_radius, _, _ = theUfSession.Modeling.AskFaceData(face.Tag)
            nx, ny, nz = direction_data

            if face.SolidFaceType.value == 2 and major_radius <= 5:
                edges = face.GetEdges()
                has_bottom_face = False
                matched = False
                bottom_idx = -1

                for edge in edges:
                    for adj_face in edge.GetFaces():
                        if adj_face.SolidFaceType.value == 1 and len(adj_face.GetEdges()) == 1:
                            _, _, adj_norm_vec, _, _, _, _ = theUfSession.Modeling.AskFaceData(adj_face.Tag)
                            adj_nx, adj_ny, adj_nz = adj_norm_vec
                            for i, f in enumerate(origin_faces):
                                if f.Tag == adj_face.Tag:
                                    bottom_idx = i
                                    break
                            if drill_orientation is None and adj_nz == 1:
                                matched = True
                            elif drill_orientation == 'y' and adj_ny == 1:
                                matched = True
                            elif drill_orientation == '-y' and adj_ny == -1:
                                matched = True
                            elif drill_orientation == 'x' and adj_nx == 1:
                                matched = True
                            elif drill_orientation == '-x' and adj_nx == -1:
                                matched = True
                            if matched:
                                has_bottom_face = True
                                break
                    if matched:
                        break

                if matched and has_bottom_face:
                    face_idx_list.append(idx)
                    face.Color = 150
                    if bottom_idx != -1:
                        face_idx_list.append(bottom_idx)
                        origin_faces[bottom_idx].Color = 150

                elif not has_bottom_face:
                    adjacent_faces = []
                    for edge in edges:
                        for af in edge.GetFaces():
                            if af.Tag != face.Tag:
                                adjacent_faces.append(af)

                    flat_face_count = sum(1 for af in adjacent_faces if af.SolidFaceType.value == 1)

                    if len(adjacent_faces) == 2 and flat_face_count == 2:
                        face_idx_list.append(idx)
                        face.Color = 150

                    elif flat_face_count == 0:
                        drill_vec = {'x': [1, 0, 0], '-x': [-1, 0, 0], 'y': [0, 1, 0], '-y': [0, -1, 0], None: [0, 0, 1]}
                        dx, dy, dz = drill_vec.get(drill_orientation, [0, 0, 1])
                        dot = abs(nx * dx + ny * dy + nz * dz)
                        if dot > 0.98:
                            face_idx_list.append(idx)
                            face.Color = 150

    # if (operation_type == "Drill"):
    #         for idx, face in enumerate(origin_faces):
    #             _, _, direction_data, _, major_radius, _, _ = theUfSession.Modeling.AskFaceData(face.Tag) 
    #             nx, ny, nz = direction_data
    #             # if (abs(nz) == 1.0):
    #                     face.Color = 150
    #                     for edge in edges:
    #                         for adj_face in adj_faces:
    #                             if adj_face.SolidFaceType.value == 1:
    #                                 if (len(adj_face.GetEdges()) == 1):
    #                                     _, _, adj_norm_vec, _, _, _, _ = theUfSession.Modeling.AskFaceData(adj_face.Tag)
    #                                     bottom_idx = 0
    #                                     for bottom_face in origin_faces:
    #                                         if (bottom_face.Tag == adj_face.Tag):
    #                                             break
    #                                         bottom_idx=bottom_idx+1
    #                                     if drill_orientation == None:
    #                                         if adj_nz == 1:
    #                                             face_idx_list.append(idx)
    #                                             face_idx_list.append(bottom_idx)
    #                                             face.Color = 150
    #                                     elif drill_orientation == 'y':
    #                                         if adj_ny == 1:
    #                                             face_idx_list.append(idx)
    #                                             face_idx_list.append(bottom_idx)
    #                                             face.Color = 150
    #                                     elif drill_orientation == '-y':
    #                                         if adj_ny == -1:
    #                                             face_idx_list.append(idx)
    #                                             face_idx_list.append(bottom_idx)
    #                                             face.Color = 150
    #                                     elif drill_orientation == 'x':
    #                                         if adj_nx == 1:
    #                                             face_idx_list.append(idx)
    #                                             face_idx_list.append(bottom_idx)
    #                                             face.Color = 150
    #                                     elif drill_orientation == '-x':
    #                                         if adj_nx == -1:
    #                                             face_idx_list.append(idx)
    #                                             face_idx_list.append(bottom_idx)
    #                                             face.Color = 150
    return face_idx_list


def visualize_face(face, color):
        """Performs: visualize face."""
        face.Color = color
        face.RedisplayObject()
def getEncInputData(faces,faces_tag):
    """Performs: get enc input data."""
    theUfSession = NXOpen.UF.UFSession.GetUFSession()
    the_session = NXOpen.Session.GetSession()

    G=nx.Graph()
    for face_tag in faces_tag:
        G.add_node(face_tag)
    
    # For face
    for face_tag in faces_tag:
        adjacent_faces_tag = theUfSession.Modeling.AskAdjacFaces(face_tag)
        for adj_face_tag in adjacent_faces_tag:
            if adj_face_tag in faces_tag:  # keep only edges between origin faces
                G.add_edge(face_tag, adj_face_tag)

    # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
    # plt.show()

    faces_vec = []
    faces_area = []
    faces_type = []

    for face in faces:
        r, isBlended = face.GetBlendData()
        face_type = face.SolidFaceType.value
        if face_type == 10: #convergent facet
            # face_vec, face_type = getConvergentFaceInfo(face)
            # faces_vec.append(face_vec.tolist())
            pass
        elif isBlended or face_type>=5:
            face_type = 5
            visualize_face(face, 200)
            faces_vec.append(getFaceVector(face.Tag))
        elif face.SolidFaceType.value == 1:
            if len(face.GetEdges()) == 1:
                area, _, _, _, _, _, _, _ = the_session.Measurement.GetFaceProperties([face], 0.98999999999999999, NXOpen.Measurement.AlternateFace.Radius, True)
                if area <= math.pi * math.pow(5.1, 2): 
                    face_type = 6
                    visualize_face(face, 6)
                    faces_vec.append(getFaceVector(face.Tag))
                else:
                    visualize_face(face, face_type * 25)
                    faces_vec.append(getFaceVector(face.Tag))
            else:
                visualize_face(face, face_type * 25)
                faces_vec.append(getFaceVector(face.Tag))
        else:
            visualize_face(face, face_type * 25)
            faces_vec.append(getFaceVector(face.Tag))

        faces_area.append(getFaceArea(face))
        faces_type.append(face_type)
    return G, faces_vec, faces_area, faces_type 
def getDecOutputData(opType, pattern_idx, tool_idx, selected_face_idx):
    """Performs: get dec output data."""
    return [opType, pattern_idx, tool_idx]+selected_face_idx
def graph_to_json(G):
    """Performs: graph to json."""
    return json.dumps(nx.readwrite.json_graph.node_link_data(G))   

def json_to_graph(graph_json):
    """Performs: json to graph."""
    return nx.readwrite.json_graph.node_link_graph(json.loads(graph_json))


def save_data(path, data_list,overwrite=False):#graph, faces_vec, faces_area, faces_type, dec_input, dec_output):
    """Performs: save data."""
    new_data_list = []
    for data in data_list:
        graph_json = graph_to_json(data[0])
        new_data = {
            'graph': graph_json,
            'faces_vec': data[1],
            'faces_area': data[2],
            'faces_type': data[3],
            'dec_input' : data[4],
            'dec_output' : data[5]
        }
        new_data_list.append(new_data)
    df_new = pd.DataFrame(new_data_list)

    if os.path.exists(path) and not overwrite:
        df_existing = pd.read_parquet(path)
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing.to_parquet(path)
    else:
        df_new.to_parquet(path)
        
def save_data_sdf(path, data_list, overwrite=False):
    """Performs: save data sdf."""
    new_data_list = []
    for data in data_list:
        enc_input_distance = (np.array(data[0][0]) * 10000).astype(np.int16).flatten()  # 64x64x64 -> flat int16
        enc_input_gradient = (np.array(data[0][1]) * 10000).astype(np.int16).flatten()  # 64x64x64x3 -> flat int16
        dec_input_distance = (np.array(data[1][0]) * 10000).astype(np.int16).flatten()  # 64x64x64 -> flat int16
        dec_input_gradient = (np.array(data[1][1]) * 10000).astype(np.int16).flatten()  # 64x64x64x3 -> flat int16
        dec_output = np.array(data[2]).flatten()  # action/output vector -> flat

        new_data = {
            'enc_input_distance': enc_input_distance,
            'enc_input_gradient': enc_input_gradient,
            'dec_input_distance': dec_input_distance,
            'dec_input_gradient': dec_input_gradient,
            'dec_output': dec_output
        }
        new_data_list.append(new_data)
    df_new = pd.DataFrame(new_data_list)

    if os.path.exists(path) and not overwrite:
        df_existing = pd.read_parquet(path)
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing.to_parquet(path)
    else:
        df_new.to_parquet(path)
def load_data(path):
    """Performs: load data."""
    df_loaded = pd.read_parquet(path)

    loaded_graphs = []
    loaded_faces_vec = []
    loaded_faces_areas = []
    loaded_faces_type = []
    loaded_dec_input =[]
    loaded_dec_output = []
    for index, row in df_loaded.iterrows():
        graph_json = row['graph']
        faces_vec = row['faces_vec']
        faces_area = row['faces_area']
        faces_type = row['faces_type']
        dec_input = row['dec_input']
        dec_output = row['dec_output']
        G = json_to_graph(graph_json)
        loaded_graphs.append(G)
        loaded_faces_vec.append(faces_vec)
        loaded_faces_areas.append(faces_area)
        loaded_faces_type.append(faces_type)
        loaded_dec_input.append(dec_input)
        loaded_dec_output.append(dec_output)
    return loaded_graphs,loaded_faces_vec,loaded_faces_areas,loaded_faces_type,loaded_dec_input,loaded_dec_output
def load_data_sdf(path):
    """Performs: load data sdf."""
    df_loaded = pd.read_parquet(path)

    data_list = []
    for _, row in df_loaded.iterrows():
        enc_input_distance = np.array(row['enc_input_distance']).reshape(64, 64, 64)
        enc_input_gradient = np.array(row['enc_input_gradient']).reshape(64, 64, 64, 3)
        dec_input_distance = np.array(row['dec_input_distance']).reshape(64, 64, 64)
        dec_input_gradient = np.array(row['dec_input_gradient']).reshape(64, 64, 64, 3)
        dec_output = np.array(row['dec_output'])  # reshape later if needed

        data = [
            enc_input_distance, enc_input_gradient,
            dec_input_distance, dec_input_gradient,
            dec_output
        ]
        data_list.append(data)

    return data_list

def get_ipw_property(session=None, work_part=None, object_blank=None, tool_name=None, points_array=None, norm_vecs_array=None, lines_array=None, savepath=None):
    """Builds a temporary operation and measures current IPW volume and face deviations."""
    if session is None:
        session = NXOpen.Session.GetSession()
    if work_part is None:
        work_part = session.Parts.Work

    # Get work part
    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Fine)
    
    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "before")

    # ----------------------------------------------
    #   Create CAM Operation
    # ----------------------------------------------
    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
    method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
    tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
    operation = work_part.CAMSetup.CAMOperationCollection.Create(nCGroup, 
                                                                method, 
                                                                tool, 
                                                                object_blank, 
                                                                "mill_contour", 
                                                                "AREA_MILL", 
                                                                NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue, 
                                                                "AREA_MILL")

    ipw = operation.GetInputIpw()
    # if savepath != None:
    #     if isinstance(savepath, list):
    #         name_base = "Raw_stock" if not savepath else f"op_{len(savepath)}"
            
    #     else:
    #         name_base = str(savepath)
    #     # --- STL Export ---
    #     out_file = f"{name_base}.stl"
    #     stl = session.DexManager.CreateStlCreator()
    #     stl.AutoNormalGen = True
    #     stl.ChordalTol = 0.05
    #     stl.AdjacencyTol = 0.05
    #     stl.Commit()
    #     stl.Destroy()
        # viewer = os.path.join(os.path.dirname(__file__), "stl_viewer.py")
        # subprocess.Popen([
        #     sys.executable, viewer,
        #     "--file", out_file,           # ?? "op_3.stl"
        #     "--title", name_base,         # ?? "op_3"
        # ])

    volume = ipw.Volume/1000 
    deviation_list, objects = get_deviation_per_face(ipw, points_array, norm_vecs_array,lines_array)
    
    #volume = getVolume(objects)
    assert volume != 0.0, "volume 0"
    session.UndoToMark(markId, "before")  
    return deviation_list, volume


def CAMFilter(dec_input_list,dec_output_list, cycle_time_list, volume_diff_list):
    """Performs: camfilter."""
    indices_to_remove = [index for index, value in enumerate(volume_diff_list) if value <= 0.0]
    for index in sorted(indices_to_remove, reverse=True):
        del dec_input_list[index]
        del dec_output_list[index]
        del cycle_time_list[index]
        del volume_diff_list[index]
    cnt = len(cycle_time_list)
    min_val = min(cycle_time_list)
    max_val = max(cycle_time_list)
    if min_val == max_val:
        sys.exit(1)
    norm_cycle_time_list = [(x - min_val) / (max_val - min_val) for x in cycle_time_list]
    
    min_val = min(volume_diff_list)
    max_val = max(volume_diff_list)
    if min_val == max_val:
        sys.exit(1)
    norm_volume_diff_list = [(x - min_val) / (max_val - min_val) for x in volume_diff_list]
    
    distances = [math.sqrt(x**2 + y**2) for x, y in zip(norm_cycle_time_list, norm_volume_diff_list)]

    low_indices = np.argsort(distances)[-round(cnt*0.7):]


    filtered_dec_input_list = [value for index, value in enumerate(dec_input_list) if index not in low_indices]
    filtered_dec_output_list = [value for index, value in enumerate(dec_output_list) if index not in low_indices]
    return filtered_dec_input_list, filtered_dec_output_list



def combine_parquet_files(directory, output_path):
    """Performs: combine parquet files."""
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    
    combined_df = pd.DataFrame()
    for file in parquet_files:
        df = pd.read_parquet(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    combined_df.to_parquet(output_path)

    return combined_df


# combined_df = combine_parquet_files(directory, output_path)


import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def resample_point_cloud(points, target_count):
    """Resamples a point cloud to `target_count` points.

    If there are too many points, it downsamples by index.
    If there are too few points, it upsamples by inserting midpoints
    of nearest-neighbor pairs.
    """
    if len(points) > 0 and not isinstance(points[0], (list, tuple, np.ndarray)):
        coords_list = []
        for p in points:
            if hasattr(p, 'Coordinates'):  # NXOpen.Point
                c = p.Coordinates
                coords_list.append([c.X, c.Y, c.Z])
            else:  # handle Point3d-like objects
                coords_list.append([p.X, p.Y, p.Z])
        points = np.array(coords_list, dtype=float)
    else:
        points = np.array(points, dtype=float)

    current_count = len(points)
    if current_count == target_count:
        return points

    elif current_count < target_count:
        tree = KDTree(points)
        while len(points) < target_count:
            for i in range(len(points)):
                if len(points) >= target_count:
                    break
                _, idx = tree.query(points[i], k=2)  # nearest two neighbors
                midpoint = (points[idx[0]] + points[idx[1]]) / 2  # insert midpoint sample
                
                points = np.vstack([points, midpoint])
                
                tree = KDTree(points)
                
        return points[:target_count]

    else:
        indices = np.linspace(0, current_count - 1, target_count, dtype=int)
        return points[indices]
def getPointCloud(faces):
    """Samples per-face point clouds and returns grouped point arrays."""
    faces_area = []
    points_array = []
    for face in faces:
        if face.SolidFaceType.value == 10:
            points, norms, lines = generate_points_convergent_face(face)
        else:
            points = generate_points_v2(face)

        resample_points = resample_point_cloud(points,100)
        points_array.append(resample_points)
        faces_area.append(getFaceArea(face))
    return faces_area, points_array


def create_cam_tool(session, work_part, tool_diameter, tool_type, tool_list):
    """Creates one CAM tool instance and appends its name to the tool list."""
    return create_tool(session, work_part, tool_diameter, tool_type, tool_list)


def classify_faces_for_operation(
    session=None,
    work_part=None,
    target_size=None,
    operation_type=None,
    origin_faces=None,
    deviation_list=None,
    drill_orientation=None,
):
    """Returns face indices selected for a given operation type."""
    return face_classify(
        session=session,
        work_part=work_part,
        target_size=target_size,
        operation_type=operation_type,
        origin_faces=origin_faces,
        deviation_list=deviation_list,
        drill_orientation=drill_orientation,
    )


def get_encoder_input_data(faces, faces_tag):
    """Builds graph and per-face geometric features for the encoder."""
    return getEncInputData(faces, faces_tag)


def measure_ipw_state(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    points_array=None,
    norm_vecs_array=None,
    lines_array=None,
    savepath=None,
):
    """Measures IPW volume and per-face deviation for the current setup."""
    return get_ipw_property(
        session=session,
        work_part=work_part,
        object_blank=object_blank,
        tool_name=tool_name,
        points_array=points_array,
        norm_vecs_array=norm_vecs_array,
        lines_array=lines_array,
        savepath=savepath,
    )


def identify_visible_faces(body, direction):
    """Returns tags of faces visible from the given view direction."""
    return identify_exterior_faces(body, direction)


def get_face_point_cloud(faces):
    """Returns area and 100-point cloud samples per face."""
    return getPointCloud(faces)


