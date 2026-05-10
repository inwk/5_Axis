import functools
import tempfile
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


def _export_nx_body_to_obj(session, body, output_path: str) -> str:
    """Exports one NX body to OBJ for Python-side mesh queries."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    obj_creator = session.DexManager.CreateWavefrontObjCreator()
    try:
        obj_creator.ExportSelectionBlock.SelectionScope = NXOpen.ObjectSelector.Scope.SelectedObjects
        obj_creator.AngularTolerance = 17.999999999999996
        obj_creator.FlattenAssemblyStructure = True
        obj_creator.ExportSelectionBlock.SelectionComp.Add(body)
        obj_creator.OutputFile = output_path
        obj_creator.FileSaveFlag = False
        obj_creator.Commit()
    finally:
        _safe_destroy(obj_creator)
    return output_path


def _load_obj_as_trimesh(obj_path: str) -> "trimesh.Trimesh":
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"OBJ did not load as a trimesh.Trimesh: {obj_path}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"OBJ mesh is empty: {obj_path}")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    # NX facet export can contain zero-area triangles.  These make
    # trimesh/ray barycentric tests unstable and can explode intersection lists.
    if mesh.faces.size:
        tri = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
        area2 = np.linalg.norm(
            np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :]),
            axis=1,
        )
        valid = np.isfinite(area2) & (area2 > 1e-12)
        if not np.all(valid):
            mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float64),
                faces=np.asarray(mesh.faces, dtype=np.int64)[valid],
                process=False,
            )
    mesh.remove_unreferenced_vertices()
    return mesh


def _contains_points_ray_parity(mesh: "trimesh.Trimesh", points: "np.ndarray") -> "np.ndarray":
    """Vectorized ray-parity fallback for point-in-closed-triangle-mesh tests.

    This avoids calling NX once per octree point.  It is intended for closed IPW
    meshes; if the exported mesh is not watertight, labels are approximate.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.zeros((points.shape[0],), dtype=bool)
    if points.size == 0:
        return labels

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if triangles.size == 0:
        return labels

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    eps = 1e-9
    inside_bbox = np.all(points >= (bounds[0] - eps), axis=1) & np.all(points <= (bounds[1] + eps), axis=1)
    active_indices = np.flatnonzero(inside_bbox)
    if active_indices.size == 0:
        return labels

    active_points = points[active_indices]
    ray_dir = np.asarray([0.5773502691896258, 0.3713906763541037, 0.7276068751089989], dtype=np.float64)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    point_chunk = max(1, int(os.getenv("OCTREE_RAY_POINT_CHUNK", "32")))
    tri_chunk = max(1, int(os.getenv("OCTREE_RAY_TRI_CHUNK", "1024")))
    counts = np.zeros((active_points.shape[0],), dtype=np.int32)
    for point_start in range(0, active_points.shape[0], point_chunk):
        pts = active_points[point_start : point_start + point_chunk]
        chunk_counts = np.zeros((pts.shape[0],), dtype=np.int32)
        for tri_start in range(0, triangles.shape[0], tri_chunk):
            tri = triangles[tri_start : tri_start + tri_chunk]
            v0 = tri[:, 0, :]
            e1 = tri[:, 1, :] - v0
            e2 = tri[:, 2, :] - v0
            h = np.cross(np.broadcast_to(ray_dir, e2.shape), e2)
            a = np.einsum("ij,ij->i", e1, h)
            valid_tri = np.abs(a) > eps
            if not np.any(valid_tri):
                continue
            v0 = v0[valid_tri]
            e1 = e1[valid_tri]
            e2 = e2[valid_tri]
            h = h[valid_tri]
            inv_a = 1.0 / a[valid_tri]

            s = pts[:, None, :] - v0[None, :, :]
            u = inv_a[None, :] * np.einsum("ptj,tj->pt", s, h)
            mask = (u >= -eps) & (u <= 1.0 + eps)
            if not np.any(mask):
                continue

            q = np.cross(s, e1[None, :, :])
            v = inv_a[None, :] * np.einsum("ptj,j->pt", q, ray_dir)
            mask &= (v >= -eps) & ((u + v) <= 1.0 + eps)
            if not np.any(mask):
                continue

            t = inv_a[None, :] * np.einsum("tj,ptj->pt", e2, q)
            mask &= t > eps
            chunk_counts += np.count_nonzero(mask, axis=1).astype(np.int32)
        counts[point_start : point_start + pts.shape[0]] = chunk_counts

    labels[active_indices] = (counts % 2) == 1
    return labels


def _contains_points_mesh(mesh: "trimesh.Trimesh", points: "np.ndarray") -> "np.ndarray":
    """Returns 1.0 for points inside a mesh and 0.0 otherwise."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mode = os.getenv("OCTREE_MESH_CONTAINS_MODE", "ray").strip().lower()
    if mode in {"ray", "fallback", "chunked_ray"}:
        return _contains_points_ray_parity(mesh, points).astype(np.float32)

    if mode not in {"trimesh", "rtree"}:
        raise ValueError(f"Unsupported OCTREE_MESH_CONTAINS_MODE: {mode!r}")

    # trimesh.contains can be fast with rtree, but on degenerate/highly-faceted
    # IPW meshes it may allocate huge intersection arrays.  Keep it chunked and
    # fall back to the bounded-memory ray implementation on any memory issue.
    chunk = max(1, int(os.getenv("OCTREE_TRIMESH_POINT_CHUNK", "128")))
    labels = np.zeros((points.shape[0],), dtype=np.float32)
    try:
        for start in range(0, points.shape[0], chunk):
            stop = min(start + chunk, points.shape[0])
            labels[start:stop] = np.asarray(mesh.contains(points[start:stop]), dtype=np.float32)
        return labels
    except (MemoryError, ModuleNotFoundError, ImportError, ValueError):
        return _contains_points_ray_parity(mesh, points).astype(np.float32)

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


def _safe_destroy(nx_object):
    """Best-effort cleanup for NX builders/creators."""
    if nx_object is None:
        return
    try:
        nx_object.Destroy()
    except Exception:
        pass


def _undo_to_mark_and_delete(session, mark_id, mark_name: str) -> None:
    """Undo temporary NX objects and remove the mark from the stack."""
    undo_error = None
    try:
        session.UndoToMark(mark_id, mark_name)
    except Exception as exc:
        undo_error = exc
    try:
        session.DeleteUndoMark(mark_id, mark_name)
    except Exception:
        pass
    if undo_error is not None:
        raise undo_error

try:
    from measurements import getFaceVector, getConvergentFaceInfo
    from measurements import getFaceArea
    from measurements import get_deviation_per_face, get_pointwise_deviation_per_face
    from measurements import getVolume
    from measurements import convert_facet_to_body, generate_points_v2, generate_points_convergent_face
    from measurements import get_body_axis_aligned_bbox, sample_octree_occupancy, query_occupancy_at_positions
except ImportError as e:
    from CAM.measurements import getFaceVector#,getConvergentFaceInfo
    from CAM.measurements import getFaceArea
    from CAM.measurements import get_deviation_per_face, get_pointwise_deviation_per_face
    from CAM.measurements import getVolume
    from CAM.measurements import convert_facet_to_body, generate_points_v2, generate_points_convergent_face
    from CAM.measurements import get_body_axis_aligned_bbox, sample_octree_occupancy, query_occupancy_at_positions
def create_tool(session, work_part, tool_diameter, tool_type, tool_list):
    """Performs: create tool."""
    tool_name = f"{tool_type}_{tool_diameter}PI"
    nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("GENERIC_MACHINE")
    if tool_name in tool_list:
        tool_list.append(tool_name)
        return
    ToolBuilder = None
    if tool_type == "STD_DRILL":
        tool = work_part.CAMSetup.CAMGroupCollection.CreateTool(nCGroup, "hole_making", tool_type, NXOpen.CAM.NCGroupCollection.UseDefaultName.FalseValue, tool_name)
        ToolBuilder = work_part.CAMSetup.CAMGroupCollection.CreateDrillStdToolBuilder(tool)
    else:
        tool = work_part.CAMSetup.CAMGroupCollection.CreateTool(nCGroup, "mill_contour", tool_type, NXOpen.CAM.NCGroupCollection.UseDefaultName.FalseValue, tool_name)
        ToolBuilder = work_part.CAMSetup.CAMGroupCollection.CreateMillToolBuilder(tool)

    try:
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
    finally:
        _safe_destroy(ToolBuilder)
    tool_list.append(tool_name)

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
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "before")
    try:
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
        return deviation_list, volume
    finally:
        _undo_to_mark_and_delete(session, markId, "before")


def get_ipw_property_detailed(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    points_array=None,
    norm_vecs_array=None,
    lines_array=None,
    savepath=None,
):
    """Builds a temporary operation and measures current IPW at both face and point levels."""
    if session is None:
        session = NXOpen.Session.GetSession()
    if work_part is None:
        work_part = session.Parts.Work

    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "before_detailed")

    try:
        nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
        method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
        tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
        operation = work_part.CAMSetup.CAMOperationCollection.Create(
            nCGroup,
            method,
            tool,
            object_blank,
            "mill_contour",
            "AREA_MILL",
            NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
            "AREA_MILL",
        )

        ipw = operation.GetInputIpw()
        volume = ipw.Volume / 1000
        deviation_list, _ = get_deviation_per_face(ipw, points_array, norm_vecs_array, lines_array)
        pointwise_list, _ = get_pointwise_deviation_per_face(ipw, points_array, norm_vecs_array, lines_array)

        assert volume != 0.0, "volume 0"
        return deviation_list, pointwise_list, volume
    finally:
        _undo_to_mark_and_delete(session, markId, "before_detailed")


def sample_ipw_octree_state(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    bbox_min=None,
    bbox_max=None,
    coarse_depth: int = 3,
    fine_depth: int = 5,
    max_nodes: int = 4096,
    bbox_padding: float = 0.05,
):
    """Creates a temporary operation and samples octree occupancy from its input IPW.

    By default, the IPW is exported once as an OBJ mesh and point containment is
    evaluated in Python batches.  Set ``OCTREE_CONTAINMENT_BACKEND=nx`` to use
    NX ``AskPointContainment`` per point instead.
    """
    if session is None:
        session = NXOpen.Session.GetSession()
    if work_part is None:
        work_part = session.Parts.Work
    if object_blank is None:
        raise ValueError("object_blank is required")
    if tool_name is None:
        raise ValueError("tool_name is required")
    if bbox_min is None or bbox_max is None:
        raise ValueError("bbox_min and bbox_max are required")

    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "before_octree")
    try:
        nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
        method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
        tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
        operation = work_part.CAMSetup.CAMOperationCollection.Create(
            nCGroup,
            method,
            tool,
            object_blank,
            "mill_contour",
            "AREA_MILL",
            NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
            "AREA_MILL",
        )

        ipw = operation.GetInputIpw()
        ipw_objects = convert_facet_to_body(ipw)
        ipw_body = ipw_objects[0]
        bbox_min_use = np.asarray(bbox_min, dtype=np.float32).reshape(3)
        bbox_max_use = np.asarray(bbox_max, dtype=np.float32).reshape(3)

        backend = os.getenv("OCTREE_CONTAINMENT_BACKEND", "mesh").strip().lower()
        contains_points_fn = None
        if backend == "mesh":
            with tempfile.TemporaryDirectory(prefix="ai_cam_ipw_mesh_") as tmp_dir:
                obj_path = _export_nx_body_to_obj(session, ipw_body, os.path.join(tmp_dir, "ipw.obj"))
                ipw_mesh = _load_obj_as_trimesh(obj_path)
                body_bbox_min = np.asarray(ipw_mesh.bounds[0], dtype=np.float32)
                body_bbox_max = np.asarray(ipw_mesh.bounds[1], dtype=np.float32)
                contains_points_fn = lambda points: _contains_points_mesh(ipw_mesh, points)

                bbox_min_use = np.minimum(bbox_min_use, body_bbox_min)
                bbox_max_use = np.maximum(bbox_max_use, body_bbox_max)
                bbox_extent = np.maximum(bbox_max_use - bbox_min_use, 1e-6)
                bbox_pad = bbox_extent * float(max(bbox_padding, 0.0))
                bbox_min_use = bbox_min_use - bbox_pad
                bbox_max_use = bbox_max_use + bbox_pad

                centers, depths, labels = sample_octree_occupancy(
                    body=ipw_body,
                    bbox_min=bbox_min_use,
                    bbox_max=bbox_max_use,
                    coarse_depth=coarse_depth,
                    fine_depth=fine_depth,
                    max_nodes=max_nodes,
                    contains_points_fn=contains_points_fn,
                )
                return centers, depths, labels, bbox_min_use.astype(np.float32), bbox_max_use.astype(np.float32)
        elif backend == "nx":
            body_bbox_min, body_bbox_max = get_body_axis_aligned_bbox(ipw_body)
        else:
            raise ValueError(f"Unsupported OCTREE_CONTAINMENT_BACKEND: {backend!r}")

        bbox_min_use = np.minimum(bbox_min_use, body_bbox_min)
        bbox_max_use = np.maximum(bbox_max_use, body_bbox_max)
        bbox_extent = np.maximum(bbox_max_use - bbox_min_use, 1e-6)
        bbox_pad = bbox_extent * float(max(bbox_padding, 0.0))
        bbox_min_use = bbox_min_use - bbox_pad
        bbox_max_use = bbox_max_use + bbox_pad

        centers, depths, labels = sample_octree_occupancy(
            body=ipw_body,
            bbox_min=bbox_min_use,
            bbox_max=bbox_max_use,
            coarse_depth=coarse_depth,
            fine_depth=fine_depth,
            max_nodes=max_nodes,
            contains_points_fn=contains_points_fn,
        )
        return centers, depths, labels, bbox_min_use.astype(np.float32), bbox_max_use.astype(np.float32)
    finally:
        _undo_to_mark_and_delete(session, markId, "before_octree")


def query_ipw_occupancy_at_positions(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    centers_xyz=None,
):
    """Queries occupancy at fixed 3-D positions for the IPW of *object_blank*.

    Creates a temporary dummy CAM operation to extract the IPW body associated
    with *object_blank*, then evaluates point-in-body containment at every
    position in *centers_xyz*.  All NX objects are cleaned up via an undo mark.

    This is the companion to :func:`sample_ipw_octree_state` ??use it to obtain
    *before-operation* labels at the same octree cell positions that were sampled
    from the *after-operation* body, enabling the monotonicity training signal.

    Args:
        session:       NX session (default: current session).
        work_part:     NX work part (default: current work part).
        object_blank:  CAM workpiece geometry object for the *before* state.
        tool_name:     Tool name string used to create the dummy operation.
        centers_xyz:   ``[K, 3]`` float array of world-coordinate positions.

    Returns:
        ``[K]`` float32 occupancy labels ??1.0 inside material, 0.0 outside.
    """
    if session is None:
        session = NXOpen.Session.GetSession()
    if work_part is None:
        work_part = session.Parts.Work
    if object_blank is None or tool_name is None or centers_xyz is None:
        raise ValueError("object_blank, tool_name, and centers_xyz are all required")

    centers = np.asarray(centers_xyz, dtype=np.float64).reshape(-1, 3)
    if centers.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "query_ipw_occ")
    try:
        nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
        method  = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
        tool    = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
        operation = work_part.CAMSetup.CAMOperationCollection.Create(
            nCGroup,
            method,
            tool,
            object_blank,
            "mill_contour",
            "AREA_MILL",
            NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
            "AREA_MILL",
        )
        ipw = operation.GetInputIpw()
        ipw_objects = convert_facet_to_body(ipw)
        ipw_body = ipw_objects[0]

        backend = os.getenv("OCTREE_CONTAINMENT_BACKEND", "mesh").strip().lower()
        if backend == "mesh":
            with tempfile.TemporaryDirectory(prefix="ai_cam_qipw_") as tmp_dir:
                obj_path = _export_nx_body_to_obj(
                    session, ipw_body, os.path.join(tmp_dir, "ipw_before.obj")
                )
                ipw_mesh = _load_obj_as_trimesh(obj_path)
                labels = _contains_points_mesh(ipw_mesh, centers)
        else:
            labels = query_occupancy_at_positions(ipw_body, centers)

        return np.asarray(labels, dtype=np.float32).reshape(-1)
    finally:
        _undo_to_mark_and_delete(session, markId, "query_ipw_occ")


def export_ipw_object_blank_to_obj(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    output_path=None,
):
    """Exports the input IPW associated with a CAM object_blank as an OBJ mesh."""
    if session is None:
        session = NXOpen.Session.GetSession()
    if work_part is None:
        work_part = session.Parts.Work
    if object_blank is None or tool_name is None or output_path is None:
        raise ValueError("object_blank, tool_name, and output_path are all required")

    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "export_ipw_obj")
    try:
        nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
        method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
        tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
        operation = work_part.CAMSetup.CAMOperationCollection.Create(
            nCGroup,
            method,
            tool,
            object_blank,
            "mill_contour",
            "AREA_MILL",
            NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
            "AREA_MILL",
        )
        ipw = operation.GetInputIpw()
        ipw_objects = convert_facet_to_body(ipw)
        if not ipw_objects:
            raise RuntimeError("GetInputIpw returned no convertible body")
        return _export_nx_body_to_obj(session, ipw_objects[0], output_path)
    finally:
        _undo_to_mark_and_delete(session, markId, "export_ipw_obj")


def _octree_cell_subsample_offsets(samples_per_axis: int) -> np.ndarray:
    """Returns normalized offsets inside one octree cell."""
    samples_per_axis = max(1, int(samples_per_axis))
    axis_offsets = (np.arange(samples_per_axis, dtype=np.float64) + 0.5) / float(samples_per_axis) - 0.5
    grid = np.stack(np.meshgrid(axis_offsets, axis_offsets, axis_offsets, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def compute_octree_fill_fractions(
    contains_points_fn,
    centers_xyz,
    depths,
    bbox_min,
    bbox_max,
    samples_per_axis: int = 2,
) -> np.ndarray:
    """Estimates per-cell material fill fraction using fixed sub-cell samples."""
    centers = np.asarray(centers_xyz, dtype=np.float64).reshape(-1, 3)
    depths_arr = np.asarray(depths, dtype=np.int32).reshape(-1)
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    count = min(centers.shape[0], depths_arr.shape[0])
    centers = centers[:count]
    depths_arr = depths_arr[:count]
    bbox_min_arr = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max_arr = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    extent = np.maximum(bbox_max_arr - bbox_min_arr, 1e-9)
    offsets = _octree_cell_subsample_offsets(samples_per_axis)
    samples_per_cell = int(offsets.shape[0])
    fill = np.zeros((count,), dtype=np.float32)
    cell_chunk = max(1, int(os.getenv("OCTREE_FILL_CELL_CHUNK", "512")))

    for start in range(0, count, cell_chunk):
        stop = min(start + cell_chunk, count)
        cell_size = extent.reshape(1, 3) / np.power(2.0, depths_arr[start:stop].astype(np.float64)).reshape(-1, 1)
        points = centers[start:stop, None, :] + offsets.reshape(1, samples_per_cell, 3) * cell_size[:, None, :]
        labels = np.asarray(contains_points_fn(points.reshape(-1, 3)), dtype=np.float32).reshape(-1, samples_per_cell)
        fill[start:stop] = labels.mean(axis=1, dtype=np.float32)
    return fill


def compute_mesh_signed_distances(mesh: "trimesh.Trimesh", points) -> np.ndarray:
    """Returns signed distance in model units; inside material is negative."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    signed = np.zeros((pts.shape[0],), dtype=np.float32)
    chunk = max(1, int(os.getenv("OCTREE_TSDF_POINT_CHUNK", "4096")))
    try:
        prox = trimesh.proximity.ProximityQuery(mesh)
        for start in range(0, pts.shape[0], chunk):
            stop = min(start + chunk, pts.shape[0])
            # Use trimesh for distance magnitude, but use our bounded ray test
            # for the sign so the convention is stable: material = negative.
            dist = np.abs(np.asarray(prox.signed_distance(pts[start:stop]), dtype=np.float64))
            inside = _contains_points_mesh(mesh, pts[start:stop]) >= 0.5
            signed[start:stop] = np.where(inside, -dist, dist).astype(np.float32)
        return signed
    except Exception:
        from scipy.spatial import cKDTree

        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        tree = cKDTree(vertices)
        for start in range(0, pts.shape[0], chunk):
            stop = min(start + chunk, pts.shape[0])
            dist, _ = tree.query(pts[start:stop], k=1)
            inside = _contains_points_mesh(mesh, pts[start:stop]) >= 0.5
            signed[start:stop] = np.where(inside, -dist, dist).astype(np.float32)
        return signed


def compute_mesh_tsdf(mesh: "trimesh.Trimesh", points, truncation: float) -> np.ndarray:
    """Returns TSDF in [-1, 1], with negative values inside material."""
    tau = float(max(float(truncation), 1e-6))
    sdf = compute_mesh_signed_distances(mesh, points)
    return np.clip(sdf / tau, -1.0, 1.0).astype(np.float32)


class IpwOccupancySnapshot:
    """In-memory mesh snapshot for repeatable occupancy, fill, and TSDF queries."""

    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(self, points):
        return _contains_points_mesh(self.mesh, points)

    def signed_distances(self, points) -> np.ndarray:
        return compute_mesh_signed_distances(self.mesh, points)

    def tsdf(self, points, truncation: float) -> np.ndarray:
        return compute_mesh_tsdf(self.mesh, points, truncation)

    def fill_fractions(self, centers_xyz, depths, bbox_min, bbox_max, samples_per_axis: int = 2) -> np.ndarray:
        return compute_octree_fill_fractions(
            self,
            centers_xyz=centers_xyz,
            depths=depths,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            samples_per_axis=samples_per_axis,
        )


def load_obj_sdf_query(obj_path: str) -> IpwOccupancySnapshot:
    """Loads an OBJ mesh and returns the same query object used for IPW snapshots."""
    return IpwOccupancySnapshot(_load_obj_as_trimesh(obj_path))


def snapshot_ipw_occupancy_query(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
):
    """Snapshots an object_blank input IPW mesh and returns a point occupancy query."""
    if session is None:
        session = NXOpen.Session.GetSession()
    if work_part is None:
        work_part = session.Parts.Work
    if object_blank is None or tool_name is None:
        raise ValueError("object_blank and tool_name are required")

    session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
    work_part = session.Parts.Work
    session.CAMSession.PathDisplay.SetIpwResolution(NXOpen.CAM.PathDisplay.IpwResolutionType.Coarse)

    markId = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "snapshot_ipw_occ")
    try:
        nCGroup = work_part.CAMSetup.CAMGroupCollection.FindObject("NC_PROGRAM")
        method = work_part.CAMSetup.CAMGroupCollection.FindObject("METHOD")
        tool = work_part.CAMSetup.CAMGroupCollection.FindObject(tool_name)
        operation = work_part.CAMSetup.CAMOperationCollection.Create(
            nCGroup,
            method,
            tool,
            object_blank,
            "mill_contour",
            "AREA_MILL",
            NXOpen.CAM.OperationCollection.UseDefaultName.TrueValue,
            "AREA_MILL",
        )
        ipw = operation.GetInputIpw()
        ipw_objects = convert_facet_to_body(ipw)
        if not ipw_objects:
            raise RuntimeError("GetInputIpw returned no convertible body")
        with tempfile.TemporaryDirectory(prefix="ai_cam_snapshot_ipw_") as tmp_dir:
            obj_path = _export_nx_body_to_obj(session, ipw_objects[0], os.path.join(tmp_dir, "ipw.obj"))
            ipw_mesh = _load_obj_as_trimesh(obj_path)
    finally:
        _undo_to_mark_and_delete(session, markId, "snapshot_ipw_occ")

    return IpwOccupancySnapshot(ipw_mesh)


def query_ipw_fill_fractions_at_cells(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    centers_xyz=None,
    depths=None,
    bbox_min=None,
    bbox_max=None,
    samples_per_axis: int = 2,
):
    """Snapshots object_blank input IPW and estimates per-octree-cell fill fraction."""
    if centers_xyz is None or depths is None or bbox_min is None or bbox_max is None:
        raise ValueError("centers_xyz, depths, bbox_min, and bbox_max are required")
    snapshot = snapshot_ipw_occupancy_query(
        session=session,
        work_part=work_part,
        object_blank=object_blank,
        tool_name=tool_name,
    )
    return snapshot.fill_fractions(
        centers_xyz=centers_xyz,
        depths=depths,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        samples_per_axis=samples_per_axis,
    )


def query_ipw_tsdf_at_positions(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    centers_xyz=None,
    truncation: float = 5.0,
):
    """Snapshots object_blank input IPW and returns TSDF values at query positions."""
    if centers_xyz is None:
        raise ValueError("centers_xyz is required")
    snapshot = snapshot_ipw_occupancy_query(
        session=session,
        work_part=work_part,
        object_blank=object_blank,
        tool_name=tool_name,
    )
    return snapshot.tsdf(centers_xyz, truncation=truncation)


def _sample_mesh_surface_near_points(
    mesh: "trimesh.Trimesh",
    count: int,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Samples points near a mesh surface with optional Gaussian jitter."""
    count = int(max(0, count))
    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    except Exception:
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        if vertices.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        idx = rng.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
        points = vertices[idx]
    if float(jitter) > 0.0:
        points = points + rng.normal(0.0, float(jitter), size=points.shape)
    return points.astype(np.float32)


def _sample_regular_bbox_grid_points(
    bbox_min,
    bbox_max,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Samples a subset from a regular bbox grid for reconstruction-aware queries."""
    count = int(max(0, count))
    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    bbox_min_arr = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max_arr = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    cells_per_axis = int(max(2, math.ceil(float(count) ** (1.0 / 3.0)) + 1))
    axes = [
        np.linspace(float(bbox_min_arr[i]), float(bbox_max_arr[i]), cells_per_axis, dtype=np.float32)
        for i in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if grid.shape[0] > count:
        keep = rng.choice(grid.shape[0], size=count, replace=False)
        grid = grid[keep]
    elif grid.shape[0] < count:
        extra = rng.choice(grid.shape[0], size=count - grid.shape[0], replace=True)
        grid = np.vstack([grid, grid[extra]])
    return grid.astype(np.float32)


def sample_sdf_transition_query_points(
    before_snapshot: IpwOccupancySnapshot,
    after_snapshot: IpwOccupancySnapshot,
    target_snapshot: IpwOccupancySnapshot | None = None,
    bbox_min=None,
    bbox_max=None,
    count: int = 16384,
    region_points=None,
    focus_center=None,
    focus_radius=None,
    surface_jitter: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Samples 3D query points for TSDF transition learning.

    Distribution prioritises the GT/predicted affected-face region when
    available, then before/after surfaces, target CAD surface, and global bbox.
    This is intentionally not octree/depth based.
    """
    total = int(max(1, count))
    rng = np.random.default_rng(int(seed))
    bbox_min_arr = np.asarray(bbox_min if bbox_min is not None else before_snapshot.mesh.bounds[0], dtype=np.float64).reshape(3)
    bbox_max_arr = np.asarray(bbox_max if bbox_max is not None else before_snapshot.mesh.bounds[1], dtype=np.float64).reshape(3)
    bbox_min_arr = np.minimum(bbox_min_arr, np.asarray(before_snapshot.mesh.bounds[0], dtype=np.float64))
    bbox_max_arr = np.maximum(bbox_max_arr, np.asarray(before_snapshot.mesh.bounds[1], dtype=np.float64))
    bbox_min_arr = np.minimum(bbox_min_arr, np.asarray(after_snapshot.mesh.bounds[0], dtype=np.float64))
    bbox_max_arr = np.maximum(bbox_max_arr, np.asarray(after_snapshot.mesh.bounds[1], dtype=np.float64))
    if target_snapshot is not None:
        bbox_min_arr = np.minimum(bbox_min_arr, np.asarray(target_snapshot.mesh.bounds[0], dtype=np.float64))
        bbox_max_arr = np.maximum(bbox_max_arr, np.asarray(target_snapshot.mesh.bounds[1], dtype=np.float64))
    extent = np.maximum(bbox_max_arr - bbox_min_arr, 1e-6)
    pad = extent * 0.03
    bbox_min_arr = bbox_min_arr - pad
    bbox_max_arr = bbox_max_arr + pad

    region_arr = None
    if region_points is not None:
        region_arr = np.asarray(region_points, dtype=np.float64).reshape(-1, 3)
        if region_arr.shape[0] <= 0:
            region_arr = None

    n_region = int(total * 0.25) if region_arr is not None else 0
    n_before = int(total * 0.20)
    n_after = int(total * 0.20)
    n_target = int(total * 0.10) if target_snapshot is not None else 0
    n_uniform = int(total * 0.15)
    n_regular = int(total * 0.10)
    n_focus = int(total * 0.10) if region_arr is None and focus_center is not None and focus_radius is not None else 0
    n_global = max(0, total - n_region - n_before - n_after - n_target - n_uniform - n_regular - n_focus)

    parts = []
    if n_region > 0 and region_arr is not None:
        idx = rng.choice(region_arr.shape[0], size=n_region, replace=region_arr.shape[0] < n_region)
        region_sample = region_arr[idx]
        if float(surface_jitter) > 0.0:
            region_sample = region_sample + rng.normal(0.0, float(surface_jitter), size=region_sample.shape)
        parts.append(region_sample.astype(np.float32))
    parts.extend([
        _sample_mesh_surface_near_points(before_snapshot.mesh, n_before, surface_jitter, rng),
        _sample_mesh_surface_near_points(after_snapshot.mesh, n_after, surface_jitter, rng),
    ])
    if n_target > 0 and target_snapshot is not None:
        parts.append(_sample_mesh_surface_near_points(target_snapshot.mesh, n_target, surface_jitter, rng))
    if n_uniform > 0:
        parts.append(rng.uniform(bbox_min_arr, bbox_max_arr, size=(n_uniform, 3)).astype(np.float32))
    if n_regular > 0:
        parts.append(_sample_regular_bbox_grid_points(bbox_min_arr, bbox_max_arr, n_regular, rng))
    if n_focus > 0:
        center = np.asarray(focus_center, dtype=np.float64).reshape(1, 3)
        radius = float(max(float(focus_radius), 1e-6))
        parts.append((center + rng.normal(0.0, radius * 0.5, size=(n_focus, 3))).astype(np.float32))
    if n_global > 0:
        parts.append(rng.uniform(bbox_min_arr, bbox_max_arr, size=(n_global, 3)).astype(np.float32))

    points = np.vstack([p for p in parts if p.size > 0]).astype(np.float32)
    if points.shape[0] < total:
        extra = rng.uniform(bbox_min_arr, bbox_max_arr, size=(total - points.shape[0], 3)).astype(np.float32)
        points = np.vstack([points, extra])
    elif points.shape[0] > total:
        keep = rng.choice(points.shape[0], size=total, replace=False)
        points = points[keep]
    return points.astype(np.float32)


def _dedupe_octree_cells(centers: np.ndarray, depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicates cells by rounded center and depth while preserving order."""
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    depths_arr = np.asarray(depths, dtype=np.int16).reshape(-1)
    count = min(centers_arr.shape[0], depths_arr.shape[0])
    centers_arr = centers_arr[:count]
    depths_arr = depths_arr[:count]
    seen: set[tuple] = set()
    keep: list[int] = []
    for idx, (center, depth) in enumerate(zip(centers_arr, depths_arr)):
        key = (
            int(depth),
            round(float(center[0]), 6),
            round(float(center[1]), 6),
            round(float(center[2]), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    if not keep:
        return centers_arr[:0], depths_arr[:0]
    keep_arr = np.asarray(keep, dtype=np.int64)
    return centers_arr[keep_arr], depths_arr[keep_arr]


def _sample_focus_octree_cells(
    focus_center,
    focus_radius,
    bbox_min,
    bbox_max,
    fine_depth: int,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples fine-depth cells around the action focus region."""
    if focus_center is None or focus_radius is None or int(max_nodes) <= 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int16)
    center = np.asarray(focus_center, dtype=np.float64).reshape(3)
    radius = float(max(float(focus_radius), 1e-9))
    bbox_min_arr = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max_arr = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    extent = np.maximum(bbox_max_arr - bbox_min_arr, 1e-9)
    depth = int(max(0, fine_depth))
    cell = extent / float(2 ** depth)
    local_min = np.maximum(center - radius, bbox_min_arr)
    local_max = np.minimum(center + radius, bbox_max_arr)
    if np.any(local_max <= local_min):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int16)

    idx_min = np.floor((local_min - bbox_min_arr) / cell).astype(np.int64)
    idx_max = np.ceil((local_max - bbox_min_arr) / cell).astype(np.int64)
    grid_max = int(2 ** depth)
    idx_min = np.clip(idx_min, 0, grid_max - 1)
    idx_max = np.clip(idx_max, 0, grid_max)
    ranges = [np.arange(idx_min[axis], idx_max[axis], dtype=np.int64) for axis in range(3)]
    if any(r.size == 0 for r in ranges):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int16)

    indices = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    centers = bbox_min_arr.reshape(1, 3) + (indices.astype(np.float64) + 0.5) * cell.reshape(1, 3)
    inside_focus = np.linalg.norm(centers - center.reshape(1, 3), axis=1) <= radius
    centers = centers[inside_focus]
    if centers.shape[0] > int(max_nodes):
        rng = np.random.default_rng(0)
        keep = rng.choice(centers.shape[0], size=int(max_nodes), replace=False)
        centers = centers[keep]
    depths = np.full((centers.shape[0],), depth, dtype=np.int16)
    return centers.astype(np.float32), depths


def sample_transition_ipw_octree_state(
    session=None,
    work_part=None,
    object_blank_after=None,
    tool_name=None,
    before_snapshot: IpwOccupancySnapshot | None = None,
    bbox_min=None,
    bbox_max=None,
    coarse_depth: int = 3,
    fine_depth: int = 5,
    max_nodes: int = 16384,
    bbox_padding: float = 0.05,
    focus_center=None,
    focus_radius=None,
):
    """Samples octree cells from before/after boundary union plus changed/focus cells."""
    if before_snapshot is None:
        raise ValueError("before_snapshot is required")
    if object_blank_after is None or tool_name is None or bbox_min is None or bbox_max is None:
        raise ValueError("object_blank_after, tool_name, bbox_min, and bbox_max are required")

    after_snapshot = snapshot_ipw_occupancy_query(
        session=session,
        work_part=work_part,
        object_blank=object_blank_after,
        tool_name=tool_name,
    )
    bbox_min_use = np.asarray(bbox_min, dtype=np.float32).reshape(3)
    bbox_max_use = np.asarray(bbox_max, dtype=np.float32).reshape(3)
    bbox_min_use = np.minimum(bbox_min_use, np.asarray(before_snapshot.mesh.bounds[0], dtype=np.float32))
    bbox_max_use = np.maximum(bbox_max_use, np.asarray(before_snapshot.mesh.bounds[1], dtype=np.float32))
    bbox_min_use = np.minimum(bbox_min_use, np.asarray(after_snapshot.mesh.bounds[0], dtype=np.float32))
    bbox_max_use = np.maximum(bbox_max_use, np.asarray(after_snapshot.mesh.bounds[1], dtype=np.float32))
    bbox_extent = np.maximum(bbox_max_use - bbox_min_use, 1e-6)
    bbox_pad = bbox_extent * float(max(bbox_padding, 0.0))
    bbox_min_use = bbox_min_use - bbox_pad
    bbox_max_use = bbox_max_use + bbox_pad

    before_centers, before_depths, _ = sample_octree_occupancy(
        body=None,
        bbox_min=bbox_min_use,
        bbox_max=bbox_max_use,
        coarse_depth=coarse_depth,
        fine_depth=fine_depth,
        max_nodes=max_nodes,
        contains_points_fn=before_snapshot,
    )
    after_centers, after_depths, _ = sample_octree_occupancy(
        body=None,
        bbox_min=bbox_min_use,
        bbox_max=bbox_max_use,
        coarse_depth=coarse_depth,
        fine_depth=fine_depth,
        max_nodes=max_nodes,
        contains_points_fn=after_snapshot,
    )
    focus_budget = max(0, int(max_nodes) // 4)
    focus_centers, focus_depths = _sample_focus_octree_cells(
        focus_center=focus_center,
        focus_radius=focus_radius,
        bbox_min=bbox_min_use,
        bbox_max=bbox_max_use,
        fine_depth=fine_depth,
        max_nodes=focus_budget,
    )

    centers_all = np.vstack([before_centers, after_centers, focus_centers]).astype(np.float32)
    depths_all = np.concatenate([before_depths, after_depths, focus_depths]).astype(np.int16)
    centers_all, depths_all = _dedupe_octree_cells(centers_all, depths_all)
    before_labels_all = np.asarray(before_snapshot(centers_all), dtype=np.float32).reshape(-1)
    after_labels_all = np.asarray(after_snapshot(centers_all), dtype=np.float32).reshape(-1)

    count = centers_all.shape[0]
    if count > int(max_nodes):
        changed = before_labels_all != after_labels_all
        fine = depths_all == int(fine_depth)
        focus_like = np.zeros((count,), dtype=bool)
        if focus_centers.shape[0] > 0:
            focus_center_arr = np.asarray(focus_center, dtype=np.float32).reshape(1, 3)
            focus_radius_f = float(max(float(focus_radius), 1e-9))
            focus_like = np.linalg.norm(centers_all - focus_center_arr, axis=1) <= focus_radius_f
        priority = changed.astype(np.int32) * 4 + focus_like.astype(np.int32) * 2 + fine.astype(np.int32)
        order = np.lexsort((np.arange(count), -priority))
        keep = order[: int(max_nodes)]
        centers_all = centers_all[keep]
        depths_all = depths_all[keep]
        before_labels_all = before_labels_all[keep]
        after_labels_all = after_labels_all[keep]

    return (
        centers_all.astype(np.float32),
        depths_all.astype(np.int16),
        after_labels_all.astype(np.float32),
        before_labels_all.astype(np.float32),
        bbox_min_use.astype(np.float32),
        bbox_max_use.astype(np.float32),
    )


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


def measure_ipw_state_detailed(
    session=None,
    work_part=None,
    object_blank=None,
    tool_name=None,
    points_array=None,
    norm_vecs_array=None,
    lines_array=None,
    savepath=None,
):
    """Measures IPW volume and both face-level and point-level deviations for the current setup."""
    return get_ipw_property_detailed(
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
