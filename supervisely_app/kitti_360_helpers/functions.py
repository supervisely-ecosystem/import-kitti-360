import os

import numpy as np
import open3d
import transforms3d
from scipy.spatial.transform.rotation import Rotation

import kitti_360_helpers
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

import sly_globals as g
import sly_progress


def download_raw_project():
    if not g.api.file.dir_exists(g.TEAM_ID, g.kitti360_remote_dir):
        raise FileExistsError(f'Directory {g.kitti360_remote_dir} not exists')

    dir_size_in_bytes = g.api.file.get_directory_size(g.TEAM_ID, g.kitti360_remote_dir)
    progress_cb = sly_progress.get_progress_cb(g.api, g.TASK_ID, f"Downloading {g.kitti360_remote_dir}",
                                               dir_size_in_bytes, is_size=True)

    remote_dir = f"{g.kitti360_remote_dir}/"
    print(remote_dir)
    print(g.kitti360_local_dir)
    print(f"File exists: {g.api.file.exists(g.TEAM_ID, remote_dir)}")
    print(f"Dir exists: {g.api.file.dir_exists(g.TEAM_ID, remote_dir)}")
    print(f"Isdir: {os.path.isdir(g.kitti360_local_dir)}")
    if not os.path.isdir(g.kitti360_local_dir):
        g.api.file.download_directory(g.TEAM_ID,
                                      remote_path=remote_dir,
                                      local_save_path=g.kitti360_local_dir,
                                      progress_cb=progress_cb)


def load_static_transformations():
    g.cam2velo = kitti_360_helpers.get_cam_to_velodyne_rigid(os.path.join(g.calibrations_path, 'calib_cam_to_velo.txt'))

    g.intrinsic_calibrations = \
        kitti_360_helpers.get_perspective_intrinsic(os.path.join(g.calibrations_path, 'perspective.txt'))


def get_kitti_360_data():
    download_raw_project()
    load_static_transformations()


def world_to_velo_transformation(obj, frame_index):
    # rotate_z = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()
    # rotate_z = np.hstack((rotate_z, np.asarray([[0, 0, 0]]).T))

    # tr0(local -> fixed_coordinates_local)
    tr0 = np.asarray([[0, -1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    # tr0(fixed_coordinates_local -> world)
    tr1 = obj.transform

    # tr2(world -> cam)
    tr2 = np.linalg.inv(g.cam2world[frame_index])

    # tr3(world -> cam)
    tr3 = g.cam2velo

    return tr3 @ tr2 @ tr1 @ tr0


def get_mesh_from_object(obj):
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(obj.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(obj.faces)

    return mesh


def convert_kitti_cuboid_to_supervisely_geometry(obj, current_frame):
    transformation_matrix = world_to_velo_transformation(obj, current_frame)

    Tdash, Rdash, Zdash, Sdash = transforms3d.affines.decompose44(transformation_matrix)

    x, y, z = Tdash[0], Tdash[1], Tdash[2]
    position = Vector3d(x, y, z)

    rotation_angles = Rotation.from_matrix(Rdash).as_euler('xyz', degrees=False)
    r_x, r_y, r_z = rotation_angles[0], rotation_angles[1], rotation_angles[2]

    rotation = Vector3d(r_x, r_y, r_z)

    w, h, l = Zdash[0], Zdash[1], Zdash[2]
    dimension = Vector3d(w, h, l)

    return Cuboid3d(position, rotation, dimension)


def visualize(geometries):
    viewer = open3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
