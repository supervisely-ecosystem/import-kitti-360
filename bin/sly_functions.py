import numpy as np
from tqdm import tqdm

import kitti_annotation
import open3d

import supervisely
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.api.module_api import ApiField

import sly_globals as g




def get_project_meta(labels, geometry=Cuboid3d):
    unique_labels = set()
    for globalId, v in labels.objects.items():
        if len(v) > 1:
            continue  # @TODO dynamic objects
        for obj in v.values():
            unique_labels.add(obj.name)
    obj_classes = [supervisely.ObjClass(k, geometry) for k in unique_labels]
    return supervisely.ProjectMeta(obj_classes=supervisely.ObjClassCollection(obj_classes))



def project(points, R, T, inverse=False):
    assert (points.ndim == R.ndim)
    assert (T.ndim == R.ndim or T.ndim == (R.ndim - 1))
    ndim = R.ndim
    if ndim == 2:
        R = np.expand_dims(R, 0)
        T = np.reshape(T, [1, -1, 3])
        points = np.expand_dims(points, 0)
    if not inverse:
        points = np.matmul(R, points.transpose(0, 2, 1)).transpose(0, 2, 1) + T
    else:
        points = np.matmul(R.transpose(0, 2, 1), (points - T).transpose(0, 2, 1))

    if ndim == 2:
        points = points[0]

    return points


def convert_world_coordinates_to_velodyne(obj, current_frame):
    R = g.world2cam[current_frame][:3, :3]
    T = g.world2cam[current_frame][:3, 3]

    # vert = np.matmul(R, (obj.vertices - T).T).T  # 1 — inv cam2world

    # R = g.cam2velodyne['image_00'][:3, :3]
    # T = g.cam2velodyne['image_00'][:3, 3]
    #
    # vert = np.matmul(R, vert.T).T + T  # 2 — cam2velodyne

    return obj.vertices - T


def convert_kitti_cuboid_to_supervisely_geometry(obj, current_frame):
    """
    TO UNDERSTAND SUPERVISELY FORMAT
    bbox = l.to_xyzwhlr()
    dim = bbox[[3, 5, 4]]
    pos = bbox[:3] + [0, 0, dim[1] / 2]
    yaw = bbox[-1]
    position = Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
    rotation = Vector3d(0, 0, float(-yaw))

    dimension = Vector3d(float(dim[0]), float(dim[2]), float(dim[1]))
    geometry = Cuboid3d(position, rotation, dimension)
    geometries.append(geometry)
    """
    # vertices = convert_world_coordinates_to_velodyne(obj, current_frame)
    vertices = obj.vertices

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(obj.faces)
    # mesh.compute_vertex_normals() #?

    min_bound, max_bound = mesh.get_min_bound(), mesh.get_max_bound()  # x, y, z

    x, y, z = min_bound[0], min_bound[1], min_bound[2]  # (max_bound[2] + min_bound[2]) / 2

    position = Vector3d(x, y, z)
    rotation = Vector3d(0, 0, 0)

    w, h, l = max_bound[0] - min_bound[0], max_bound[1] - min_bound[1], max_bound[2] - min_bound[2]
    dimension = Vector3d(w, h, l)

    # open3d.visualization.draw_geometries([mesh])

    return Cuboid3d(position, rotation, dimension)


def frames_to_figures_dict(annotations_object, project_meta):
    frame2figures = {}
    frame2objs = {}

    for globalId, v in annotations_object.objects.items():
        if len(v) > 1:
            continue  # @TODO dynamic objects
        for obj in v.values():
            pcobj = supervisely.PointcloudObject(project_meta.get_obj_class(obj.name))
            for frame_index in range(obj.start_frame, 5):
            # for frame_index in range(obj.start_frame, obj.end_frame):
                geometry = convert_kitti_cuboid_to_supervisely_geometry(obj, frame_index)
                frame2figures.setdefault(frame_index, []).append(supervisely.PointcloudFigure(pcobj, geometry))
                frame2objs.setdefault(frame_index, []).append(pcobj)

    return frame2figures, frame2objs


def convert_kitty_to_supervisely(annotations_object, project_meta):
    frames2figures, frames2objs = frames_to_figures_dict(annotations_object, project_meta)

    frames2annotations = {}

    for frame_index in frames2figures.keys():
        figures_on_frame = frames2figures.get(frame_index, [])
        objs_on_frame = frames2objs.get(frame_index, [])

        frames2annotations[frame_index] = supervisely.PointcloudAnnotation(PointcloudObjectCollection(objs_on_frame),
                                                                           figures_on_frame)

    return frames2annotations


def get_annotations_in_supervisely_format(shapes_path):
    kitti_annotations_object = kitti_annotation.Annotation3D(shapes_path)
    project_meta = get_project_meta(kitti_annotations_object)
    return convert_kitty_to_supervisely(kitti_annotations_object, project_meta), project_meta


def upload_pcl_project():
    project = g.api.project.create(g.WORKSPACE_ID,
                                   g.project_name,
                                   type=supervisely.ProjectType.POINT_CLOUDS,
                                   change_name_if_conflict=True)

    project_fs = supervisely.PointcloudProject.read_single(g.project_dir_path)

    g.api.project.update_meta(project.id, project_fs.meta.to_json())
    supervisely.logger.info("Project {!r} [id={!r}] has been created".format(project.name, project.id))

    uploaded_objects = KeyIdMap()

    for dataset_fs in project_fs:
        dataset = g.api.dataset.create(project.id, dataset_fs.name, change_name_if_conflict=True)
        supervisely.logger.info("dataset {!r} [id={!r}] has been created".format(dataset.name, dataset.id))

        pbar = tqdm(desc='uploading items', total=len(dataset_fs))
        progress_items_cb = pbar.update

        for item_name in dataset_fs:
            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)

            item_meta = {}
            pointcloud = g.api.pointcloud.upload_path(dataset.id, item_name, item_path, item_meta)

            # validate_item_annotation
            ann_json = supervisely.io.json.load_json_file(ann_path)
            ann = supervisely.PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            # ignore existing key_id_map because the new objects will be created
            g.api.pointcloud.annotation.append(pointcloud.id, ann, uploaded_objects)

            # upload related_images if exist
            related_items = dataset_fs.get_related_images(item_name)
            if len(related_items) != 0:
                rimg_infos = []
                for img_path, meta_json in related_items:
                    # img_name = supervisely.fs.get_file_name(img_path)
                    img = g.api.pointcloud.upload_related_image(img_path)[0]
                    rimg_infos.append({ApiField.ENTITY_ID: pointcloud.id,
                                       ApiField.NAME: meta_json[ApiField.NAME],
                                       ApiField.HASH: img,
                                       ApiField.META: meta_json[ApiField.META]})

                g.api.pointcloud.add_related_images(rimg_infos)
            progress_items_cb(1)

    g.my_app.show_modal_window(f"'{project.name}' project has been successfully imported.")


def load_pose_to_world_data(pose_file_path):
    pose2world_rows = np.loadtxt(pose_file_path)

    pose2world_data = np.reshape(pose2world_rows[:, 1:], [-1, 3, 4])
    frames_numbers = list(np.reshape(pose2world_rows[:, :1], (-1)).astype(int))
    pose2world = {}

    current_data = pose2world_data[0]
    for frame_index in range(1, frames_numbers[-1] + 1):
        if frame_index in frames_numbers:
            mapped_index = frames_numbers.index(frame_index)
            current_data = pose2world_data[mapped_index]

        pose2world[frame_index] = current_data

        if frame_index > 20:  # DEBUG
            break
    return pose2world


def get_bin_file_by_path(bin_file_path):
    return np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)


def convert_bin_to_pcd(bin, save_filepath):
    points = bin[:, 0:3]
    intensity = bin[:, -1]
    intensity_fake_rgb = np.zeros((intensity.shape[0], 3))
    intensity_fake_rgb[:, 0] = intensity
    pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    pc.colors = open3d.utility.Vector3dVector(intensity_fake_rgb)
    open3d.io.write_point_cloud(save_filepath, pc)


def apply_transformation(transformation, points, inverse=False):
    R = transformation[:3, :3]
    T = transformation[:3, 3]

    if not inverse:
        return np.matmul(R, points.transpose()).transpose() + T
    else:
        return np.matmul(R, (points - T).transpose()).transpose()

