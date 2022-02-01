import glob
import os
import shutil

import numpy as np

import kitti_360_helpers
import pointcloud_episode
import sly_globals as g
import supervisely
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

from kitti_360_helpers import get_kitti_360_data
from pointcloud_episode import upload_pcl_episodes_project


def get_project_meta(labels, geometry=Cuboid3d):
    unique_labels = set()
    for globalId, v in labels.objects.items():
        if len(v) > 1:
            continue  # @TODO dynamic objects
        for obj in v.values():
            unique_labels.add(obj.name)
    obj_classes = [supervisely.ObjClass(k, geometry) for k in unique_labels]
    return supervisely.ProjectMeta(obj_classes=supervisely.ObjClassCollection(obj_classes))


def frames_to_figures_dict(annotations_object, project_meta):
    frame2figures = {}
    ann_objects = set()

    for globalId, v in annotations_object.objects.items():
        if len(v) > 1:
            continue  # @TODO dynamic objects
        for obj in v.values():
            pcobj = supervisely.PointcloudObject(project_meta.get_obj_class(obj.name))
            ann_objects.add(pcobj)

            # for frame_index in range(obj.start_frame, 5):
            for frame_index in range(obj.start_frame, obj.end_frame):
                geometry = kitti_360_helpers.convert_kitti_cuboid_to_supervisely_geometry(obj, frame_index)
                frame2figures.setdefault(frame_index, []).append(supervisely.PointcloudFigure(pcobj, geometry,
                                                                                              frame_index=frame_index))
    return frame2figures, ann_objects


def convert_kitty_to_supervisely(annotations_object, project_meta):
    frames2figures, ann_objects = frames_to_figures_dict(annotations_object, project_meta)

    frames_list = []

    for frame_index in range(0, list(frames2figures.keys())[-1] + 1):
        figures_on_frame = frames2figures.get(frame_index, [])
        frames_list.append(supervisely.Frame(frame_index, figures_on_frame))

        if frame_index == 20:  # DEBUG
            break

    frames_collection = supervisely.FrameCollection(frames_list)
    return supervisely.PointcloudEpisodeAnnotation(frames_count=len(frames_list),
                                                   objects=PointcloudObjectCollection(ann_objects),
                                                   frames=frames_collection)


def get_annotations_in_supervisely_format(shapes_path):
    kitti_annotations_object = kitti_360_helpers.Annotation3D(shapes_path)
    project_meta = get_project_meta(kitti_annotations_object)
    return convert_kitty_to_supervisely(kitti_annotations_object, project_meta), project_meta


def update_project_meta(pcl_episodes_project, project_meta):
    old_meta = pcl_episodes_project.meta
    updated_meta = old_meta.merge(project_meta)
    pcl_episodes_project.set_meta(updated_meta)


def process_annotations(seq_to_process, pcl_episodes_project, pcl_episodes_dataset):
    path_to_kitti_annotations = os.path.join(g.bboxes_path, f'{seq_to_process}.xml')
    if os.path.isfile(path_to_kitti_annotations):
        episode_annotations, project_meta = get_annotations_in_supervisely_format(
            shapes_path=path_to_kitti_annotations)

        pcl_episodes_dataset.set_ann(episode_annotations)
        update_project_meta(pcl_episodes_project, project_meta)

    else:
        supervisely.logger.info(f'Annotations for {seq_to_process} not found')


def convert_kitti360_to_supervisely_pcl_episodes_project():
    pcl_episodes_project = pointcloud_episode.create_empty_pcl_episodes_project()
    pcl_episodes_project.set_meta(supervisely.ProjectMeta())

    seq_to_process = sorted([seq_name for seq_name in os.listdir(g.seq_dir_path)  # filter directories
                             if os.path.isdir(os.path.join(g.seq_dir_path, seq_name))])

    for current_seq in seq_to_process:  # for each episode
        frame2pcl = {}

        g.cam2world = kitti_360_helpers.get_cam_to_world_rigid(os.path.join(g.poses_path.format(current_seq),
                                                                            'cam0_to_world.txt'))

        bins_paths = sorted(glob.glob(os.path.join(g.bins_dir_path.format(current_seq), '*.bin')))  # pointclouds paths
        pcl_episodes_dataset = pcl_episodes_project.create_dataset(f'{current_seq}')

        for frame_num, current_bin_path in enumerate(bins_paths, start=0):
            pcl_path, pcl_filename = pointcloud_episode.bin_to_pcl(current_bin_path)  # add pointcloud
            pcl_episodes_dataset.add_item_file(pcl_filename, pcl_path)

            img_filename = pcl_filename.replace('.pcd', '.png')
            image_path = os.path.join(g.photocontext_path.format(current_seq), img_filename)

            image_info = get_image_info(image_name=img_filename)

            if os.path.isfile(image_path):  # if image existing
                add_photo_context(pcl_episodes_dataset=pcl_episodes_dataset,  # add photo context
                                  pcl_name=pcl_filename,
                                  image_path=image_path,
                                  img_info=image_info)

            frame2pcl[frame_num] = pcl_filename  # updating mappings

            if frame_num == 5:
                break  # DEBUG

        process_annotations(current_seq, pcl_episodes_project, pcl_episodes_dataset)
        pointcloud_episode.save_frame_to_pcl_mapping(pcl_episodes_dataset, frame2pcl)  # save frame2pcl mapping


# move to submodules
def get_related_images_dir_path_by_dataset(pcl_episodes_dataset, pointcloud_name):
    related_images_path = pcl_episodes_dataset.get_related_images_path(pointcloud_name)
    os.makedirs(related_images_path, exist_ok=True)

    return related_images_path


def add_photo_context(pcl_episodes_dataset, pcl_name, image_path, img_info):
    """
    @TODO move to SDK
    """
    related_images_dir_path = get_related_images_dir_path_by_dataset(pcl_episodes_dataset, pcl_name)
    os.makedirs(related_images_dir_path, exist_ok=True)

    sly_path_img = os.path.join(related_images_dir_path, pcl_name.replace('.pcd', '.png'))

    shutil.copy(src=image_path, dst=sly_path_img)  # copy image to project
    if img_info is not None:
        supervisely.json.dump_json_file(img_info, sly_path_img + '.json')  # add img_info to project


def get_image_info(image_name, camera_num=0):
    intrinsic_matrix = g.intrinsic_calibrations[f'P_rect_0{camera_num}'][:3, :3]  # now only for zero_cam
    extrinsic_matrix = np.linalg.inv(g.cam2velo)[:3, :4]

    return {
        "name": image_name,
        "meta": {
            "deviceId": f'P_rect_0{camera_num}',
            "sensorsData": {
                "extrinsicMatrix": list(extrinsic_matrix.flatten().astype(float)),
                "intrinsicMatrix": list(intrinsic_matrix.flatten().astype(float))
            }
        }
    }
