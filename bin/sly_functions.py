import glob
import os

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

            for frame_index in range(obj.start_frame, 5):  # DEBUG
                # for frame_index in range(obj.start_frame, obj.end_frame):
                geometry = kitti_360_helpers.convert_kitti_cuboid_to_supervisely_geometry(obj, frame_index)
                frame2figures.setdefault(frame_index, []).append(supervisely.PointcloudFigure(pcobj, geometry,
                                                                                              frame_index=frame_index))
    return frame2figures, ann_objects


def convert_kitty_to_supervisely(annotations_object, project_meta):
    frames2figures, ann_objects = frames_to_figures_dict(annotations_object, project_meta)

    frames_list = []

    for frame_index in frames2figures.keys():
        figures_on_frame = frames2figures.get(frame_index, [])
        frames_list.append(supervisely.Frame(frame_index, figures_on_frame))

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
        supervisely.logger.info(f'Annotations for {seq_to_process} not found;')


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
            pcl_path, filename = pointcloud_episode.bin_to_pcl(current_bin_path)  # save pcl to episode
            pcl_episodes_dataset.add_item_file(filename, pcl_path)

            frame2pcl[frame_num] = filename

            if frame_num == 5:
                break  # DEBUG

        process_annotations(current_seq, pcl_episodes_project, pcl_episodes_dataset)
        pointcloud_episode.save_frame_to_pcl_mapping(pcl_episodes_dataset, frame2pcl)  # save frame2pcl mapping
