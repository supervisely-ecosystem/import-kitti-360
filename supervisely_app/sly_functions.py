import glob
import os

import kitti_360_helpers
import pointcloud_episode
import sly_globals as g
import sly_progress
import supervisely
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection



def get_project_meta(labels, geometry=Cuboid3d):
    unique_labels = set()
    for globalId, v in labels.objects.items():
        if len(v) > 0:
            unique_labels.add(list(v.values())[0].name)

    obj_classes = [supervisely.ObjClass(k, geometry) for k in unique_labels]
    return supervisely.ProjectMeta(obj_classes=supervisely.ObjClassCollection(obj_classes))


def frames_to_figures_dict(annotations_object, project_meta):
    frame2figures = {}
    ann_objects = set()

    progress_cb = sly_progress.get_progress_cb(g.api, g.TASK_ID, f"Converting annotations",
                                               total=len(annotations_object.objects.keys()))

    for globalId, v in annotations_object.objects.items():
        if len(v) > 0:
            first_object = list(v.values())[0]
            pcobj = supervisely.PointcloudObject(project_meta.get_obj_class(first_object.name))
            ann_objects.add(pcobj)

            if len(v) > 1:  # dynamic objects {frame: annotation}
                start_frame, end_frame = sorted(list(v.keys()))[0], sorted(list(v.keys()))[-1]
            else:  # static objects {-placeholder: annotation}
                start_frame, end_frame = first_object.start_frame, first_object.end_frame

            actual_object = v[start_frame] if len(v) > 1 else first_object  # get first dynamic / static object
            for frame_index in range(start_frame, end_frame):
                if v.get(frame_index, None) is not None:
                    actual_object = v.get(frame_index)

                geometry = kitti_360_helpers.convert_kitti_cuboid_to_supervisely_geometry(actual_object,
                                                                                          frame_index)
                frame2figures.setdefault(frame_index, []).append(supervisely.PointcloudFigure(pcobj, geometry,
                                                                                              frame_index=frame_index))

            progress_cb(1)

    return frame2figures, ann_objects


def convert_kitty_to_supervisely(annotations_object, project_meta):
    frames2figures, ann_objects = frames_to_figures_dict(annotations_object, project_meta)

    frames_list = []

    progress_cb = sly_progress.get_progress_cb(g.api, g.TASK_ID, f"Mapping annotations",
                                               total=list(frames2figures.keys())[-1] + 1)

    for frame_index in range(0, list(frames2figures.keys())[-1] + 1):
        figures_on_frame = frames2figures.get(frame_index, [])
        frames_list.append(supervisely.Frame(frame_index, figures_on_frame))

        progress_cb(1)

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

        progress_cb = sly_progress.get_progress_cb(g.api, g.TASK_ID, f"Processing pointclouds",
                                                   total=len(bins_paths))

        for frame_num, current_bin_path in enumerate(bins_paths, start=0):
            pcl_path, pcl_filename = pointcloud_episode.bin_to_pcl(current_bin_path)  # add pointcloud
            pcl_episodes_dataset.add_item_file(pcl_filename, pcl_path)

            img_filename = pcl_filename.replace('.pcd', '.png')
            image_path = os.path.join(g.photocontext_path.format(current_seq), img_filename)

            image_info = pointcloud_episode.get_image_info(image_name=img_filename)

            if os.path.isfile(image_path):  # if image existing
                pointcloud_episode.add_photo_context(pcl_episodes_dataset=pcl_episodes_dataset,  # add photo context
                                                     pcl_name=pcl_filename,
                                                     image_path=image_path,
                                                     img_info=image_info)

            frame2pcl[frame_num] = pcl_filename  # updating mappings

            progress_cb(1)

        process_annotations(current_seq, pcl_episodes_project, pcl_episodes_dataset)
        pointcloud_episode.save_frame_to_pcl_mapping(pcl_episodes_dataset, frame2pcl)  # save frame2pcl mapping
