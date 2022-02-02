import os
import shutil

import numpy as np
import open3d

import sly_globals as g
import supervisely
from supervisely.io.json import dump_json_file
from supervisely.project.pointcloud_episode_project import upload_pointcloud_episode_project


def save_frame_to_pcl_mapping(episode_ds, frame2pcl):
    frame2pcl_map_path = episode_ds.get_frame_pointcloud_map_path()
    dump_json_file(frame2pcl, frame2pcl_map_path)


def convert_bin_to_pcd(bin, save_filepath):
    points = bin[:, 0:3]
    intensity = bin[:, -1]
    intensity_fake_rgb = np.zeros((intensity.shape[0], 3))
    intensity_fake_rgb[:, 0] = intensity
    pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    pc.colors = open3d.utility.Vector3dVector(intensity_fake_rgb)
    open3d.io.write_point_cloud(save_filepath, pc)


def bin_to_pcl(current_bin_path):
    bin_file = get_bin_file_by_path(current_bin_path)  # read bin

    pathname, extension = os.path.splitext(current_bin_path)
    pcl_path = f'{pathname}.pcd'
    filename = os.path.basename(pcl_path)

    convert_bin_to_pcd(bin_file, pcl_path)  # convert to pcd and save
    return pcl_path, filename


def upload_pcl_episodes_project():
    project_id, project_name = upload_pointcloud_episode_project(g.project_dir_path, g.api, g.WORKSPACE_ID,
                                                                 project_name=g.project_name, log_progress=True)
    g.api.task.set_output_project(g.TASK_ID, project_id, project_name)


def get_bin_file_by_path(bin_file_path):
    return np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)


def create_empty_pcl_episodes_project():
    if os.path.isdir(g.project_dir_path):
        shutil.rmtree(g.project_dir_path, ignore_errors=False)
    pcl_project = supervisely.PointcloudEpisodeProject(g.project_dir_path, supervisely.OpenMode.CREATE)

    return pcl_project


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
                "intrinsicMatrix": list(intrinsic_matrix.flatten().astype(float)),
                "extrinsicMatrix": list(extrinsic_matrix.flatten().astype(float))
            }
        }
    }
