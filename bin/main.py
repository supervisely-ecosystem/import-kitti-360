import glob
import os
import shutil

import numpy as np

import supervisely

import sly_globals as g
import sly_functions as f

if __name__ == '__main__':

    f.get_kitti_360_data()
    pcl_project = f.create_empty_pcl_episodes_project()



    pcl_dataset = pcl_project.create_dataset('main')

    frames2annotations, project_meta = f.get_annotations_in_supervisely_format(
        shapes_path='../data_3d_bboxes/train/2013_05_28_drive_0000_sync.xml')
    pcl_project.set_meta(project_meta)

    bin_files_paths = sorted(glob.glob(os.path.join(g.bins_dir_path, '*')))[:4]  # DEBUG

    for frame_index, bin_file_path in enumerate(bin_files_paths):
        item_name = supervisely.fs.get_file_name(bin_file_path) + ".pcd"
        item_path = pcl_dataset.generate_item_path(item_name)

        bin_file = f.get_bin_file_by_path(bin_file_path)
        f.convert_bin_to_pcd(bin_file, item_path)

        frame_annotations = frames2annotations.get(frame_index)
        pcl_dataset.add_item_file(item_name, item_path, ann=frame_annotations)


    f.upload_pcl_project()
