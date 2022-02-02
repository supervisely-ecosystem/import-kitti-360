import os

import numpy as np
from tqdm import tqdm

from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid, loadCalibrationCameraToPose, \
    loadPerspectiveIntrinsic

import sly_progress
import sly_globals as g


def get_perspective_intrinsic(filepath):
    return loadPerspectiveIntrinsic(filepath)


def get_cam_to_velodyne_rigid(filepath):
    return loadCalibrationRigid(filepath)


def get_cam_to_world_rigid(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError

    cam2world_rows = np.loadtxt(file_path)
    cam2world_rigid = np.reshape(cam2world_rows[:, 1:], (-1, 4, 4))
    frames_numbers = list(np.reshape(cam2world_rows[:, :1], (-1)).astype(int))
    cam2world = {}

    current_rigid = cam2world_rigid[0]

    progress_cb = sly_progress.get_progress_cb(g.api, g.TASK_ID, f"Loading poses matrices",
                                               total=frames_numbers[-1] + 1)

    for frame_index in range(0, frames_numbers[-1]):
        if frame_index in frames_numbers:
            mapped_index = frames_numbers.index(frame_index)
            current_rigid = cam2world_rigid[mapped_index]

        # (Tr(cam -> world))
        cam2world[frame_index] = current_rigid
        progress_cb(1)
    return cam2world


if __name__ == '__main__':
    intrinsic_filepath = os.path.join(g.calibrations_path, 'perspective.txt')
    intrinsic_calibrations = get_perspective_intrinsic(intrinsic_filepath)['P_rect_00']
