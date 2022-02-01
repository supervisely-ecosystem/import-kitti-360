import os

import numpy as np
from tqdm import tqdm

from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid, loadCalibrationCameraToPose, \
    loadPerspectiveIntrinsic

import sly_globals as g


def get_perspective_intrinsic(filepath):
    return loadPerspectiveIntrinsic(filepath)


def get_cam_to_velodyne_rigid(filepath):
    return loadCalibrationRigid(filepath)


def get_cam_to_world_rigid(file_path):
    kitti360Path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..')

    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')

    if not os.path.isfile(file_path):
        raise FileNotFoundError
    cam2world_rows = np.loadtxt(file_path)
    cam2world_rigid = np.reshape(cam2world_rows[:, 1:], (-1, 4, 4))
    frames_numbers = list(np.reshape(cam2world_rows[:, :1], (-1)).astype(int))
    cam2world = {}

    current_rigid = cam2world_rigid[0]
    for frame_index in tqdm(range(1, frames_numbers[-1] + 1), desc='loading cam2world rigids'):
        if frame_index in frames_numbers:
            mapped_index = frames_numbers.index(frame_index)
            current_rigid = cam2world_rigid[mapped_index]

        # (Tr(cam -> world))
        cam2world[frame_index] = current_rigid

        if frame_index == 20:  # DEBUG
            break

    return cam2world


if __name__ == '__main__':
    intrinsic_filepath = os.path.join(g.calibrations_path, 'perspective.txt')
    intrinsic_calibrations = get_perspective_intrinsic(intrinsic_filepath)['P_rect_00']
