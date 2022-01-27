# cam_0 to velo
import os

import numpy as np
from tqdm import tqdm

from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid, loadCalibrationCameraToPose


def get_camera_to_velodyne_rigid():
    kitti360Path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..')

    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

    # all cameras to system center
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # all cameras to velodyne to
    TrCamToVelo = {}
    for k, v in TrCamToPose.items():
        # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo[k] = TrCam0ToVelo @ TrCamkToCam0

    return TrCamToVelo


def loadCamToWorldRigid(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError
    cam2world_rows = np.loadtxt(file_path)
    cam2world_rigid = np.reshape(cam2world_rows[:, 1:], (-1, 4, 4))
    frames_numbers = list(np.reshape(cam2world_rows[:, :1], (-1)).astype(int))
    world2cam = {}

    # current_rigid = np.linalg.inv(cam2world_rigid[0])
    current_rigid = cam2world_rigid[0]
    for frame_index in tqdm(range(1, frames_numbers[-1] + 1), desc='loading cam2world rigids'):
        if frame_index in frames_numbers:
            mapped_index = frames_numbers.index(frame_index)
            current_rigid = cam2world_rigid[mapped_index]
        # Tr(world -> cam) = inv(Tr(cam -> world))
        world2cam[frame_index] = current_rigid

        if frame_index == 20:
            break

    return world2cam


def get_world_to_velodyne_rigid_by_frames():
    kitti360Path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..')

    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

    seq = '2013_05_28_drive_0000_sync'
    fileCameraToWorld = os.path.join(kitti360Path, 'data_poses', seq, 'cam0_to_world.txt')

    # Tr(world -> cam)
    TrWorldToCam = loadCamToWorldRigid(fileCameraToWorld)

    TrWorldToVelo = {}
    for frame_index, TrCam2world in TrWorldToCam.items():
        # Tr(world -> velo) = Tr(world -> cam) @ Tr(cam -> velo)
        # TrWorldToVelo[frame_index] = TrWorldToCam[frame_index]
        # TrWorldToVelo[frame_index] = TrWorldToCam[frame_index] @ TrCam0ToVelo
        TrWorldToVelo[frame_index] = TrWorldToCam[frame_index]

    return TrWorldToVelo


if __name__ == '__main__':
    get_camera_to_velodyne_rigid()
    get_world_to_velodyne_rigid_by_frames()
