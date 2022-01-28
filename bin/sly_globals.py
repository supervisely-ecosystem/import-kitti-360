import os
import shutil

from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid, loadCalibrationCameraToPose
from bin.coord_systems_transforamations import get_camera_to_velodyne_rigid, get_world_to_velodyne_rigid_by_frames

import supervisely


my_app = supervisely.AppService()
api: supervisely.Api = my_app.public_api

# shutil.rmtree(my_app.data_dir, ignore_errors=False)  # DEBUG

TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

kitti360_remote_dir = os.environ["modal.state.kittiPath"]
kitti360_local_dir = os.path.join(my_app.data_dir, 'kitti360data_raw')

project_name = os.path.basename(os.path.normpath(kitti360_remote_dir))
project_dir_path = os.path.join(my_app.data_dir, project_name)


seq_dir_path = os.path.join(kitti360_local_dir, 'data_3d_raw')
bins_dir_path = os.path.join(kitti360_local_dir, 'data_3d_raw/{}/velodyne_points/data/')  # paste seq name
bboxes_path = os.path.join(kitti360_local_dir, 'data_3d_bboxes/train/')


# -----------------
# DEBUG

cam2velodyne = get_camera_to_velodyne_rigid()
world2cam = get_world_to_velodyne_rigid_by_frames()

kitti360Path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..')

fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)


fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_00']
