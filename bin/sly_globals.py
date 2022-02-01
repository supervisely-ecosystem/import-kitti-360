import os
import shutil

import kitti_360_helpers


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

calibrations_path = os.path.join(kitti360_local_dir, 'calibration/')
poses_path = os.path.join(kitti360_local_dir, 'data_poses/{}/')  # paste seq name


cam2world = None
cam2velo = None




