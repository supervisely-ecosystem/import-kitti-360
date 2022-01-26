import os
import supervisely

import sly_functions as f

my_app = supervisely.AppService()
api: supervisely.Api = my_app.public_api

TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
INPUT_FILE = os.environ["modal.state.slyFile"]

project_name = 'pcl_project'
project_dir_path = os.path.join(my_app.data_dir, project_name)
bins_dir_path = '/Users/qanelph/Desktop/velodyne_points/data'
bboxes_path = '/Users/qanelph/Desktop/work/supervisely/engine/kitti360Scripts/data_3d_bboxes/train/2013_05_28_drive_0000_sync.xml'

pose2world = f.load_pose_to_world_data(
    pose_file_path='/Users/qanelph/Downloads/data_poses/2013_05_28_drive_0000_sync/poses.txt')
