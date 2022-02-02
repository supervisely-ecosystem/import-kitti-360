import kitti_360_helpers
import sly_functions as f
import pointcloud_episode


if __name__ == '__main__':
    kitti_360_helpers.get_kitti_360_data()  # download data from server
    f.convert_kitti360_to_supervisely_pcl_episodes_project()  # convert kitti to supervisely
    pointcloud_episode.upload_pcl_episodes_project()  # upload converted project to server



