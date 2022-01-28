import sly_functions as f

if __name__ == '__main__':
    f.get_kitti_360_data()  # download data from server
    f.convert_kitti360_to_supervisely_pcl_episodes_project()  # convert kitti to supervisely
    f.upload_pcl_episodes_project()  # upload converted project to server

