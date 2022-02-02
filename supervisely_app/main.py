import kitti_360_helpers
import pointcloud_episode

import sly_functions as f
import sly_globals as g

if __name__ == '__main__':
    kitti_360_helpers.get_kitti_360_data()                      # download data from server
    f.convert_kitti360_to_supervisely_pcl_episodes_project()    # convert kitti to supervisely
    pointcloud_episode.upload_pcl_episodes_project()            # upload converted project to server

    g.my_app.stop()



