<div align="center" markdown>
<img .png"/>


# Import KITTI-360

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Usage">Usage</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

https://supervisely-dev.deepsystems.io/ecosystem/apps/supervisely-ecosystem/import-kitti-360/supervisely_app

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/import-kitti-360/supervisely_app)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/import-kitti-360/)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-kitti-360/supervisely_app&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-kitti-360/supervisely_app&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-kitti-360/supervisely_app&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Application imports Supervisely Pointcloud Episodes from KITTI-360 format

Application key points:  
- `Raw Velodyne Pointclouds` supported
- `3D BBoxes` supported
- `2D Rectified Images from Perspective Camera (00)` supported

# Usage

1. Prepare folder in KITTI-360 structure ([**DOWNLOAD LITE EXAMPLE HERE**]())

```
Example_1
├── calibration
│   ├── calib_cam_to_pose.txt  
│   ├── calib_cam_to_velo.txt  
│   ├── calib_sick_to_velo.txt
│   ├── image_02.yaml
│   ├── image_03.yaml
│   └── perspective.txt
├── data_2d_raw
│ └── 2013_05_28_drive_0000_sync — name of sequence 
│      └── image_00
│           └── data_rect
│               ├── 0000000000.png
│               ├── 0000000001.png
│               ├── 0000000002.png
├── data_3d_bboxes
│   └── train
│       └── 2013_05_28_drive_0000_sync.xml — name of sequence (.xml)
├── data_3d_raw
│   └── 2013_05_28_drive_0000_sync — name of sequence
│       └── velodyne_points
│           └── data
│               ├── 0000000000.bin
│               ├── 0000000001.bin
│               ├── 0000000002.bin
└── data_poses
    └── 2013_05_28_drive_0000_sync — name of sequence
        ├── cam0_to_world.txt
        └── poses.txt
```

2. Upload folder to Team Files

# How to Run
1. Add [Import KITTI-360](https://ecosystem.supervise.ly/apps/import-images-from-csv) to your team from Ecosystem.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-images-from-csv" src="https://imgur.com/Cqe7fjv.png" width="450px" style='padding-bottom: 20px'/>  

2. Run app from the context menu of the [prepared folder](#usage):

<img src=".png" width="100%"/>

