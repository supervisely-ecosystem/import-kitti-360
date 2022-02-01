#!/usr/bin/python
#

from __future__ import print_function, absolute_import, division
import os
import json
from skimage import io, filters
import numpy as np
from collections import namedtuple
from collections import defaultdict
from matplotlib import cm
import xml.etree.ElementTree as ET
import glob
import struct

# get current date and time
import datetime
import locale

# A point in a polygon
Point = namedtuple('Point', ['x', 'y'])


from abc import ABCMeta, abstractmethod
from kitti360scripts.helpers.labels     import labels, id2label, kittiId2label, name2label

MAX_N = 1000
def local2global(semanticId, instanceId):
    globalId = semanticId*MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int)
    else:
        return int(globalId)

def global2local(globalId):
    semanticId = globalId // MAX_N
    instanceId = globalId % MAX_N
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int), instanceId.astype(np.int)
    else:
        return int(semanticId), int(instanceId)

annotation2global = defaultdict()

# Abstract base class for annotation objects
class KITTI360Object:
    __metaclass__ = ABCMeta

    def __init__(self):
        # the label
        self.label    = ""

        # colormap
        self.cmap = cm.get_cmap('Set1')
        self.cmap_length = 9 

    def getColor(self, idx):
        if idx==0:
            return np.array([0,0,0])
        return np.asarray(self.cmap(idx % self.cmap_length)[:3])*255.

    def assignColor(self):
        if self.semanticId>=0:
            self.semanticColor = id2label[self.semanticId].color
            if self.instanceId>0:
                self.instanceColor = self.getColor(self.instanceId)
            else:
                self.instanceColor = self.semanticColor


# Class that contains the information of a single annotated object as 3D bounding box
class KITTI360Bbox3D(KITTI360Object):
    # Constructor
    def __init__(self):
        KITTI360Object.__init__(self)
        # the polygon as list of points
        self.vertices  = []
        self.faces  = []
        self.lines = [[0,5],[1,4],[2,7],[3,6],
                      [0,1],[1,3],[3,2],[2,0],
                      [4,5],[5,7],[7,6],[6,4]]

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # the window that contains the bbox
        self.start_frame = -1
        self.end_frame = -1

        # timestamp of the bbox (-1 if statis)
        self.timestamp = -1

        # projected vertices
        self.vertices_proj = None
        self.meshes = []

        # name
        self.name = '' 

    def __str__(self): 
        return self.name

    def generateMeshes(self):
        self.meshes = []
        if self.vertices_proj:
            for fidx in range(self.faces.shape[0]):
                self.meshes.append( [ Point(self.vertices_proj[0][int(x)], self.vertices_proj[1][int(x)]) for x in self.faces[fidx]] )
                
    def parseOpencvMatrix(self, node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data = node.find('data').text.split(' ')
    
        mat = []
        for d in data:
            d = d.replace('\n', '')
            if len(d)<1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find('transform'))
        R = transform[:3, :3]
        T = transform[:3, 3]
        vertices = self.parseOpencvMatrix(child.find('vertices'))
        faces = self.parseOpencvMatrix(child.find('faces'))

        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices
        self.faces = faces
        self.R = R
        self.T = T

        self.transform = transform

    def parseBbox(self, child):
        semanticIdKITTI = int(child.find('semanticId').text)
        self.semanticId = kittiId2label[semanticIdKITTI].id
        self.instanceId = int(child.find('instanceId').text)
        # self.name = str(child.find('label').text)
        self.name = kittiId2label[semanticIdKITTI].name

        self.start_frame = int(child.find('start_frame').text)
        self.end_frame = int(child.find('end_frame').text)

        self.timestamp = int(child.find('timestamp').text)

        self.annotationId = int(child.find('index').text) + 1

        global annotation2global
        annotation2global[self.annotationId] = local2global(self.semanticId, self.instanceId)
        self.parseVertices(child)

    def parseStuff(self, child):
        classmap = {'driveway': 'parking', 'ground': 'terrain', 'unknownGround': 'ground', 
                    'railtrack': 'rail track'}
        label = child.find('label').text 
        if label in classmap.keys():
            label = classmap[label]

        self.start_frame = int(child.find('start_frame').text)
        self.end_frame = int(child.find('end_frame').text)

        self.semanticId = name2label[label].id
        self.instanceId = 0 
        self.parseVertices(child)


# Class that contains the information of the point cloud a single frame
class KITTI360Point3D(KITTI360Object):
    # Constructor
    def __init__(self):
        KITTI360Object.__init__(self)

        self.vertices = []

        self.vertices_proj = None

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # name
        self.name = '' 

        # color
        self.semanticColor = None
        self.instanceColor = None

    def __str__(self): 
        return self.name


    def generateMeshes(self):
        pass


# Meta class for KITTI360Bbox3D
class Annotation3D:
    def __init__(self, labelPath):
        # load annotation
        tree = ET.parse(labelPath)
        root = tree.getroot()

        self.objects = defaultdict(dict)

        self.num_bbox = 0

        for child in root:
            if child.find('transform') is None:
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            globalId = local2global(obj.semanticId, obj.instanceId)
            self.objects[globalId][obj.timestamp] = obj
            self.num_bbox+=1

        globalIds = np.asarray(list(self.objects.keys()))
        semanticIds, instanceIds = global2local(globalIds)
        for label in labels:
            if label.hasInstances:
                print(f'{label.name:<30}:\t {(semanticIds==label.id).sum()}')
        print(f'Loaded {len(globalIds)} instances')
        print(f'Loaded {self.num_bbox} boxes')

    def __call__(self, semanticId, instanceId, timestamp=None):
        globalId = local2global(semanticId, instanceId)
        if globalId in self.objects.keys():
            # static object
            if len(self.objects[globalId].keys())==1: 
                if -1 in self.objects[globalId].keys():
                    return self.objects[globalId][-1]
                else:
                    return None
            # dynamic object
            else:
                return self.objects[globalId][timestamp]
        else:
            return None


def projectVeloToImage(cam_id=0, seq=0):
    from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
    from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
    from PIL import Image
    import matplotlib.pyplot as plt

    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '..', '..')

    sequence = '2013_05_28_drive_%04d_sync' % seq

    # perspective camera
    if cam_id in [0, 1]:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye camera
    elif cam_id in [2, 3]:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
    else:
        raise RuntimeError('Unknown camera ID!')

    # object for parsing 3d raw data
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)

    # cam_0 to velo
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

    # all cameras to system center
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # velodyne to all cameras
    TrVeloToCam = {}
    for k, v in TrCamToPose.items():
        # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        # Tr(velo -> cam_k)
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

    # take the rectification into account for perspective cameras
    if cam_id == 0 or cam_id == 1:
        TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    cm = plt.get_cmap('jet')

    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    for frame in range(0, 18, 1):
        points = velo.loadVelodyneData(frame)
        points[:, 3] = 1

        # transfrom velodyne points to camera coordinate
        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:, :3]
        # project to image space
        u, v, depth = camera.cam2image(pointsCam.T)
        u = u.astype(np.int)
        v = v.astype(np.int)

        # prepare depth map for visualization
        depthMap = np.zeros((camera.height, camera.width))
        depthImage = np.zeros((camera.height, camera.width, 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0), v < camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(np.logical_and(mask, depth > 0), depth < 30)
        depthMap[v[mask], u[mask]] = depth[mask]
        layout = (2, 1) if cam_id in [0, 1] else (1, 2)
        sub_dir = 'data_rect' if cam_id in [0, 1] else 'data_rgb'
        fig, axs = plt.subplots(*layout, figsize=(18, 12))

        # load RGB image for visualization
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir,
                                 '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            raise RuntimeError('Image file %s does not exist!' % imagePath)

        colorImage = np.array(Image.open(imagePath)) / 255.
        depthImage = cm(depthMap / depthMap.max())[..., :3]
        colorImage[depthMap > 0] = depthImage[depthMap > 0]

        axs[0].imshow(depthMap, cmap='jet')
        axs[0].title.set_text('Projected Depth')
        axs[0].axis('off')
        axs[1].imshow(colorImage)
        axs[1].title.set_text('Projected Depth Overlaid on Image')
        axs[1].axis('off')
        plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
        plt.show()


# a dummy example
if __name__ == "__main__":
    ann = Annotation3D()


