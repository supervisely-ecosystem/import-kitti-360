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
import sly_progress
import sly_globals as g

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

        progress_cb = sly_progress.get_progress_cb(g.api, g.TASK_ID, f"Loading kitty annotations",
                                                   total=len(root))

        for child in root:
            if child.find('transform') is None:
                progress_cb(1)
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            globalId = local2global(obj.semanticId, obj.instanceId)
            self.objects[globalId][obj.timestamp] = obj
            self.num_bbox+=1
            
            progress_cb(1)

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



# a dummy example
if __name__ == "__main__":
    ann = Annotation3D()


