import torch, torchvision
print(torch.__version__,torch.cuda.is_available())
device = torch.device("cuda")
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore") 

import numpy as np
import random
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt

import cv2
from PIL import Image
from skimage import measure
import sklearn

import pycocotools as pycoco
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

np.set_printoptions(threshold=sys.maxsize)

class trained_model:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final_check.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.60
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.predictor = DefaultPredictor(self.cfg)
        #check_metadata = MetadataCatalog.get("checknet_train").set(thing_classes = ["no", 'yes'])
        #MetadataCatalog.get("fishnet_val").set(thing_classes = ["fish", 'blue', 'yellow'])
        #cfg.DATASETS.TEST = ("fishnet_val",)

    def make_prediction(self,image):
        #im = cv2.imread(d["file_name"])
        outputs = self.predictor(image)
        return image,outputs

    
    def show_prediction(self,im,outputs):
        plt.figure()

        r =c= 1
        #f, axarr = plt.subplots(r,c,figsize=(20,12))
        #plt.xticks([])
        #plt.xticks([])
        v = Visualizer(im[:, :, ::-1],
                        metadata=MetadataCatalog.get("checknet_train"), 
                        scale=0.8, 
                        instance_mode=ColorMode.IMAGE_BW
                      )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_im = Image.fromarray(v.get_image()[:, :, ::-1])
        #axarr.imshow(pred_im)
        #plt.show()
        return pred_im
    
    def get_json_result(self,outputs):
        scores = [k.item() for k in outputs["instances"].to("cpu").scores]
        classes = [y.item() for y in outputs["instances"].to("cpu").pred_classes]


        vertices = {'v1':[],'v2':[],'v3':[],'v4':[]}
        for box in range(len(scores)):
            box_vertices = [y.item() for y in [k for k in outputs["instances"].to("cpu").pred_boxes[box]][0]]
            i=0
            for v in vertices:
                vertices[v].append(box_vertices[i])
                i+=1

        v_frame = pd.DataFrame(vertices) 
        v_frame['scores'] =scores
        v_frame['classes'] =classes
        v_frame = v_frame.sort_values(by = ['v2','v4','v1','v3'],ascending=True).reset_index(drop=True)
        v_frame2 =v_frame[['classes','scores']].to_dict(orient='dict')
        return v_frame2