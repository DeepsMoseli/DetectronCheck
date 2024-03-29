# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 02:22:22 2020

@author: Deeps
"""

import json
import os
from PIL import Image
import pickle
path = os.getcwd()

# We mostly care about the x and y coordinates of each region

anno_file = "\..\check_box_48_done.json"
#peter = '\..\\via_100_peter.json'
image_path = "C:/Users/Deeps/Documents/School/UH manoa/ICS 314 Software Engineering/DetetronCheck/images/"


def annotations_file(jsonfile):
    complete_annotations = []
    
    annotations = json.load(open(path+ jsonfile))
    annotations = list(annotations.values())[1]
    
    image_id  = list(range(1,len(annotations)+1))
    #image_id  = list(range(478,478+len(annotations)+1)) for adding
    
    file_name = [annotations[k]['filename'] for k in annotations]
    
    annotations = [annotations[k] for k in annotations if annotations[k]['regions']]
    annotations = [annotations[k]['regions'] for k in range(len(annotations))]
    #annotations = [k['regions'][k]["shape_attributes"] for k in annotations]
    
    for k in range(len(annotations)):
        dict_hold = {"file_name":file_name[k],
                     "size": Image.open(image_path + file_name[k]).size,
                     "image_id":image_id[k],
                     "annotations":annotations[k]}
        complete_annotations.append(dict_hold)
    
    return complete_annotations


annotations = annotations_file(anno_file)
with open( "check_box_annotations_48.pickle" , 'wb') as file:
    pickle.dump(annotations,file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

