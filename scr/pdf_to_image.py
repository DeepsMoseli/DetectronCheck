import pandas as pd
import numpy as np
import os
import  random
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms

path = os.getcwd()

from pdf2image import convert_from_path, convert_from_bytes

files = ['LC %s'%i for i in range(1,5)]

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


def transform(image):
    all_results = []
    all_trans= []
    size_ = image.size
    ransize = np.random.randint(8,12)/10
    ransize2 = np.random.randint(9,13)/10
    all_trans.append(transforms.RandomRotation(8))
    all_trans.append(transforms.RandomRotation(9))
    all_trans.append(transforms.Resize((int(size_[0]*ransize), int(size_[1]*ransize))))
    all_trans.append(transforms.Resize((int(size_[0]*ransize2), int(size_[1]*ransize2))))
    all_trans.append(transforms.ColorJitter(brightness=2))
    all_trans.append(transforms.ColorJitter(brightness=1.5))
    all_trans.append(transforms.ColorJitter(contrast=2))
    all_trans.append(transforms.ColorJitter(saturation=2))
    all_trans.append(transforms.ColorJitter(saturation=1))
    all_trans.append(transforms.ColorJitter(hue=0.1))
    all_trans.append(transforms.ColorJitter(hue=0.08))
    all_results.append(image)
    for tr in all_trans:
        all_results.append(tr(image))
    return all_results
    


#rand = random.sample(files,1)[0]
for rand in files:
    images = convert_from_bytes(open(path + '\\pdf_data\\' +rand + '.pdf' , 'rb').read())[0]
    i = 0
    for j in transform(images):
        j.save(path + '\\images\\'+rand+'_%s'%i + '.jpg')
        i+=1
