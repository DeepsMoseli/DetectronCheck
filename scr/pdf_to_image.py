import pandas as pd
import numpy as numpy
import os
import  random

path = os.getcwd()

from pdf2image import convert_from_path, convert_from_bytes

files = ['LC%s'%i for i in range(1,5)]

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

rand = random.sample()
images = convert_from_bytes(open(path + '\pdf_data\\' rand + '.pdf'+ , 'rb').read())