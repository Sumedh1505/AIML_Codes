# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:53:46 2019

@author: 205557
"""

from PIL import Image
import imutils
import os
from imutils import paths
import numpy as np
import cv2
from tqdm import tqdm
import glob
from pdf2image import convert_from_path
import shutil
import cv2

class converter():
    '''
    this class helps in converting pdf files and multipage tif files into jpeg or png
    source_liby: basically the directory which has either the pdf or tif files.
    '''
    def __init__(self, source_liby):
        self.source_liby= source_liby
        self.exceptlistjpg=[]
        self.exceptlistpng=[]        
    def create_jpegs(self):
        file_names_tif = glob.glob(self.source_liby+'/**/*.tif', recursive=True)
        file_names_pdf=  glob.glob(self.source_liby+'/**/*.pdf', recursive=True)
        if len(file_names_tif) > 0:
            for j in tqdm(file_names_tif, miniters=1000, mininterval = 300):
                try:
                #print(imageliby+directory+'/'+j)
                    im = Image.open(j) 
                    dpi_orig= im.info['dpi']
                    dpi_half= (int(dpi_orig[0]/2),int(dpi_orig[1]/2))
                    dpi_pt25= (int(dpi_orig[0]/4),int(dpi_orig[1]/4))
                    size= im.size
                    size_half= (round(size[0]/2.0), round(size[1]/2.0))
                    size_pt25= (round(size[0]/4.0), round(size[1]/4.0))
                    size_half= (round(size[0]/2.0), round(size[1]/2.0))
                    cnt= im.n_frames
                    for k in range(cnt):
                        im.seek(k)
                        im.save(j[:-4]+'-fromtif-page'+str(k+1)+'_orig.jpg', dpi= dpi_orig)  
                        im_half=im.resize(size_half)
                        im_half.save(j[:-4]+'-fromtif-page'+str(k+1)+'_half.jpg', dpi= dpi_half)
                        im_pt25=im.resize(size_pt25)
                        im_pt25.save(j[:-4]+'-fromtif-page'+str(k+1)+'_pt25.jpg', dpi= dpi_pt25)
                except:
                    self.exceptlistjpg.append(j)
        else:
            pass
        if len(file_names_pdf) > 0:
            for j in tqdm(file_names_pdf, miniters=1000, mininterval = 300):
                try:
                    pages = convert_from_path(j, 300)
                    pdf_file = j[:-4]
                    for page in pages:
                       page.save("%s-frompdf-page%d.jpg" % (pdf_file,pages.index(page)+1), "JPEG")        
                except:
                    self.exceptlistjpg.append(j)
        else:
            pass        
    def create_pngs(self):
        file_names_tif = glob.glob(self.source_liby+'/**/*.tif', recursive=True)
        file_names_pdf=  glob.glob(self.source_liby+'/**/*.pdf', recursive=True)
        if len(file_names_tif) > 0:
            for j in tqdm(file_names_tif, miniters=1000, mininterval = 300):
                try:
                #print(imageliby+directory+'/'+j)
                    im = Image.open(j) 
                    dpi_orig= im.info['dpi']
                    dpi_half= (int(dpi_orig[0]/2),int(dpi_orig[1]/2))
                    dpi_pt25= (int(dpi_orig[0]/4),int(dpi_orig[1]/4))
                    size= im.size
                    size_half= (round(size[0]/2.0), round(size[1]/2.0))
                    size_pt25= (round(size[0]/4.0), round(size[1]/4.0))
                    size_half= (round(size[0]/2.0), round(size[1]/2.0))
                    cnt= im.n_frames
                    for k in range(cnt):
                        im.seek(k)
                        im.save(j[:-4]+'-fromtif-page'+str(k+1)+'_orig.png', dpi= dpi_orig)  
                        im_half=im.resize(size_half)
                        im_half.save(j[:-4]+'-fromtif-page'+str(k+1)+'_half.png', dpi= dpi_half)
                        im_pt25=im.resize(size_pt25)
                        im_pt25.save(j[:-4]+'-fromtif-page'+str(k+1)+'_pt25.png', dpi= dpi_pt25)
                except:
                    self.exceptlistjpg.append(j)
        else:
            pass

def move_files(input_liby, output_liby, type_file, condition=None):
    """
    function that moves files of a given type from an input library to a defined output library
    Inputs:
        input_liby: Folder where the files reside. Usually multiple types of files would be here. 
        output_liby:Folder to which the files need to be moved to. 
        type_file: The extension of the files which need to be moved. it can take values jpg, png, html etc
        condition: the condition on which files needs to move. It will be a part of the file name that needs to be statisfied for a file to move
    Outputs:
    None. it creates and saves images
    
    """
    files= os.listdir(input_liby)
    files= [x for x in files if ( (x[-3:]==type_file) | (x[-4:]==type_file) ) ]
    if condition != None:
        files=[x for x in files if condition in x]
    for file in files:
        shutil.move(input_liby+file, output_liby)

def rotate_images(input_liby, type_file):
    """
    function that moves files of a given type from an input library to a defined output library
    Inputs:
        input_liby: Folder where the files reside.These would be image files. 
        The rotated images would be saved in the same directory.
        type_file: the extension of the files which need to be moved. it can take values jpg, png etc
    Outputs:
        None. it creates and saves images
    """
    files= os.listdir(input_liby)
    files= [x for x in files if x[-3:]==type_file]
    imageROI = cv2.imread(output_liby+file)
    for angle in np.arange(0, 360, 15):
        rotated = imutils.rotate_bound(imageROI, angle)
        cv2.imwrite(input_liby+str(angle)+'_'+file, rotated)
