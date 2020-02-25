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

class rotate_images():
    """
    class that moves files of a given type from an input library to a defined output library
    Inputs:
        input_liby: Folder where the files reside.These would be image files. 
        The rotated images would be saved in the same directory.
        type_file: the extension of the files which need to be moved. it can take values jpg, png etc
    Outputs:
        None. it creates and saves images
    """
	def __init__(self,input_liby):
		self.input_liby= input_liby
		self.files= os.listdir(input_liby)
	def rotate(self, file_type, 
		files= [x for x in files if x[-3:]==type_file]
		imageROI = cv2.imread(output_liby+file)
		for angle in np.arange(0, 360, 15):
			rotated = imutils.rotate_bound(imageROI, angle)
			cv2.imwrite(input_liby+str(angle)+'_'+file, rotated)
