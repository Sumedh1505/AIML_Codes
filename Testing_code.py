# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:23:01 2020

@author: 205557
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append("D:/PROJECTS_ROOT/AIML_utilities/")
import spivision
from spivision.For_Object_Detection import analysis_objectdetection
from spivision.For_Object_Detection import pascol_voc_converter_utils
from spivision.For_Object_Detection import plot_annotation_utils
from utilities.utils import create_folder
help(pascol_voc_converter_utils)
help(plot_annotation_utils)

'''
converting the xml files into csv for object detection 
'''

xml_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
out_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
pascol_voc_converter_utils.convert_pascol_voc_xml_to_csv(xml_folder_path, out_path=out_path, return_df=False)

xml_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/test/"
out_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/test.csv"
pascol_voc_converter_utils.convert_pascol_voc_xml_to_csv(xml_folder_path, out_path=out_path, return_df=False)


''' Testing convert_csv_2_pascol_voc_xml. Create a folder dummy in the location mentioned below'''

xml_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/dummy/"
create_folder(xml_folder_path)
csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
image_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"

pascol_voc_converter_utils.convert_csv_2_pascol_voc_xml(csv_path, image_path, xml_folder_path,labelname='class')

''' Testing the plotting and annotation tools '''
csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
annotated_files_out_folder_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/annotatedimages/"
original_images_input_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
create_folder(annotated_files_out_folder_path)
plot_annotation_utils.plot_annotation(csv_path, annotated_files_out_folder_path, original_images_input_folder_path, labelname='class')

'''testing the label wise annotations'''
plot_annotation_utils.plot_annotation_labelwise(csv_path, annotated_files_out_folder_path, original_images_input_folder_path,labelname='class', first_5_only=False)

''' testing annotations of both gt and predicted on the same image all labels '''
pred_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
gt_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"

annotated_files_out_folder_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/annotatedimages/"
original_images_input_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
plot_annotation_utils.plot_annotation_gt_pred(pred_csv_path,gt_csv_path, annotated_files_out_folder_path, original_images_input_folder_path,colorgt=(255,0,0),colorpred=(0,0,255), labelname='class')


''' testing annotations of both gt and predicted on the same image label wise '''
pred_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
gt_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"

annotated_files_out_folder_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/annotatedimages/"
original_images_input_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
plot_annotation_utils.plot_annotation_labelwise_gt_pred(pred_csv_path,gt_csv_path, annotated_files_out_folder_path, original_images_input_folder_path,colorgt=(255,0,0),colorpred=(0,0,255),labelname='class', first_5_only=False)

