# -*- coding: utf-8 -*-
"""
Created on Wen Dec 18 2018
@author: Rt-Rakesh

This script is a utility to plot all the bbox in the images along with the labels.
Usage: Should be used by calling in jupyter notbooks.
"""

import os
import pandas as pd
import cv2
from tqdm import tqdm


def plot_rec(coor, img, label='', color=(255,0,0),text_place='up'):
    """
    This Function plots the annoations on the images along with label.
    Args:
    1.coor: --tuple The coornidates of the  bbox, it should have the data in the following format (xmin,ymin,xmax,ymax).
    2.image: --np array The image object containing the image in np.array must be provided.
    3.label: -- str The label for the bbox to be mentioned here.
    4.text_place: --str The place where the label text is to be placed. Up means top left and down means bottom left.
    Returns:
    The image with the annotaions and label written on the image.
    """
    x1, y1, x2, y2 = coor
    draw_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=3)
    if text_place== 'up':
        cv2.putText(draw_img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    if text_place== 'down':
        cv2.putText(draw_img, label, (int(x1), int(y2)), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

    return draw_img


def plot_annotation_labelwise(csv_path, annotated_files_out_folder_path, original_images_input_folder_path,color,labelname='label',text_place='up', first_5_only=False):
    """
    This Function plots the annotations on the images along with label and saves it labelwise.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,labelname).
    2.annotated_files_out_folder_path: --str The path to directory where the new annotated images will be saved.
    3.original_images_input_folder_path: --str The path to images directory.
    4.color: tuple. The color that is to be used for the rectangle and the label
    5.labelname: --str The labelname used for the labels.Default is 'label'
    6.text_place: --str The place where the label text is to be placed. Up means top left and down means bottom left.
    7.first_5_only: --Boolean Default: False This parameter by default will allow for plotting of all the annotations.Chenge it to True to plot only 5 images per label.
    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    data_df = pd.read_csv(csv_path)
    lable_list = set(data_df[labelname])
    for i in tqdm(lable_list, desc='Processing labels for all images.'):
        path = os.path.join(annotated_files_out_folder_path, 'labelwise_annotations', str(i))
        if not os.path.exists(path):
            os.makedirs(path)
        if first_5_only:
            temp_df = data_df.loc[data_df[labelname] == i].head()
        else:
            temp_df = data_df.loc[data_df[labelname] == i]
        if len(temp_df) > 0:
            path_unique= list(temp_df['path'].unique())
            
            for j, t in enumerate(path_unique):
                image_path = os.path.join(original_images_input_folder_path, str(t))
                img = cv2.imread(image_path)
                temp_df_path= temp_df[temp_df['path']==t]
                for l,m in temp_df_path.iterrows():
                    x1 = m.xmin
                    y1 = m.ymin
                    x2 = m.xmax
                    y2 = m.ymax
                    label = str(m[labelname])
                    anno_image = plot_rec((x1, y1, x2, y2), img, label, color= color)
                cv2.imwrite(os.path.join(path, str(t)), anno_image)


def plot_annotation(csv_path, annotated_files_out_folder_path, original_images_input_folder_path,color=(255,0,0), labelname='label', text_place='up'):
    """
    This Function plots the annoations on the images along with label.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,labelname).
    2.annotated_files_out_folder_path: --str The path to directory where the new annotated images will be saved.
    3.original_images_input_folder_path: --str The path to images directory.
    4.color: tuple. The color that is to be used for the rectangle and the label
    5.labelname: --str The labelname used for the labels.Default is 'label'
    6.text_place: --str The place where the label text is to be placed. Up means top left and down means bottom left.
    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    data_df = pd.read_csv(csv_path)
    image_list = set(data_df.path)
    path = os.path.join(annotated_files_out_folder_path, 'annotated_images')
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm(image_list):
        temp_df = data_df.loc[data_df['path'] == i]
        image_path = os.path.join(original_images_input_folder_path, str(i))
        img = cv2.imread(image_path)
        if len(temp_df) > 0:
            for j, t in temp_df.iterrows():
                x1 = t.xmin
                y1 = t.ymin
                x2 = t.xmax
                y2 = t.ymax
                label = str(t[labelname])
                anno_img = plot_rec((x1, y1, x2, y2), img, label, color= color,text_place= text_place)
            cv2.imwrite(os.path.join(path, str(i)), anno_img)


def plot_annotation_labelwise(csv_path, annotated_files_out_folder_path, original_images_input_folder_path,color=(255,0,0),labelname='label',text_place='up', first_5_only=False):
    """
    This Function plots the annotations on the images along with label and saves it labelwise.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,labelname).
    2.annotated_files_out_folder_path: --str The path to directory where the new annotated images will be saved.
    3.original_images_input_folder_path: --str The path to images directory.
    4.color: tuple. The color that is to be used for the rectangle and the label
    5.labelname: --str The labelname used for the labels.Default is 'label'
    6.first_5_only: --Boolean Default: False This parameter by default will allow for plotting of all the annotations.Chenge it to True to plot only 5 images per label.
    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    data_df = pd.read_csv(csv_path)
    lable_list = set(data_df[labelname])
    for i in tqdm(lable_list, desc='Processing labels for all images.'):
        path = os.path.join(annotated_files_out_folder_path, 'labelwise_annotations', str(i))
        if not os.path.exists(path):
            os.makedirs(path)
        if first_5_only:
            temp_df = data_df.loc[data_df[labelname] == i].head()
        else:
            temp_df = data_df.loc[data_df[labelname] == i]
        if len(temp_df) > 0:
            path_unique= list(temp_df['path'].unique())
            
            for j, t in enumerate(path_unique):
                image_path = os.path.join(original_images_input_folder_path, str(t))
                img = cv2.imread(image_path)
                temp_df_path= temp_df[temp_df['path']==t]
                for l,m in temp_df_path.iterrows():
                    x1 = m.xmin
                    y1 = m.ymin
                    x2 = m.xmax
                    y2 = m.ymax
                    label = str(m[labelname])
                    anno_image = plot_rec((x1, y1, x2, y2), img, label, color= color)
                cv2.imwrite(os.path.join(path, str(t)), anno_image)

def plot_annotation_gt_pred(pred_csv_path,gt_csv_path, annotated_files_out_folder_path, original_images_input_folder_path,colorgt=(255,0,0),colorpred=(0,0,255), labelname='label'):
    """
    This Function plots the annoations on the images along with label for both the preddicted and gt.
    Args:
    1.pred_csv path: --str The path to the predicted csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    2.gt_csv_path: --str The path to the ground truth csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    2.annotated_files_out_folder_path: --str The path to directory where the new annotated images will be saved.
    3.original_images_input_folder_path: --str The path to images directory.
    4.colorgt: tuple. The color that is to be used for the rectangle and the label for the ground truth
    5.colorpred: tuple. The color that is to be used for the rectangle and the label for the predicted
    6.labelname: --str The labelname used for the labels.Default is 'label'

    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    plot_annotation(gt_csv_path, annotated_files_out_folder_path, original_images_input_folder_path,colorgt, labelname=labelname, text_place= 'up')
    plot_annotation(pred_csv_path, annotated_files_out_folder_path, annotated_files_out_folder_path+'annotated_images/',colorpred, labelname=labelname,text_place='down')


def plot_annotation_labelwise_gt_pred(pred_csv_path,gt_csv_path, annotated_files_out_folder_path, original_images_input_folder_path,colorgt=(255,0,0),colorpred=(0,0,255),labelname='label', first_5_only=False):
    """
    This Function plots the annotations on the images along with label and saves it labelwise.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,labelname).
    2.annotated_files_out_folder_path: --str The path to directory where the new annotated images will be saved.
    3.original_images_input_folder_path: --str The path to images directory.
    4.colorgt: tuple. The color that is to be used for the rectangle and the label for the ground truth
    5.colorpred: tuple. The color that is to be used for the rectangle and the label for the predicted
    6.labelname: --str The labelname used for the labels.Default is 'label'
    7.first_5_only: --Boolean Default: False This parameter by default will allow for plotting of all the annotations.Chenge it to True to plot only 5 images per label.
    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    pred_df = pd.read_csv(pred_csv_path)
    gt_df = pd.read_csv(gt_csv_path)
    pred_df['set']='pred'
    gt_df['set']='gt'
    data_df= pd.concat([gt_df,pred_df])
    lable_list = set(gt_df[labelname])
    for i in tqdm(lable_list, desc='Processing labels for all images.'):
        path = os.path.join(annotated_files_out_folder_path, 'labelwise_annotations', str(i))
        if not os.path.exists(path):
            os.makedirs(path)
        if first_5_only:
            temp_df = data_df.loc[data_df[labelname] == i].head()
        else:
            temp_df = data_df.loc[data_df[labelname] == i]
        if len(temp_df) > 0:
            path_unique= list(temp_df['path'].unique())
            
            for j, t in enumerate(path_unique):
                image_path = os.path.join(original_images_input_folder_path, str(t))
                img = cv2.imread(image_path)
                temp_df_path= temp_df[temp_df['path']==t]
                for l,m in temp_df_path.iterrows():
                    x1 = m.xmin
                    y1 = m.ymin
                    x2 = m.xmax
                    y2 = m.ymax
                    label = str(m[labelname])
                    if m.set == 'pred':
                        anno_image = plot_rec((x1, y1, x2, y2), img, label, color= colorpred, text_place='down')
                    if m.set == 'gt':
                        anno_image = plot_rec((x1, y1, x2, y2), img, label, color= colorgt, text_place='up')
                cv2.imwrite(os.path.join(path, str(t)), anno_image)