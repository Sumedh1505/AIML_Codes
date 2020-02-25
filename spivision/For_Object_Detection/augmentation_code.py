#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from data_aug import *
from bbox_util import *
from PIL import Image


# In[2]:


def create_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Output directory is created!!!")
   
        


# In[3]:


def read_data(path, bbox_df):
    img = cv2.imread(path)
    img_name = os.path.basename(path)
    group = bbox_df.groupby('new_path')
    img_csv = group.get_group(img_name)  
    img_csv.sort_values('label',ascending=False,inplace=True)
    bboxes = img_csv.to_numpy()
    orig_label_height = bboxes[:,-1]
    orig_label_width = bboxes[:,-2]
    label = bboxes[:,[-4]]
    bboxes = bboxes[:,1:-4]
    no_of_labels = np.size(bboxes,0)
    return(img_name,img,bboxes,label,orig_label_height,orig_label_width,no_of_labels)
    
    


# In[4]:


def transform_data(transforms,img,bboxes):
    img_new,bboxes_new = transforms(img,bboxes)
    bboxes_new = bboxes_new.astype(int)
    return(img_new,bboxes_new)
    


# In[5]:


def write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir):
    ann_df = pd.DataFrame(bboxes_new, columns = ['xmin','ymin','xmax','ymax'])  
    count = ann_df.count()
    if((count[1] == no_of_labels)):
        ann_df['path']=mod_img_name
        ann_df['label']=label
        ann_df.sort_values('label',ascending=False,inplace=True)
        ann_df['label_width']=ann_df['xmax']-ann_df['xmin']
        ann_df['label_height']=ann_df['ymax']-ann_df['ymin']
        ann_df['original_label_width']=orig_label_width
        ann_df['original_label_height']=orig_label_height
        ann_df['ratio_width']=ann_df['label_width']/ann_df['original_label_width']
        ann_df['ratio_height']=ann_df['label_height']/ann_df['original_label_height']
        width_mean = ann_df['ratio_width'].mean()
        height_mean = ann_df['ratio_height'].mean()
        img_wo = img_new.copy()
        img_new = draw_rect(img_new, bboxes_new,color=[0,255,0])
        mod_img_ann = 'ann'+mod_img_name
        if((width_mean>0.8)&(height_mean>0.8)):
            all_df=all_df.append(ann_df,ignore_index=True)
            print("writing image!!!!")
            cv2.imwrite(os.path.join(out_dir,mod_img_name),img_wo)
            #cv2.imwrite(os.path.join(out_dir,mod_img_ann),img_new)
    return(all_df)
    


# In[7]:


def convert_bit_depth(out_dir, img_format, bit_depth):
    print("Convertion in progress!!")
    img_ext = '*.'+img_format
    out_img_list=glob.glob(out_dir+img_ext) 
    for i in out_img_list:
        new_i = os.path.join(out_dir,os.path.basename(i))
        image =Image.open(i)
        MOD_IM=image.convert(bit_depth)
        MOD_IM.save(new_i,img_format)


# In[8]:


def augment_image_bbox(input_img_dir,csv_path,img_format,out_dir,augment_list=['scale'],aug_min=0.03,aug_interval=0.04,aug_max=0.1,convert_image_depth='no'):
    create_dir(out_dir) 
    img_ext = '*.'+img_format
    img_list = glob.glob(os.path.join(input_img_dir,img_ext))
    PIL_image=Image.open(img_list[0])
    image_mode =PIL_image.mode
    scale_list = np.arange(aug_min,aug_max,aug_interval)
    bbox_df = pd.read_csv(csv_path)
    bbox_df['new_path'] = [os.path.basename(i) for i in bbox_df['path']]
    bbox_df['label_width']=bbox_df['xmax']-bbox_df['xmin']
    bbox_df['label_height']=bbox_df['ymax']-bbox_df['ymin']
    all_df = pd.DataFrame()
    for augment_name in augment_list:
        for i in tqdm(img_list):  
            img_name,img1,bboxes1,label,orig_label_height,orig_label_width,no_of_labels = read_data(i,bbox_df)
            img_name = img_name[:-4]
            if(augment_name=='scale'):
                for j in tqdm(range(len(scale_list))):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomScale(scale_list[j],diff=False)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_scale_false_'+str(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
                
                
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomScale(scale_list[j],diff=True)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_scale_true_'+str(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
                    
            elif(augment_name=='translate'): 
                 for j in tqdm(range(len(scale_list))):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomTranslate(scale_list[j],diff=False)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_trans_false_'+str(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
                    
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomTranslate(scale_list[j],diff=True)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_trans_true_'+str(scale_list[j])+'.'+img_format
                    all_df= write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
                    
            
                
            elif(augment_name=='trans_scale'):
                for j in tqdm(range(len(scale_list))):
                    for k in range(len(scale_list)):
                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomScale(scale_list[j],diff=True)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_true_'+str(scale_list[k])+'_scale_true_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
                        
                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomScale(scale_list[j],diff=False)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_true_'+str(scale_list[k])+'_scale_false_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
                       
                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomScale(scale_list[j],diff=False)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_false_'+str(scale_list[k])+'_scale_false_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
                        
                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomScale(scale_list[j],diff=True)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_false_'+str(scale_list[k])+'_scale_true_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,orig_label_height,orig_label_width,no_of_labels,out_dir)
            
            
                
           
    all_df=all_df[['path','xmin','ymin','xmax','ymax','label']]
    final_csv_name='augment_all.csv'
    all_df.to_csv(os.path.join(out_dir,final_csv_name),index=False)
    if(convert_image_depth=='yes'):
        convert_bit_depth(out_dir,img_format,image_mode)
    print("Augmentation is done successfully!!!")    
    


# In[10]:


input_img_dir = r"/home/developer/my_project_dir/vignesh_projects/project_invoice_pipeline/train_images"
csv_path = r"/home/developer/my_project_dir/vignesh_projects/project_invoice_pipeline/traindata.csv"
img_format = 'jpg'
out_dir = r"/home/developer/my_project_dir/vignesh_projects/project_invoice_pipeline/augmentation_train"
augment_list = ['scale']
aug_min = 0.01
aug_interval = 0.05
aug_max = 0.24
convert_image_depth = 'no'

augment_image_bbox(input_img_dir,csv_path,img_format,out_dir,augment_list,aug_min,aug_interval,aug_max)

                    


# In[ ]:




