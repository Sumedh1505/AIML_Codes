# -*- coding: utf-8 -*-
"""
Created on Wen Dec 12 2018
@author: sivaselvi

This script is a utility for test data analysis
Usage: Could be used by calling in jupyter notbooks.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np

class analysis_obj():
    '''
    This class takes in a ground truth csv file and a predicted csv file and has methods to output the 
    analysis file which can then be used for calculating the F-Values
       
    '''
    def __init__(self,gt,pred,var):
        '''
        Inputs:
            gt: dataframe consisting of the ground truth labels and coordinates.path, xmin,xmax,ymin, 
            ymax,label
            pred: dataframe of predictions with the same structure as gt.
            
        '''
        self.gt= gt.copy()
        self.pred=pred.copy()
        self.var=var
        self.gt['gt_serialnum']= self.gt.groupby(['path']).cumcount() 
        self.gt.rename(columns={'label':'label_gt'},inplace= True)
        self.pred['pred_serialnum']= self.gt.groupby(['path']).cumcount() 
        self.pred.rename(columns={'label':'label_pred'},inplace= True)        
    @staticmethod         
    def area(boxes):
        """Computes area of boxes.
        Args:
        boxes: Numpy array with shape [N, 4] holding N boxes
        Returns:
        a numpy array with shape [N*1] representing box areas
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    @staticmethod 
    def intersection(boxes1, boxes2):
        """Compute pairwise intersection areas between boxes.
        Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes
        Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
        """
        [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
        [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

        all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
        all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
        intersect_heights = np.maximum(
          np.zeros(all_pairs_max_ymin.shape),
          all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
        all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
        intersect_widths = np.maximum(
          np.zeros(all_pairs_max_xmin.shape),
          all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths
    
    @staticmethod 
    def intersection_tuples(boxes1, boxes2):
        """Compute pairwise intersection areas between two boxes.
        Args:
        boxes1: a tuple with 4 elements in order (y1,x1,y2,x2)
        boxes2: a tuple with 4 elements in order (y1,x1,y2,x2)
        Returns:
        a value of the intersection area
        """
        [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
        [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

        min_ymax = np.minimum(y_max1, y_max2)
        max_ymin = np.maximum(y_min1, y_min2)
        intersect_heights = np.maximum(   0,min_ymax - max_ymin)
        min_xmax = np.minimum(x_max1, x_max2)
        max_xmin = np.maximum(x_min1, x_min2)
        intersect_widths = np.maximum(0,min_xmax - max_xmin)
        return intersect_heights * intersect_widths


    def compute_iou(self,boxes1, boxes2):
        """Computes intersection-over-union between a pair of boxes. Rounded off to 5 decimal places
        Args:
        boxes1: a tuple with 4 elements in order (y1,x1,y2,x2)
        boxes2: a tuple with 4 elements in order (y1,x1,y2,x2)

        Returns:
        IOU: floating value, rounded to 5 decimal places
        """
        boxes1 = np.expand_dims(np.array(boxes1), axis=0)
        boxes2 = np.expand_dims(np.array(boxes2), axis=0)
        intersect = self.intersection_tuples(boxes1, boxes2)
        area1 = self.area(boxes1)
        area2 = self.area(boxes2)
        union = np.expand_dims(area1, axis=1) + np.expand_dims(
          area2, axis=0) - intersect
        iou = intersect / union
        return round(iou[0][0], 5)

    def compute_iomin(self,boxes1, boxes2):
        """Computes intersection-over-union between a pair of boxes. Rounded off to 5 decimal places
        Args:
        boxes1: a tuple with 4 elements in order (y1,x1,y2,x2)
        boxes2: a tuple with 4 elements in order (y1,x1,y2,x2)

        Returns:
        IOMIN: floating value, rounded to 5 decimal places. Indicates the proportion of the smaller area that
        is also the intersection
        """
        boxes1 = np.expand_dims(np.array(boxes1), axis=0)
        boxes2 = np.expand_dims(np.array(boxes2), axis=0)
        intersect = self.intersection_tuples(boxes1, boxes2)
        area1 = self.area(boxes1)
        area2 = self.area(boxes2)
        union = np.expand_dims(area1, axis=1) + np.expand_dims(
          area2, axis=0) - intersect
        iomin = intersect / (min(area1,area2)+0.000001)
        return round(iomin[0][0], 5)
        

    @staticmethod 
    def create_coordinates(df):
        """Computes the mid point of a rectangle which has xmin,ymin,xmax and ymax
        Args:
        df: DataFrame whcih consists of the coordinates.
        Returns:
        df: DataFrame which has the midpoint coordiantes x and y in addition to the original 4 numbers of the
        rectangle
        """
        df['x']= (df['xmin']+df['xmax'])/2
        df['y']= (df['ymin']+df['ymax'])/2
        return(df)
        
    @staticmethod         
    def return_sub(pred, gt, path, var):
        """returns the subset of dataframe with respect to a particular image
        Args:
        gt: dataframe consisting of the ground truth labels and coordinates.path, xmin,xmax,ymin, 
        ymax,label
        pred: dataframe of predictions with the same structure as gt.
        path: the name of the image or file to which the dataframes need to be submitted. 
        var: the name of the variable in the dataframes that has the image or file names
        Returns:
        pred_sub: Subset of the pred dataframe
        gt_sub: Subset of the ground truth dataframe.
        """
        pred_sub=pred[pred[var]==path]
        gt_sub=gt[gt[var]==path]   
        pred_sub=pred_sub.reset_index(drop= True)
        gt_sub=gt_sub.reset_index(drop= True)    
        pred_sub['serial_no']=range(len(pred_sub))
        gt_sub['serial_no']=range(len(gt_sub))
        #pred_sub=pred_sub[['serial_no','x','y']]
        #gt_sub=gt_sub[['serial_no','x','y']]
        return(pred_sub,gt_sub)

    def analysis(self):      
        self.pred= self.create_coordinates(self.pred)
        self.gt= self.create_coordinates(self.gt)
        gtpaths= self.gt[self.var].unique()
        pred_merged= pd.DataFrame()
        for path in tqdm(gtpaths):
            pred_sub, gt_sub=self.return_sub(self.pred, self.gt, path,self.var)
            if pred_sub.shape[0] > 0:
                nbrs = NearestNeighbors(n_neighbors=min(3,gt_sub.shape[0]), algorithm='brute').fit(gt_sub[['x','y']])
                distances, indices = nbrs.kneighbors(pred_sub[['x','y']])
                pred_tuples=list(pred_sub[['ymin','xmin','ymax','xmax']].itertuples(index=False, name=None))
                gt_tuples=list(gt_sub[['ymin','xmin','ymax','xmax']].itertuples(index=False, name=None))
                len(gt_tuples)
                resultlist=[]
                ioulist=[]
                for j in range(indices.shape[0]):
                    iou=[]
                    for k,l in enumerate(indices[j,:]):
                        iou.append((self.compute_iou(pred_tuples[j],gt_tuples[l])))
                    if max(iou) > 0.5:
                        resultlist.append(indices[j,np.argmax(iou)])
                        ioulist.append(max(iou))
                    else:
                        resultlist.append(99999)
                        ioulist.append(max(iou))
                pred_sub['gt_serialnum']=resultlist
                pred_sub['ioumax']=ioulist
                pred_merged= pred_merged.append(pred_sub)
            pred_merged.dropna(axis=0, how='any', thresh=None, subset=['gt_serialnum'], inplace=True)
            pred_merged1= pred_merged.copy()
            pred_merged_match=pred_merged[pred_merged['gt_serialnum'] != 99999]
            pred_merged_nomatch=pred_merged[pred_merged['gt_serialnum'] == 99999]
            pred_merged_match= pred_merged_match.sort_values(['path','gt_serialnum','ioumax'], ascending=[True, True, False])
            pred_merged_match['repeatnum']= pred_merged_match.groupby(['path','gt_serialnum']).cumcount() 
            pred_merged_match.loc[(pred_merged_match['repeatnum'] != 0),'gt_serialnum']=99999
            pred_merged_nomatch['repeatnum']=0
            pred_merged= pd.concat([pred_merged_match, pred_merged_nomatch])    
            analysis= pd.merge(pred_merged,self.gt, on=['path','gt_serialnum'], how='outer', indicator= True)
            analysis['corpred']=0
            analysis['gt']=0
            analysis['pred']=0
            analysis.loc[analysis['_merge'] != 'left_only','gt']=1
            analysis.loc[(analysis['_merge']== 'both') & (analysis['label_gt']==analysis['label_pred']),'corpred']=1
            analysis.loc[analysis['_merge'] != 'right_only','pred']=1

        return(analysis,pred_merged,pred_merged1)





