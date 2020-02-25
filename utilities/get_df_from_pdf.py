# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:38:35 2020

@author: 205557
"""

from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import re
import json
import ast
import subprocess
from pdf2image import convert_from_path
import glob

class get_df_from_pdf():
    """
    This is a class that should be used for converting pdf files and getting the lines and the coordinates into 
    a data frame. 
    Requires the library where the pdf files are kept. 
    """
    
    def __init__(self, inputlib,multiple):
        self.input_lib= inputlib
        self.multiple= multiple
        #self.output_html_lib= outputhtmllib
        #self.output_csv_lib= outputcsvlib
        
    @staticmethod
    def count_pages(pdfpath):
        """
        Calculate the number of pages in a pdf. Defined as a static method since it is 
        used only inside the class. 
        Inputs:
            pdfpath: path to the pdf file. 
        Returns:
            number of pages in pdf file.
        """
        pages = convert_from_path(pdfpath, 300)
        return len(pages)
    
    @staticmethod   
    def apply_multiple(df,multiple):
        for j in ['xmin','xmax','ymin','ymax']:
            df[j]= df[j]*multiple
            return df
        
    def create_df(self):
        """
        Function to convert all the pdfs in the input folder into corresponding htmls and then extract words and 
        their coordinates into a dataframe. 
        Inputs:
            Object
        Returns:
            Dataframe with information including the pagenumber
        """
        alldf= pd.DataFrame()
        pdffiles= glob.glob(self.input_lib+'/**/*.pdf', recursive=True)
        for pdf_file in pdffiles:
            pdf_page_count= self.count_pages(pdf_file)
            for pg in range(1,pdf_page_count+1):
                pg = str(pg)
                cmd = ['pdftotext','-bbox-layout','-f', pg, pdf_file, pdf_file[:-4]+'_'+pg+'.html']
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        
                o, e = proc.communicate()
                page = open(pdf_file[:-4]+'_'+pg+'.html',encoding="utf8")
                soup = BeautifulSoup(page.read(),'html.parser')
                out_html_file_path=pdf_file[:-4]+'_'+pg+'.html'
                lines = soup.find_all('line')
                pdf_file= pdf_file.replace("\\","/")
                path= pdf_file.split("/")[-1]
                path= path[:-4]+"-from-pdf-page"+pg+'.jpg'
                td_list = []
                for line in lines:
                    req_td_dict = {}
                    req_td_dict['path'] =path
                    #req_td_dict['page']= int(pg)
                    req_td_dict['xmin'] = round(float(line['xmin']))
                    req_td_dict['ymin'] = round(float(line['ymin']))
                    req_td_dict['xmax'] = round(float(line['xmax']))
                    req_td_dict['ymax'] = round(float(line['ymax']))
                    req_td_dict['label'] = line.text.replace('\n',' ')
                    td_list.append(req_td_dict)
                df1 = pd.DataFrame(td_list)
                alldf= alldf.append(df1)
        alldf_multi= self.apply_multiple(alldf, self.multiple)
        return alldf_multi,alldf