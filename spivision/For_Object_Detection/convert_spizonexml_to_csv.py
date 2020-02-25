# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:34:27 2020

@author: 205557
"""

from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import re
import json
import ast
import warnings
warnings.filterwarnings('ignore')

def convert_spizonexml_to_csv(file_path, multiple= 3.125):
    """
    Function to convert the .zone file from spizone annotations. .zone is converted into .xml
    Inputs:
        file_path: Path to the xml file.The xml file will contain information on multiple pages.
        multiple: Multiple to convert the coordinates fo pdf to jpg or tif
        
    """
    page = open(file_path)
    soup = BeautifulSoup(page.read(),'html.parser')
    pages = soup.find('page')
    tr_list = []
    for pagenum, pgs  in enumerate(pages,1):
        for tags in pgs.findAll('tr'):
            for td in tags.findAll('td'):
                td_region = td['region']
                td_id = td['id']
        #         print(td_region)
                td_region = td_region.replace("=",":")
                res = re.findall(r'\w+', td_region) 
                for r in res:
                    if r.isalpha():
                        td_region = td_region.replace(r, "'"+r+"'")        

                td_dict = ast.literal_eval(td_region)

                if td_id.startswith('1-'):
                    td_dict['label'] = 'header_cell'
                else:
                    td_dict['label'] = 'row_cell'

                req_td_dict = {}
                req_td_dict['path'] = str(pagenum)+'.tif'
                req_td_dict['xmin'] = round(td_dict['X'])
                req_td_dict['ymin'] = round(td_dict['Y'])
                req_td_dict['xmax'] = round(td_dict['X'] + td_dict['Width'])
                req_td_dict['ymax'] = round(td_dict['Y'] + td_dict['Height'])
                req_td_dict['label'] = td_dict['label']
                tr_list.append(req_td_dict)
    df1 = pd.DataFrame(tr_list)
    return(df1)

