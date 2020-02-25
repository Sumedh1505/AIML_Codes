# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:27:12 2020

@author: 205557
"""
import os

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)