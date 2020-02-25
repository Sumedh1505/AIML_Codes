# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:36:31 2020

@author: 205557
"""
import PyPDF2
import os

def subset_pdf(inputpath,pdffile,pagelist,savepath):
	'''
	Function to take an input pdf and subset the pdf to the required pages
	Inputs:
		inputpath: folder in which the pdf file exists
		pdffile: name of the pdffile. Please send with extension(.pdf)
		pagelist: list of pagenumbers that need to be subsetted. Page numbers start from 0. So if you need first and 3rd page of a pdf, then
				  you enter the list as [0,2]
		savepath: folder in which you want to save the subsetted pdf. 
	Outputs:
		saved file in the save path directory with the name of pdffilename_subset.pdf
	
	'''
    pdfFileObj = open(inputpath+pdffile, 'rb') 
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
    pdf_writer = PyPDF2.PdfFileWriter()
    for pagenum in pagelist:
        pdf_writer.addPage(pdfReader.getPage(pagenum))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        output_filename = '{}_subset.pdf'.format(pdffile[:-4])
        with open(savepath+output_filename,'wb') as out:
            pdf_writer.write(out)
