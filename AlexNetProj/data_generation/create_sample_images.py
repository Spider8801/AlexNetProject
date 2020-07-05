#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:55:57 2020

@author: Vaibhav
"""

from PIL import Image 
import numpy as np
import random
random.seed(100)  
import matplotlib.pyplot as plt
from scipy import ndimage#
from matplotlib import cm
import os
from tqdm import tqdm
import pandas as pd


class generate_data:
    def __init__(self, *args,  **kwargs):
        self.base_image=kwargs['base_image'] if 'base_image' in kwargs.keys() else 'base_field.png' #Name of the file from which to extract data
        self.row_size_arc_min=kwargs['row_size_arc_min'] if 'row_size_arc_min' in kwargs.keys() else 15 #Size of row of base image in arc-min
        self.fov_arc_min=kwargs['fov_arc_min'] if 'fov_arc_min' in kwargs.keys() else 6 #Desired fov in arc-min
        self.max_shift_arc_min=kwargs['max_shift_arc_min'] if 'max_shift_arc_min' in kwargs.keys() else 5 #Maximum shift in the field
        self.num_images_to_generate=kwargs['num_images_to_generate'] if 'num_images_to_generate' in kwargs.keys() else 50000 #Number of images to generate
        self.data_directory=kwargs['data_directory'] if 'data_directory' in kwargs.keys() else 'field_image_data' #Name of the directory where to store data
        
        self.base_image_processing()
        self.get_maximum_shift()
        
        
        CHECK_FOLDER = os.path.isdir(self.data_directory)
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(self.data_directory)
            print("created folder : ", self.data_directory)

        
        image_file_name=list()
        x_shift=list()
        y_shift=list()    
        iteration_list=list(range(self.num_images_to_generate))        
        for image_sequence_number in tqdm(iteration_list):
            self.generate_random_shift_values()
            self.shift_image()
            self.filename=self.data_directory+'/data_'+str(image_sequence_number).zfill(6)+'.png'
            self.save_image_data()
            image_file_name.append('data_'+str(image_sequence_number).zfill(6)+'.png')
            x_shift.append(self.random_num2*self.one_px_sz)
            y_shift.append(self.random_num1*self.one_px_sz)
        
        # dictionary of lists  
        data_dict = {'image_file_name': image_file_name, 'x_shift': x_shift, 'y_shift': y_shift}
        df = pd.DataFrame(data_dict) 
        # saving the dataframe 
        df.to_csv(self.data_directory+'/a_list_of_data.csv', index=False) 
        print('\n'+'Data prepared. Check data directory: '+self.data_directory)
        

    def base_image_processing(self):
        img = Image.open(self.base_image)     # Read image  
        im=np.array(img) #Convert to numpy array
        im=im/np.max(im) #Normalize the array
    
        self.base_im_array=im[:,:,0]     # Using only first image 

    def get_maximum_shift(self):    
        rows, cols=self.base_im_array.shape
        self.center_px_row=round(rows/2)
        self.center_px_col=round(cols/2)
        self.one_px_sz=self.row_size_arc_min/rows #Size of one pixel in minutes
    
        self.fov_px_half=round(self.fov_arc_min/self.one_px_sz/2) #Half FOV in pixels
        self.max_shift_in_field_px=self.max_shift_arc_min/self.one_px_sz #Unit in pixel



    def generate_random_shift_values(self):
        self.random_num1=random.uniform(-1*self.max_shift_in_field_px, self.max_shift_in_field_px)
        self.random_num2=random.uniform(-1*self.max_shift_in_field_px, self.max_shift_in_field_px)


    def shift_image(self):
        shifted_image=ndimage.interpolation.shift(self.base_im_array, (self.random_num1, self.random_num2))
        cropped_image=shifted_image[self.center_px_row-self.fov_px_half:self.center_px_row+self.fov_px_half, self.center_px_col-self.fov_px_half:self.center_px_col+self.fov_px_half]
        cropped_image= Image.fromarray(np.uint8(cm.gist_earth(cropped_image)*255))
        self.sample_image = cropped_image.resize((28, 28))

    
    def save_image_data(self):
        self.sample_image.save(self.filename)
        

if __name__ == '__main__':
    generate_data()

