## Authors: Sanmathi Parvatikar and Arijit Ghosh ##
## Creating modalities based on preferences ##

## This code uses the InverseRenderNet paper code and hence... ##
## you need to download it in the same disk. ##
import glob
import numpy as np
from PIL import Image
from subprocess import call
import os
from tqdm import tqdm

class CreateModality:
    '''
    Creates different modality using
    InverseRenderNet from selected frames.
    '''

    def __init__(self , inp_dir , output_dir ,
                 lighting = True , lighting_dir = 'lighting' ,
                 albedo = True , albedo_dir = 'albedo' ,
                 normal_map = True , normal_map_dir = 'normal_map',
                 shading = True , shading_dir = 'shading' ,
                 shadow = True , shadow_dir = 'shadow'):

        self.output_dir = output_dir

        self.lighting = lighting
        self.lighting_dir = lighting_dir
        self.lighting_name = None

        self.albedo = albedo
        self.albedo_dir = albedo_dir
        self.albedo_name = None

        self.normal_map = normal_map
        self.normal_map_dir = normal_map_dir
        self.normal_map_name = None

        self.shading = shading
        self.shading_dir = shading_dir
        self.shading_name = None

        self.shadow = shadow
        self.shadow_dir = shadow_dir
        self.shadow_name = None

        self.files = glob.glob(inp_dir + '/*.jpg')

        white_img = np.uint8(np.ones((256, 256)) * 255.)

        img = Image.fromarray(white_img)

        img.save('mask.jpg')


    def create(self):
        '''
        Based on a given directory containing frames,
        creates the user defined modalities and saves them
        on disk.
        '''

        loop = tqdm(self.files)
        for each_img in loop:

            img_name = os.path.basename(each_img)

            if self.lighting:
                self.lighting_name = self.lighting_dir + '/' + img_name

            if self.albedo:
                self.albedo_name = self.albedo_dir + '/' + img_name

            if self.normal_map:
                self.normal_map_name = self.normal_map_dir + '/' + img_name

            if self.shadow:
                self.shadow_name = self.shadow_dir + '/' + img_name

            if self.shading:
                self.shading_name = self.shading_dir + '/' + img_name

            call(
                'python test.py --image {} --mask mask.jpg --model model_ckpt --output {} --lighting {} --lighting_name {} --albedo {} --albedo_name {} --shading {} --shading_name {} --shadow {} --shadow_name {} --normal_map {} --normal_map_name {}'.format(
                    each_img,
                    self.output_dir,
                    self.lighting,
                    self.lighting_name,
                    self.albedo,
                    self.albedo_name,
                    self.shading,
                    self.shading_name,
                    self.shadow,
                    self.shadow_name,
                    self.normal_map,
                    self.normal_map_name
                    ),
                shell=True)

        print('Modalities are created!!')

a = CreateModality(inp_dir='example_frames' , output_dir='example_out')
a.create()