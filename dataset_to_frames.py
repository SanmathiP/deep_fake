## Authors: Sanmathi Parvatikar and Arijit Ghosh ##

## Importing necessary packages ##

import ffmpeg
import os
import glob
from tqdm import tqdm


class Preprocessing:
    '''
    Creates a fixed number of frames from a given video.
    '''

    def __init__(self, num_frames, root_dir, res_dir):
        '''
        num_frames : Number of frames that needs to be extracted.
        root_dir : Folder containing the video file.
        '''

        self.num_frames = num_frames

        self.files = glob.glob(root_dir + './*.mp4')

        self.res_dir = res_dir

    def create_frame(self, file):
        '''
        Given the filename, file, extracts specific number of frames given in the constructor
        using self.num_frames.
        '''

        probe = ffmpeg.probe(file)

        time = float(probe['streams'][0]['duration'])

        interval = int(time // self.num_frames)

        interval_ranges = [(i * interval, (i + 1) * interval) for i in range(self.num_frames)]

        output_name = os.path.basename(file)[:-4]

        for idx, each_interval_range in enumerate(interval_ranges):
            out_name = self.res_dir + '/' + output_name + str(idx) + '.jpg'
            ffmpeg.input(file, ss=each_interval_range[0]).filter('scale', 256, 256). \
                output(out_name, vframes=1).run()



    def vid_2_img(self):
        '''
        Creates fixed frames for multiple videos in the given root folder.
        '''

        loop = tqdm(self.files)

        for file in loop:
            self.create_frame(file)

        print('Everything is completed!!')


a = Preprocessing(5 , 'dfdc_train_part_1' , 'frames_data')

a.vid_2_img()