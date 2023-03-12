"""Processes a video dataset to create certain number of frames from each videos. 
"""

## Importing necessary packages ##

import ffmpeg
import os
import glob
from tqdm import tqdm


class Preprocessing:
    """
    Creates a fixed number of frames from a given video.

    Attributes
    ----------
    num_frames : int
            Number of frames to break a video into.

    root_dir : str
            Video dataset directory.

    res_dir : str
            Output directory.

    Methods
    -------
    create_frame
            Given a video filename, generates num_frames number of equally
            spaced frames.

    vid_2_img 
            Given a video dataset, for each video calls create_frame method
            to generate equally spaced frames. After this method call, all
            the videos would be broken down into their consequent frames.

    """

    def __init__(self, num_frames, root_dir, res_dir):
        """Constructor.
        num_frames : Number of frames that needs to be extracted.
        root_dir : Folder containing the video file.
        """

        self.num_frames = num_frames

        self.files = glob.glob(root_dir + "./*.mp4")

        self.res_dir = res_dir

    def create_frame(self, file):
        """Given the filename extracts specific number of frames given by the attribute
        num_frames.

        Parameters
        ----------
        file : str
            The video filename.
        """

        probe = ffmpeg.probe(file)

        time = float(probe["streams"][0]["duration"])

        interval = int(time // self.num_frames)

        interval_ranges = [
            (i * interval, (i + 1) * interval) for i in range(self.num_frames)
        ]

        output_name = os.path.basename(file)[:-4]

        for idx, each_interval_range in enumerate(interval_ranges):
            out_name = self.res_dir + "/" + output_name + str(idx) + ".jpg"
            ffmpeg.input(file, ss=each_interval_range[0]).filter(
                "scale", 256, 256
            ).output(out_name, vframes=1).run()

    def vid_2_img(self):
        """Creates fixed frames for multiple videos in the given root folder."""

        loop = tqdm(self.files)

        for file in loop:
            self.create_frame(file)

        print("Everything is completed!!")
