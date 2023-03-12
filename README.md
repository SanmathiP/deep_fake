


## Fake it till you can't make it: A sneak peak to DeepFake Detection only using images ##



**An experimental go through for detecting Deepfake videos as well images by utilizing the modalities generated by Inverserendernet++ network.**

This project tries to tackle the problem of DeepFake detection (for both image and videos) by breaking them into modalities generated by Inverserendernet++ network. The workflow is summarized below.

- Break video into exactly 5 frames/images. (Not necessary for doing image based deepfake detection.)
- For each image generate the 5 modalities using the Inverserendernet Network.
- At this time we trained two types of network: simple classification network with single modality as input (and) ensemble network which takes all the modalities as input to classify as real/fake.
- For images the modality based ensemble is as good as the normal RGB based. 


## Architecture ##

To be added!!

## Dependencies ##

To run the code, the following dependencies are particularly necessary.
- tensorflow 1.12.0
- Python 3.6
- skimage
- cv2
- numpy
- Pytorch 1.12.0
- ffmpeg
- glob 
- tqdm
- PIL
- torchmetrics
- tensorboardX
- pandas
- typing

## Usage ##

The project is heavily built on the InverseRenderNet++ implementaion. 
- To start off clone their repository from here: https://github.com/YeeU/InverseRenderNet_v2. 
- You also need to download the pretrained weights from here: https://drive.google.com/uc?export=download&id=1hGIoK3Pemtg3eYjFy_CBK-R37D3gA0VC and unzip it. 
- After cloning the repository , downloading and unzipping the pretrained weights the project structure would look like this:
```bash
InverseRenderNet_v2
│   README.md
│   test.py    
│   ...
└───model_ckpt
│      model.ckpt.meta
│      model.ckpt.index
│      ...
└───iiw_model_ckpt
│      model.ckpt.meta
│      model.ckpt.index
│      ...
└───diode_model_ckpt
│      model.ckpt.meta
│      model.ckpt.index
│      ...
```
- For our project, replace the <code>test.py</code> with the <code>test.py</code> given in our repository.

Now we are going to branch out into 2 scopes that we have devised our project : video based and image based.

**Video Based**
- In case of videos, the frames can be generated by using the utility function <code>dataset_to_frames.py</code> given in the video_preprocessing directory. 
- In case of training an ensemble based model, run <code>train_model_ensemble_video.py</code> given in the video_based_training directory. 
- In case of training a single modality model, run <code>train_model_normal_video.py</code> given in the video_based_training directory. 

**Image Based**
- In case of images, we need to generate fake images. 
- For our project we used FFHQ dataset to get 5000 real images.
- We used a StyleGAN2-ADA to generate 5000 fake images.
- To generate these fake images, clone this repository: https://github.com/NVlabs/stylegan2-ada-pytorch.git .
- For the ffhq weights go to this website: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ and download <code>ffhq.pkl</code> file.  
- Use the <code>generate.py</code> to generate the fake images. We used the command: 
```bash
python generate.py --network=ffhq.pkl --seeds=5000-10000 --outdir=fake
```
- Similar to the video based training, to train an ensemble model for the image based model, run <code>train_model_ensemble.py</code> given in the image_based_training directory. 
- To train a single modality model for the image based model, run <code>train_model.py</code> given in the image_based_training directory. 
