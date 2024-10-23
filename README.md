**Code for Node Detection and Tracking in microscopic Phase Images for Cell Force Microscopy**

This codebase expects input in either Phase image format or Fluroscent image format. If Phase images are used as input then a Pix2Pix model is used to generate fake fluroscent images and then the node detection and tracking network is used for tracking nodes to calucalute displacement and cell forces. 

***For training:***
1. Train Pix2Pix using code from [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository, also placed in *Pix2Pix/*
    Refer to tips  in *Pix2Pix/pytorch-CycleGAN-and-pix2pix/docs/tips.md*
2. Node detection network with the train script in *src/*

***For Inference:****
1. Run DataPrepforPix2Pix to prepare data in the format Pix2Pix expects for conversion from Phase data to Flurocent data. if phase data as input else start from step 3
2. Run generate_fluro.py to generate fluroscnet images from phase images using Pix2Pix trained model. 
*update pix2pix_path, fake_fluro_path, phase_path and num_images (counted from 0 so 1+the actual last id you have)*
3. Run DataPrep.ipynb for preparing fluroscent images for node tracking. 
*Run DataPrep cell 5 if starting from step 1 or  Run DataPrep cell 4 if starting from step 3* 
*update video_name_list and master_Folder_name*
4. Run detection_tracking.sh for node detection and tracking 
*update video path in the loop definition*
