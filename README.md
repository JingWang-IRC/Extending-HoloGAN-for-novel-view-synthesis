# Publication
This research has been published at IEEE/SICE International Symposium on System Integration 2022 (SII 2022). You can find my paper in this repository. The formal URL to the paper will be updated soon.

# Extending-HoloGAN-for-novel-view-synthesis
This is a work extends HoloGAN for novel view synthesis. HoloGAN is a GAN trained in
an entirely unsupervised manner, and can generate images of the same content from various viewpoints.
However, HoloGAN can not specify what content to generate. Therefore, we attempt to solve this problem.
The details of HoloGAN can be found in [HoloGAN_homepage](https://www.monkeyoverflow.com/hologan-unsupervised-learning-of-3d-representations-from-natural-images).
![image](https://github.com/xxxiaojing/Extending-HoloGAN-for-novel-view-synthesis/blob/main/images/generated_images_four_methods.png)

# Dataset
We only trained the neural networks on [CompCar dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/)
As for the pre-processing, we used program *separate_train_test_data/ separate_train_test_data.py*

# Usage
The code of HoloGAN is in the directory original_HoloGAN_Compcar, while the code of HoloGAN+AE is in the directory HoloGAN_plus_auto-encoder_Compcar. Both of them have very similar contents so the usages of them are almost the same.

In the corresponding directory, main.py is the main program that call many functions in the directory tools and model_HoloGAN.py.
The architecture and training programs are in model_HoloGAN.py.
The file config_HoloGAN.json contains the hyperparameters (configuration) for training.

If you want to use our codes, first, you should download CompCar dataset first and use separate_train_test_data/separate_train_test_data.py to pre-process the images. 
Then put the images in a directory in the directory original_HoloGAN_Compcar or HoloGAN_plus_auto-encoder_Compcar.

Last, type the following code to run the program:  
`
python main.py ./config_HoloGAN.json --dataset cars --input_fname_pattern “.jpg”--output_height 128 --train=“True” –rotate_azimuth=“False”
`

After training, you can type the following code to generate some samples:  
`
python main.py ./config_HoloGAN.json --dataset cars --input_fname_pattern “.jpg”--output_height 128 --train=“False” –rotate_azimuth=“True”
`  

# Hyperparameter
```
image_path:  
			Full path to the dataset directory. All images should under a folder. 
gpu:  
			Index number of the GPU to use. Default: 0.  
batch_size:  
			Batch size. Defaults is 32.  
max_epochs:  
			Number of epochs to train. Defaults is 25.  
epoch_step:  
			Number of epochs to train before starting to decrease the learning rate. Default is 12.  
z_dim:  
			Dimension of the noise vector. Defaults is 200.
d_eta:  
			Learning rate of the discriminator.Default is 0.00005
g_eta:  
			Learning rate of the generator.Default is 0.00005
beta1:  
			Beta 1 for the Adam optimiser. Default is 0.5
beta2:  
			Beta 2 for the Adam optimiser. Default is 0.999
discriminator:  
			Name of the discriminator to use. 
generator:  
			Name of the generator to use. 
view_func:  
			Name of the view sampling function to use.
train_func:  
			Name of the train function to use.
build_func:  
			Name of the build function to use.
style_disc:  
			Use Style discriminator. Useful for training images at 128.
sample_z:  
			Distribution to sample the noise fector. Default is "uniform".
add_D_noise:  
			Add noise to the input of the discriminator. Default is "true".
DStyle_lambda:  
			Lambda for the style discriminator loss. Default is 1.0
lambda_latent:  
			Lambda for the identity regulariser.Useful for training images at 128. Default is 1.0.
lambda_camera_pose:  
			Lambda for camera pose loss. Default is 1.0.
lambda_reconstruction:  
			Lambda for reconstruction loss. Default is 1.0.  
ele_low:  
			The low range of elevation angle. Default is 60.
ele_high:  
			The high range of elevation angle. Default is 95.
azi_low:  
			The low range of azimuth angle. Default is 0.
azi_high:  
			The high range of azimuth angle. Default is 360.
scale_low:  
			The low range of scaling. Default is 0.8
scale_high:  
			The high range of scaling. Default is 1.5
x_low:  
			Default is 0. HoloGAN did not use x-axi translation, therefore 0.
x_high:  
			Default is 0. HoloGAN did not use x-axi translation, therefore 0.
y_low:  
			Default is 0. HoloGAN did not use y-axi translation, therefore 0.
y_high:  
			Default is 0. HoloGAN did not use y-axi translation, therefore 0.
z_low:  
			Default is 0. HoloGAN did not use z-axi translation, therefore 0.
z_high:  
			Default is 0. HoloGAN did not use z-axi translation, therefore 0.
with_translation:  
			To use translation in 3D transformation. Default is "false".
with_scale:  
			To use scaling in 3D transformation. Default is "true".
output_dir:   
			Full path to the output directory.
```

# Other
The version of Tensorflow is 1.14.0.  
The GPU used for training is GTX 1080 Ti.
A training of 25 epochs cost about 2 days.
In addition, the pre-trained weights for original HoloGAN and HoloGAN plus auto-encoder can be find in
https://drive.google.com/drive/folders/1PrPqrYhbTMLV3q1yWfr3Exhd4r9inIHC?usp=sharing

