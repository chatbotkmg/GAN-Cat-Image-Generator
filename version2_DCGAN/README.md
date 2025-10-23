
# GAN Cat Image Generator - DCGAN Version 2

This project is an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate cat images. The goal is to showcase the transition from a fully connected (FC) network in earlier versions to a convolutional (DC) network that can generate higher-quality images. The project is implemented in PyTorch and uses a custom dataset of cat head images. 

### Key Features:
- **Generator Architecture**: Transition from Fully Connected (FC) layers to Transposed Convolutional (Deconvolutional) layers to preserve spatial hierarchies in images.
- **Discriminator Architecture**: Utilizes deep convolutional layers for better image classification.
- **FID Score**: Introduces FID (Frechet Inception Distance) as a metric for assessing the quality of generated images.

## Project Structure

The project directory structure is as follows:

/version2_DCGAN/
├── cat_generatored/ # Folder to save generated cat images
├── temp_real/ # Folder to store real images for FID calculation
├── temp_fake/ # Folder to store fake images for FID calculation
├── README.md # Project README file
├── train.py # Training script
└── ... # Other files like models, utils, etc.



## Motivation

In the first version of this project, the generator was based on a **fully connected (FC)** network. However, FC networks fail to capture spatial features in images, leading to low-quality outputs. In this version (v2), we've switched to **Deep Convolutional Networks (DC)**, specifically using **Transposed Convolutions** (Deconvolutional layers) in the generator. This allows us to retain the spatial structure of images, improving the quality of the generated images significantly.

### Key Improvements in v2:
- **Transition to Transposed Convolutions**: The generator now uses a series of transposed convolutional layers, enabling it to generate more detailed images with spatial coherence.
- **Batch Normalization**: Added batch normalization layers to both the generator and discriminator for stable training and improved performance.

## Dataset

This project uses a custom dataset of **cat head images**. The images should be placed in the folder specified by the `data_dir` parameter in the `train.py` script. You can use your own dataset or download one containing cat head images.

### Dataset Directory Structure:

/cat_head/
├── CAT_00/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── CAT_01/
└── ...


## Training

To train the GAN model, run the following command:

```bash
python train.py
This will start the training process, and the generator will begin producing cat images after each epoch. The generated images will be saved in the cat_generatored/ directory.

Training Process
The model is trained for 500 epochs.

The FID score is calculated every 5 epochs to evaluate the quality of the generated images.

Real and fake images are saved periodically for further analysis.

The loss for both the discriminator and the generator is monitored during training.

Current Issues and Limitations
Despite the improvements from the fully connected (FC) network, the current model still faces several challenges:

Mode Collapse: The generator still exhibits signs of mode collapse, meaning it tends to produce a limited variety of images despite training for many epochs.

FID Score: The Frechet Inception Distance (FID) score, which measures the similarity between real and generated images, has plateaued at around 270. While this is an improvement, it is still far from the ideal value.

Training Stability: Although batch normalization was added, the training process still suffers from occasional instability, which could be improved with more advanced techniques.

Version 3 Plan
For Version 3, we plan to introduce several advanced techniques to improve the performance of the model:

Complex Network Architectures: We plan to introduce more complex architectures like ResNet blocks or U-Net structures for the generator to improve feature extraction and synthesis.

Wasserstein GAN with Gradient Penalty (WGAN-GP): This method aims to provide a more stable training process and reduce issues like mode collapse. We believe that WGAN-GP will significantly enhance the quality of the generated images and stabilize training.

Higher-Resolution Images: In the next version, we also plan to increase the resolution of the generated images, moving from 64x64 pixels to higher resolutions like 128x128 or 256x256.

Results
The results of the training are stored in the cat_generatored/ folder. You can find generated images for every epoch, named like:


cat_generatored/generated_epoch_5.png
cat_generatored/generated_epoch_10.png
The FID score is also displayed in the terminal output every 5 epochs to track the progress of the model. A lower FID score indicates better image quality.



