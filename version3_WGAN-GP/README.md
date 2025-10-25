
GAN-Cat-Image-Generator

This project implements a Generative Adversarial Network (GAN) for generating realistic cat images, utilizing the WGAN-GP (Wasserstein GAN with Gradient Penalty) algorithm. The goal of this project is to generate high-quality cat images from random noise, and to evaluate the model's performance using metrics such as FID (Fréchet Inception Distance).

Project Overview

This repository contains a working implementation of WGAN-GP using PyTorch. The model consists of a Generator and a Discriminator, both implemented as neural networks. The Generator is trained to produce realistic cat images, while the Discriminator is tasked with distinguishing between real and generated images.

Key Achievements

WGAN-GP Successfully Implemented: We have successfully implemented the WGAN-GP architecture, which improves upon traditional GANs by adding a gradient penalty to the discriminator’s loss function. This ensures more stable training and better convergence.

Stability of Generator and Discriminator: Both the Generator and Discriminator have shown significantly improved stability during training. The use of WGAN-GP has minimized the issue of mode collapse, and the network is able to generate diverse and high-quality images.

FID Score Improvement: The FID score (Fréchet Inception Distance), a metric used to evaluate the quality of generated images, has seen a significant reduction during training. A lower FID score indicates that the generated images are becoming more similar to real images in terms of feature distributions.

Model Architecture

The model is built on the WGAN-GP architecture, which consists of the following components:

Generator: The Generator takes a random vector as input and transforms it into a cat image through a series of fully connected and transposed convolution layers. Batch Normalization is used to stabilize the learning process.

Discriminator: The Discriminator evaluates whether a given image is real or fake. It consists of several convolutional layers, followed by a fully connected layer that outputs a single scalar value indicating the "realness" of the image.

Gradient Penalty: The gradient penalty is added to the discriminator's loss function to improve training stability. This term penalizes large gradients, ensuring smoother and more stable updates to the Discriminator.

Training

The training process involves iterating over real and fake images in batches:

Discriminator Training: In each iteration, the Discriminator is trained to distinguish between real and generated (fake) images. The Discriminator loss consists of two parts: the loss on real images and the loss on fake images, plus the gradient penalty.

Generator Training: The Generator is trained to maximize the Discriminator’s score for fake images, thereby encouraging it to generate more realistic images.

Metrics

Generator Loss: The loss function used for training the Generator aims to maximize the discriminator’s score for fake images.

Discriminator Loss: The Discriminator loss combines the loss on real and fake images, and the gradient penalty to improve training stability.

FID Score: The Fréchet Inception Distance (FID) is computed every few epochs to monitor the progress of the model. Lower FID values correspond to better-quality generated images.

Results

After training the model for 500 epochs, the following improvements have been observed:

The FID score has dropped significantly, indicating that the Generator has improved in generating more realistic images over time.

The Generator is capable of producing diverse cat images with better textures, shapes, and details compared to the earlier epochs.

The Discriminator has become more accurate in distinguishing real and generated images, ensuring a better learning process for the Generator.

Directory Structure
/GAN-Cat-Image-Generator/
    ├── /cat_head/                # Folder containing the cat head dataset
    ├── /version3_WGAN-GP/         # Folder for the final WGAN-GP implementation
        ├── /cat_generatored/      # Generated images
        ├── /temp_real/            # Real images
        ├── /temp_fake/            # Fake images
    ├── /saved_models/             # Folder to store saved models (Generator, Discriminator)
    ├── /notebooks/                # Jupyter notebooks for visualization and analysis
    └── README.md                  # This file

Installation

Clone the repository:

git clone https://github.com/yourusername/GAN-Cat-Image-Generator.git
cd GAN-Cat-Image-Generator


Install dependencies:

pip install -r requirements.txt


Download the cat dataset:
You can upload your own dataset to the /cat_head/ folder. It should contain cat images in .jpg format.

Start training:

python train.py

Evaluation

The evaluation of the model is done using the FID score, which can be calculated after each training epoch. A lower FID score indicates better performance in generating realistic images. The FID score is computed between real and generated images, and it helps in understanding how close the generated images are to real images.

Future Improvements

Improve Image Resolution: The current model generates images at 128x128 resolution. Future improvements could involve increasing the resolution of generated images by adjusting the network architecture.

Incorporate More Advanced Models: Explore other GAN variants like StyleGAN or BigGAN for even higher quality image generation.

Fine-tuning with Transfer Learning: Pre-train the model on a large dataset and fine-tune it for the specific cat image generation task to improve performance.
