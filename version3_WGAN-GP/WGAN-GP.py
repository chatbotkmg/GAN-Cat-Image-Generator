
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.init as init
from pytorch_fid import fid_score  
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable


# ----------------------
# Cat Dataset
# ----------------------
class CatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        # Traverse all subdirectories to find all jpg files
        for subdir, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.image_paths.append(os.path.join(subdir, file))

        # Check if images were successfully loaded
        if len(self.image_paths) == 0:
            print(f"🚨 No image files found in {data_dir}, please check the path!")
        else:
            print(f"✅ Found {len(self.image_paths)} images!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0



# ----------------------
# Generator
# ----------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(100, 256)  
        self.fc2 = nn.Linear(256, 512)  
        self.fc3 = nn.Linear(512, 1024)  
        self.fc4 = nn.Linear(1024, 4*4*1024)  

        # Transposed convolution layers (Deconv)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        # Activation functions and normalization
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # BatchNorm2d layers
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(3)  # No BatchNorm needed for the final layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = x.view(-1, 1024, 4, 4)
        
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        
        return x

    def init_weights(self):
        # Initialize fully connected layers
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')

        # Initialize transposed convolution layers (Deconv layers)
        init.kaiming_normal_(self.deconv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.deconv2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.deconv3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.deconv4.weight, mode='fan_in', nonlinearity='relu')
        
        # Uniform initialization for the final layer's transposed convolution
        init.uniform_(self.deconv5.weight, -0.05, 0.05)  # Uniform distribution initialization for the final layer

        # Initialize BatchNorm layers
        init.constant_(self.bn1.weight, 1)  # gamma
        init.constant_(self.bn1.bias, 0)    # beta
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)
        init.constant_(self.bn3.weight, 1)
        init.constant_(self.bn3.bias, 0)
        init.constant_(self.bn4.weight, 1)
        init.constant_(self.bn4.bias, 0)
        init.constant_(self.bn5.weight, 1)
        init.constant_(self.bn5.bias, 0)

# Create generator and initialize weights
generator = Generator()
generator.init_weights()

# ----------------------
# Discriminator
# ----------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1)  # 128x128x3 -> 64x64x128
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)  # 64x64x128 -> 32x32x256
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)  # 32x32x256 -> 16x16x512
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)  # 16x16x512 -> 8x8x1024
        
        # Fully connected layer
        self.fc1 = nn.Linear(1024 * 8 * 8, 1)  # Flatten and enter fully connected layer

        # Activation function
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        
        x = x.view(-1, 1024 * 8 * 8)  # Flatten
        x = self.fc1(x)  # Output a single scalar
        return x

    def init_weights(self):
        # Initialize convolution layers
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')

        # Initialize fully connected layer
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

# Create discriminator and initialize weights
discriminator = Discriminator()
discriminator.init_weights()



# Gradient penalty
def gradient_penalty(discriminator, real_samples, fake_samples, device):
    epsilon = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones(d_interpolated.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Discriminator loss function
def discriminator_loss(discriminator, real_samples, fake_samples, device, lambda_gp=10):
    # WGAN loss: calculate the difference between real and fake sample scores
    real_validity = discriminator(real_samples)
    fake_validity = discriminator(fake_samples)
    
    # Gradient penalty
    gp = gradient_penalty(discriminator, real_samples, fake_samples, device)

    # Total discriminator loss: WGAN loss + gradient penalty
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
    return d_loss

# Generator loss function
def generator_loss(discriminator, fake_samples):
    # Generator loss: maximize the discriminator's score for fake samples
    fake_validity = discriminator(fake_samples)
    g_loss = -torch.mean(fake_validity)
    return g_loss


# ----------------------
# Data processing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Set correct path
data_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/cat_head'  # Update path to cat head dataset
cat_dataset = CatDataset(data_dir, transform=transform)

# Create data loader
cat_loader = DataLoader(cat_dataset, batch_size=64, shuffle=True)

# Verify data loading
print(f"Dataset loaded successfully: {len(cat_loader.dataset)} images")


# ----------------------
# Initialize network and optimizer
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Initialize optimizers
optimizer_d = optim.RMSprop(discriminator.parameters(), lr=0.0002)
optimizer_g = optim.RMSprop(generator.parameters(), lr=0.0001)

# Learning rate scheduler, ReduceLROnPlateau adjusts the learning rate based on loss
scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)

# ----------------------
# Save image function (3x3)
# ----------------------
def save_image_grid(images, epoch, prefix, grid_size=3, show_first=False):
    images_np = images.permute(0,2,3,1).detach().cpu().numpy()

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i, ax in enumerate(axes.flatten()):
        if i < images_np.shape[0]:
            ax.imshow((images_np[i]+1)/2)
            ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()

    if show_first:
        plt.imshow((images_np[0]+1)/2)
        plt.axis('off')
        plt.show()


save_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/version3_WGAN-GP/cat_generatored/'
real_save_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/version3_WGAN-GP/temp_real/'
fake_save_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/version3_WGAN-GP/temp_fake/'

# Create folders (if they don't exist)
os.makedirs(real_save_dir, exist_ok=True)
os.makedirs(fake_save_dir, exist_ok=True)
# ----------------------
# Save a set of real images as reference before training
# ----------------------
real_samples, _ = next(iter(cat_loader))
real_samples = real_samples.to(device)
save_image_grid(real_samples[:9], epoch=0, prefix='real_before_training', grid_size=3, show_first=True)

# Training loop
# ----------------------
num_epochs = 500
lambda_gp = 10  # Weight of gradient penalty

for epoch in range(num_epochs):
    start_time = time.time()
    d_loss_total = 0
    g_loss_total = 0
    correct_generated = 0
    total_images = 0

    if epoch == 0:
        # Save real images only on the first epoch
        real_images = next(iter(cat_loader))[0][:9]
        for i in range(9):
            img = transforms.ToPILImage()(real_images[i])
            img.save(os.path.join(real_save_dir, f'real_{i+1}.jpg'))

    for real_images, _ in cat_loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # --- Discriminator --- 
        optimizer_d.zero_grad()  # Clear discriminator gradients
        output_real = discriminator(real_images)
        d_loss_real = -output_real.mean()  # WGAN uses negative mean as loss
        noise = torch.randn(batch_size, 100, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())  # Don't compute gradients for generated images
        d_loss_fake = output_fake.mean()  # WGAN uses mean as loss
        d_loss = d_loss_real + d_loss_fake
        
        # Compute gradient penalty
        gp = gradient_penalty(discriminator, real_images, fake_images.detach(), device)
        d_loss += lambda_gp * gp
        d_loss.backward()  # Backprop and update discriminator
        optimizer_d.step()

        # --- Generator --- 
        optimizer_g.zero_grad()  # Clear generator gradients
        output_fake = discriminator(fake_images)  # Compute loss for generated images
        g_loss = -output_fake.mean()  # Generator's goal is to maximize the discriminator's output
        g_loss.backward()  # Backprop for generator
        optimizer_g.step()  # Update generator

        # Accumulate loss and accuracy
        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()
        correct_generated += (output_fake > 0).sum().item()
        total_images += batch_size

    epoch_time = time.time() - start_time
    epoch_d_loss = d_loss_total / len(cat_loader)
    epoch_g_loss = g_loss_total / len(cat_loader)
    epoch_accuracy = 100 * correct_generated / total_images

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, "
          f"Time: {epoch_time:.2f}s")

    # Save and display images every 5 epochs (generated + real)
    if (epoch + 1) % 5 == 0:
        save_image_grid(fake_images, epoch+1, prefix='generated', grid_size=3, show_first=True)
        for j in range(9):
            img = transforms.ToPILImage()(fake_images[j])
            img.save(os.path.join(fake_save_dir, f'fake_{epoch+1}_{j+1}.jpg'))
        
        # Calculate FID
        fid_value = fid_score.calculate_fid_given_paths([real_save_dir, fake_save_dir], batch_size=9, dims=2048, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], FID: {fid_value:.4f}")


