
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pytorch_fid import fid_score  # Import fid_score to calculate FID

# ----------------------
# Cat Dataset
# ----------------------
class CatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        # Traverse all subdirectories and find all jpg files
        for subdir, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.image_paths.append(os.path.join(subdir, file))

        # Check if images were loaded successfully
        if len(self.image_paths) == 0:
            print(f"🚨 No image files found in path {data_dir}, please check the path setting!")
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
        self.fc1 = nn.Linear(100, 256)  # From noise to 256 dimensions
        self.fc2 = nn.Linear(256, 512)  # 512 dimensions
        self.fc3 = nn.Linear(512, 1024)  # 1024 dimensions
        self.fc4 = nn.Linear(1024, 4*4*1024)  # Output feature map (4x4x1024)

        # Transposed convolution layers (Deconv)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)  # 4x4x1024 -> 8x8x512
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)   # 8x8x512 -> 16x16x256
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)   # 16x16x256 -> 32x32x128
        self.deconv4 = nn.ConvTranspose2d(128, 3, 4, 2, 1)     # 32x32x128 -> 64x64x3

        # Activation functions and normalization
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # BatchNorm2d layers
        self.bn1 = nn.BatchNorm2d(512)  # Corresponding to deconv1 output channels
        self.bn2 = nn.BatchNorm2d(256)  # Corresponding to deconv2 output channels
        self.bn3 = nn.BatchNorm2d(128)  # Corresponding to deconv3 output channels
        self.bn4 = nn.BatchNorm2d(3)    # Corresponding to deconv4 output channels, no need for batch norm in last layer

    def forward(self, x):
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = x.view(-1, 1024, 4, 4)  # Reshape to (batch_size, 1024, 4, 4)
        
        # Use transposed convolution layers to generate the image
        x = self.relu(self.bn1(self.deconv1(x)))  # Add BatchNorm
        x = self.relu(self.bn2(self.deconv2(x)))  # Add BatchNorm
        x = self.relu(self.bn3(self.deconv3(x)))  # Add BatchNorm
        x = self.tanh(self.deconv4(x))  # Last layer doesn't have BatchNorm, use tanh activation
        return x


# ----------------------
# Discriminator
# ----------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1)  # 64x64x3 -> 32x32x128
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)  # 32x32x128 -> 16x16x256
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)  # 16x16x256 -> 8x8x512
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)  # 8x8x512 -> 4x4x1024
        self.fc1 = nn.Linear(1024 * 4 * 4, 1)  # Flatten before going into the fully connected layer
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = x.view(-1, 1024 * 4 * 4)  # Flatten
        x = self.sigmoid(self.fc1(x))  # Output a probability between 0 and 1
        return x


# ----------------------
# Data Processing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Set the correct path
data_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/cat_head'  # Update the path to the cat head dataset
cat_dataset = CatDataset(data_dir, transform=transform)

# Create data loader
cat_loader = DataLoader(cat_dataset, batch_size=64, shuffle=True)

# Verify data loading
print(f"Dataset loaded successfully: {len(cat_loader.dataset)} images")


# ----------------------
# Initialize Network and Optimizers
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# ----------------------
# Create Directories
# ----------------------
save_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/version2_DCGAN/cat_generatored/'
real_save_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/version2_DCGAN/temp_real/'
fake_save_dir = '/kaggle/working/cat_gan/GAN-Cat-Image-Generator/version2_DCGAN/temp_fake/'

# Clear directories (create them if they don't exist)
os.makedirs(real_save_dir, exist_ok=True)
os.makedirs(fake_save_dir, exist_ok=True)

# ----------------------
# Save Image Function (3x3)
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

real_samples, _ = next(iter(cat_loader))
real_samples = real_samples.to(device)
save_image_grid(real_samples[:9], epoch=0, prefix='real_before_training', grid_size=3, show_first=True)

# ----------------------
# Training Loop
# ----------------------
num_epochs = 500
for epoch in range(num_epochs):
    start_time = time.time()
    d_loss_total = 0
    g_loss_total = 0
    correct_generated = 0
    total_images = 0
    if epoch == 0:
        # Randomly select 9 real images from the dataset and save to temp_real folder
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
        optimizer_d.zero_grad()
        output_real = discriminator(real_images)
        d_loss_real = criterion(output_real, real_labels)
        noise = torch.randn(batch_size, 100, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(output_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # --- Generator ---
        optimizer_g.zero_grad()
        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_g.step()

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()
        correct_generated += (output > 0.5).sum().item()
        total_images += batch_size

    epoch_time = time.time() - start_time
    epoch_d_loss = d_loss_total / len(cat_loader)
    epoch_g_loss = g_loss_total / len(cat_loader)
    epoch_accuracy = 100 * correct_generated / total_images

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, "
          f"Generator Accuracy: {epoch_accuracy:.2f}%, "
          f"Time: {epoch_time:.2f}s")

    # Save and display images every 5 epochs (generated images)
    if (epoch + 1) % 5 == 0:
        save_image_grid(fake_images, epoch+1, prefix='generated', grid_size=3, show_first=True)
        for j in range(9):
            img = transforms.ToPILImage()(fake_images[j])
            img.save(os.path.join(fake_save_dir, f'fake_{epoch+1}_{i+1}_{j+1}.jpg'))

        # Calculate FID every 5 epochs
        fid_value = fid_score.calculate_fid_given_paths([real_save_dir, fake_save_dir], batch_size=9, dims=2048, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], FID: {fid_value:.4f}")

