
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x.view(-1, 28, 28)  # Reshape to image size 28x28

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Record the start time of each epoch

    # Track loss and accuracy
    d_loss_total = 0
    g_loss_total = 0
    correct_generated_images = 0
    total_images = 0

    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)

        # Train discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        optimizer_d.zero_grad()

        # Real images
        output = discriminator(real_images)
        d_loss_real = criterion(output, real_labels)
        d_loss_real.backward()

        # Fake images
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        output = discriminator(fake_images.detach())  # No gradient computation for fake images
        d_loss_fake = criterion(output, fake_labels)
        d_loss_fake.backward()

        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)  # Generator wants discriminator to think fake images are real
        g_loss.backward()

        optimizer_g.step()

        # Accumulate loss and accuracy
        d_loss_total += d_loss_real.item() + d_loss_fake.item()
        g_loss_total += g_loss.item()

        # Calculate generator accuracy
        predicted = output > 0.5  # If discriminator's output is greater than 0.5, image is classified as "real"
        correct_generated_images += predicted.sum().item()
        total_images += batch_size

    epoch_end_time = time.time()  # Record the end time of each epoch
    epoch_duration = epoch_end_time - epoch_start_time  # Calculate epoch duration

    # Output training status for each epoch
    epoch_d_loss = d_loss_total / len(train_loader)
    epoch_g_loss = g_loss_total / len(train_loader)
    epoch_accuracy = 100 * correct_generated_images / total_images

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, "
          f"Generator Accuracy: {epoch_accuracy:.2f}%, "
          f"Time: {epoch_duration:.2f} seconds")

    # Display generated images every 5 epochs
    if (epoch + 1) % 5 == 0:
        fake_images = fake_images.view(-1, 28, 28).detach().numpy()  # Convert fake images from tensor to NumPy array
        plt.imshow(fake_images[0], cmap='gray')  # Display the first generated image
        plt.show()



