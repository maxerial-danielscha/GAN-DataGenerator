

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import config

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# 
print("Is cuda available")
torch.cuda.is_available()


# Root directory for dataset
dataroot = config.data_path

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 1

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
number_of_channels = 1

# Size of z latent vector (i.e. size of generator input)
gen_input_size = 100

# Size of feature maps in generator
gen_feature_map_size = 128

# Size of feature maps in discriminator
disc_feature_map_size = 128

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                               transforms.Grayscale(num_output_channels=1)
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
print(real_batch[0].shape)


# 
# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: Z vector (latent vector) going into a transposed convolution
            nn.ConvTranspose2d(gen_input_size, gen_feature_map_size * 32, 4, 1, 0, bias=False),  # Output: (ngf*32) x 4 x 4
            nn.BatchNorm2d(gen_feature_map_size * 32),
            nn.ReLU(True),
            # State size: (ngf*32) x 4 x 4
            nn.ConvTranspose2d(gen_feature_map_size * 32, gen_feature_map_size * 16, 4, 2, 1, bias=False),  # Output: (ngf*16) x 8 x 8
            nn.BatchNorm2d(gen_feature_map_size * 16),
            nn.ReLU(True),
            # State size: (ngf*16) x 8 x 8
            nn.ConvTranspose2d(gen_feature_map_size * 16, gen_feature_map_size * 8, 4, 2, 1, bias=False),  # Output: (ngf*8) x 16 x 16
            nn.BatchNorm2d(gen_feature_map_size * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 16 x 16
            nn.ConvTranspose2d(gen_feature_map_size * 8, gen_feature_map_size * 4, 4, 2, 1, bias=False),  # Output: (ngf*4) x 32 x 32
            nn.BatchNorm2d(gen_feature_map_size * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 32 x 32
            nn.ConvTranspose2d(gen_feature_map_size * 4, gen_feature_map_size * 2, 4, 2, 1, bias=False),  # Output: (ngf*2) x 64 x 64
            nn.BatchNorm2d(gen_feature_map_size * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 64 x 64
            nn.ConvTranspose2d(gen_feature_map_size * 2, gen_feature_map_size, 4, 2, 1, bias=False),  # Output: (ngf) x 128 x 128
            nn.BatchNorm2d(gen_feature_map_size),
            nn.ReLU(True),
            # State size: (ngf) x 128 x 128
            nn.ConvTranspose2d(gen_feature_map_size, number_of_channels, 4, 2, 1, bias=False),  # Output: (nc) x 256 x 256
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)



# 
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

# 
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input shape: (nc) x 256 x 256
            nn.Conv2d(number_of_channels, disc_feature_map_size, 4, 2, 1, bias=False),  # Output: (ndf) x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_feature_map_size, disc_feature_map_size * 2, 4, 2, 1, bias=False),  # Output: (ndf*2) x 64 x 64
            nn.BatchNorm2d(disc_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_feature_map_size * 2, disc_feature_map_size * 4, 4, 2, 1, bias=False),  # Output: (ndf*4) x 32 x 32
            nn.BatchNorm2d(disc_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_feature_map_size * 4, disc_feature_map_size * 8, 4, 2, 1, bias=False),  # Output: (ndf*8) x 16 x 16
            nn.BatchNorm2d(disc_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_feature_map_size * 8, disc_feature_map_size * 16, 4, 2, 1, bias=False),  # Output: (ndf*16) x 8 x 8
            nn.BatchNorm2d(disc_feature_map_size * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_feature_map_size * 16, disc_feature_map_size * 32, 4, 2, 1, bias=False),  # Output: (ndf*32) x 4 x 4
            nn.BatchNorm2d(disc_feature_map_size * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_feature_map_size * 32, 1, 4, 1, 0, bias=False),  # Output: 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)


# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, gen_input_size, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu)#.view(-1)
        print(output.shape)
        
        output = output.view(-1)
        # Calculate loss on all-real batch
        print(output.shape)
        
        print(real_cpu.shape)
        print(label.shape)
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, gen_input_size, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1
# 
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
