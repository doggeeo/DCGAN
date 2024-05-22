import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ConvDiscriminator(nn.Module):
    def __init__(
        self,
        features: int
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, features*3, kernel_size=4, padding=1, stride=2),  # 32x32
            nn.LeakyReLU(0.2),

            nn.Conv2d(features*3, features*6, kernel_size=4, padding=1, stride=2),  # 16x16
            nn.BatchNorm2d(features*6),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features*6, features*12, kernel_size=4, padding=1, stride=2),  # 8x8
            nn.BatchNorm2d(features*12),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features*12, features*24, kernel_size=4, padding=1, stride=2),  # 4x4
            nn.BatchNorm2d(features*24),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features*24, 1, kernel_size=4, stride=2), # 1x1 хз зачем нужен stride
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class ConvGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        features: int
    ):
        super().__init__()


        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features*16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features*16),
            nn.ReLU(), # 4x4 rgb

            nn.ConvTranspose2d(features*16, features*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.ReLU(), # 8x8

            nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.ReLU(), # 16x16

            nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.ReLU(),  # 32x32

            nn.ConvTranspose2d(features*2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # 64x64
        )

    def forward(self, x):
        return self.layers(x)



in_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
data_path = "cats"
batch_size=100
data = torchvision.datasets.ImageFolder(root=data_path,transform=in_transform)
loader = DataLoader(data, batch_size=batch_size, shuffle=True)
learning_rate=0.0004
img_size=64
latent_dim = 100
discriminator = ConvDiscriminator(features=64).to(device)
discriminator.load_state_dict(torch.load("discriminator.pth"))
discriminator.eval()
generator = ConvGenerator(latent_dim=100, features=64).to(device)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()
optim_gen=optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5,0.999)) # возможно вместо 0.999 нужно 0.9
optim_disc=optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5,0.999)) # возможно вместо 0.999 нужно 0.9
criterion=nn.BCELoss()
const_noise = torch.randn((8, latent_dim, 1, 1)).to(device)
num_epochs=3000
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
generator.train()
discriminator.train()
step=0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake = generator(noise)

        #Discriminator
        disc_real = discriminator(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        discriminator.zero_grad()
        loss_disc.backward()
        optim_disc.step()

        #Generator
        output = discriminator(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_gen.backward()
        optim_gen.step()
        print(epoch,batch_idx)
        if batch_idx % 20 == 0:
            with torch.no_grad():
                fake = generator(const_noise)
                img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            torch.save(generator.state_dict(), 'generator.pth')
            torch.save(discriminator.state_dict(), 'discriminator.pth')
            step += 1
