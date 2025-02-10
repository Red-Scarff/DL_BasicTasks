import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import os

def image_grid(images, n_rows):
    """Helper function to visualize a batch of images."""
    import matplotlib.pyplot as plt
    n_images = len(images)
    n_cols = n_images // n_rows
    if n_images % n_rows != 0: n_cols += 1
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(image, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    return plt

class DNet(nn.Module):
    """
    We have implemented the Discriminator network for you.
    Note: 
        (i) The input image is flattened in the forward function to a vector of size 784;
        (ii) The output of the network is a single value, which is the logit of the input image being real, 
        which means you need to use the binary cross entropy loss with logits to train the discriminator.
    """
    def __init__(self, in_features, hiddim, out_features=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hiddim = hiddim

        # Discriminator will down-sample the input producing a binary output
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hiddim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=hiddim, out_features=hiddim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=hiddim, out_features=out_features),
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class GNet(nn.Module):
    """
    You need to implement the Generator network.
        The architecture of the Generator network should be the same as the Discriminator network, 
        with only one difference: the final output layer should have a Tanh activation function.
    """
    def __init__(self, in_features, hiddim, out_shape):
        super(GNet, self).__init__()
        out_features = np.prod(out_shape)
        self.out_features = out_features
        self.out_shape = out_shape
        self.in_features = in_features
        
        self.hiddim = hiddim

        # Implement self.net with nn.Sequential() method, shown as below:
        # self.net = nn.Sequential(...)
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hiddim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=hiddim, out_features=hiddim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=hiddim, out_features=out_features),
            nn.Tanh()  # 添加Tanh激活函数
        )
        
    def forward(self, x):
        """Returns a batch of generated images of shape [batch_size, *self.out_shape]."""
        return self.net(x).view(-1, *self.out_shape)
    
if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    """

    # hyper-parameters setting.
    in_shape = (28, 28)
    indim = np.prod(in_shape)
    hiddim = 100
    latentdim = 16
    epoch = 50
    batch_size = 16

    # Tune the learning rate for the generator and the discriminator.
    # You should observe the loss of the discriminator to be close to 0.69=ln(2) at the beginning of the training.
    # lr_g = ???
    # lr_d = ???
    lr_g = 0.0002
    lr_d = 0.0002
    
    data_dir = '../data'
    checkpoint_dir = '../checkpoints'
    log_dir = '../logs'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # preparing dataset
    # we transform the pixels of the image to be {-1, 1}, 
    # which is consistent with the output of the generator.
    tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, transform=tranform, download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # model instantiation
    # model_g = GNet(???)
    # model_d = DNet(???)
    model_g = GNet(latentdim, hiddim, in_shape).to(device)
    model_d = DNet(indim, hiddim).to(device)

    # optimizer instantiation with both adam optimizer.
    # optimizer_g = torch.optim.Adam(???)
    # optimizer_d = torch.optim.Adam(???)
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr_g)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr_d)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.5)

    # fixed set of latents z, used to visualize the generated images
    # and compare the quality of the generated images.
    fixed_z = torch.randn((100, latentdim)).to(device)

    losses_g, losses_d = [], []
    step = 0
    for ep in range(epoch):
        model_d.train()
        model_g.train()
        print(f'epoch[{ep}]')
        for x, _ in train_loader:
            # squeeze the x
            real_x = torch.squeeze(x, dim=1)
            real_x = real_x.to(device)

            # generate the fake x using the generator
            # latent vector z sampling from the normal distribution
            # z = ???
            # fake_x = ???
            z = torch.randn(batch_size, latentdim, device=device)  # 随机生成噪声
            fake_x = model_g(z)

            # train the discriminator with binary cross entropy loss.
            # 1. concatenate the real_x and fake_x along the batch dimension.
            #    Note: you need to detach the fake_x from the computational graph, using .detach().clone()
            x_concat = torch.cat((real_x, fake_x.detach().clone()), dim=0)
            # 2. concatenate the real_y and fake_y along the batch dimension.
            real_y = torch.ones(batch_size,1, device=device)
            fake_y = torch.zeros(batch_size,1, device=device)
            y_concat = torch.cat((real_y, fake_y), dim=0)
            # 3. compute the logits of the concatenated x.
            x_concat_logits = model_d(x_concat)
            # 4. compute the binary cross entropy loss with logits
            loss_d = F.binary_cross_entropy_with_logits(x_concat_logits, y_concat)
            # 5. append the loss to losses_d
            losses_d.append(loss_d.item())
            # 6. update the discriminator for one step
            optimizer_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), max_norm=1.0)
            optimizer_d.step()

            # update the generator for one step
            # 1. compute the logits of the fake_x
            fake_x_logits = model_d(fake_x)
            # 2. compute the binary cross entropy loss with logits, for only the fake_x
            loss_g = F.binary_cross_entropy_with_logits(fake_x_logits, real_y)
            # 3. append the loss to losses_g
            losses_g.append(loss_g.item())
            # 4. update the generator for one step
            optimizer_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
            optimizer_g.step()
            
            # log the losses.
            step += 1
            if step % 100 == 0:
                print(f'step {step} | loss_d=[{loss_d:.4f}] | loss_g=[{loss_g:.4f}]')
        
        model_d.eval()
        model_g.eval()
        # visualize the generated images of the fixed zs. 
        with torch.no_grad():
            fake_x = model_g(fixed_z) * 0.5 + 0.5
            fake_x = fake_x.cpu().numpy().reshape(-1, 28, 28)
            plt = image_grid(fake_x, n_rows=10)
            plt.suptitle(f'epoch {ep}')
            plt.savefig(os.path.join(log_dir, f'epoch_{ep}.png'))

        scheduler_g.step()
        scheduler_d.step()

    # checkpoint the model.
    torch.save(model_d.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pt'))
    torch.save(model_g.state_dict(), os.path.join(checkpoint_dir, 'generator.pt'))
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(range(len(losses_d)), losses_d, 'g', label='Discriminator Losses')
    plt.plot(range(len(losses_g)), losses_g, 'b', label='Generator Losses')
    plt.title('Training Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'losses.png'))