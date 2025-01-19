import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import os


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.2):
        """
        Here, we use a simple MLP encoder and decoder, and parameterize the latent
        The encoder and decoder are both MLPs with 2 hidden layers, whose activation functions are all ReLU, i.e.,
        encoder: input_dim -> hidden_dim -> hidden_dim -> ??? (what is ??? here, as we need to output both mu and var?)
        decoder: latent_dim -> hidden_dim -> hidden_dim -> input_dim.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # TODO: instantiate the encoder and decoder.
        # 实例化编码器
        # 实例化编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和方差，所以输出维度是 latent_dim * 2
        )
        
        # 实例化解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出范围在 [0, 1] 之间
        )

    def encode(self, x):
        """
        Probabilistic encoding of the input x to mu and sigma.
        Note:
            sigma needs to be (i) diagnal and (ii) non-negative,
            but Linear() layer doesn't give you that, so you need to transform it.
        Hint:
            (i) modeling sigma in the form of var,
            (ii) use torch.log1p(torch.exp()) to ensure the non-negativity of var.
        """
        x = self.encoder(x)
        mu, var = torch.chunk(x, 2, dim=1)  # 分割成mu和var
        var = torch.log1p(torch.exp(var))  # 转换成非负数
        return mu, var

    def reparameterize(self, mu, var):
        """
        Reparameterization trick, return the sampled latent variable z.
        Note:
            var is the variance, sample with std.
        """
        # 从方差转换为标准差
        std = torch.sqrt(var + 1e-10)  # 加上一个小的常数以防止数值不稳定
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        z = mu + eps * std  # 重参数化
        return z

    def decode(self, z):
        """
        Generation with the decoder.
        """
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        """
        The forward function of the VAE.
        Returns:
            (i) x_hat, the reconstructed input;
            (ii) mu, the mean of the latent variable;
            (iii) var, the variance of the latent variable.
        """
        mu, var = self.encode(x)  # 编码
        z = self.reparameterize(mu, var)  # 重参数化
        x_hat = self.decode(z)  # 解码
        return x_hat, mu, var


def loss_function(x, x_hat, mu, log_var):
    """
    VAE 的总损失：重建损失 + KL 散度损失
    """
    # 重建损失（使用二元交叉熵损失）
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    
    # KL 散度
    # 目标是将潜在变量的分布接近标准正态分布
    # D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 这里的 log_var 是 log(sigma^2)，因此需要进行一些变换。
    KL_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 总损失
    loss = BCE + KL_div
    return loss, BCE, KL_div

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    """

    # hyper-parameters setting.
    indim = 28 * 28
    hiddim = 100
    latentdim = 2 
    epoch = 10
    batch_size = 64
    lr = 1e-4
    dropout_prob = 0.2  # Dropout 概率
    weight_decay = 1e-5  # L2 正则化（权重衰减）
    
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # preparing dataset
    dataset = datasets.MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # model instantiation
    model = VAE(indim, hiddim, latentdim).cuda()

    model.apply(init_weights) 

    # optimizer instantiation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    reconstruct_losses, kl_losses = [], []
    for ep in range(epoch):

        model.train()
        running_loss = 0.0
        running_reconstruct_loss = 0.0
        running_kl_loss = 0.0

        for x, _ in tqdm(train_loader):
            # flatten the x
            x = x.view(-1, indim).cuda()

            # foward of the x
            x_hat, mu, log_var = model(x)

            # calculate the reconstruction loss,
            # where we use binary cross entropy loss as the reconstruction loss, not the MSE.
            # calculate the kl-divergence loss.
            
            loss, recon_loss, kl_loss = loss_function(x, x_hat, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            running_loss += loss.item()
            running_reconstruct_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()

            # log the losses.
            reconstruct_losses.append(running_reconstruct_loss / len(train_loader.dataset))
            kl_losses.append(running_kl_loss / len(train_loader.dataset))
        
        print(f'epoch[{ep}]')
        print(f'losses | rec=[{reconstruct_losses[-1]:.4f}] | kl=[{kl_losses[-1]:.4f}]')

    # checkpoint the model.
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'vae_epoch_{ep+1}.pth'))
    
    import matplotlib.pyplot as plt
    plt.plot(range(len(reconstruct_losses)), reconstruct_losses, 'g', label='Reconstruction Losses')
    # plt.plot(range(len(kl_losses)), kl_losses, 'b', label='KL-Div Losses')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'losses.png'))