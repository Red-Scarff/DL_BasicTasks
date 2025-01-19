import torch
import torch.nn.functional as F

from vae import VAE

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


if __name__ == '__main__':
    indim = 28 * 28
    hiddim = 100
    latentdim = 2
    
    # load the model weights.
    model = VAE(indim, hiddim, latentdim)
    model.load_state_dict(torch.load('./checkpoints/vae_epoch_10.pth'))

    # data generation
    with torch.no_grad():
        z = torch.arange(-2, 2, 0.1)
        N = len(z)
        z1, z2 = torch.meshgrid(z, z)
        latents = torch.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])
        images = F.sigmoid(model.decode(latents)).numpy().reshape(-1, 28, 28) # shape: [400, 28, 28]
        plt = image_grid(images, N)
        plt.show()
    
    ## The following code is for 
    # data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    # test_loader = DataLoader(dataset=data, batch_size=400, shuffle=False)
    # iterator = iter(test_loader)
    # images = next(iterator)[0].reshape(400, 28, 28).numpy()
    # plt = image_grid(images, 20)
    # plt.show()