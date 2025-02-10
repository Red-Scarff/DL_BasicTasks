import numpy as np
import torch
import torch.nn.functional as F

from gan import GNet

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
    latentdim = 16
    in_shape = (28, 28)
    indim = np.prod(in_shape)
    hiddim = 100
    
    # load the model weights.
    model = GNet(in_features=latentdim, hiddim=hiddim, out_shape=in_shape)
    model.load_state_dict(torch.load('../checkpoints/generator.pt'))

    # data generation
    with torch.no_grad():
        N = 10
        latents = torch.randn((10 * 10, latentdim))
        # images = F.sigmoid(model(latents)).numpy().reshape(-1, 28, 28) # shape: [400, 28, 28]
        images = model(latents).numpy().reshape(-1, 28, 28) # shape: [400, 28, 28]
        plt = image_grid(images, N)
        plt.show()
    
    ## The following code is for 
    # data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    # test_loader = DataLoader(dataset=data, batch_size=400, shuffle=False)
    # iterator = iter(test_loader)
    # images = next(iterator)[0].reshape(400, 28, 28).numpy()
    # plt = image_grid(images, 20)
    # plt.show()