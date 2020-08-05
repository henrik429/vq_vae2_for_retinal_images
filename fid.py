########################################################################################
# Imported from https://github.com/hukkelas/pytorch-frechet-inception-distance
########################################################################################

import torch
from torch import nn
from torchvision.models import inception_v3
from skimage import io
import multiprocessing
import numpy as np
import glob
import os
from scipy import linalg
from tqdm import tqdm

from utils.utils import setup
from utils.models import Mode, VQ_VAE_2, VQ_VAE
from torchvision.utils import make_grid, save_image

import warnings


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" + \
                                             ", but got {}".format(x.shape)
        x = x * 2 - 1  # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def get_activations(images, batch_size, device="cpu"):
    """
    Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
    """
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" + \
                                              ", but got {}".format(images.shape)

    num_images = images.shape[0]
    inception_network = PartialInceptionNetwork()
    inception_network = inception_network.to(device)
    inception_network.eval()
    n_batches = int(np.ceil(num_images / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = ims.to(device)
        activations = inception_network(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format(
            (ims.shape[0], 2048), activations.shape)
        inception_activations[start_idx:end_idx, :] = activations
    return inception_activations


def calculate_activation_statistics(images, batch_size, device="cpu"):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer
                of the inception model.

    """
    act = get_activations(images, batch_size, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def preprocess_image(im):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
    Return:
        im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
    """
    assert im.shape[2] == 3
    assert len(im.shape) == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    im = np.resize(im, (299, 299, 3))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im)
    assert im.max() <= 1.0
    assert im.min() >= 0.0
    assert im.dtype == torch.float32
    assert im.shape == (3, 299, 299)

    return im


def preprocess_images(images, use_multiprocessing):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
    """
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for im in images:
                job = pool.apply_async(preprocess_image, (im,))
                jobs.append(job)
            final_images = torch.zeros(images.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                im = job.get()
                final_images[idx] = im  # job.get()
    else:
        # final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
        final_images = torch.zeros(images.shape[0], 3, 299, 299)
        for idx, img in enumerate(images):
            final_images[idx] = preprocess_image(img)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float32
    return final_images


def calculate_fid(images1, images2, use_multiprocessing, batch_size, device="cpu"):
    """ Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    images1 = preprocess_images(images1, use_multiprocessing)
    images2 = preprocess_images(images2, use_multiprocessing)
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size, device)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size, device)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def load_images(path, num_images):
    """ Loads all .png or .jpg images from a given path
    Warnings: Expects all images to be of same dtype and shape.
    Args:
        path: relative path to directory
    Returns:
        final_images: np.array of image dtype and shape.
    """
    image_paths = []
    image_extensions = ["png", "jpg", "jpeg"]
    for ext in image_extensions:
        print("Looking for images in", os.path.join(path, "*.{}".format(ext)))
        for impath in glob.glob(os.path.join(path, "*.{}".format(ext)))[:num_images]:
            image_paths.append(impath)
    first_image = io.imread(image_paths[0])
    W, H = first_image.shape[:2]
    image_paths.sort()
    image_paths = image_paths
    final_images = np.zeros((len(image_paths), H, W, 3), dtype=first_image.dtype)
    for idx, impath in enumerate(image_paths):
        im = io.imread(impath)
        assert im.dtype == final_images.dtype
        final_images[idx] = im
    return final_images


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/models.py", config="./config.json")
    input_path = FLAGS.input
    valid_path = FLAGS.valid
    device = FLAGS.device if torch.cuda.is_available() else "cpu"
    mode = Mode.vq_vae if FLAGS.mode == 1 else Mode.vq_vae_2

    network_dir = f'{FLAGS.path_prefix}/models/{FLAGS.network_name}'

    os.makedirs(f"{network_dir}/generated_train_images", exist_ok=True)
    os.makedirs(f"{network_dir}/generated_valid_images", exist_ok=True)
    use_multiprocessing = False
    batch_size = 64
    num_images = 12800
    n_batches = num_images // batch_size

    if mode == Mode.vq_vae:
        num_emb = FLAGS.num_emb
        emb_dim = FLAGS.emb_dim
        size_latent_space = FLAGS.size_latent_space ** 2

        vq_vae = VQ_VAE(
            training=True,
            hidden_channels=FLAGS.hidden_channels,
            num_emb=num_emb,
            emb_dim=emb_dim,
            ema=FLAGS.exponential_moving_averages,
            commitment_cost=0.25,
            gamma=0.99,
            epsilon=1e-5
        )

        vq_vae.load_state_dict(torch.load(f"{network_dir}/{FLAGS.network_name}.pth"))
        vq_vae.to(device=device)
    else:
        num_emb = {"bottom": FLAGS.num_emb_bottom, "top": FLAGS.num_emb_top}
        emb_dim = {"bottom": FLAGS.emb_dim_bottom, "top": FLAGS.emb_dim_top}
        size_latent_space = {"bottom": FLAGS.size_latent_space_bottom ** 2, "top": FLAGS.size_latent_space_top ** 2}

        vq_vae = VQ_VAE_2(
            training=True,
            hidden_channels=FLAGS.hidden_channels,
            num_emb=num_emb,
            emb_dim=emb_dim,
            ema=FLAGS.exponential_moving_averages,
            commitment_cost=0.25,
            gamma=0.99,
            epsilon=1e-5
        ).to(device=device)
        vq_vae.load_state_dict(torch.load(f"{network_dir}/{FLAGS.network_name}.pth"))
        vq_vae.to(device=device)

    file = open(f"{network_dir}/fid_{FLAGS.network_name}.txt", 'w')

    for path in [input_path, valid_path]:
        assert len(os.listdir(path)) >= num_images
        real_images = load_images(path, num_images)
        generated_images = np.zeros_like(real_images)
        with torch.no_grad():
            data = torch.tensor((real_images)).permute(0,3,1,2).float()
            for batch_idx in tqdm(range(n_batches)):
                start_idx = batch_size * batch_idx
                end_idx = batch_size * (batch_idx + 1)

                reconstructions = vq_vae(data[start_idx:end_idx].to(device)).cpu().detach()
                generated_images[start_idx:end_idx] = reconstructions.permute(0,2,3,1).numpy()
                if path == input_path and reconstructions.size(0) > 50:
                    grid = make_grid(torch.cat((data[start_idx:end_idx][0:50:5], reconstructions[0:50:5]), dim=0), nrow=10)
                    save_image(grid,
                               f"{network_dir}/generated_train_images/gen_train_imgs_{batch_idx}.png",
                               normalize=True)
                elif path == valid_path and reconstructions.size(0) > 50:
                    grid = make_grid(torch.cat((data[start_idx:end_idx][0:50:5], reconstructions[0:50:5]), dim=0), nrow=10)
                    save_image(grid,
                               f"{network_dir}/generated_valid_images/gen_valid_imgs_{batch_idx}.png",
                               normalize=True)

        fid_value = calculate_fid(real_images, generated_images, use_multiprocessing, batch_size, device)

        if path==input_path:
            print(f"\nThe FID Score of the train data is: {fid_value}\n")
            file.write(f"\nThe FID Score of the train data is: {fid_value}\n")
        else:
            print(f"\nThe FID Score of the valid data is: {fid_value}\n")
            file.write(f"\nThe FID Score of the valid data is: {fid_value}\n")
