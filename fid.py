########################################################################################
# Imported from https://github.com/hukkelas/pytorch-frechet-inception-distance
########################################################################################

import torch
from skimage import io, img_as_float32
import numpy as np
import os
from scipy import linalg
from tqdm import tqdm

from utils.inception import InceptionV3
from utils.utils import setup
import warnings



def get_activations(images, batch_size, device="cpu"):
    """
    Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
    """
    num_images = images.shape[0]
    inception_network = InceptionV3()
    inception_network = inception_network.to(device)
    inception_network.eval()
    print(num_images)
    n_batches = int(np.floor(num_images / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = ims.to(device)
        activations = inception_network(ims)[0].view(ims.shape[0], -1)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expected output shape to be: {}, but was: {}".format(
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


def calculate_fid(images1, images2, batch_size, device="cpu"):
    """ Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size, device)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size, device)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/models.py", config="./config.json")
    input_path = FLAGS.input
    valid_path = FLAGS.valid
    device = FLAGS.device if torch.cuda.is_available() else "cpu"
    network_dir = f'{FLAGS.path_prefix}/models/{FLAGS.network_name}'

    print("\nCalculate FID-Score...")
    file = open(f"{network_dir}/fid_{FLAGS.network_name}.txt", 'w')
    gen_train_path = f"{network_dir}/generated_train_images/"
    gen_valid_path = f"{network_dir}/generated_valid_images/"

    num_images = len(os.listdir(gen_train_path))
    batch_size = 32

    for path1, path2 in [(gen_train_path, input_path), (gen_valid_path, valid_path)]:
        generated_images = torch.zeros((num_images, 3, 256, 256))
        real_images = torch.zeros((num_images, 3, 256, 256))
        for i, jpg in enumerate(os.listdir(path1)):
            generated_images[i] = torch.tensor(img_as_float32(io.imread(path1+jpg))).permute(2, 0, 1).float()
            real_images[i] = torch.tensor(img_as_float32(io.imread(path2+jpg))).permute(2, 0, 1).float()

        assert generated_images.shape == (generated_images.shape[0], 3, 256, 256)
        assert generated_images.max() <= 1.0
        assert generated_images.min() >= 0.0
        assert generated_images.dtype == torch.float32

        assert real_images.shape == (real_images.shape[0], 3, 256, 256)
        assert real_images.max() <= 1.0
        assert real_images.min() >= 0.0
        assert real_images.dtype == torch.float32

        fid_value = calculate_fid(real_images, generated_images, batch_size, device)

        if path2 == input_path:
            print(f"\nThe FID Score of the train data is: {fid_value}\n")
            file.write(f"\nThe FID Score of the train data is: {fid_value}\n")
        else:
            print(f"\nThe FID Score of the valid data is: {fid_value}\n")
            file.write(f"\nThe FID Score of the valid data is: {fid_value}\n")

