"""
Trigger training here
"""
import os
import sys
import torch
import time
import numpy as np
from skimage import io
from skimage import img_as_ubyte

from utils.dataloader import Dataloader
from utils.utils import setup
from utils.models import VQ_VAE_Training, Mode


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/models.py", config="./config.json")
    input_path = FLAGS.input
    valid_path = FLAGS.valid
    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    network_dir = f'{FLAGS.path_prefix}/models/{FLAGS.network_name}'
    os.makedirs(network_dir, exist_ok=True)
    batch_size = FLAGS.batch_size

    if FLAGS.network_name in os.listdir(network_dir):
        input1 = input("\nNetwork already exists. Are you sure to proceed? ([y]/n) ")
        if not input1 in ['y', 'yes']:
            sys.exit()

    train_data = Dataloader(input_path, batch_size=FLAGS.batch_size)
    valid_data = Dataloader(valid_path, batch_size=FLAGS.batch_size)

    for data, in train_data:
        image_size = data.size(2)
        break

    if FLAGS.mode == 1:
        mode = Mode.vq_vae
    elif FLAGS.mode == 2:
        mode = Mode.vq_vae_2
    elif FLAGS.mode == 3:
        mode = Mode.vae
    elif FLAGS.mode == 4:
        mode = Mode.custom_vq_vae_2

    if mode == Mode.vq_vae:
        num_emb=FLAGS.num_emb
        emb_dim=FLAGS.emb_dim
        size_latent_space = FLAGS.size_latent_space ** 2
        reduction_factor = image_size // FLAGS.size_latent_space

    elif mode == Mode.vq_vae_2 or mode == Mode.custom_vq_vae_2:
        num_emb = {"top": FLAGS.num_emb_top, "bottom": FLAGS.num_emb_bottom}
        emb_dim = {"top": FLAGS.emb_dim_top, "bottom": FLAGS.emb_dim_bottom}
        size_latent_space = {"top": FLAGS.size_latent_space_top ** 2,
                             "bottom": FLAGS.size_latent_space_bottom ** 2}
        reduction_factor = {"top": FLAGS.size_latent_space_bottom // FLAGS.size_latent_space_top,
                            "bottom": image_size // FLAGS.size_latent_space_bottom}
    elif mode == Mode.vae:
        reduction_factor = image_size // FLAGS.size_before_fc
        num_emb = None
        emb_dim = None

    vq_vae = VQ_VAE_Training(
        train_data,
        valid=valid_data,
        mode=mode,
        training=True,
        reduction_factor=reduction_factor,
        hidden_channels=FLAGS.hidden_channels,
        num_emb=num_emb,
        emb_dim=emb_dim,
        z_dim=FLAGS.z_dim,
        image_size=image_size,
        encoding_channels=FLAGS.encoding_channels,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=None,
        max_epochs=FLAGS.maxepochs,
        device=device,
        network_dir=network_dir,
        network_name=FLAGS.network_name,
        report_interval=FLAGS.reportinterval,
        checkpoint_interval=FLAGS.checkpointinterval,
        verbose=True,
        ema=FLAGS.exponential_moving_averages,
        commitment_cost=0.25,
        gamma=0.99,
        epsilon=1e-5
    )

    print("\nStart Training...")
    time_start = time.time()
    vq_vae.train()
    print('\nTraining with %i epochs done! Time elapsed: %.2f hours' % (FLAGS.maxepochs, (time.time() - time_start)/360))

    os.system(f"cp utils/models.py {network_dir}/models.py ")
    os.system(f"cp config.json {network_dir}/config.json ")

    os.makedirs(f"{network_dir}/generated_train_images", exist_ok=True)
    os.makedirs(f"{network_dir}/generated_valid_images", exist_ok=True)

    num_images = 10240
    n_batches = num_images // batch_size

    for path in [input_path, valid_path]:
        image_list = os.listdir(path)
        try:
            image_list.remove(".snakemake_timestamp")
        except ValueError:
            pass

        assert len(os.listdir(path)) >= num_images

        image_list = image_list[:num_images]
        first_image = io.imread(path + image_list[0])
        W, H = first_image.shape[:2]
        data = np.zeros((len(image_list), H, W, 3), dtype=first_image.dtype)
        for idx, jpeg in enumerate(image_list):
            im = io.imread(path + jpeg)
            assert im.dtype == data.dtype
            data[idx] = im

        with torch.no_grad():
            data = torch.tensor(data).permute(0, 3, 1, 2).float()
            for i in range(n_batches):
                reconstructions = vq_vae.vae(data[i * batch_size:(i + 1) * batch_size].to(device)).cpu().detach()
                for k, reconstruction in enumerate(reconstructions.permute(0, 2, 3, 1)):
                    assert reconstruction.shape == (W, H, 3)
                    if path == input_path:
                        io.imsave(f"{network_dir}/generated_train_images/{image_list[i*batch_size+k]}",
                                  img_as_ubyte(reconstruction.numpy()))
                    else:
                        io.imsave(f"{network_dir}/generated_valid_images/{image_list[i*batch_size+k]}",
                                  img_as_ubyte(reconstruction.numpy()))

