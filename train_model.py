"""
Trigger training here
"""
import os
import sys
import torch
import time

from utils.dataloader import Dataloader
from utils.utils import setup
from utils.models import VQ_VAE_Training, Mode
from utils.introspection import visualize_latent_space


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/models.py", config="./config.json")
    input_path = FLAGS.input
    valid_path = FLAGS.valid
    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    network_dir = f'{FLAGS.path_prefix}/models/{FLAGS.network_name}'
    os.makedirs(network_dir, exist_ok=True)

    if FLAGS.network_name in os.listdir(network_dir):
        input1 = input("\nNetwork already exists. Are you sure to proceed? ([y]/n) ")
        if not input1 in ['y', 'yes']:
            sys.exit()

    train_data = Dataloader(input_path, batch_size=FLAGS.batch_size)
    valid_data = Dataloader(valid_path, batch_size=FLAGS.batch_size)

    mode = Mode.vq_vae if FLAGS.mode == 1 else Mode.vq_vae_2

    if mode == Mode.vq_vae:
        num_emb=FLAGS.num_emb
        emb_dim=FLAGS.emb_dim
        size_latent_space = FLAGS.size_latent_space ** 2
    else:
        num_emb = {"top": FLAGS.num_emb_top, "bottom": FLAGS.num_emb_bottom}
        emb_dim = {"top": FLAGS.emb_dim_top, "bottom": FLAGS.emb_dim_bottom}
        size_latent_space = {"top": FLAGS.size_latent_space_top ** 2,
                             "bottom": FLAGS.size_latent_space_bottom ** 2
        }

    vq_vae = VQ_VAE_Training(
        train_data,
        valid=valid_data,
        mode=mode,
        training=True,
        hidden_channels=FLAGS.hidden_channels,
        num_emb=num_emb,
        emb_dim=emb_dim,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=None,
        max_epochs=FLAGS.maxepochs,
        device=device,
        network_dir = network_dir,
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
