"""
Trigger training here
"""
import torch

from utils.dataloader import Dataloader
from utils.utils import setup
from utils.models import VAE, VQ_VAE, VQ_VAE_2, Mode
from utils.introspection import visualize_latent_space


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/models.py", config="./config.json")
    device = FLAGS.device if torch.cuda.is_available() else "cpu"
    network_dir = f'{FLAGS.path_prefix}/models/{FLAGS.network_name}'
    mode = Mode.vq_vae if FLAGS.mode == 1 else Mode.vq_vae_2

    # Visualization of the latent space
    test_path = FLAGS.test
    test_data = Dataloader(test_path, batch_size=FLAGS.batch_size)
    for data, in test_data:
        image_size = data.size(2)
        break

    reduction_factor = image_size // FLAGS.size_latent_space

    if mode == Mode.vq_vae:
        num_emb=FLAGS.num_emb
        emb_dim=FLAGS.emb_dim
        size_latent_space = FLAGS.size_latent_space ** 2
        vae = VQ_VAE (
                        hidden_channels=FLAGS.hidden_channels,
                        num_emb=num_emb,
                        emb_dim=emb_dim,
                        reduction_factor=reduction_factor
                        )

    elif mode == Mode.vq_vae_2:
        num_emb = {"top": FLAGS.num_emb_top, "bottom": FLAGS.num_emb_bottom}
        emb_dim = {"top": FLAGS.emb_dim_top, "bottom": FLAGS.emb_dim_bottom}
        size_latent_space = {"top": FLAGS.size_latent_space_top ** 2,
                             "bottom": FLAGS.size_latent_space_bottom ** 2}
        reduction_factor = {"top": FLAGS.size_latent_space_bottom // FLAGS.size_latent_space_top,
                            "bottom": image_size // FLAGS.size_latent_space_bottom}

        vae = VQ_VAE_2 (
                        hidden_channels=FLAGS.hidden_channels,
                        num_emb=num_emb,
                        emb_dim=emb_dim,
                        reduction_factor_bottom=reduction_factor["bottom"],
                        reduction_factor_top=reduction_factor["top"]
                        )
    elif mode == Mode.vae:
        vae = VAE(
            hidden_channels=FLAGS.hidden_channels,
            z_dim=FLAGS.z_dim,
            image_size=image_size,
            reduction_factor=reduction_factor,
            encoding_channels=FLAGS.encodings_channels
            )

    vae.load_state_dict(torch.load(f"{network_dir}/{FLAGS.network_name}.pth"))
    vae.to(device=device)
    vae.eval()

    visualize_latent_space(test_path,
                           FLAGS.csv,
                           mode=mode,
                           batch_size=FLAGS.batch_size,
                           device=device,
                           emb_dim=emb_dim,
                           num_emb=num_emb,
                           vae=vae,
                           network_dir=network_dir,
                           size_latent_space=size_latent_space,
                           max_degree=FLAGS.maxdegree
                           )

