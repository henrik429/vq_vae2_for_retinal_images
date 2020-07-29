"""
Trigger training here
"""
import torch

from utils.dataloader import Dataloader
from utils.utils import setup
from utils.models import VQ_VAE, VQ_VAE_2, Mode
from utils.introspection import visualize_latent_space


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/models.py", config="./config.json")
    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    network_dir = f'{FLAGS.path_prefix}/models/{FLAGS.network_name}'

    mode = Mode.vq_vae if FLAGS.mode == 1 else Mode.vq_vae_2

    if mode == Mode.vq_vae:
        num_emb=FLAGS.num_emb
        emb_dim=FLAGS.emb_dim
        size_latent_space = FLAGS.size_latent_space ** 2

        vq_vae = VQ_VAE (
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
        #vq_vae.eval()
        #print(vq_vae.vector_quantization.embedding)
        ind = vq_vae.encode(torch.randn((10, 3, 256, 256)).to(device)).cpu().detach().numpy()
        print(ind[0:100000:50])
    else:
        num_emb = {"bottom": FLAGS.num_emb_bottom, "top": FLAGS.num_emb_top}
        emb_dim = {"bottom": FLAGS.emb_dim_bottom, "top": FLAGS.emb_dim_top}
        size_latent_space = {"bottom": FLAGS.size_latent_space_bottom ** 2, "top": FLAGS.size_latent_space_top ** 2}

        vq_vae = VQ_VAE_2 (
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

    # Visualization of the latent space
    test_path = FLAGS.test
    test_data = Dataloader(test_path, batch_size=FLAGS.batch_size)

    visualize_latent_space(test_data,
                           test_path,
                           FLAGS.csv,
                           mode=mode,
                           batch_size=FLAGS.batch_size,
                           device=device,
                           num_emb=num_emb,
                           vq_vae=vq_vae,
                           network_dir=network_dir,
                           network_name=FLAGS.network_name,
                           size_latent_space=size_latent_space,
                           max_degree=FLAGS.maxdegree
                            )

