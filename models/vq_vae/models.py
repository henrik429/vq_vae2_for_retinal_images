import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from enum import IntEnum

# from utils.functions import vector_quantization


class Mode (IntEnum):
    vq_vae = 1
    vq_vae_2 = 2
    # vq_vae3 = 3


class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, kernel_size=1)
        )

    def forward(self, input):
        return F.relu(input + self.res(input))


class QuarterEncoder(nn.Module):
    """
    An encoder that cuts the input size down by a factor of 4.
    """
    def __init__(self, hidden_channels=256, padding=1):
        super().__init__()

        # Formula of new "Image" Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=4, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
        )

        self.residual = Residual_Block(channels=hidden_channels)
        self.residual2 = Residual_Block(channels=hidden_channels)

    def forward(self, x):
        return self.residual2(self.residual(self.cnn(x)))


class HalfEncoder(nn.Module):
    """
    An encoder that cuts the input size down by a factor of 2.
    """
    def __init__(self, hidden_channels=256, padding=1):
        super().__init__()

        # Formula of new "Image" Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels)
        )

        self.residual = Residual_Block(channels=hidden_channels)
        self.residual2 = Residual_Block(channels=hidden_channels)

    def forward(self, x):
        return self.residual2(self.residual(self.cnn(x)))


class QuarterDecoder(nn.Module):
    """
    An encoder that up-samples the input size by a factor of 4.
    """
    def __init__(self, hidden_channels=256, padding=1):
        super().__init__()

        self.residual = Residual_Block(channels=hidden_channels)
        self.residual2 = Residual_Block(channels=hidden_channels)

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.ConvTranspose2d(hidden_channels, out_channels=hidden_channels, kernel_size=4, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cnn(self.residual2(self.residual(x)))


class HalfDecoder(nn.Module):
    """
    An encoder that up-samples the input size by a factor of 4.
    """
    def __init__(self, hidden_channels=256, padding=1):
        super().__init__()

        self.residual = Residual_Block(channels=hidden_channels)
        self.residual2 = Residual_Block(channels=hidden_channels)

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels)
        )

    def forward(self, x):
        return self.cnn(self.residual2(self.residual(x)))


class Vector_Quantization(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_emb,
                 training=False,
                 ema=False,
                 gamma=0.99,
                 epsilon=1e-5):
        super().__init__()

        if emb_dim is None:
            emb_dim = {"top": 100, "bottom": 100}
        if num_emb is None:
            num_emb = {"top": 150, "bottom": 300}

        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.embedding = nn.Parameter(torch.Tensor(self.num_emb, self.emb_dim))
        self.embedding.data.normal_()
        self.embedding.data.uniform_(-self.num_emb, self.num_emb)

        self.register_buffer("embed", self.embedding)
        self.register_buffer("cluster_size", torch.zeros(self.num_emb))
        self.register_buffer("embed_avg", self.embedding.clone())

        self.gamma = gamma
        self.epsilon = epsilon

        self.ema = ema
        self.trainining = training

    def forward(self, z_e):
        # Dimension of Encoder Output z_e: (B,C,H,W)
        # Flatten z_e to (B*H*W,C)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flatten = z_e.view(-1, self.emb_dim)

        # Compute the distances to the embedded vectors
        # Calculation of distances from z_e_flatten to embeddings e_j using the Euclidean Distance:
        # (z_e_flatten - emb)^2 = z_e_flatten^2 + emb^2 - 2 emb * z_e_flatten
        emb_sqr = torch.sum(self.embedding.pow(2), dim=1)
        z_e_sqr = torch.sum(z_e_flatten.pow(2), dim=1, keepdim=True)

        distances = torch.addmm(torch.add(emb_sqr, z_e_sqr), z_e_flatten, self.embedding.t(), alpha=-2.0, beta=1.0)

        # Store indices as integer values
        indices = torch.argmin(distances, dim=1)

        z_q_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q = z_q_flatten.view(z_e.shape)

        # Indices as one-hot vectors
        z_one_hot = F.one_hot(indices, num_classes=self.num_emb).float()
        """
        To obtain z_q, z_one_hot can be alternatively multiplied by the embedded weights. 
        z_q = torch.matmul(z_one_hot, emb)
        """

        z_q = z_e + (z_q - z_e).detach()

        if self.ema:
            # Use EMA to update the embedding vectors:
            self.exponential_moving_averages(z_one_hot, z_e_flatten)

        return z_e, z_q, indices, z_one_hot

    def exponential_moving_averages(self, z, z_e_flatten):
        embed_onehot_sum = z.sum(0)
        embed_sum = torch.matmul(z_e_flatten.t(), z)

        self.cluster_size.data.mul_(self.gamma).add_(embed_onehot_sum, alpha=1 - self.gamma)

        self.embed_avg.data.mul_(self.gamma).add_(embed_sum.t(), alpha=1 - self.gamma)

        # Laplace smoothing of the cluster size
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_emb * self.epsilon) * n

        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embedding.data.copy_(embed_normalized)


class VQ_VAE(nn.Module):
    def __init__(self, hidden_channels, num_emb, emb_dim,
                 training=False, ema=False, commitment_cost=0.25, gamma=0.99, epsilon=1e-5):
        super().__init__()

        self.enc = QuarterEncoder(hidden_channels=hidden_channels)
        self.dec = QuarterDecoder(hidden_channels=hidden_channels)
        self.vector_quantization = Vector_Quantization(emb_dim,
                                                       num_emb,
                                                       training=training,
                                                       ema=ema,
                                                       gamma=gamma,
                                                       epsilon=epsilon
                                                       )


        self.conv_to_emb_dim = nn.Conv2d(hidden_channels, emb_dim, kernel_size=1)
        self.conv_from_emb_dim = nn.Conv2d(emb_dim, hidden_channels, kernel_size=1)

        self.training = training
        if self.training:
            self.total_loss = None
            self.reconstruction_loss = None
            self.ema = ema
            self.commmitment_loss = None
            self.commitment_cost = commitment_cost
            if not self.ema:
                self.codebook_loss = None

    def encode(self, x):
        """
        Function only used to generate latent space for visualization.
        :param x: Input Image
        :return z: Latent Space
        """
        z_e = self.enc(x)
        z_e = self.conv_to_emb_dim(z_e)

        _, _, indices, _ = self.vector_quantization(z_e)
        return indices

    def forward(self, x):
        z_e = self.enc(x)
        z_e = self.conv_to_emb_dim(z_e)

        z_e, z_q, indices, z_one_hot = self.vector_quantization(z_e)
        z_q_conv = self.conv_from_emb_dim(z_q.permute(0, 3, 1, 2).contiguous().float())

        reconstruction = self.dec(z_q_conv)

        if self.training:
            # Calculate Losses
            self.reconstruction_loss = F.binary_cross_entropy(reconstruction, x)
            self.commmitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_e)
            if self.ema:
                # Use EMA to update the embedding vectors:
                self.total_loss = self.reconstruction_loss + self.commmitment_loss
            else:
                # Otherwise:
                self.codebook_loss = F.mse_loss(z_q, z_e.detach())
                self.total_loss = self.reconstruction_loss + self.codebook_loss + \
                                  self.commitment_cost * self.commmitment_loss

        return reconstruction


class VQ_VAE_2(nn.Module):
    def __init__(self, hidden_channels,
                 emb_dim,
                 num_emb,
                 training=False,
                 ema=False,
                 commitment_cost=0.25,
                 gamma=0.99,
                 epsilon=1e-5):
        super().__init__()

        self.encoders = {
                         "top": HalfEncoder(hidden_channels=hidden_channels),
                         "bottom": QuarterEncoder(hidden_channels=hidden_channels)
        }
        self.decoders = {
                         "top": HalfDecoder(hidden_channels=hidden_channels),
                         "bottom": QuarterDecoder(hidden_channels=hidden_channels)
        }

        self.conv_to_emb_dim_top = nn.Conv2d(hidden_channels, emb_dim["top"], kernel_size=1)
        self.conv_to_emb_dim_bottom = nn.Conv2d(2*hidden_channels, emb_dim["bottom"], kernel_size=1)

        self.conv_from_emb_dim_top= nn.Conv2d(emb_dim["top"], hidden_channels, kernel_size=1)
        self.conv_from_emb_dim_bottom = nn.Conv2d(emb_dim["bottom"], hidden_channels, kernel_size=1)

        # Initialize for both levels an Embedding Space
        self.num_emb = num_emb
        self.emb_dim = emb_dim

        self.emb_top = Vector_Quantization(emb_dim["top"], num_emb["top"],
                                           training=training,
                                           ema=ema,
                                           gamma=gamma,
                                           epsilon=epsilon,
                                           )
        self.emb_bottom = Vector_Quantization(emb_dim["bottom"], num_emb["bottom"],
                                              training=training,
                                              ema=ema,
                                              gamma=gamma,
                                              epsilon=epsilon,
                                              )

        if self.training:
            self.total_loss = None
            self.reconstruction_loss = None
            self.commmitment_loss = None
            self.commitment_cost = commitment_cost
            self.ema = ema
            if not self.ema:
                self.codebook_loss = None

    def encode(self, x):
        """
        Function only used to generate latent space for visualization.
        :param x: Input Image
        :return z: Latent Space
        """
        z_e_bottom = self.encoders["bottom"](x)
        z_e_top = self.encoders["top"](z_e_bottom)
        z_e_top = self.conv_to_emb_dim_top(z_e_top)

        _, z_q_top, indices_top, _ = self.emb_top(z_e_top)

        z_q_top_conv = self.conv_from_emb_dim_top(z_q_top.permute(0, 3, 1, 2).contiguous())

        # Upsample z_q_top by a factor of 2, concatenate it with z_e_bottom and quantize.
        z_e_top_upsampled = self.decoders["top"](z_q_top_conv)
        z_e_bottom = torch.cat((z_e_top_upsampled, z_e_bottom), dim=1)
        z_e_bottom = self.conv_to_emb_dim_bottom(z_e_bottom)

        _, _, indices_bottom, _ = self.emb_bottom(z_e_bottom)

        return indices_top, indices_bottom

    def forward(self, x):
        z_e_bottom = self.encoders["bottom"](x)
        z_e_top = self.encoders["top"](z_e_bottom)
        z_e_top = self.conv_to_emb_dim_top(z_e_top)

        z_e_top, z_q_top, indices_top, z_top_one_hot = self.emb_top(z_e_top)

        z_q_top_conv = self.conv_from_emb_dim_top(z_q_top.permute(0, 3, 1, 2).contiguous())

        # Upsample z_q_top by a factor of 2, concatenate it with z_e_bottom and quantize.
        z_e_top_upsampled = self.decoders["top"](z_q_top_conv)
        z_e_bottom = torch.cat((z_e_top_upsampled, z_e_bottom), dim=1)
        z_e_bottom = self.conv_to_emb_dim_bottom(z_e_bottom)

        z_e_bottom, z_q_bottom, indices_bottom, z_bottom_one_hot = self.emb_bottom(z_e_bottom)

        z_q_bottom_conv = self.conv_from_emb_dim_bottom(z_q_bottom.permute(0, 3, 1, 2).contiguous())
        reconstruction = self.decoders["bottom"](z_q_bottom_conv)

        if self.training:
            # Calculate Losses
            self.reconstruction_loss = F.mse_loss(x, reconstruction)
            commmitment_loss_top = F.mse_loss(z_q_top.detach(), z_e_top)
            commmitment_loss_bottom = F.mse_loss(z_q_bottom.detach(), z_e_bottom)
            self.commmitment_loss = self.commitment_cost * (commmitment_loss_bottom + commmitment_loss_top)

            if self.ema:
                self.total_loss = self.reconstruction_loss + self.commmitment_loss
            else:
                codebook_loss_top = F.mse_loss(z_q_top, z_e_top.detach())
                codebook_loss_bottom = F.mse_loss(z_q_bottom, z_e_bottom.detach())
                self.codebook_loss = codebook_loss_top + codebook_loss_bottom

                self.total_loss = self.reconstruction_loss + self.codebook_loss +  \
                                  self.commitment_cost * self.commmitment_loss
        return reconstruction


class VQ_VAE_Training:
    """
    VQ-VAE setup.
    """
    def __init__(self, data, valid=None,
                 mode=Mode.vq_vae,
                 training=False,
                 hidden_channels=256,
                 num_emb=None,
                 emb_dim=None,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 max_epochs=50,
                 device="cpu",
                 network_dir = "./models/network",
                 network_name = "network",
                 report_interval=10,
                 checkpoint_interval=500,
                 verbose=True,
                 ema=True,
                 commitment_cost=0.25,
                 gamma=0.99,
                 epsilon=1e-5):

        if emb_dim is None:
            emb_dim = {"top": 50, "bottom": 100}
        if num_emb is None:
            num_emb = {"top": 150, "bottom": 300}

        self.training = training
        self.verbose = verbose
        self.network_name = network_name
        self.network_dir = network_dir
        self.writer = SummaryWriter(f"{self.network_dir}")
        self.report_interval = report_interval
        self.checkpoint_interval = checkpoint_interval

        self.train_data = data
        self.valid_data = valid
        self.max_epochs = max_epochs
        self.device = device

        self.epoch_id = 0
        self.step_id = 0

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 5e-4}

        if mode == Mode.vq_vae:
            self.vq_vae = VQ_VAE(hidden_channels=hidden_channels,
                                 num_emb=num_emb,
                                 emb_dim=emb_dim,
                                 device=device,
                                 training=training,
                                 ema=ema,
                                 commitment_cost=commitment_cost,
                                 gamma=gamma,
                                 epsilon=epsilon)

        else:
            self.vq_vae = VQ_VAE_2(hidden_channels=hidden_channels,
                                   num_emb=num_emb,
                                   emb_dim=emb_dim,
                                   training=training,
                                   ema=ema,
                                   commitment_cost=commitment_cost,
                                   gamma=gamma,
                                   epsilon=epsilon)

        self.vq_vae.to(self.device)
        for name, param in self.vq_vae.named_parameters():
            if param.device.type != "cuda":
                print('param {}, not on GPU'.format(name))
            else:
                print('param {}, on GPU'.format(name))

        self.optimizer = optimizer(self.vq_vae.parameters(),  **optimizer_kwargs  )

    def step(self, data):
        """Performs a single step of VQ-VAE training.
        Args:
          data: data points used for training."""

        self.optimizer.zero_grad()
        data = data.to(self.device)

        reconstruction = self.vq_vae(data)

        if self.verbose:
            self.writer.add_scalar("reconstruction loss", self.vq_vae.reconstruction_loss, self.step_id)
            if not self.vq_vae.ema:
                self.writer.add_scalar("codebook loss", self.vq_vae.codebook_loss, self.step_id)
            self.writer.add_scalar("commitment loss", self.vq_vae.commmitment_loss, self.step_id)
        self.writer.add_scalar("total loss", self.vq_vae.total_loss, self.step_id)

        if self.step_id % 100 == 0 and data.size(0) > 50:
            self.writer.add_images("target", data[0:16], self.step_id)
            self.writer.add_images("reconstruction", reconstruction[0:10], self.step_id)

            if self.epoch_id > 4:
                grid = make_grid(torch.cat((data[0:50:5], reconstruction[0:50:5]), dim=0), nrow=5)
                save_image(grid, f"{self.network_dir}/results-{self.epoch_id}-{self.step_id}.png", normalize=True)

        if self.step_id % 80 == 0 and data.size(0) > 50:
            self.writer.add_images("target", data[0:16], self.step_id)
            self.writer.add_images("reconstruction", reconstruction[0:10], self.step_id)

            if self.epoch_id > 4:
                grid = make_grid(torch.cat((data[0:50:5], reconstruction[0:50:5]), dim=0), nrow=5)
                save_image(grid, f"{self.network_dir}/results-{self.epoch_id}-{self.step_id}.png", normalize=True)

        self.vq_vae.total_loss.backward()
        self.optimizer.step()

    def valid_step(self, data):
        """Performs a single step of VAE validation.
        Args:
          data: data points used for validation."""
        with torch.no_grad():
            if self.training:
                data = data.to(self.device)
                _ = self.vq_vae(data)
                self.writer.add_scalar("valid loss", self.vq_vae.total_loss, self.step_id)

    def train(self):
        """Trains a VQ-VAE until the maximum number of epochs is reached."""
        if not self.training:
            raise Exception("Network can not be trained, if argument 'training' is not True!")

        best_loss = -1
        for epoch_id in tqdm(range(self.max_epochs)):
            self.epoch_id = epoch_id
            if self.valid_data is not None:
                valid_iter = iter(self.valid_data)

            for data, in self.train_data:
                self.step(data)
                if self.step_id % self.checkpoint_interval == 0 and self.epoch_id == 0 \
                    or (self.vq_vae.total_loss < best_loss):

                    path = f'{self.network_dir}/{self.network_name}.pth'
                    torch.save(self.vq_vae.state_dict(), path)

                if self.valid_data is not None and self.step_id % self.report_interval == 0:
                    try:
                        vdata, = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(self.valid_data)
                        vdata, = next(valid_iter)
                    self.valid_step(vdata)
                self.step_id += 1


if __name__ == '__main__':
    vq_vae = VQ_VAE_2(hidden_channels=256)

    #for param in vq_vae.parameters():
    #    print(param)
    a = torch.randn((4, 3, 256, 256))
    z, reconstruction = vq_vae(a)
    print(reconstruction.shape)