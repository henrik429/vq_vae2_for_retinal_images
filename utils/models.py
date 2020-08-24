import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from enum import IntEnum
import os
import numpy as np
# from utils.functions import vector_quantization


class Mode (IntEnum):
    vq_vae = 1
    vq_vae_2 = 2
    vae = 3


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


class AbstractEncoder(nn.Module):
    """
    An encoder that cuts the input size down by a factor x.
    """
    def __init__(self, in_channels=256, hidden_channels=256, reduction_factor=4, num_res_blocks=1):
        super().__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels)]

        layers = []
        for r in range(int(np.log(reduction_factor)/np.log(2))):
            if r == 0:
                layers.extend(conv_block(in_channels=in_channels, out_channels=hidden_channels))
            else:
                layers.extend(conv_block(in_channels=hidden_channels, out_channels=hidden_channels))
            for n in range(num_res_blocks):
                layers.append(Residual_Block(hidden_channels))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class AbstractDecoder(nn.Module):
    """
    An decoder that up-samples the input size down by a factor x.
    """
    def __init__(self, hidden_channels=256, upscale_factor=4, num_res_blocks=1):
        super().__init__()

        def conv_block(hidden_channels, kernel_size=4, stride=2, padding=1):
            return [nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(hidden_channels)]

        layers = []
        for r in range(int(np.log(upscale_factor)/np.log(2))):
            for n in range(num_res_blocks):
                layers.append(Residual_Block(hidden_channels))
            layers.extend(conv_block(hidden_channels=hidden_channels))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class Vector_Quantization(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_emb,
                 training=False,
                 ema=False,
                 gamma=0.99,
                 epsilon=1e-5
                 ):
        super().__init__()

        if emb_dim is None:
            emb_dim = {"top": 100, "bottom": 100}
        if num_emb is None:
            num_emb = {"top": 150, "bottom": 300}

        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.embedding = nn.Parameter(torch.Tensor(self.num_emb, self.emb_dim))
        self.embedding.data.uniform_(-1/self.num_emb, 1/self.num_emb)

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
        # (z_e_flatten - emb)^2 = z_e_flatten^2 + emb^2 - 2 * emb * z_e_flatten
        emb_sqr = torch.sum(self.embedding.pow(2), dim=1)
        z_e_sqr = torch.sum(z_e_flatten.pow(2), dim=1, keepdim=True)

        distances = torch.addmm(torch.add(emb_sqr, z_e_sqr), z_e_flatten, self.embedding.t(), alpha=-2.0, beta=1.0)

        # Store indices as integer values
        indices = torch.argmin(distances, dim=1)

        z_q_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q = z_q_flatten.view(z_e.shape)
        # Alternatively, to obtain z_q, z_one_hot can be  multiplied by the embedded weights.
        # z_q = torch.matmul(z_one_hot, emb)

        # Indices as one-hot vectors
        z_one_hot = F.one_hot(indices, num_classes=self.num_emb).float()

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
                 reduction_factor=16, training=False, ema=False, commitment_cost=0.25, gamma=0.99, epsilon=1e-5):
        super().__init__()

        self.encoder = AbstractEncoder(in_channels=3, hidden_channels=hidden_channels, reduction_factor=reduction_factor)
        self.decoder = AbstractDecoder(hidden_channels=hidden_channels, upscale_factor=reduction_factor)

        self.vector_quantization = Vector_Quantization(emb_dim,
                                                       num_emb,
                                                       training=training,
                                                       ema=ema,
                                                       gamma=gamma,
                                                       epsilon=epsilon
                                                       )

        self.conv_to_emb_dim = nn.Conv2d(hidden_channels, emb_dim, kernel_size=1)
        self.conv_from_emb_dim = nn.Conv2d(emb_dim, hidden_channels, kernel_size=1)

        self.last_layer = nn.Conv2d(hidden_channels, 3, kernel_size=1)

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
        z_e = self.encoder(x)

        z_e = self.conv_to_emb_dim(z_e)

        _, z_q, indices, _ = self.vector_quantization(z_e)
        return z_q, indices

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.conv_to_emb_dim(z_e)

        z_e, z_q, indices, z_one_hot = self.vector_quantization(z_e)
        z_q_conv = self.conv_from_emb_dim(z_q.permute(0, 3, 1, 2).contiguous().float())
        reconstruction = torch.sigmoid(self.last_layer(self.decoder(z_q_conv)))

        if self.training:
            # Calculate Losses
            self.reconstruction_loss = F.mse_loss(reconstruction, x)
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
                 reduction_factor_top=8,
                 reduction_factor_bottom=4,
                 training=False,
                 ema=False,
                 commitment_cost=0.25,
                 gamma=0.99,
                 epsilon=1e-5):
        super().__init__()

        self.bottom_encoder = AbstractEncoder(in_channels=3, hidden_channels=hidden_channels, reduction_factor=reduction_factor_bottom)
        self.top_encoder = AbstractEncoder(in_channels=hidden_channels, hidden_channels=hidden_channels, reduction_factor=reduction_factor_top)

        self.top_decoder = AbstractDecoder(hidden_channels=hidden_channels, upscale_factor=reduction_factor_top)
        self.bottom_decoder = AbstractDecoder(hidden_channels=hidden_channels, upscale_factor=reduction_factor_bottom)

        self.conv_to_emb_dim_top = nn.Conv2d(hidden_channels, emb_dim["top"], kernel_size=1)
        self.conv_to_emb_dim_bottom = nn.Conv2d(hidden_channels, emb_dim["bottom"], kernel_size=1)

        self.conv_from_emb_dim_top= nn.Conv2d(emb_dim["top"], hidden_channels, kernel_size=1)
        self.conv_from_emb_dim_bottom = nn.Conv2d(emb_dim["bottom"], hidden_channels, kernel_size=1)

        self.conv_concatenation = nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=1)
        self.last_layer = nn.Conv2d(hidden_channels, 3, kernel_size=1)

        # Initialize for both levels an Embedding Space
        self.num_emb = num_emb
        self.emb_dim = emb_dim

        self.vector_quantization_top = Vector_Quantization(emb_dim["top"], num_emb["top"],
                                                           training=training,
                                                           ema=ema,
                                                           gamma=gamma,
                                                           epsilon=epsilon,
                                                           )
        self.vector_quantization_bottom = Vector_Quantization(emb_dim["bottom"], num_emb["bottom"],
                                                              training=training,
                                                              ema=ema,
                                                              gamma=gamma,
                                                              epsilon=epsilon,
                                                              )

        self.training = training
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
        z_e_bottom = self.bottom_encoder(x)

        z_e_top = self.top_encoder(z_e_bottom)
        z_e_top = self.conv_to_emb_dim_top(z_e_top)

        _, z_q_top, indices_top, _ = self.vector_quantization_top(z_e_top)

        z_e_bottom = self.conv_to_emb_dim_bottom(z_e_bottom)
        _, z_q_bottom, indices_bottom, _ = self.vector_quantization_bottom(z_e_bottom)

        return z_q_bottom, z_q_top, indices_bottom, indices_top

    def forward(self, x):
        z_e_bottom = self.bottom_encoder(x)

        z_e_top = self.top_encoder(z_e_bottom)
        z_e_top = self.conv_to_emb_dim_top(z_e_top)

        z_e_top, z_q_top, indices_top, z_top_one_hot = self.vector_quantization_top(z_e_top)

        z_q_top_conv = self.conv_from_emb_dim_top(z_q_top.permute(0, 3, 1, 2).contiguous())

        # Upsample z_q_top by a factor of 2, quantize and concatenate it with z_e_bottom.
        z_e_top_upsampled = self.top_decoder(z_q_top_conv)
        z_e_bottom = self.conv_to_emb_dim_bottom(z_e_bottom)

        z_e_bottom, z_q_bottom, indices_bottom, z_bottom_one_hot = self.vector_quantization_bottom(z_e_bottom)
        z_q_bottom_conv = self.conv_from_emb_dim_bottom(z_q_bottom.permute(0, 3, 1, 2).contiguous())

        z_q_bottom_conv = torch.cat((z_e_top_upsampled, z_q_bottom_conv), dim=1)
        z_q_bottom_conv = self.conv_concatenation(z_q_bottom_conv)
        reconstruction = torch.sigmoid(self.last_layer(self.bottom_decoder(z_q_bottom_conv)))

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


class VAE(nn.Module):
    def __init__(self, hidden_channels=128,
                 image_size=256,
                 encoding_channels=8,
                 reduction_factor=16,
                 z_dim=32,
                 kl_weight=1,
                 training=False):
        super(VAE, self).__init__()

        self.size_after_conv = (image_size//reduction_factor)
        encoding_size = (self.size_after_conv**2)*encoding_channels

        self.encoding_channels=encoding_channels
        self.encoder = AbstractEncoder(in_channels=3, hidden_channels=hidden_channels, reduction_factor=reduction_factor)
        self.conv_to_enc_channels = nn.Conv2d(hidden_channels, encoding_channels, kernel_size=1)

        self.fc_layer_encoder = nn.Sequential(
            nn.Linear(encoding_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

        self.fc_layer_decoder = nn.Sequential(
            nn.Linear(z_dim, encoding_size),
            nn.BatchNorm1d(encoding_size),
            nn.ReLU()
        )
        self.conv_to_hidden_channels = nn.Conv2d(encoding_channels, hidden_channels, kernel_size=1)

        self.decoder = AbstractDecoder(hidden_channels=hidden_channels, upscale_factor=reduction_factor)
        self.last_layer = nn.Conv2d(hidden_channels, 3, kernel_size=1)

        self.kl_weight = kl_weight
        self.training=training
        if self.training:
            self.total_loss = None
            self.reconstruction_loss = None
            self.kl_loss = None

    def encode(self, x):
        features = self.conv_to_enc_channels(self.encoder(x))
        features = features.view(-1, np.prod(features.shape[1:]))
        features = self.fc_layer_encoder(features)

        mean = self.mean(features)
        logvar = self.logvar(features)
        z = self.reparametrize(mean, logvar)
        return mean, logvar, z

    def forward(self, x):
        mean, logvar, z = self.encode(x)
        reconstruction = self.fc_layer_decoder(z).reshape(z.size(0), self.encoding_channels, self.size_after_conv, self.size_after_conv)
        reconstruction = torch.sigmoid(self.last_layer(self.decoder(self.conv_to_hidden_channels(reconstruction))))

        if self.training:
            # Calculate Losses
            self.reconstruction_loss = F.binary_cross_entropy(reconstruction, x)
            self.kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim = 1), dim = 0)
            self.total_loss = self.reconstruction_loss + self.kl_weight*self.kl_loss

        return reconstruction

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


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
                 z_dim=32,
                 image_size=256,
                 encoding_channels=8,
                 reduction_factor=8,
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

        self.training = training
        self.verbose = verbose
        self.network_name = network_name
        self.network_dir = network_dir
        self.writer = SummaryWriter(f"{self.network_dir}")
        self.report_interval = report_interval
        self.checkpoint_interval = checkpoint_interval
        os.makedirs(f"{self.network_dir}/results", exist_ok=True)
        os.makedirs(f"{self.network_dir}/valid_results/", exist_ok=True)

        self.train_data = data
        self.valid_data = valid
        self.max_epochs = max_epochs
        self.device = device

        self.epoch_id = 0
        self.step_id = 0

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 5e-4}

        self.mode = mode
        if self.mode == Mode.vq_vae:
            self.vae = VQ_VAE(hidden_channels=hidden_channels,
                              num_emb=num_emb,
                              emb_dim=emb_dim,
                              reduction_factor=reduction_factor,
                              training=training,
                              ema=ema,
                              commitment_cost=commitment_cost,
                              gamma=gamma,
                              epsilon=epsilon).to(self.device)

        elif self.mode == Mode.vq_vae_2:
            self.vae = VQ_VAE_2(hidden_channels=hidden_channels,
                                num_emb=num_emb,
                                emb_dim=emb_dim,
                                reduction_factor_bottom=reduction_factor["bottom"],
                                reduction_factor_top=reduction_factor["top"],
                                training=training,
                                ema=ema,
                                commitment_cost=commitment_cost,
                                gamma=gamma,
                                epsilon=epsilon).to(self.device)

        elif self.mode == Mode.vae:
            self.vae = VAE(hidden_channels=hidden_channels,
                           image_size=image_size,
                           encoding_channels=encoding_channels,
                           reduction_factor=reduction_factor,
                           z_dim=z_dim)


        for name, param in self.vae.named_parameters():
            if param.device.type != "cuda":
                print('param {}, not on GPU'.format(name))
            else:
                print('param {}, on GPU'.format(name))

        self.optimizer = optimizer(self.vae.parameters(), **optimizer_kwargs)
        self.num_emb = num_emb

    def step(self, data):
        """Performs a single step of VQ-VAE training.
        Args:
          data: data points used for training."""
        self.optimizer.zero_grad()
        data = data.to(self.device)

        reconstruction = self.vae(data)

        if self.verbose:
            self.writer.add_scalar("reconstruction loss", self.vae.reconstruction_loss, self.step_id)

            if self.mode == Mode.vq_vae or self.mode == Mode.vq_vae_2:
                if not self.vae.ema:
                    self.writer.add_scalar("codebook loss", self.vae.codebook_loss, self.step_id)
                self.writer.add_scalar("commitment loss", self.vae.commmitment_loss, self.step_id)
            else:
                self.writer.add_scalar("kl loss", self.vae.kl_loss, self.step_id)
            self.writer.add_scalar("total loss", self.vae.total_loss, self.step_id)

        if self.step_id % 400 == 0 and data.size(0) > 50:
            self.writer.add_images("target", data[0:8], self.step_id)
            self.writer.add_images("reconstruction", reconstruction[0:8], self.step_id)

        if self.epoch_id > 8 and self.step_id % 400 == 0:
            grid = make_grid(torch.cat((data[0:50:5], reconstruction[0:50:5]), dim=0), nrow=10)
            save_image(grid, f"{self.network_dir}/results/result-{self.epoch_id}-{self.step_id}.png", normalize=True)

        self.vae.total_loss.backward()
        self.optimizer.step()

    def valid_step(self, data):
        """Performs a single step of VAE validation.
        Args:
          data: data points used for validation."""
        with torch.no_grad():
            if self.training:
                data = data.to(self.device)
                reconstruction = self.vae(data)
                self.writer.add_scalar("valid loss", self.vae.total_loss, self.step_id)

                if self.step_id % 400 == 0 and data.size(0) > 50:
                    self.writer.add_images("valid_target", data[0:16], self.step_id)
                    self.writer.add_images("valid_reconstruction", reconstruction[0:16], self.step_id)

                if self.epoch_id > 8 and self.step_id % 400 == 0:
                    grid = make_grid(torch.cat((data[0:50:5], reconstruction[0:50:5]), dim=0), nrow=10)
                    save_image(grid, f"{self.network_dir}/valid_results/valid_result-{self.epoch_id}-{self.step_id}.png",
                               normalize=True)

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
                if self.epoch_id == 0 or (self.vae.total_loss < best_loss):

                    path = f'{self.network_dir}/{self.network_name}.pth'
                    torch.save(self.vae.state_dict(), path)

                if self.valid_data is not None and self.step_id % self.report_interval == 0:
                    try:
                        vdata, = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(self.valid_data)
                        vdata, = next(valid_iter)
                    self.valid_step(vdata)
                self.step_id += 1

        path = f'{self.network_dir}/{self.network_name}.pth'
        torch.save(self.vae.state_dict(), path)

        if self.mode == Mode.vq_vae:
            self.writer.add_embedding(self.vae.vector_quantization.embedding,
                                      metadata = list(range(0, self.num_emb)),
                                      global_step=self.step_id)
        elif self.mode == Mode.vq_vae_2:
            self.writer.add_embedding(self.vae.vector_quantization_bottom.embedding,
                                      metadata = list(range(0, self.num_emb["bottom"])),
                                      global_step=self.step_id)


class classifier(nn.Module):
    def __init__(self, size_flatten_encodings=32*32*8, num_targets=5):
        super(classifier, self).__init__()
        self.fc_layer = nn.Linear(size_flatten_encodings, num_targets)

    def forward(self, x):
        return torch.sigmoid(self.fc_layer(x))


if __name__ == '__main__':
    vae = VAE(hidden_channels=10, image_size=256, encoding_channels=8, reduction_factor=16, z_dim=32)
    print(vae(torch.rand(4, 3, 256, 256)).shape)
    """
    vq_vae = VQ_VAE_2(hidden_channels=10,
                emb_dim={"top":20, "bottom": 50},
                num_emb={"top":100, "bottom": 150})

    a = torch.randn((4, 3, 256, 256))
    reconstruction = vq_vae(a)
    print(reconstruction.shape)
    """
