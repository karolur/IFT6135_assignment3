import math
import torch
from torch import nn

### WARNING: DO NOT EDIT THE CLASS NAME, INITIALIZER, AND GIVEN INPUTS AND ATTRIBUTES. OTHERWISE, YOUR TEST CASES CAN FAIL. ###


class ConvVAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128, base_channels=64):
        super().__init__()
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.encoder = self._build_encoder(base_channels, image_channels)
        feature_dim = base_channels * 4
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.latent_to_features = nn.Linear(latent_dim, feature_dim)
        self.decoder = self._build_decoder(base_channels, image_channels)

    def _encoder_block(self, in_channels, out_channels, stride, kernel_size, use_batchnorm):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1 if stride > 1 else 0,
                bias=not use_batchnorm,
            ),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        block = nn.Sequential(*layers)
        return block

    def _decoder_block(self, in_channels, out_channels, stride, kernel_size, use_batchnorm):
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1 if stride > 1 else 0,
                bias=not use_batchnorm,
            ),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        block = nn.Sequential(*layers)
        return block

    def _build_encoder(self, base_channels, image_channels):
        encoder = nn.Sequential(
            self._encoder_block(image_channels, base_channels, stride=2, kernel_size=4, use_batchnorm=True), # 3 channels to 64 filters. res 16x16
            self._encoder_block(base_channels, base_channels * 2, stride=2, kernel_size=4, use_batchnorm=True), # 64 filters to 2*64. res 8x8
            self._encoder_block(base_channels * 2, base_channels * 4, stride=2, kernel_size=4, use_batchnorm=True), # 64*2 to 64*4. res 4x4
            self._encoder_block(base_channels * 4, base_channels * 4, stride=1, kernel_size=4, use_batchnorm=True), # 64*4. res 4x4
        )
        return encoder

    def _build_decoder(self, base_channels, image_channels):
        decoder = nn.Sequential(
            self._decoder_block(base_channels * 4, base_channels * 4, stride=1, kernel_size=4, use_batchnorm=True),
            self._decoder_block(base_channels * 4, base_channels * 2, stride=2, kernel_size=4, use_batchnorm=True),
            self._decoder_block(base_channels * 2, base_channels, stride=2, kernel_size=4, use_batchnorm=True),
            nn.ConvTranspose2d(base_channels, image_channels * 2, kernel_size=4, stride=2, padding=1),
        )
        return decoder

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h) # mean vector
        logvar = self.fc_logvar(h) # log-variance vector
        # both mu and logvar make up the diagonal gaussian q(z|h)
        return mu, logvar

    def reparameterize(self, mu, logvar, sample=True):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.latent_to_features(z)
        h = h.view(z.size(0), -1, 1, 1)
        recon = self.decoder(h)
        mean, logvar = recon.chunk(2, dim=1)
        return mean, logvar

    def _gaussian_nll(self, x, mean, logvar):
        """
        nll = negative log likelihood
        equation from https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        but added the pi term in the end bc of autograder
        """
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)  # clamp logvar between -10 and 10 for stability
        var = torch.exp(logvar_clamped)  # exponentiate the clamped log-variance
        log_term = math.log(2 * math.pi) + logvar_clamped  # compute log-term for Gaussian NLL 
        squared_error = (x - mean) ** 2  # squared reconstruction error
        nll = 0.5 * (log_term + squared_error / var) # negative log likelihood
        return nll.flatten(start_dim=1).sum(dim=1) # [batch_size] because sum over dim 1

    def _kl_divergence(self, mu, logvar):
        """
        KL divergence between q(z|x) = N(mu, diag(exp(logvar)))
        and p(z) = N(0, I)
        I'm using the pytorch logic by omiting the pi terms 
        KL(q || p) = -0.5 * sum_i(1 + logvar_i - mu_i^2 - exp(logvar_i))

        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # compute KL divergence between diagonal Gaussian and standard normal distribution
        return kl # [batch_size] because sum over dim 1

    def forward(self, batch):
        x = batch["images"] # [batch_size, 3, 32, 32]
        mu, logvar = self.encode(x)  # encode the input
        z = self.reparameterize(mu, logvar)  # sample from the latent distribution
        recon_mu, recon_logvar = self.decode(z)  # decode the latent sample
        recon_loss = self._gaussian_nll(x, recon_mu, recon_logvar)  # compute reconstruction loss (per sample)
        kl = self._kl_divergence(mu, logvar)  # compute KL divergence term (per sample)
        loss = (recon_loss + kl).mean() ## mean is used to aggregate the total loss across the batch
        return {
            "loss": loss,
            "reconstruction_mean": recon_mu,
            "reconstruction_logvar": recon_logvar,
            "latent_mu": mu,
            "latent_logvar": logvar,
            "latent_sample": z,
            "kl": kl,
            "reconstruction_loss": recon_loss,
        }
