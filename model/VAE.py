import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, dim_code, ch=3, n_feature_map=(32, 64, 128)) -> object:
        super().__init__()

        self.extractor_encoder = nn.Sequential(
            nn.Conv2d(ch, n_feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(n_feature_map[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(n_feature_map[0], n_feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(n_feature_map[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(n_feature_map[1], n_feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(n_feature_map[2]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        self.in_mu = nn.Linear(n_feature_map[2] * 5 * 5, dim_code)
        self.in_logsigma = nn.Linear(n_feature_map[2] * 5 * 5, dim_code)
        self.out_h = nn.Linear(dim_code, n_feature_map[2] * 5 * 5)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (n_feature_map[2], 5, 5)),

            nn.ConvTranspose2d(n_feature_map[2], n_feature_map[1],
                               kernel_size=3,
                               stride=2),
            nn.BatchNorm2d(n_feature_map[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(n_feature_map[1], n_feature_map[0],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(n_feature_map[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(n_feature_map[0], ch,
                               kernel_size=3,
                               stride=2)
        )

    def encode(self, x):
        x = self.extractor_encoder(x)
        mu = self.in_mu(x)
        logsigma = self.in_logsigma(x)

        return mu, logsigma

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            esp = torch.randn(mu.size(), device=mu.device)
            z = mu + esp * torch.exp(logsigma * 0.5)
            return z
        else:
            return mu

    def decode(self, z):
        z = self.out_h(z)
        z = F.relu(z)
        reconstruction = self.decoder(z)
        reconstruction = torch.sigmoid(reconstruction)
        return reconstruction

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.gaussian_sampler(mu, logsigma)
        reconstruction = self.decode(z)

        return mu, logsigma, reconstruction
