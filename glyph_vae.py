
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlyphVAE(nn.Module):
    """
    Conditional VAE for glyph generation.
    Condition = style_vector (32-dim) + char_label (one-hot 36-dim)
    Input/Output = 32x32 grayscale character image
    """
    def __init__(self, latent_dim=128, style_dim=32, n_chars=36):
        super().__init__()
        self.latent_dim = latent_dim
        condition_dim   = style_dim + n_chars  # 32 + 36 = 68

        # ── Encoder ────────────────────────────────────────────────────────
        # self.enc_conv = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 32→16
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16→8
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8→4
        #     nn.ReLU(),
        # )
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16→8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 8→4
            nn.ReLU(),
        )
        # self.enc_flat = nn.Linear(128 * 4 * 4 + condition_dim, 256)
        # self.enc_mu   = nn.Linear(256, latent_dim)
        # self.enc_logvar = nn.Linear(256, latent_dim)
        self.enc_flat   = nn.Linear(256*4*4 + condition_dim, 512)
        self.enc_mu     = nn.Linear(512, latent_dim)
        self.enc_logvar = nn.Linear(512, latent_dim)

        # ── Decoder ────────────────────────────────────────────────────────
        # self.dec_input = nn.Linear(latent_dim + condition_dim, 128 * 4 * 4)
        # self.dec_conv  = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4→8
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 8→16
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 16→32
        #     nn.Tanh()
        # )
        self.dec_input = nn.Linear(latent_dim + condition_dim, 256*4*4)
        self.dec_conv  = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4→8
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8→16
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16→32
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 32→64
                nn.Tanh()
            )

    def encode(self, img, condition):
        feat = self.enc_conv(img).flatten(1)               # [B, 2048]
        feat = torch.cat([feat, condition], dim=1)         # [B, 2048+68]
        feat = F.relu(self.enc_flat(feat))
        return self.enc_mu(feat), self.enc_logvar(feat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = F.relu(self.dec_input(x))
        x = x.view(-1, 256, 4, 4) # 256->128
        return self.dec_conv(x)                            # [B, 1, 32, 32]

    def forward(self, img, style_vec, char_onehot):
        condition = torch.cat([style_vec, char_onehot], dim=1)  # [B, 68]
        mu, logvar = self.encode(img, condition)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, condition)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, style_vec, char_idx, n_chars=36, sigma_mult=1.0, device='cuda'):
        """Sample a glyph at inference time"""
        char_onehot = F.one_hot(
            torch.tensor([char_idx], device=device), n_chars
        ).float()
        condition = torch.cat([style_vec, char_onehot], dim=1)

        # Build per-char mu/logvar from learned bank (set after training)
        mu     = self.char_mu[char_idx].unsqueeze(0).to(device)
        logvar = self.char_logvar[char_idx].unsqueeze(0).to(device)
        std    = torch.exp(0.5 * logvar) * sigma_mult
        z      = mu + std * torch.randn_like(std)

        return self.decode(z, condition)
