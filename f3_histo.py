import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        emb = torch.sin(t.unsqueeze(-1) * torch.exp(-torch.linspace(0, 10, self.dim // 2, device=t.device)))
        emb = torch.cat([emb, torch.cos(emb)], dim=-1)
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        return emb

class TMSSM(nn.Module):
    def __init__(self, dim, rank_dim, num_states):
        super().__init__()
        self.dim = dim
        self.rank_dim = rank_dim
        self.num_states = num_states
        self.linear = nn.Linear(dim, rank_dim + 2 * num_states)
        self.linear_delta = nn.Linear(rank_dim, dim)
        self.A = nn.Parameter(torch.log(torch.randn(dim, num_states)))
        self.D = nn.Parameter(torch.ones(dim))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, u, t_emb):
        batch, seq_len, dim = u.shape
        u_prime = u + t_emb.unsqueeze(1) * self.scale
        
        x_proj = self.linear(u_prime)
        delta, B_prime, C_prime = torch.split(x_proj, [self.rank_dim, self.num_states, self.num_states], dim=-1)
        delta_prime = F.softplus(self.linear_delta(delta))
        
        delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta_prime, self.A))
        delta_B_u = torch.einsum('bln,bld->bldn', B_prime, u_prime) * delta_prime.unsqueeze(-1)
        
        z = torch.zeros(batch, dim, self.num_states, device=u.device)
        v = []
        for i in range(seq_len):
            z = delta_A[:, i] @ z + delta_B_u[:, i]
            v_i = torch.einsum('bdn,bn->bd', z, C_prime[:, i]) * torch.sigmoid(t_emb)
            v.append(v_i)
        
        v = torch.stack(v, dim=1) + u_prime * self.D
        return v

class TMM(nn.Module):
    def __init__(self, dim, num_states=16):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim // 2)
        self.linear2 = nn.Linear(dim, dim // 2)
        self.dep_conv = nn.Conv1d(dim // 2, dim // 2, kernel_size=3, padding=1)
        self.dil_conv = nn.Conv1d(dim // 2, dim // 2, kernel_size=3, padding=2, dilation=2)
        self.tm_ssm = TMSSM(dim // 2, rank_dim=dim // 4, num_states=num_states)
        self.linear_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, t_emb):
        x1 = self.linear1(x)
        x1 = F.silu(self.dep_conv(x1.transpose(1, 2)).transpose(1, 2))
        x1 = self.tm_ssm(x1, t_emb)
        
        x2 = self.linear2(x)
        x2 = F.silu(self.dil_conv(x2.transpose(1, 2)).transpose(1, 2))
        
        x_out = torch.cat([x1, x2], dim=-1)
        x_out = self.norm(self.linear_out(x_out))
        return x_out

class DGFA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.gat1 = GATConv(dim, dim // num_heads, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(dim, dim // num_heads, heads=num_heads, dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.mlp_agg = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(),
            nn.Linear(dim // 2, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, edge_index):
        h = features
        h = F.leaky_relu(self.gat1(h, edge_index))
        h = F.leaky_relu(self.gat2(h, edge_index))
        
        weights = torch.softmax(self.mlp(h).squeeze(-1), dim=0)
        agg_feature = torch.sum(weights.unsqueeze(-1) * h, dim=0)
        agg_feature = self.norm(self.mlp_agg(agg_feature))
        return agg_feature

class ACB(nn.Module):
    def __init__(self, num_classes, feature_dim=2048):
        super().__init__()
        self.num_classes = num_classes
        self.inception = inception_v3(pretrained=True, aux_logits=False).eval()
        self.feature_dim = feature_dim
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.register_buffer('fid_scores', torch.ones(num_classes))

    def compute_fid(self, features, class_idx):
        prototype = self.class_prototypes[class_idx]
        mu1 = torch.mean(features, dim=0)
        mu2 = prototype
        sigma1 = torch.cov(features.t())
        sigma2 = torch.cov(prototype.unsqueeze(0).t())
        
        diff = mu1 - mu2
        covmean = sqrtm((sigma1 @ sigma2).cpu().numpy())
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        covmean = torch.tensor(covmean, device=features.device)
        
        fid = diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def forward(self, features, labels):
        batch_size = features.shape[0]
        weights = torch.ones(batch_size, device=features.device)
        
        for cls in range(self.num_classes):
            cls_mask = (labels == cls)
            if cls_mask.sum() > 1:
                cls_features = features[cls_mask]
                fid = self.compute_fid(cls_features, cls)
                self.fid_scores[cls] = fid
                weights[cls_mask] = 1.0 / (fid + 1e-6)
        
        weights = weights / weights.sum() * batch_size
        return weights

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, 3, padding=1)
        self.dec1 = nn.Conv2d(768, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(384, 128, 3, padding=1)
        self.dec3 = nn.Conv2d(192, 64, 3, padding=1)
        self.dec4 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)
        self.norm4 = nn.BatchNorm2d(512)

    def forward(self, x):
        e1 = F.relu(self.norm1(self.enc1(x)))
        e2 = F.relu(self.norm2(self.enc2(self.pool(e1))))
        e3 = F.relu(self.norm3(self.enc3(self.pool(e2))))
        e4 = F.relu(self.norm4(self.enc4(self.pool(e3))))
        
        d1 = F.relu(self.dec1(torch.cat([self.up(e4), e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([self.up(d1), e2], dim=1)))
        d3 = F.relu(self.dec3(torch.cat([self.up(d2), e1], dim=1)))
        out = self.dec4(d3)
        return out, [e1, e2, e3, e4]

class F3Histo(nn.Module):
    def __init__(self, num_classes, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.unet = UNet()
        self.tmm = TMM(dim=512)
        self.dgfa = DGFA(dim=512)
        self.acb = ACB(num_classes=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.time_embed = TimestepEmbedding(512)
        
        self.beta = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t=None):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        
        t_emb = self.time_embed(t)
        
        # Forward diffusion
        noise = torch.randn_like(x)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        x_t = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        
        # Reconstruction branch
        noise_pred, features = self.unet(x_t)
        
        # Classification branch
        feature_vectors = []
        for f in features:
            f = F.adaptive_avg_pool2d(f, (1, 1)).view(f.shape[0], -1)
            f = self.tmm(f, t_emb)
            feature_vectors.append(f)
        
        # DGFA
        features = torch.stack(feature_vectors, dim=0)  # [K, batch, dim]
        batch_size = features.shape[1]
        edge_index = torch.tensor([[i, i+1] for i in range(len(feature_vectors)-1)], dtype=torch.long, device=x.device).t()
        edge_index = edge_index.repeat(1, batch_size)
        
        agg_features = []
        for i in range(batch_size):
            batch_features = features[:, i, :]
            agg_feature = self.dgfa(batch_features, edge_index)
            agg_features.append(agg_feature)
        
        agg_features = torch.stack(agg_features)
        class_pred = self.classifier(agg_features)
        
        return noise_pred, class_pred, agg_features