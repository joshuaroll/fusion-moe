#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from utils import *
import wandb

# Alternative 1: Attention Mechanism
class Attention(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super(Attention, self).__init__()
        # Attention layer to compute scores
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, z1, feat_embed):
        # Compute attention scores
        # feat_embed: [batch_size, embed_dim]
        attention_scores = F.softmax(self.attention(feat_embed), dim=1)  # [batch_size, 1]

        # Weighted sum of features based on attention scores
        weighted_feat = attention_scores * feat_embed  # [batch_size, embed_dim]

        # Concatenate z1 with weighted_feat
        z1_feat = torch.cat([z1, weighted_feat], dim=1)  # [batch_size, latent_dim + embed_dim]
        return z1_feat

# Alternative 2: Gated Mechanism
class GatedFusion(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, latent_dim + embed_dim),
            nn.Sigmoid()
        )

    def forward(self, z1, feat_embed):
        combined = torch.cat([z1, feat_embed], dim=1)
        gate = self.gate(combined)
        fused = gate * combined  # Element-wise multiplication
        return fused

class BilinearInteraction(nn.Module):
    def __init__(self, latent_dim, embed_dim, output_dim):
        super(BilinearInteraction, self).__init__()
        # Optional projection layer to match dimensions
        self.project_embed = nn.Linear(embed_dim, latent_dim)
        self.bilinear = nn.Bilinear(latent_dim, latent_dim, output_dim)

    def forward(self, z1, feat_embed):
        # Project feat_embed to latent_dim
        feat_embed_proj = self.project_embed(feat_embed)
        # Perform bilinear interaction
        return self.bilinear(z1, feat_embed_proj)

# Alternative 4: Residual Fusion
class ResidualFusion(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super(ResidualFusion, self).__init__()
        self.linear = nn.Linear(latent_dim + embed_dim, latent_dim)

    def forward(self, z1, feat_embed):
        combined = torch.cat([z1, feat_embed], dim=1)
        fused = self.linear(combined)
        return z1 + fused  # Residual connection

# Alternative 5: Mixture of Experts (MoE)
class MixtureOfExperts(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        # Define experts, each outputting latent_dim + embed_dim
        self.experts = nn.ModuleList([nn.Linear(latent_dim + embed_dim, latent_dim + embed_dim) for _ in range(num_experts)])
        
        # Gate to generate weights for each expert
        self.gate = nn.Linear(latent_dim + embed_dim, num_experts)

    def forward(self, z1, feat_embed):
        # Concatenate z1 and feat_embed along the last dimension
        combined = torch.cat([z1, feat_embed], dim=1)  # Shape: [batch_size, latent_dim + embed_dim]
        
        # Compute gate weights
        gate_weights = F.softmax(self.gate(combined), dim=1)  # Shape: [batch_size, num_experts]
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(combined) for expert in self.experts], dim=1)  # Shape: [batch_size, num_experts, latent_dim + embed_dim]
        
        # Fuse expert outputs using gate weights
        fused = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # Shape: [batch_size, latent_dim + embed_dim]
        
        return fused

# Alternative 6: Transformer-Based Fusion
class TransformerFusion(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_heads, num_layers):
        super(TransformerFusion, self).__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim + embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Output projection layer to match z1 + embed_dim
        self.output_layer = nn.Linear(latent_dim + embed_dim, latent_dim + embed_dim)

    def forward(self, z1, feat_embed):
        # Concatenate z1 and feat_embed
        combined = torch.cat([z1, feat_embed], dim=1).unsqueeze(1)  # Shape: [batch_size, 1, latent_dim + embed_dim]
        
        # Pass through transformer encoder
        fused = self.transformer(combined)  # Shape: [batch_size, 1, latent_dim + embed_dim]
        
        # Extract and project the first token to match z1 + embed_dim
        fused = fused.squeeze(1)  # Shape: [batch_size, latent_dim + embed_dim]
        fused = self.output_layer(fused)  # Shape: [batch_size, latent_dim + embed_dim]
        
        return fused
    