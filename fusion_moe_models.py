import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredSparseMoE(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_experts=4, k=1, sparsity_coef=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.sparsity_coef = sparsity_coef
        
        # Separate projection layers for x (gene) and y (cell/drug/dose)
        self.x_proj = nn.Linear(latent_dim, latent_dim)
        self.y_proj = nn.Linear(embed_dim, embed_dim)
        
        # Factorized experts: Each expert processes x and y separately
        self.experts_x = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(num_experts)
        ])
        self.experts_y = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)
        ])
        
        # Cross-modality gating: Combines x and y to select experts
        self.gate = nn.Linear(latent_dim + embed_dim, num_experts)
        
        # Interaction layer (optional)
        self.interaction = nn.Linear(latent_dim + embed_dim, latent_dim + embed_dim)

    def forward(self, x, y):
        # Project inputs to stabilize training
        x_proj = self.x_proj(x)  # [batch, latent_dim]
        y_proj = self.y_proj(y)  # [batch, embed_dim]
        
        # Cross-modality gating: Combine x and y to select experts
        gate_input = torch.cat([x_proj, y_proj], dim=1)  # [batch, latent_dim + embed_dim]
        gate_logits = self.gate(gate_input)  # [batch, num_experts]
        
        # Add noise during training
        if self.training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 1e-2
        
        # Select top-k experts
        topk_val, topk_indices = torch.topk(gate_logits, self.k, dim=1)  # [batch, k]
        topk_gates = F.softmax(topk_val, dim=1)  # [batch, k]
        
        # Process x and y through selected experts
        batch_size = x.size(0)
        expert_outputs = []
        for i in range(self.num_experts):
            # Process x and y through expert i
            expert_x_out = self.experts_x[i](x_proj)  # [batch, latent_dim]
            expert_y_out = self.experts_y[i](y_proj)  # [batch, embed_dim]
            
            # Combine outputs (e.g., concatenation or interaction)
            combined = torch.cat([expert_x_out, expert_y_out], dim=1)  # [batch, latent_dim + embed_dim]
            expert_outputs.append(self.interaction(combined))  # Optional
            
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, D]
        
        # Gather outputs from selected experts
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.k)
        selected_outputs = expert_outputs[batch_indices, topk_indices]  # [batch, k, D]
        
        # Weighted sum of expert outputs
        fused_output = torch.sum(selected_outputs * topk_gates.unsqueeze(2), dim=1)
        
        # Sparsity loss (L1 on gate weights)
        sparsity_loss = torch.mean(torch.norm(self.gate.weight, dim=1))
        self.auxiliary_loss = self.sparsity_coef * sparsity_loss
        
        return fused_output

    def get_auxiliary_loss(self):
        return self.auxiliary_loss


class SparseMoE(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_experts=4, k=1, sparsity_coef=0.1):
        super(SparseMoE, self).__init__()
        self.input_dim = latent_dim + embed_dim
        self.num_experts = num_experts
        self.k = k  # Sparse MoE typically uses k=1
        self.sparsity_coef = sparsity_coef  # Coefficient for sparsity loss
        
        self.experts = nn.ModuleList([
            nn.Linear(self.input_dim, self.input_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(self.input_dim, num_experts)
        self.noise_epsilon = 1e-2  # Small noise to prevent mode collapse

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)  # [batch, input_dim]
        
        # Compute gating logits + noise for training
        gate_logits = self.gate(inp)  # [batch, num_experts]
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise
        
        # Select top-K experts (default k=1 for sparse MoE)
        topk_val, topk_indices = torch.topk(gate_logits, self.k, dim=1)  # [batch, k]
        topk_gates = F.softmax(topk_val, dim=1)  # [batch, k]

        # Compute outputs for all experts
        all_expert_outputs = torch.stack([expert(inp) for expert in self.experts], dim=1)  # [batch, num_experts, input_dim]
        
        # Gather outputs of selected experts
        batch_size = inp.size(0)
        batch_indices = torch.arange(batch_size, device=inp.device).unsqueeze(1).expand(-1, self.k)
        topk_expert_outputs = all_expert_outputs[batch_indices, topk_indices]  # [batch, k, input_dim]

        # Compute final weighted output
        topk_gates = topk_gates.unsqueeze(2)  # [batch, k, 1]
        fused_output = torch.sum(topk_expert_outputs * topk_gates, dim=1)  # [batch, input_dim]

        # Compute **only sparsity loss** (no balance loss)
        sparsity_loss = self.compute_sparsity_loss()

        # Dynamically update auxiliary loss (only sparsity loss)
        self.auxiliary_loss = self.sparsity_coef * sparsity_loss

        return fused_output

    def compute_sparsity_loss(self):
        """
        Computes sparsity loss: Ensures experts are used selectively and 
        prevents overuse of a single expert.
        """
        gate_weights = self.gate.weight  # [num_experts, input_dim]
        sparsity_loss = torch.mean(torch.norm(gate_weights, dim=1))  # L1 penalty on expert selection
        return sparsity_loss

    def get_auxiliary_loss(self):
        """Returns the dynamically updated auxiliary loss (only sparsity)."""
        return self.auxiliary_loss  # Now only includes sparsity loss

class BalancedMoE(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_experts=4, k=2, sparsity_coef=0.1):
        super(BalancedMoE, self).__init__()
        self.input_dim = latent_dim + embed_dim
        self.num_experts = num_experts
        self.k = k  # Balanced MoE typically uses k=2
        self.sparsity_coef = sparsity_coef
        
        self.experts = nn.ModuleList([
            nn.Linear(self.input_dim, self.input_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(self.input_dim, num_experts)
        self.noise_epsilon = 1e-2
        self.auxiliary_loss = torch.tensor(0.0)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)  # [batch, input_dim]
        
        # Compute gating logits + noise for training
        gate_logits = self.gate(inp)  # [batch, num_experts]
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise
        
        # Select top-K experts (default k=2 for balanced MoE)
        topk_val, topk_indices = torch.topk(gate_logits, self.k, dim=1)  # [batch, k]
        topk_gates = F.softmax(topk_val, dim=1)  # [batch, k]

        # Compute outputs for all experts
        all_expert_outputs = torch.stack([expert(inp) for expert in self.experts], dim=1)  # [batch, num_experts, input_dim]
        
        # Gather outputs of selected experts
        batch_size = inp.size(0)
        batch_indices = torch.arange(batch_size, device=inp.device).unsqueeze(1).expand(-1, self.k)
        topk_expert_outputs = all_expert_outputs[batch_indices, topk_indices]  # [batch, k, input_dim]

        # Compute final weighted output
        topk_gates = topk_gates.unsqueeze(2)  # [batch, k, 1]
        fused_output = torch.sum(topk_expert_outputs * topk_gates, dim=1)  # [batch, input_dim]

        # Compute both balance loss and sparsity loss
        balance_loss = self.compute_balance_loss(inp) 
        sparsity_loss = self.compute_sparsity_loss()

        # Dynamically update auxiliary loss
        self.auxiliary_loss = balance_loss + self.sparsity_coef * sparsity_loss

        return fused_output

    def compute_balance_loss(self, inp):
        """
        Computes balance loss dynamically to ensure experts are used evenly.
        """
        gate_probs = F.softmax(self.gate(inp), dim=1)  # [batch, num_experts]
        importance = torch.sum(gate_probs, dim=0)      # [num_experts]
        one_hot = torch.zeros_like(gate_probs)
        one_hot.scatter_(1, torch.topk(gate_probs, self.k, dim=1)[1], 1)
        load = torch.sum(one_hot, dim=0).float()       # [num_experts]

        # Compute coefficient of variation for importance and load
        importance_mean, importance_std = torch.mean(importance), torch.std(importance)
        load_mean, load_std = torch.mean(load), torch.std(load)
        balance_loss = (importance_std / (importance_mean + 1e-6)) + (load_std / (load_mean + 1e-6))

        return balance_loss  # Now dynamically computed in forward pass

    def compute_sparsity_loss(self):
        """
        Computes sparsity loss: Ensures experts are used selectively and 
        prevents overuse of a single expert.
        """
        gate_weights = self.gate.weight  # [num_experts, input_dim]
        sparsity_loss = torch.mean(torch.norm(gate_weights, dim=1))  # L1 penalty on expert selection
        return sparsity_loss

    def get_auxiliary_loss(self):
        """Returns the dynamically updated auxiliary loss (balance + sparsity)."""
        return self.auxiliary_loss  # Now correctly updated per forward pass
