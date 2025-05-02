# Fusion-MoE: Multimodal Data Fusion with Mixture of Experts

This repository contains code for combining multimodal data using the Mixture of Experts (MoE) approach within the MultiDCP framework. It has beenn tested in multiple other frameworks for generalized use as well. The primary script, `fusion_moe_models.py`, implements models and utilities for fusing data from multiple modalities to improve predictive performance. It is meant to be integrated into multimodal machinne learning frameworks.

## Features
- Implements Mixture of Experts for multimodal data fusion.
- Supports flexible integration of different data modalities.
- Designed for optimal use within the MultiDCP framework currently.

## File Overview
### `fusion_moe_models.py`
This file contains the core implementation of the Fusion-MoE models. It includes:
- Model definitions for multimodal data fusion.
- Training and evaluation utilities.
- Customizable configurations for different datasets and tasks.

### `baseline_fusion_models.py`
This file contains alternative generic fusion mechanisms for comparison purposes.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/joshuaroll/fusion-moe.git
   cd fusion-moe
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your multimodal dataset.
2. Configure the model parameters in `fusion_moe_models.py`.
3. Run the script to train and evaluate the model as you would normally.

## Example for MultiDCP Framework
```python
parser.add_argument('--fusion_type', type=str, default='concat', 
                        help='Type of fusion layer to use: concat, attention, gated, bilinear, residual, moe, transformer')
    
# ...

# Dynamically add auxiliary loss based on fusion type
if hasattr(model.multidcp.fusion_layer, 'get_auxiliary_loss'):
    aux_loss = model.multidcp.fusion_layer.get_auxiliary_loss()
    
    if model.fusion_type == "sparse_moe":
        loss_t += model.multidcp.fusion_layer.sparsity_coef * aux_loss  # Sparsity loss only
    elif model.fusion_type == "structured_moe":
        loss_t += model.multidcp.fusion_layer.sparsity_coef * aux_loss  # Sparsity loss only
    elif model.fusion_type == "balanced_moe":
        loss_t += model.multidcp.moe_balance_coef * aux_loss  # Balance + sparsity loss
    
    # Log comparative metrics for auxiliary loss
    wandb.log({'MoE Auxiliary Loss': aux_loss.item()}, step=epoch)
```

## MultiDCP Model Initialization
```python
def __init__(self, device, model_param_registry):
        super(MultiDCP, self).__init__()
        self.set_attr_from_dict(model_param_registry)
        assert self.drug_emb_dim == self.gene_emb_dim, 'Embedding size mismatch'
        self.drug_fp = NeuralFingerprint(self.drug_input_dim['atom'], self.drug_input_dim['bond'], 
                                         self.conv_size, self.drug_emb_dim,
                                         self.degree, device)
        self.gene_embed = nn.Linear(self.gene_input_dim, self.gene_emb_dim)
        self.drug_gene_attn = DrugGeneAttention(self.gene_emb_dim, self.gene_emb_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=self.dropout, device=device)
        self.cell_id_emb_dim = 50
        
        # Define fusion layer
        self.fusion_type = model_param_registry.get('fusion_type', 'concat')
        print(f"Using fusion type: {self.fusion_type}")
        latent_dim = self.gene_emb_dim + self.drug_emb_dim
        embed_dim = self.cell_id_emb_dim
        
        if self.fusion_type == "attention":
            self.fusion_layer = Attention(latent_dim, embed_dim)
        elif self.fusion_type == "gated":
            self.fusion_layer = GatedFusion(latent_dim, embed_dim)
        elif self.fusion_type == "bilinear":
            self.fusion_layer = BilinearInteraction(latent_dim, embed_dim, output_dim=latent_dim + embed_dim)
        elif self.fusion_type == "residual":
            self.fusion_layer = ResidualFusion(latent_dim, embed_dim)
        elif self.fusion_type == "moe":
            self.fusion_layer = MixtureOfExperts(latent_dim, embed_dim, num_experts=3)
        elif self.fusion_type == "transformer":
            self.fusion_layer = TransformerFusion(latent_dim, embed_dim, num_heads=4, num_layers=2)
        elif self.fusion_type == "default":
            self.fusion_layer = nn.Sequential(
                nn.Linear(latent_dim + embed_dim, latent_dim),
                nn.ReLU()
            )
        elif self.fusion_type == "sparse_moe":
            self.fusion_layer = SparseTopKMoE(latent_dim, embed_dim, num_experts=4)
        elif self.fusion_type == "balanced_moe":
            self.fusion_layer = BalancedMoE(latent_dim, embed_dim, num_experts=4)
        elif self.fusion_type == "lora_moe":
            self.fusion_layer = LoRAMoE(latent_dim, embed_dim, num_experts=4)
        elif self.fusion_type == "expert_moe":
            self.fusion_layer = ExpertChoiceMoE(latent_dim, embed_dim, num_experts=4)
        else:
            self.fusion_layer = None  # Default to concatenation
        
        if self.linear_encoder_flag:
            self.encoder = LinearEncoder(self.cell_id_input_dim)
        else:
            self.encoder = TransformerEncoder(self.cell_id_input_dim)

        self.pert_idose_embed = nn.Linear(self.pert_idose_input_dim, self.pert_idose_emb_dim)
        self.linear_dim = self.drug_emb_dim + self.gene_emb_dim + self.cell_id_emb_dim + self.pert_idose_emb_dim

        self.linear_1 = nn.Linear(self.linear_dim, self.hid_dim)
        self.relu = nn.ReLU()
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
