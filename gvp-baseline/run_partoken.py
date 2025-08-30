import os
import argparse
from utils.proteinshake_dataset import create_dataloader, get_dataset
from partoken_model import ParTokenModel 
import torch.nn as nn
from utils.utils import set_seed  
from utils.train import train_pnc_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="enzymecommission",
    choices=["enzymecommission", "proteinfamily", "scope", "geneontology"],
)

parser.add_argument(
    "--split",
    type=str,
    default="structure",
    choices=["random", "sequence", "structure"],
)

parser.add_argument(
    "--split_similarity_threshold",
    type=float,
    default=0.7,
)

parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)

parser.add_argument(
    "--max_clusters", type=int, default=30, help="Maximum number of clusters for PnC partitioner"
)

parser.add_argument(
    "--cluster_size_max", type=int, default=15, help="Maximum nodes per cluster for PnC partitioner"
)

parser.add_argument(
    "--tau_init", type=float, default=1.0, help="Initial temperature for Gumbel-Softmax"
)

parser.add_argument(
    "--tau_min", type=float, default=0.1, help="Minimum temperature for Gumbel-Softmax"
)

parser.add_argument(
    "--tau_decay", type=float, default=0.95, help="Temperature decay rate for Gumbel-Softmax"
)

# Updated arguments to match efficient_pnc parameters
parser.add_argument(
    "--k_hop", type=int, default=2, help="k-hop neighborhood constraint for connectivity"
)

parser.add_argument(
    "--enable_connectivity", action="store_true", default=True, 
    help="Enable k-hop connectivity constraint (default: True)"
)

parser.add_argument(
    "--disable_connectivity", dest="enable_connectivity", action="store_false",
    help="Disable k-hop connectivity constraint"
)

parser.add_argument(
    "--num_gcn_layers", type=int, default=3, help="Number of GCN layers for inter-cluster message passing"
)

parser.add_argument(
    "--use_wandb",
    action="store_true",
    help="Use Weights & Biases for experiment tracking"
)


parser.add_argument(
    "--nhid", type=int, default=50, help="Hidden dimension for partitioner context network"
)

# Add termination threshold parameter
parser.add_argument(
    "--termination_threshold", type=float, default=0.95, help="Early termination threshold"
)

parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size for training"
)

parser.add_argument(
    "--num_workers", type=int, default=8, help="Number of data loading workers"
)

parser.add_argument(
    "--epochs", type=int, default=150, help="Number of training epochs"
)

parser.add_argument(
    "--lr", type=float, default=1e-4, help="Learning rate for optimizer"
)



args = parser.parse_args()

def main():
    dataset_name = args.dataset_name
    split = args.split
    split_similarity_threshold = args.split_similarity_threshold
    seed = args.seed
    max_clusters = args.max_clusters
    tau_init = args.tau_init
    tau_min = args.tau_min
    tau_decay = args.tau_decay
    k_hop = args.k_hop
    enable_connectivity = args.enable_connectivity
    num_gcn_layers = args.num_gcn_layers
    cluster_size_max = args.cluster_size_max
    nhid = args.nhid
    termination_threshold = args.termination_threshold
    use_wandb = args.use_wandb

    set_seed(seed)

    # Get datasets
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=dataset_name,
        split=split,
        split_similarity_threshold=split_similarity_threshold,
        data_dir="./data",
    )

    # # Create efficient PnC model
    # model = GVPGradientSafeHardGumbelModel(
    #     node_in_dim=(6, 3),
    #     node_h_dim=(100, 16),   
    #     edge_in_dim=(32, 1),
    #     edge_h_dim=(32, 1),
    #     num_classes=num_classes,
    #     seq_in=False,
    #     num_layers=3,
    #     drop_rate=0.1,
    #     pooling="sum",
    #     max_clusters=max_clusters,
    #     termination_threshold=termination_threshold  # Add termination threshold
    # )
    
    # # Update partitioner parameters to match command line arguments
    # if hasattr(model, 'partitioner'):
    #     model.partitioner.tau_init = tau_init
    #     model.partitioner.tau_min = tau_min
    #     model.partitioner.tau_decay = tau_decay
    #     model.partitioner.k_hop = k_hop
    #     model.partitioner.cluster_size_max = cluster_size_max
    #     model.partitioner.enable_connectivity = enable_connectivity
    #     model.partitioner.termination_threshold = termination_threshold
        
    #     # Update context network hidden dimension if needed
    #     current_nhid = model.partitioner.context_gru.hidden_size
    #     if nhid != current_nhid:
    #         ns = model.partitioner.selection_mlp[0].in_features - current_nhid  # nfeat
            
    #         # Recreate networks with new hidden dimension
    #         model.partitioner.context_gru = nn.GRU(ns, nhid, batch_first=True)
    #         model.partitioner.context_init = nn.Linear(ns, nhid)
    #         model.partitioner.selection_mlp = nn.Sequential(
    #             nn.Linear(ns + nhid, nhid),
    #             nn.ReLU(),
    #             nn.Dropout(0.1),
    #             nn.Linear(nhid, 1)
    #         )
    #         model.partitioner.size_predictor = nn.Sequential(
    #             nn.Linear(ns + nhid + 1, nhid),  # +1 for max_possible_size
    #             nn.ReLU(),
    #             nn.Dropout(0.1),
    #             nn.Linear(nhid, model.partitioner.cluster_size_max - model.partitioner.cluster_size_min + 1)
    #         )

    # print(f"Dataset: {dataset_name}")
    # print(f"Split: {split}")
    # print(f"Number of classes: {num_classes}")
    # print(f"Model: GVPGradientSafeHardGumbelModel (Efficient PnC)")
    # print(f"PnC max clusters: {max_clusters}")
    # print(f"Cluster size max: {cluster_size_max}")
    # print(f"k-hop constraint: {k_hop} (enabled: {enable_connectivity})")
    # print(f"GCN layers: {num_gcn_layers}")
    # print(f"Hidden dimension: {nhid}")
    # print(f"Termination threshold: {termination_threshold}")
    # print(f"Temperature schedule: init={tau_init}, min={tau_min}, decay={tau_decay}")
    # print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # train_pnc_model(
    #     model,
    #     train_dataset,
    #     val_dataset,
    #     test_dataset,
    #     args,
    #     epochs=150,
    #     lr=1e-4,
    #     batch_size=64,
    #     num_workers=4,
    #     models_dir="./models",
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     use_wandb=use_wandb
    # )


if __name__ == "__main__":
    main()
