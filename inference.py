import torch
import yaml
from utils.proteinshake_dataset import get_dataset, create_dataloader
from bioblob_resume_lightning import BioBlobsTrainingCodebookModule


def get_model():
    checkpoint_path = "./outputs/codebook-resume/enzymecommission/structure/2025-09-19-12-42-17/best-partoken-epoch=00-val_acc=0.567.ckpt"
    model = BioBlobsTrainingCodebookModule.load_from_checkpoint(
        checkpoint_path,
    )

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

def get_data(dataset_name='ec', split='test', split_type='structure'):
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=dataset_name,
        split=split,
        test_mode=False,
    )

    if dataset_name == 'ec':
        from proteinshake.datasets import EnzymeCommissionDataset
        ps_dataset = EnzymeCommissionDataset()
    if dataset_name == 'go':
        from proteinshake.datasets import GeneOntologyDataset
        ps_dataset = GeneOntologyDataset()

    if split == 'train':
        dataset = train_dataset
    elif split == 'val': 
        dataset = train_dataset
    elif split == 'test': 
        dataset = train_dataset
    else:
        raise ValueError("Invalid split")

    loader = create_dataloader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    return loader, ps_dataset
    

def do_inference(model, loader, dataset):
    model.eval()
    
    all_embeddings = []
    all_assignments = []
    all_pdbids = []
    all_resnums = []

    pdbid_to_protein = {prot['protein']['ID']: prot for prot in\
                                               dataset.proteins()}

    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            for attr in ['node_s', 'node_v', 'edge_s', 'edge_v', 'edge_index', 'batch', 'y']:
                if hasattr(batch, attr):
                    setattr(batch, attr, getattr(batch, attr).to(model.device))
            if hasattr(batch, 'seq'):
                batch.seq = batch.seq.to(model.device)

            all_pdbids.extend(list(batch.name))
            for prot_name in batch.name:
                all_resnums.extend(pdbid_to_protein[prot_name]['residue']['residue_number'])

            # Forward pass through the model
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            seq = batch.seq if hasattr(batch, 'seq') and hasattr(model.model, 'sequence_embedding') else None

            # Get cluster embeddings and codebook assignments by calling model.model directly
            # Process through GVP layers to get node features
            h_V_encoded = model.model.node_encoder(h_V)
            h_E_encoded = model.model.edge_encoder(h_E)

            for layer in model.model.gvp_layers:
                h_V_encoded = layer(h_V_encoded, batch.edge_index, h_E_encoded)

            node_features = model.model.output_projection(h_V_encoded)

            # Convert to dense format for partitioning
            from torch_geometric.utils import to_dense_batch
            dense_x, mask = to_dense_batch(node_features, batch.batch)
            dense_index, _ = to_dense_batch(
                torch.arange(node_features.size(0), device=node_features.device), batch.batch
            )

            # Apply partitioner to get cluster features
            cluster_features, assignment_matrix = model.model.partitioner(
                dense_x, None, mask, edge_index=batch.edge_index, batch_vec=batch.batch, dense_index=dense_index
            )

            # Get valid cluster mask
            cluster_valid_mask = (assignment_matrix.sum(dim=1) > 0)

            # Get codebook assignments
            _, code_indices, _, _ = model.model.codebook(cluster_features, mask=cluster_valid_mask)

            # Store results - only keep valid clusters
            for b in range(cluster_features.shape[0]):
                valid_mask_b = cluster_valid_mask[b]
                if valid_mask_b.any():
                    valid_embeddings = cluster_features[b][valid_mask_b]  # [num_valid_clusters, D]
                    valid_indices = code_indices[b][valid_mask_b]         # [num_valid_clusters]

                    all_embeddings.append(valid_embeddings.cpu())
                    all_assignments.append(valid_indices.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [total_valid_clusters, D]
    all_assignments = torch.cat(all_assignments, dim=0)  # [total_valid_clusters]

    print("Cluster embeddings shape:", all_embeddings.shape)
    print("Code assignments shape:", all_assignments.shape)
    print("Number of unique codes used:", len(torch.unique(all_assignments[all_assignments >= 0])))
    return all_embeddings, all_assignments, all_pdbids, all_resnums

if __name__ == "__main__":

    loader, dataset = get_data(split='test')
    model = get_model()
    Z, assignments, all_pdbids, all_resnums = do_inference(model, loader, dataset)

    pass
