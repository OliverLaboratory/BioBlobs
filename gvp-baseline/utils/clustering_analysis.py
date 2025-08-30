import torch
import numpy as np



def collect_dataset_clustering_stats(model, dataloader, device='cpu'):
    """
    Collect comprehensive clustering statistics across an entire dataset.
    
    Args:
        model: Trained OptimizedGVPModel
        dataloader: DataLoader containing protein data
        device: Device to run inference on
        
    Returns:
        Dictionary with detailed clustering statistics for histogram plotting
    """
    model.eval()
    model.to(device)
    
    # Collect all cluster sizes across the dataset
    all_cluster_sizes = []
    all_coverage_scores = []
    all_cluster_counts = []
    protein_stats = []
    
    print("Collecting clustering statistics across dataset...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Extract data based on your dataloader format
            # Adjust these lines based on your actual data format
            if isinstance(batch_data, dict):
                h_V = batch_data['node_features']
                edge_index = batch_data['edge_index'] 
                h_E = batch_data['edge_features']
                batch = batch_data.get('batch', None)
                seq = batch_data.get('seq', None)
            else:
                # Assume tuple/list format: (h_V, edge_index, h_E, batch, seq, labels)
                h_V, edge_index, h_E = batch_data[:3]
                batch = batch_data[3] if len(batch_data) > 3 else None
                seq = batch_data[4] if len(batch_data) > 4 else None
            
            # Move to device
            if isinstance(h_V, tuple):
                h_V = (h_V[0].to(device), h_V[1].to(device))
            else:
                h_V = h_V.to(device)
            
            edge_index = edge_index.to(device)
            
            if isinstance(h_E, tuple):
                h_E = (h_E[0].to(device), h_E[1].to(device))
            else:
                h_E = h_E.to(device)
            
            if batch is not None:
                batch = batch.to(device)
            if seq is not None:
                seq = seq.to(device)
            
            # Get clustering statistics
            stats = model.get_clustering_stats(h_V, edge_index, h_E, seq, batch)
            assignment_matrix = stats['assignment_matrix']  # [B, N, S]
            
            # Extract cluster sizes for each protein in the batch
            for b in range(assignment_matrix.size(0)):
                protein_assignment = assignment_matrix[b]  # [N, S]
                cluster_sizes = protein_assignment.sum(dim=0)  # [S] - size of each cluster
                
                # Only keep non-empty clusters
                non_empty_sizes = cluster_sizes[cluster_sizes > 0]
                
                if len(non_empty_sizes) > 0:
                    all_cluster_sizes.extend(non_empty_sizes.cpu().numpy().tolist())
                    
                    # Store per-protein statistics
                    protein_stats.append({
                        'batch_idx': batch_idx,
                        'protein_idx': b,
                        'cluster_sizes': non_empty_sizes.cpu().numpy(),
                        'num_clusters': len(non_empty_sizes),
                        'coverage': stats['avg_coverage'],
                        'avg_cluster_size': non_empty_sizes.float().mean().item(),
                        'max_cluster_size': non_empty_sizes.max().item(),
                        'min_cluster_size': non_empty_sizes.min().item()
                    })
            
            # Collect batch-level statistics
            all_coverage_scores.append(stats['avg_coverage'])
            all_cluster_counts.append(stats['avg_clusters'])
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx} batches...")
    
    # Convert to numpy arrays for easier analysis
    all_cluster_sizes = np.array(all_cluster_sizes)
    all_coverage_scores = np.array(all_coverage_scores)
    all_cluster_counts = np.array(all_cluster_counts)
    
    print(f"Dataset analysis complete! Processed {len(protein_stats)} proteins.")
    
    return {
        'cluster_sizes': all_cluster_sizes,
        'coverage_scores': all_coverage_scores, 
        'cluster_counts': all_cluster_counts,
        'protein_stats': protein_stats,
        'summary': {
            'total_proteins': len(protein_stats),
            'total_clusters': len(all_cluster_sizes),
            'avg_cluster_size': all_cluster_sizes.mean(),
            'std_cluster_size': all_cluster_sizes.std(),
            'min_cluster_size': all_cluster_sizes.min(),
            'max_cluster_size': all_cluster_sizes.max(),
            'cluster_size_percentiles': {
                'p25': np.percentile(all_cluster_sizes, 25),
                'p50': np.percentile(all_cluster_sizes, 50),
                'p75': np.percentile(all_cluster_sizes, 75),
                'p90': np.percentile(all_cluster_sizes, 90),
                'p95': np.percentile(all_cluster_sizes, 95)
            },
            'avg_coverage': all_coverage_scores.mean(),
            'avg_clusters_per_protein': all_cluster_counts.mean()
        }
    }


def plot_clustering_histograms(dataset_stats, save_path=None):
    """
    Create comprehensive histograms of clustering statistics.
    
    Args:
        dataset_stats: Output from collect_dataset_clustering_stats
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cluster Size Distribution
    axes[0, 0].hist(dataset_stats['cluster_sizes'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Cluster Size (number of residues)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Cluster Size Distribution')
    axes[0, 0].axvline(dataset_stats['summary']['avg_cluster_size'], color='red', linestyle='--', 
                       label=f"Mean: {dataset_stats['summary']['avg_cluster_size']:.2f}")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Coverage Distribution
    axes[0, 1].hist(dataset_stats['coverage_scores'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Coverage Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Protein Coverage Distribution')
    axes[0, 1].axvline(dataset_stats['summary']['avg_coverage'], color='red', linestyle='--',
                       label=f"Mean: {dataset_stats['summary']['avg_coverage']:.3f}")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Number of Clusters per Protein
    axes[1, 0].hist(dataset_stats['cluster_counts'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Clusters per Protein Distribution')
    axes[1, 0].axvline(dataset_stats['summary']['avg_clusters_per_protein'], color='red', linestyle='--',
                       label=f"Mean: {dataset_stats['summary']['avg_clusters_per_protein']:.2f}")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cluster Size vs Number of Clusters (scatter plot)
    protein_cluster_counts = [len(p['cluster_sizes']) for p in dataset_stats['protein_stats']]
    protein_avg_sizes = [p['avg_cluster_size'] for p in dataset_stats['protein_stats']]
    
    axes[1, 1].scatter(protein_cluster_counts, protein_avg_sizes, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Number of Clusters per Protein')
    axes[1, 1].set_ylabel('Average Cluster Size')
    axes[1, 1].set_title('Clusters vs Average Size per Protein')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET CLUSTERING SUMMARY")
    print("="*60)
    print(f"Total proteins analyzed: {dataset_stats['summary']['total_proteins']:,}")
    print(f"Total clusters formed: {dataset_stats['summary']['total_clusters']:,}")
    print(f"Average cluster size: {dataset_stats['summary']['avg_cluster_size']:.2f} ± {dataset_stats['summary']['std_cluster_size']:.2f}")
    print(f"Cluster size range: {dataset_stats['summary']['min_cluster_size']} - {dataset_stats['summary']['max_cluster_size']}")
    print(f"Average coverage: {dataset_stats['summary']['avg_coverage']:.3f}")
    print(f"Average clusters per protein: {dataset_stats['summary']['avg_clusters_per_protein']:.2f}")
    print("\nCluster size percentiles:")
    for p, v in dataset_stats['summary']['cluster_size_percentiles'].items():
        print(f"  {p}: {v:.2f}")
    print("="*60)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig







def example_dataset_analysis():
    """
    Example of how to use the dataset clustering analysis functions.
    This creates a mock dataset and demonstrates the analysis workflow.
    """
    print("\n" + "="*60)
    print("DATASET CLUSTERING ANALYSIS EXAMPLE")
    print("="*60)
    
    # Create model
    model = create_optimized_model()
    
    # Create a mock dataset with multiple batches
    class MockDataset:
        def __init__(self, num_batches=10, batch_size=4, max_nodes=50):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.max_nodes = max_nodes
            
        def __len__(self):
            return self.num_batches
            
        def __iter__(self):
            for _ in range(self.num_batches):
                total_nodes = self.batch_size * self.max_nodes
                
                # Create synthetic protein data
                h_V = (
                    torch.randn(total_nodes, 6),
                    torch.randn(total_nodes, 3, 3)
                )
                
                # Create chain-like connectivity
                edge_list = []
                for b in range(self.batch_size):
                    start_idx = b * self.max_nodes
                    for i in range(self.max_nodes - 1):
                        edge_list.extend([
                            [start_idx + i, start_idx + i + 1],
                            [start_idx + i + 1, start_idx + i]
                        ])
                        
                        # Add some long-range connections
                        if i % 5 == 0 and i + 5 < self.max_nodes:
                            edge_list.extend([
                                [start_idx + i, start_idx + i + 5],
                                [start_idx + i + 5, start_idx + i]
                            ])
                
                edge_index = torch.tensor(edge_list).t().contiguous()
                h_E = (
                    torch.randn(edge_index.size(1), 32),
                    torch.randn(edge_index.size(1), 1, 3)
                )
                batch = torch.repeat_interleave(torch.arange(self.batch_size), self.max_nodes)
                
                yield {
                    'node_features': h_V,
                    'edge_index': edge_index,
                    'edge_features': h_E,
                    'batch': batch
                }
    
    # Create mock dataloader
    mock_dataset = MockDataset(num_batches=5, batch_size=3, max_nodes=40)
    
    print("Analyzing mock dataset...")
    
    # Collect statistics across the dataset
    dataset_stats = collect_dataset_clustering_stats(model, mock_dataset)
    
    # Plot histograms
    plot_clustering_histograms(dataset_stats, save_path='clustering_analysis.png')
    
    print("\n✅ Dataset analysis example complete!")
    print("\nTo use with your real dataset:")
    print("1. Create your DataLoader")
    print("2. Call: dataset_stats = collect_dataset_clustering_stats(model, dataloader)")
    print("3. Call: plot_clustering_histograms(dataset_stats)")



if __name__ == "__main__":
    example_dataset_analysis()