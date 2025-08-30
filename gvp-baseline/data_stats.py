import json
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from utils.proteinshake_dataset import create_dataloader, get_dataset

def analyze_dataset_statistics(dataset_name="enzymecommission", split="structure", split_similarity_threshold=0.7, data_dir="./data"):
    """
    Analyze statistics of the protein dataset.
    """
    print(f"Analyzing {dataset_name} dataset...")
    print("=" * 60)
    
    # Create dataset-specific directory
    dataset_data_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_data_dir, exist_ok=True)
    
    # Get datasets
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=dataset_name,
        split=split,
        split_similarity_threshold=split_similarity_threshold,
        data_dir=data_dir,
    )
    
    # Combine all datasets for overall statistics
    all_structures = []
    
    # Extract structures from datasets - use data_list instead of structures
    for dataset, name in [(train_dataset, "train"), (val_dataset, "val"), (test_dataset, "test")]:
        print(f"{name.capitalize()} dataset size: {len(dataset)}")
        for i in range(len(dataset)):
            # Get the raw structure data from data_list
            structure = dataset.data_list[i]
            all_structures.append(structure)
    
    print(f"\nTotal number of proteins: {len(all_structures)}")
    print(f"Number of classes: {num_classes}")
    
    # Analyze sequence lengths (number of residues)
    sequence_lengths = []
    coord_lengths = []
    labels = []
    
    for structure in all_structures:
        seq_len = len(structure['seq'])
        coord_len = len(structure['coords'])
        
        sequence_lengths.append(seq_len)
        coord_lengths.append(coord_len)
        labels.append(structure['label'])
    
    # Convert to numpy arrays for easier analysis
    sequence_lengths = np.array(sequence_lengths)
    coord_lengths = np.array(coord_lengths)
    labels = np.array(labels)
    
    # Basic statistics
    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH STATISTICS")
    print("=" * 60)
    print(f"1. Total number of proteins: {len(sequence_lengths)}")
    print(f"2. Maximum number of residues: {np.max(sequence_lengths)}")
    print(f"3. Average number of residues: {np.mean(sequence_lengths):.2f}")
    print(f"4. Median number of residues: {np.median(sequence_lengths):.2f}")
    print(f"5. Standard deviation: {np.std(sequence_lengths):.2f}")
    print(f"6. Minimum number of residues: {np.min(sequence_lengths)}")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(sequence_lengths, p)
        print(f"  {p}th percentile: {value:.1f}")
    
    # Coordinate length statistics (should match sequence length in most cases)
    print("\n" + "=" * 60)
    print("COORDINATE LENGTH STATISTICS")
    print("=" * 60)
    print(f"Maximum coordinate length: {np.max(coord_lengths)}")
    print(f"Average coordinate length: {np.mean(coord_lengths):.2f}")
    print(f"Median coordinate length: {np.median(coord_lengths):.2f}")
    
    # Check if sequence and coordinate lengths match
    length_mismatch = np.sum(sequence_lengths != coord_lengths)
    print(f"Proteins with seq/coord length mismatch: {length_mismatch}")
    
    # Class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    label_counts = Counter(labels)
    print(f"Number of classes: {len(label_counts)}")
    
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(labels)) * 100
        print(f"Class {label}: {count} proteins ({percentage:.2f}%)")
    
    # Length distribution by bins
    print("\n" + "=" * 60)
    print("LENGTH DISTRIBUTION BY BINS")
    print("=" * 60)
    
    # Define bins
    bins = [0, 50, 100, 200, 300, 500, 1000, float('inf')]
    bin_labels = ['0-50', '51-100', '101-200', '201-300', '301-500', '501-1000', '1000+']
    
    for i in range(len(bins)-1):
        if bins[i+1] == float('inf'):
            count = np.sum(sequence_lengths > bins[i])
            print(f"{bin_labels[i]}: {count} proteins ({count/len(sequence_lengths)*100:.2f}%)")
        else:
            count = np.sum((sequence_lengths > bins[i]) & (sequence_lengths <= bins[i+1]))
            print(f"{bin_labels[i]}: {count} proteins ({count/len(sequence_lengths)*100:.2f}%)")
    
    # Save detailed statistics to file in dataset-specific directory
    stats_file = os.path.join(dataset_data_dir, f"{dataset_name}_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Dataset Statistics: {dataset_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Split: {split}\n")
        f.write(f"Split similarity threshold: {split_similarity_threshold}\n")
        f.write(f"Total proteins: {len(all_structures)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Max residues: {np.max(sequence_lengths)}\n")
        f.write(f"Average residues: {np.mean(sequence_lengths):.2f}\n")
        f.write(f"Median residues: {np.median(sequence_lengths):.2f}\n")
        f.write(f"Std deviation: {np.std(sequence_lengths):.2f}\n")
        f.write(f"Min residues: {np.min(sequence_lengths)}\n")
        f.write("\nPercentiles:\n")
        for p in percentiles:
            value = np.percentile(sequence_lengths, p)
            f.write(f"{p}th: {value:.1f}\n")
        f.write("\nClass distribution:\n")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(labels)) * 100
            f.write(f"Class {label}: {count} ({percentage:.2f}%)\n")
    
    print(f"\nDetailed statistics saved to: {stats_file}")
    
    # Create visualization - pass dataset-specific directory
    create_visualizations(sequence_lengths, labels, label_counts, dataset_name, dataset_data_dir)
    
    return {
        'total_proteins': len(all_structures),
        'max_residues': np.max(sequence_lengths),
        'avg_residues': np.mean(sequence_lengths),
        'median_residues': np.median(sequence_lengths),
        'std_residues': np.std(sequence_lengths),
        'min_residues': np.min(sequence_lengths),
        'num_classes': num_classes,
        'class_distribution': dict(label_counts),
        'sequence_lengths': sequence_lengths,
        'labels': labels
    }


def create_visualizations(sequence_lengths, labels, label_counts, dataset_name, dataset_data_dir):
    """
    Create and save visualizations of the dataset statistics.
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name.capitalize()} Dataset Statistics', fontsize=16, fontweight='bold')
    
    # 1. Histogram of sequence lengths
    axes[0, 0].hist(sequence_lengths, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Number of Residues')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Protein Lengths')
    axes[0, 0].axvline(np.mean(sequence_lengths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(sequence_lengths):.1f}')
    axes[0, 0].axvline(np.median(sequence_lengths), color='green', linestyle='--', 
                       label=f'Median: {np.median(sequence_lengths):.1f}')
    axes[0, 0].legend()
    
    # 2. Box plot of sequence lengths
    axes[0, 1].boxplot(sequence_lengths, vert=True)
    axes[0, 1].set_ylabel('Number of Residues')
    axes[0, 1].set_title('Box Plot of Protein Lengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Class distribution bar plot
    classes = list(label_counts.keys())
    counts = list(label_counts.values())
    bars = axes[1, 0].bar(classes, counts)
    axes[1, 0].set_xlabel('Class Label')
    axes[1, 0].set_ylabel('Number of Proteins')
    axes[1, 0].set_title('Class Distribution')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                       f'{count}', ha='center', va='bottom', fontsize=10)
    
    # 4. Log-scale histogram for better visualization of long tail
    axes[1, 1].hist(sequence_lengths, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Number of Residues')
    axes[1, 1].set_ylabel('Frequency (log scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Distribution of Protein Lengths (Log Scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot in dataset-specific directory
    plot_file = os.path.join(dataset_data_dir, f"{dataset_name}_statistics.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    plt.show()


def main():
    """
    Main function to run dataset analysis.
    """
    # Analyze enzyme commission dataset
    stats = analyze_dataset_statistics(
        dataset_name="enzymecommission",
        split="structure",
        split_similarity_threshold=0.7,
        data_dir="./data"
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1. Number of data: {stats['total_proteins']}")
    print(f"2. Max number of residues in a protein: {stats['max_residues']}")
    print(f"3. Average number of residues: {stats['avg_residues']:.2f}")


if __name__ == "__main__":
    main()