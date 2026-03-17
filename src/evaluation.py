"""
Evaluation module for clustering algorithms
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ClusterEvaluator:
    """Evaluates clustering performance using multiple metrics"""
    
    def __init__(self, config=None):
        """
        Initialize ClusterEvaluator with optional configuration
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config if config else {}
        self.metrics_history = {}
        self.results_dir = Path('results/metrics')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, X, labels, proba=None):
        """
        Evaluate clustering performance
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            proba: Cluster probabilities (for GMM)
        
        Returns:
            dict: Evaluation metrics
        """
        # Handle noise points (for DBSCAN)
        if -1 in labels:
            mask = labels != -1
            X_valid = X[mask]
            labels_valid = labels[mask]
            noise_points = (labels == -1).sum()
            noise_percentage = (noise_points / len(labels)) * 100
        else:
            X_valid = X
            labels_valid = labels
            noise_percentage = 0
        
        n_clusters = len(np.unique(labels_valid))
        
        # If only one cluster or invalid, return default values
        if n_clusters < 2 or len(X_valid) < 2:
            return {
                'n_clusters': n_clusters,
                'silhouette': -1,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0,
                'noise_percentage': noise_percentage,
                'cluster_sizes': self._get_cluster_sizes(labels)
            }
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(X_valid, labels_valid)
        except Exception as e:
            print(f"    Warning: Silhouette calculation failed: {e}")
            silhouette = -1
        
        try:
            davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
        except Exception as e:
            print(f"    Warning: Davies-Bouldin calculation failed: {e}")
            davies_bouldin = float('inf')
        
        try:
            calinski = calinski_harabasz_score(X_valid, labels_valid)
        except Exception as e:
            print(f"    Warning: Calinski-Harabasz calculation failed: {e}")
            calinski = 0
        
        # Calculate additional metrics
        cluster_sizes = self._get_cluster_sizes(labels)
        
        # Calculate cluster separation
        separation = self._calculate_cluster_separation(X_valid, labels_valid)
        
        results = {
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'noise_percentage': noise_percentage,
            'cluster_sizes': cluster_sizes,
            'separation': separation
        }
        
        return results
    
    def _get_cluster_sizes(self, labels):
        """Get size of each cluster"""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.astype(str), counts.tolist()))
    
    def _calculate_cluster_separation(self, X, labels):
        """Calculate average distance between cluster centers"""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0
        
        centers = []
        for label in unique_labels:
            mask = labels == label
            centers.append(X[mask].mean(axis=0))
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0
    
    def save_results(self, metrics, algorithm_name):
        """
        Save evaluation results to JSON file
        
        Args:
            metrics: Dictionary of evaluation metrics
            algorithm_name: Name of the algorithm
        """
        # Create a copy to avoid modifying original
        results = metrics.copy()
        
        # Convert numpy types to Python types
        results = self._convert_numpy_types(results)
        
        # Add timestamp
        from datetime import datetime
        results['timestamp'] = datetime.now().isoformat()
        results['algorithm'] = algorithm_name
        
        # Save to file
        filename = self.results_dir / f'{algorithm_name}_metrics.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"    ✅ Evaluation results saved to {filename}")
        
        # Store in history
        self.metrics_history[algorithm_name] = results
        
        return filename
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def visualize_clusters(self, X, labels, algorithm_name):
        """
        Create visualization of clusters
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
            labels: Cluster labels
            algorithm_name: Name of the algorithm
        """
        # Check if X is DataFrame or numpy array
        if hasattr(X, 'values'):  # It's a pandas DataFrame
            X_array = X.values
        else:  # It's already a numpy array
            X_array = X
        
        # Use first two dimensions for visualization
        if X_array.shape[1] >= 2:
            X_vis = X_array[:, :2]
        else:
            print(f"    Warning: Cannot visualize {algorithm_name} - insufficient dimensions")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Handle noise points
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # Noise points
                plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                           c='black', marker='x', s=30, alpha=0.6, label='Noise')
            else:
                plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                           c=[colors[i]], s=50, alpha=0.7, edgecolor='black', linewidth=0.5,
                           label=f'Cluster {label}')
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{algorithm_name} Clustering Results')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path('results/cluster_plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        filename = plots_dir / f'{algorithm_name}_clusters.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Visualization saved to {filename}")
    
    def compare_algorithms(self):
        """
        Compare all evaluated algorithms
        
        Returns:
            DataFrame: Comparison of algorithms with metrics
        """
        if not self.metrics_history:
            print("No algorithms have been evaluated yet")
            return None
        
        comparison = []
        for algo_name, metrics in self.metrics_history.items():
            comparison.append({
                'Algorithm': algo_name,
                'Clusters': metrics['n_clusters'],
                'Silhouette': metrics['silhouette'],
                'Davies-Bouldin': metrics['davies_bouldin'],
                'Calinski-Harabasz': metrics['calinski_harabasz'],
                'Noise %': metrics.get('noise_percentage', 0)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Silhouette', ascending=False).reset_index(drop=True)
        
        # Save comparison
        filename = self.results_dir / 'algorithm_comparison.csv'
        df.to_csv(filename, index=False)
        print(f"✅ Algorithm comparison saved to {filename}")
        
        return df