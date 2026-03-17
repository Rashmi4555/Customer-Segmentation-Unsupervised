"""
Hierarchical clustering implementation
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class HierarchicalClustering:
    """Hierarchical clustering with automatic K selection"""
    
    def __init__(self, config=None):
        """
        Initialize Hierarchical clustering
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config if config else {}
        self.model = None
        self.optimal_k = None
        self.labels = None
        
        # Get configuration with defaults
        hier_config = self.config.get('clustering', {}).get('hierarchical', {})
        self.max_clusters = hier_config.get('max_clusters', 10)
        self.linkage = hier_config.get('linkage', 'ward')
        self.affinity = hier_config.get('affinity', 'euclidean')
        # Note: hierarchical.py is looking for 'method' but sklearn uses 'linkage'
        # We'll support both for compatibility
        self.method = hier_config.get('method', self.linkage)
    
    def find_optimal_k(self, X):
        """
        Find optimal number of clusters using silhouette score
        
        Args:
            X: Feature matrix
            
        Returns:
            int: Optimal number of clusters
        """
        logger.info("Finding optimal number of clusters")
        
        silhouette_scores = []
        
        for k in range(2, self.max_clusters + 1):
            logger.debug(f"Testing K={k}")
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=self.linkage
            )
            labels = model.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
                logger.debug(f"  K={k}, Silhouette={score:.4f}")
            else:
                silhouette_scores.append(-1)
                logger.debug(f"  K={k}, Invalid clustering")
        
        # Find best K
        best_idx = np.argmax(silhouette_scores)
        self.optimal_k = range(2, self.max_clusters + 1)[best_idx]
        logger.info(f"Optimal K = {self.optimal_k} (Silhouette: {silhouette_scores[best_idx]:.4f})")
        
        return self.optimal_k
    
    def fit_predict(self, X):
        """
        Fit hierarchical clustering and predict clusters
        
        Args:
            X: Feature matrix
            
        Returns:
            dict: Results with labels and cluster information
        """
        logger.info("Fitting hierarchical clustering")
        
        # Find optimal K
        self.find_optimal_k(X)
        
        # Train final model
        self.model = AgglomerativeClustering(
            n_clusters=self.optimal_k,
            linkage=self.linkage
        )
        
        self.labels = self.model.fit_predict(X)
        
        # Calculate cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        cluster_sizes = dict(zip(unique.astype(str), counts.tolist()))
        
        results = {
            'labels': self.labels,
            'n_clusters': self.optimal_k,
            'model': self.model,
            'cluster_sizes': cluster_sizes
        }
        
        logger.info(f"Hierarchical clustering completed: {self.optimal_k} clusters")
        
        return results
    
    def plot_dendrogram(self, X, max_display=100):
        """
        Plot dendrogram (for visualization)
        
        Args:
            X: Feature matrix
            max_display: Maximum number of samples to display
        """
        # Sample if too large
        if len(X) > max_display:
            indices = np.random.choice(len(X), max_display, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Calculate linkage matrix
        linkage_matrix = linkage(X_sample, method=self.linkage)
        
        # Plot
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, truncate_mode='level', p=5)
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()