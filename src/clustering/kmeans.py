"""
K-Means clustering implementation
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

class KMeansClustering:
    """K-Means clustering with automatic K selection"""
    
    def __init__(self, config=None):
        """
        Initialize K-Means clustering
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config if config else {}
        self.model = None
        self.optimal_k = None
        self.labels = None
        
        # Get configuration with defaults
        kmeans_config = self.config.get('clustering', {}).get('kmeans', {})
        self.max_clusters = kmeans_config.get('max_clusters', 10)
        self.random_state = kmeans_config.get('random_state', 42)
        self.n_init = kmeans_config.get('n_init', 10)
        self.n_clusters_range = kmeans_config.get('n_clusters_range', list(range(2, 11)))
    
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
        
        for k in self.n_clusters_range:
            logger.debug(f"Testing K={k}")
            kmeans = KMeans(
                n_clusters=k, 
                random_state=self.random_state,
                n_init=self.n_init
            )
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
                logger.debug(f"  K={k}, Silhouette={score:.4f}")
            else:
                silhouette_scores.append(-1)
                logger.debug(f"  K={k}, Invalid clustering")
        
        # Find best K
        best_idx = np.argmax(silhouette_scores)
        self.optimal_k = self.n_clusters_range[best_idx]
        logger.info(f"Optimal K = {self.optimal_k} (Silhouette: {silhouette_scores[best_idx]:.4f})")
        
        return self.optimal_k
    
    def fit_predict(self, X):
        """
        Fit K-Means and predict clusters
        
        Args:
            X: Feature matrix
            
        Returns:
            dict: Results with labels and cluster information
        """
        logger.info("Fitting K-Means clustering")
        
        # Find optimal K
        self.find_optimal_k(X)
        
        # Train final model
        self.model = KMeans(
            n_clusters=self.optimal_k,
            random_state=self.random_state,
            n_init=self.n_init
        )
        
        self.labels = self.model.fit_predict(X)
        
        # Calculate cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        cluster_sizes = dict(zip(unique.astype(str), counts.tolist()))
        
        results = {
            'labels': self.labels,
            'n_clusters': self.optimal_k,
            'model': self.model,
            'cluster_sizes': cluster_sizes,
            'inertia': self.model.inertia_
        }
        
        logger.info(f"K-Means completed: {self.optimal_k} clusters")
        
        return results
    
    def get_cluster_centers(self):
        """Get cluster centers"""
        if self.model:
            return self.model.cluster_centers_
        return None
    
    def predict(self, X):
        """Predict clusters for new data"""
        if self.model:
            return self.model.predict(X)
        return None