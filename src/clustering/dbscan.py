"""
DBSCAN clustering implementation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging
from pathlib import Path

class DBSCANClustering:
    """DBSCAN clustering with automatic parameter tuning"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.best_params = None
        
    def find_optimal_params(self, X):
        """Find optimal epsilon and min_samples parameters"""
        self.logger.info("Finding optimal DBSCAN parameters")
        
        eps_range = self.config['clustering']['dbscan']['eps_range']
        min_samples_range = self.config['clustering']['dbscan']['min_samples_range']
        
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                # Count clusters (excluding noise)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Calculate silhouette score (excluding noise)
                if n_clusters > 1:
                    # Filter out noise points for silhouette calculation
                    non_noise_mask = labels != -1
                    if sum(non_noise_mask) > 1:
                        try:
                            silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                        except:
                            silhouette = -1
                    else:
                        silhouette = -1
                else:
                    silhouette = -1
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': n_noise / len(X),
                    'silhouette': silhouette
                })
        
        results_df = pd.DataFrame(results)
        
        # Filter results with reasonable clusters
        valid_results = results_df[
            (results_df['n_clusters'] >= 2) & 
            (results_df['n_clusters'] <= 10) &
            (results_df['noise_ratio'] < 0.3)  # Less than 30% noise
        ]
        
        if len(valid_results) > 0:
            # Select best based on silhouette score
            best_idx = valid_results['silhouette'].idxmax()
            self.best_params = {
                'eps': valid_results.loc[best_idx, 'eps'],
                'min_samples': int(valid_results.loc[best_idx, 'min_samples'])
            }
        else:
            # Use default parameters
            self.best_params = {'eps': 0.5, 'min_samples': 5}
            self.logger.warning("No valid parameters found. Using defaults.")
        
        self.logger.info(f"Optimal parameters: {self.best_params}")
        
        # Plot k-distance graph for epsilon selection
        self._plot_k_distance(X)
        
        return self.best_params
    
    def _plot_k_distance(self, X, k=4):
        """Plot k-distance graph for epsilon selection"""
        neigh = NearestNeighbors(n_neighbors=k)
        neighbors = neigh.fit(X)
        distances, indices = neighbors.kneighbors(X)
        
        # Sort distances
        k_distances = np.sort(distances[:, k-1], axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.axhline(y=self.best_params['eps'], color='r', linestyle='--', 
                   label=f'Selected eps = {self.best_params["eps"]}')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-distance')
        plt.title('k-Distance Graph for Epsilon Selection')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_dir = Path('results/cluster_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'dbscan_k_distance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def fit_predict(self, X):
        """Fit DBSCAN and predict clusters"""
        self.logger.info("Fitting DBSCAN clustering")
        
        # Find optimal parameters if not already found
        if self.best_params is None:
            self.find_optimal_params(X)
        
        # Train final model
        self.model = DBSCAN(
            eps=self.best_params['eps'],
            min_samples=self.best_params['min_samples']
        )
        
        labels = self.model.fit_predict(X)
        
        # Analyze results
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.logger.info(f"DBSCAN complete. Found {n_clusters} clusters, "
                       f"{n_noise} noise points ({n_noise/len(X)*100:.1f}%)")
        
        return {
            'labels': labels,
            'model': self.model,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'params': self.best_params
        }