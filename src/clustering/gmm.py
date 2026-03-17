"""
Gaussian Mixture Model clustering implementation
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import logging
from pathlib import Path

class GMMClustering:
    """Gaussian Mixture Model clustering with BIC/AIC selection"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.best_n_components = None
        
    def find_optimal_components(self, X):
        """Find optimal number of components using BIC and AIC"""
        self.logger.info("Finding optimal number of components for GMM")
        
        n_components_range = self.config['clustering']['gmm']['n_components_range']
        covariance_type = self.config['clustering']['gmm']['covariance_type']
        
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n_components in n_components_range:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                max_iter=self.config['clustering']['gmm']['max_iter'],
                n_init=self.config['clustering']['gmm']['n_init'],
                random_state=42
            )
            
            gmm.fit(X)
            labels = gmm.predict(X)
            
            # Calculate scores
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
            
            if n_components > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = np.nan
            
            silhouette_scores.append(silhouette)
            
            self.logger.info(f"Components={n_components}: "
                           f"BIC={gmm.bic(X):.2f}, AIC={gmm.aic(X):.2f}, "
                           f"Silhouette={silhouette:.3f}")
        
        # Find optimal number of components
        # BIC and AIC: lower is better
        best_bic_idx = np.argmin(bic_scores)
        best_aic_idx = np.argmin(aic_scores)
        
        # Silhouette: higher is better
        valid_silhouette = [s for s in silhouette_scores if not np.isnan(s)]
        if valid_silhouette:
            best_silhouette_idx = np.argmax(valid_silhouette)
        else:
            best_silhouette_idx = best_bic_idx
        
        # Combine decisions (prefer silhouette when available)
        if len(valid_silhouette) > 0:
            self.best_n_components = n_components_range[best_silhouette_idx]
        else:
            self.best_n_components = n_components_range[best_bic_idx]
        
        self.logger.info(f"Optimal components selected: {self.best_n_components}")
        
        # Plot selection curves
        self._plot_selection_curves(n_components_range, bic_scores, 
                                   aic_scores, silhouette_scores)
        
        return self.best_n_components
    
    def _plot_selection_curves(self, n_components_range, bic_scores, 
                              aic_scores, silhouette_scores):
        """Plot BIC, AIC, and silhouette scores"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # BIC plot
        axes[0].plot(n_components_range, bic_scores, 'bo-')
        axes[0].axvline(x=self.best_n_components, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('BIC Score')
        axes[0].set_title('BIC (Lower is better)')
        axes[0].grid(True)
        
        # AIC plot
        axes[1].plot(n_components_range, aic_scores, 'go-')
        axes[1].axvline(x=self.best_n_components, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('AIC Score')
        axes[1].set_title('AIC (Lower is better)')
        axes[1].grid(True)
        
        # Silhouette plot
        axes[2].plot(n_components_range, silhouette_scores, 'ro-')
        axes[2].axvline(x=self.best_n_components, color='r', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Number of Components')
        axes[2].set_ylabel('Silhouette Score')
        axes[2].set_title('Silhouette Score (Higher is better)')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('results/cluster_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'gmm_component_selection.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def fit_predict(self, X):
        """Fit GMM and predict clusters"""
        self.logger.info("Fitting Gaussian Mixture Model")
        
        # Find optimal number of components if not already found
        if self.best_n_components is None:
            self.find_optimal_components(X)
        
        # Train final model
        self.model = GaussianMixture(
            n_components=self.best_n_components,
            covariance_type=self.config['clustering']['gmm']['covariance_type'],
            max_iter=self.config['clustering']['gmm']['max_iter'],
            n_init=self.config['clustering']['gmm']['n_init'],
            random_state=42
        )
        
        labels = self.model.fit_predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Calculate cluster responsibilities
        responsibilities = self.model.predict_proba(X)
        
        self.logger.info(f"GMM complete. Found {self.best_n_components} components")
        
        return {
            'labels': labels,
            'model': self.model,
            'proba': probabilities,
            'responsibilities': responsibilities,
            'n_components': self.best_n_components,
            'bic': self.model.bic(X),
            'aic': self.model.aic(X)
        }