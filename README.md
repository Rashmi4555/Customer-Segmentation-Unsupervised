## 🎯 Project Title
AI-Driven Customer Intelligence System for Strategic Business Decision Making 

## 📌 Problem Statement
Retail businesses need to understand customer purchasing behaviour to improve marketing strategies, increase revenue and retain customers.

The objective of this project is to segment customers into meaningful groups based on purchasing patterns using unsupervised machine learning techniques.


## 📂 Dataset Description
This project uses the Online Retail dataset containing transactional data of customers.

### Features
- InvoiceNo – Transaction ID
- StockCode – Product Code
- Description – Product Name
- Quantity – Number of items purchased
- InvoiceDate – Purchase date
- UnitPrice – Price per item
- CustomerID – Unique customer identifier
- Country – Customer location


## 🤖 Algorithms Used

### 🔹 KMeans Clustering
- Partitions customers into K clusters
- Fast and efficient
- Best performing algorithm in this project

### 🔹 DBSCAN
- Density-based clustering
- Detects noise and outliers
- No need to define number of clusters

### 🔹 Hierarchical Clustering
- Creates cluster hierarchy
- Useful for dendrogram visualization


## ▶️ How to Run Project
- pip install -r requirements.txt
- python main.py


## 🧠 main.py Functionality

The main script performs:

- Loads dataset  
- Runs preprocessing  
- Performs feature engineering  
- Trains clustering models  

### Prints:
- Silhouette Score  
- Number of clusters  

### Saves:
- Cluster assignments  
- Evaluation metrics  
- Visualization outputs  

---

## 📈 Key Results

- **Number of Clusters Found:** 4  
- **Best Algorithm:** KMeans  
- **Silhouette Score:** ~0.52  

### 💡 Business Insights

- Identified high-value customers for loyalty programs  
- Detected at-risk customers for retention campaigns  
- Found frequent low-spending customers for upselling  
- Enabled targeted marketing strategies  

---


## 📊 Sample Visualizations

### Executive Dashboard  
![Dashboard](https://github.com/Rashmi4555/Customer-Segmentation-Unsupervised/blob/main/Screenshot%202026-03-16%20040738.png?raw=true)

### PCA Cluster Visualization  
![PCA](results/cluster_plots/pca_analysis.png)

### KMeans Clustering  
![KMeans](results/cluster_plots/kmeans_clusters.png)

### Hierarchical Dendrogram  
![Dendrogram](results/cluster_plots/dendrogram.png)

### Algorithm Comparison  
![Comparison](results/cluster_plots/algorithm_comparison.png)
---

## 📁 Project Structure
customer-segmentation-unsupervised/
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       ├── customers_processed.csv
│       └── customer_features.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clustering_models.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_visualization.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluation.py
│   ├── utils.py
│   │
│   └── clustering/
│       ├── kmeans.py
│       ├── dbscan.py
│       ├── hierarchical.py
│       └── gmm.py
│
├── results/
│   ├── cluster_assignments.csv
│   │
│   ├── cluster_plots/
│   │   ├── executive_dashboard.png
│   │   ├── pca_analysis.png
│   │   ├── kmeans_clusters.png
│   │   ├── dendrogram.png
│   │   ├── algorithm_comparison.png
│   │   └── rfm_distributions.png
│   │
│   └── metrics/
│       ├── final_model_selection.json
│       └── segment_profiles.csv
│
├── reports/
│   ├── final_report.pdf
│   └── presentation.pptx
│
├── logs/
│   └── sample_run.log
│
├── config.yaml
├── main.py
├── requirements.txt
└── README.md
