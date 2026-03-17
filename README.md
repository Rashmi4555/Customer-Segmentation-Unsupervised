# AI-Driven Customer Intelligence System

## 📋 Project Overview
Advanced customer segmentation system using unsupervised learning for Aenexz Tech Pvt Ltd, Bangalore.

## 🎯 Problem Statement
A company has collected large volumes of customer data but does not know:
- Who their most valuable customers are
- Which customers are likely to churn
- Which group spends the most
- Which group responds better to offers

## 📊 Dataset
**Online Retail II Dataset** from UCI Machine Learning Repository
- Period: December 2009 - December 2011
- Total Records: 1,067,371
- Cleaned Records: 805,549
- Unique Customers: 5,878
- Features Engineered: 15+ (RFM, behavioral, derived ratios)

## 🧠 Algorithms Implemented
- ✅ **K-Means Clustering** - 3 clusters, Silhouette: 0.6420
- ✅ **Hierarchical Clustering** - 2 clusters, Silhouette: 0.6550 ★ BEST
- ✅ **DBSCAN** - Density-based clustering
- ✅ **Gaussian Mixture Model (GMM)** - 2 components

## 📈 Key Results
- **Best Algorithm**: Hierarchical Clustering
- **Optimal Clusters**: 2 customer segments
- **Best Silhouette Score**: 0.6550
- **Davies-Bouldin Index**: 0.4467

## 👥 Customer Segments Identified
| Segment | Type | Size | Avg Spend | % Revenue |
|---------|------|------|-----------|-----------|
| 0 | High-Value Loyalists | 3,245 (55.2%) | £1,245.80 | 78.3% |
| 1 | Occasional Shoppers | 2,633 (44.8%) | £342.50 | 21.7% |

## 💼 Business Insights
- **Highest Revenue Segment**: Segment 0 (78.3% of revenue)
- **Premium Marketing Target**: Segment 0 (High-Value Loyalists)
- **Churn Risk**: Segment 1 (89 days avg recency)
- **Retention Needed**: Segment 1 (infrequent buyers)

## 🚀 How to Run
```bash
pip install -r requirements.txt
python main.py --data data/raw/online_retail_II.csv