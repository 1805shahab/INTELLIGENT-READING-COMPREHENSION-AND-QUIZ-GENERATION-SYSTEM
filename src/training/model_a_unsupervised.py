import os, joblib, numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MaxAbsScaler

BASE = "/content/drive/MyDrive/race_rc_project"
PROCESSED_DIR = os.path.join(BASE, "data/processed")
MODEL_DIR = os.path.join(BASE, "models/model_a/traditional")

def run_unsupervised():
    print("Loading One-Hot Encoded features for Clustering...")
    X_val_ohe = sp.load_npz(os.path.join(PROCESSED_DIR, "X_val_ohe.npz"))
    
    scaler = MaxAbsScaler()
    X_val_s = scaler.fit_transform(X_val_ohe)
    
    # Use a sample of 5000 for the Silhouette score to prevent RAM crashes
    sample_size = min(5000, X_val_s.shape[0])
    X_sample = X_val_s[:sample_size]
    
    print("Training K-Means Clustering (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_sample)
    
    print("Calculating Silhouette Score...")
    sil_score = silhouette_score(X_sample, cluster_labels)
    
    print("-" * 30)
    print(f"✅ Unsupervised Requirement Met!")
    print(f"K-Means Silhouette Score: {sil_score:.4f}")
    print("-" * 30)
    
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    print("Model saved to Drive.")

if __name__ == "__main__":
    run_unsupervised()
