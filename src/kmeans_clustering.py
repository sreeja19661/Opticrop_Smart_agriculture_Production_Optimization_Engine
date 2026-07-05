import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/Crop_recommendation.csv")

print("Dataset Loaded Successfully")

# -----------------------------
# Features Only
# -----------------------------
X = df.drop("label", axis=1)

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

print("Calculating WCSS...")

for i in range(1,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# -----------------------------
# Plot Elbow Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.savefig("screenshots/elbow_graph.png")

plt.show()

print("Elbow Graph Saved")

# -----------------------------
# Final KMeans Model
# -----------------------------
optimal_clusters = 4

kmeans = KMeans(
    n_clusters=optimal_clusters,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# -----------------------------
# Add Cluster Column
# -----------------------------
df["Cluster"] = clusters

print(df.head())

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

print("KMeans Model Saved Successfully")
# Save clustered dataset
df.to_csv("dataset/Crop_clustered.csv", index=False)

print("Clustered dataset saved successfully.")
print("\n========== Crops in Each Cluster ==========\n")

for cluster in sorted(df["Cluster"].unique()):
    crops = df[df["Cluster"] == cluster]["label"].unique()

    print(f"Cluster {cluster}")

    for crop in crops:
        print("   -", crop)

    print()
    print("\nCluster Summary")

summary = df.groupby("Cluster")["label"].value_counts()

print(summary)