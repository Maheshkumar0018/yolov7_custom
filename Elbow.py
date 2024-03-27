import matplotlib.pyplot as plt

# Function to perform K-means clustering and calculate WCSS
def calculate_wcss(data, max_clusters=10):
    wcss = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Load and preprocess frames from the example video
video_path = "example.mp4"  # Change this to the path of your video
original_frames = load_frames_from_video(video_path)

# Extract visual features using ResNet-50
features = []
for frame in original_frames:
    features.append(extract_features(frame, resnet_model))
features = np.array(features)

# Perform PCA for dimensionality reduction
reduced_features = apply_pca(features)

# Calculate WCSS for different number of clusters
wcss_values = calculate_wcss(reduced_features)

# Plot the elbow curve
plt.plot(range(1, len(wcss_values) + 1), wcss_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
