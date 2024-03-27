import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

# Function to extract visual features using ResNet-50
def extract_features(image, model):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        features = model(image)
    
    return features.numpy().flatten()

# Function to perform PCA for dimensionality reduction
def apply_pca(features, n_components=50):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    reduced_features = pca.transform(features)
    return reduced_features

# Function to perform clustering
def perform_clustering(features, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

# Load ResNet-50 model
resnet_model = resnet50(pretrained=True)
# Remove the classification layer
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))

# Assume you have original frames data (replace this with your actual keyframe data)
# Here, we are loading frames from a video as an example
def load_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

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

# Perform clustering
cluster_labels = perform_clustering(reduced_features)

# Print cluster labels
print("Cluster labels:", cluster_labels)

# Function to perform K-means clustering and select nearest frames for video summarization
def select_nearest_frames(original_frames, cluster_labels, num_clusters=10, num_summary_frames=1000):
    centroids = np.array([np.mean(original_frames[cluster_labels == i], axis=0) for i in range(num_clusters)])
    nearest_cluster_indices = pairwise_distances_argmin_min(original_frames, centroids)[0]
    nearest_frames_indices = []
    total_frames_per_cluster = [int(num_summary_frames * np.sum(nearest_cluster_indices == i) / len(nearest_cluster_indices)) for i in range(num_clusters)]
    for cluster_idx in range(num_clusters):
        cluster_indices = np.where(nearest_cluster_indices == cluster_idx)[0]
        distances_to_centroid = pairwise_distances(original_frames[cluster_indices], [centroids[cluster_idx]])
        nearest_frames = cluster_indices[np.argsort(distances_to_centroid.ravel())][:total_frames_per_cluster[cluster_idx]]
        nearest_frames_indices.append(nearest_frames)
    selected_indices = np.concatenate(nearest_frames_indices)
    selected_indices = selected_indices[:num_summary_frames]
    return selected_indices

# Example usage:
# selected_indices = select_nearest_frames(original_frames, cluster_labels, num_clusters=10, num_summary_frames=1000)
