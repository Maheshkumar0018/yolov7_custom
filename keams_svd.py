import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Function to extract features using SVD
def extract_features(images, num_components=50):
    features = []
    for image in images:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate histogram of grayscale image
        hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])
        hist = hist.flatten()
        # Extract the most significant components
        components = hist[:num_components]
        features.append(components)
    
    return np.array(features)

# Function to perform KMeans clustering
def perform_clustering(features, num_clusters):
    # Handle NaN values in features
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    features = imputer.fit_transform(features)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform KMeans clustering
    clustering = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = clustering.fit_predict(scaled_features)
    
    return cluster_labels

# Function to select keyframes based on clustering
def select_keyframes(cluster_labels, num_keyframes_per_cluster=2):
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    total_frames = len(cluster_labels)
    print("Total number of clusters:", len(unique_clusters))
    print("Total number of frames:", total_frames)
    
    keyframe_indices = []
    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            # Choose the top num_keyframes_per_cluster frames from the cluster
            selected_indices = cluster_indices[:num_keyframes_per_cluster]
            keyframe_indices.extend(selected_indices)
            print(f"Cluster {cluster_id}: {len(selected_indices)} frame(s) selected - {selected_indices}")
    
    return keyframe_indices, unique_clusters

# Function to summarize video using selected keyframes
def summarize_video(video_path, keyframes):
    summary = []
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('summary_kmeans.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index in keyframes:
            summary.append(frame)
            out.write(frame)
        frame_index += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return summary

# Main function for video summarization
def video_summarization(video_path, num_keyframes=10, num_clusters=10, num_keyframes_per_cluster=2):
    # Load video and extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Extract features using SVD
    features = extract_features(frames)
    
    # Perform KMeans clustering
    cluster_labels = perform_clustering(features, num_clusters)
    
    # Select keyframes based on clustering
    keyframe_indices, unique_clusters = select_keyframes(cluster_labels, num_keyframes_per_cluster)

    for cluster_id in range(len(unique_clusters)):
        indices = keyframe_indices[cluster_id * num_keyframes_per_cluster: (cluster_id + 1) * num_keyframes_per_cluster]
        num_frames = len(indices)
        print(f"Cluster {cluster_id}: {num_frames} frame(s) selected - {indices}")

    
    # Generate video summary using selected keyframes
    summary = summarize_video(video_path, keyframe_indices)
    
    return summary

# Example usage
video_path = './test.mp4'
summary = video_summarization(video_path, num_keyframes=2, num_clusters=10, num_keyframes_per_cluster=2)
