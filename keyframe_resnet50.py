import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from torchvision.models import resnet50
from torchvision import transforms

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
    # print('reduced features: ',reduced_features)
    return reduced_features

# Function to perform clustering
def perform_clustering(features, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)
    # print('Cluster labels: ',cluster_labels)
    return cluster_labels

# Function to select keyframes based on a certain criteria
def select_keyframes(cluster_labels, num_keyframes=5):
    # num_keyframes -  number of keyframes to select during the summarization process
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    keyframe_indices = []
    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            # Choose a representative frame from the cluster
            representative_index = cluster_indices[np.argmax(counts[cluster_id])]
            keyframe_indices.append(representative_index)
    
    return keyframe_indices

# Function to summarize video using selected keyframes
def summarize_video(video_path, keyframes):
    summary = []
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('summary.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    
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

def main(video_path):
    # Load ResNet-50 with pretrained weights
    weights_path = 'resnet50-0676ba61.pth'
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load(weights_path))
    # Replace the final fully connected layer with an Identity layer
    model.fc = torch.nn.Identity()
    
    features = []
    cap = cv2.VideoCapture(video_path)
    
    # Count the number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in the video:", total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features.append(extract_features(frame, model))
    cap.release()
    
    features = np.array(features)
    
    # Apply PCA for dimensionality reduction
    reduced_features = apply_pca(features)
    #print("Shape of reduced features after PCA:", reduced_features.shape)
    
    # Perform clustering on reduced feature space
    cluster_labels = perform_clustering(reduced_features)
    #print("Number of unique clusters:", len(np.unique(cluster_labels)))
    
    # Select keyframes based on clustering
    keyframe_indices = select_keyframes(cluster_labels)
    #print("Number of selected keyframes:", len(keyframe_indices))
    
    # Generate video summary using selected keyframes
    summary = summarize_video(video_path, keyframe_indices)
    
    return summary

# Example usage
video_path = './test.mp4'
summary = main(video_path)



# # Load ResNet-50 with pretrained weights
# model = resnet50(pretrained=True)
# model.fc = torch.nn.Identity()
