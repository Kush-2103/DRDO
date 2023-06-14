import os
import face_recognition
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import dlib

def get_face_encodings(image):
    face_locations = face_recognition.face_locations(image, model="cnn")
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings

def perform_dbscan_clustering(encodings):
    # Perform face clustering using DBSCAN algorithm
    dbscan_clustering = DBSCAN(metric="euclidean", n_jobs=-1)
    dbscan_clustering.fit(encodings)
    labels = dbscan_clustering.labels_

    # Create a dictionary to store image filenames for each cluster label
    clusters = {}
    for label, filename in zip(labels, filenames):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)

    return clusters

def perform_kmeans_clustering(encodings, num_clusters):
    # Perform face clustering using K-Means algorithm
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    kmeans_clustering.fit(encodings)
    labels = kmeans_clustering.labels_

    # Create a dictionary to store image filenames for each cluster label
    clusters = {}
    for label, filename in zip(labels, filenames):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)

    return clusters

def perform_spectral_clustering(encodings, num_clusters):
    # Perform face clustering using Spectral Clustering algorithm
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
    labels = spectral_clustering.fit_predict(encodings)

    # Create a dictionary to store image filenames for each cluster label
    clusters = {}
    for label, filename in zip(labels, filenames):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)

    return clusters

# Folder containing the images
folder_path = r"C:\Users\91701\Desktop\eyed\Face-Clustering\dataset"

# Get the list of image files in the folder
image_filenames = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize arrays to store face encodings and corresponding image filenames
encodings = []
filenames = []
for filename in image_filenames:
    image = face_recognition.load_image_file(filename)
    face_encodings = get_face_encodings(image)
    if len(face_encodings) > 0:
        encodings.extend(face_encodings)
        filenames.extend([filename] * len(face_encodings))

# Convert the list of face encodings to a numpy array
encodings = np.array(encodings)

# Perform face clustering using DBSCAN
dbscan_clusters = perform_dbscan_clustering(encodings)

# Perform face clustering using K-Means
kmeans_clusters = perform_kmeans_clustering(encodings, num_clusters=5)  # Specify the number of clusters for K-Means

# Perform face clustering using Spectral Clustering
spectral_clusters = perform_spectral_clustering(encodings, num_clusters=5)  # Specify the number of clusters for Spectral Clustering

# Visualize the DBSCAN face clusters
fig, axs = plt.subplots(len(dbscan_clusters), figsize=(8, 8 * len(dbscan_clusters)))
for i, (label, filenames) in enumerate(dbscan_clusters.items()):
    axs[i].set_title(f"DBSCAN Cluster {label}")
    axs[i].axis("off")
    for filename in filenames:
        image = plt.imread(filename)
        axs[i].imshow(image)
plt.tight_layout()
plt.show()

# Visualize the K-Means face clusters
fig, axs = plt.subplots(len(kmeans_clusters), figsize=(8, 8 * len(kmeans_clusters)))
for i, (label, filenames) in enumerate(kmeans_clusters.items()):
    axs[i].set_title(f"K-Means Cluster {label}")
    axs[i].axis("off")
    for filename in filenames:
        image = plt.imread(filename)
        axs[i].imshow(image)
plt.tight_layout()
plt.show()

# Visualize the Spectral Clustering face clusters
fig, axs = plt.subplots(len(spectral_clusters), figsize=(8, 8 * len(spectral_clusters)))
for i, (label, filenames) in enumerate(spectral_clusters.items()):
    axs[i].set_title(f"Spectral Cluster {label}")
    axs[i].axis("off")
    for filename in filenames:
        image = plt.imread(filename)
        axs[i].imshow(image)
plt.tight_layout()
plt.show()
