import os
import face_recognition
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import dlib
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    uploaded_file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)

    # Save the uploaded file
    uploaded_file.save(file_path)

    # Perform face clustering on the uploaded file
    image = face_recognition.load_image_file(file_path)
    face_encodings = get_face_encodings(image)
    if len(face_encodings) > 0:
        encodings = np.array(face_encodings)
        filenames = [file_path] * len(face_encodings)

        # Perform face clustering using DBSCAN
        dbscan_clusters = perform_dbscan_clustering(encodings)

        # Perform face clustering using K-Means
        kmeans_clusters = perform_kmeans_clustering(encodings, num_clusters=5)  # Specify the number of clusters for K-Means

        # Perform face clustering using Spectral Clustering
        spectral_clusters = perform_spectral_clustering(encodings, num_clusters=5)  # Specify the number of clusters for Spectral Clustering

        return render_template('index.html', dbscan_clusters=dbscan_clusters, kmeans_clusters=kmeans_clusters, spectral_clusters=spectral_clusters)

    return render_template('index.html', error_message='No faces found in the uploaded image.')

if __name__ == '__main__':
    app.run(debug=True)
