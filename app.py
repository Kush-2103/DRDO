import os
import face_recognition
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import dlib
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templateFiles", static_folder="staticFiles")
app.config['UPLOAD_FOLDER'] = 'uploads' 

def get_face_encodings(image):
    face_locations = face_recognition.face_locations(image, model="cnn")
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings

def perform_clustering(encodings, algorithm, num_clusters=None):
    if algorithm == 'dbscan':
        clustering_algorithm = DBSCAN(metric="euclidean", n_jobs=-1)
    elif algorithm == 'kmeans':
        clustering_algorithm = KMeans(n_clusters=num_clusters)
    elif algorithm == 'spectral':
        clustering_algorithm = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
    else:
        raise ValueError("Invalid algorithm selected.")

    labels = clustering_algorithm.fit_predict(encodings)

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
    # Get the uploaded files
    uploaded_files = request.files.getlist('file')
    algorithm = request.form.get('algorithm')
    num_clusters = None

    if algorithm == 'kmeans' or algorithm == 'spectral':
        num_clusters = int(request.form.get('num_clusters', 5))

    filenames = []
    encodings = []

    # Save the uploaded files and extract face encodings
    for uploaded_file in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        # Perform face clustering on each uploaded file
        image = face_recognition.load_image_file(file_path)
        face_encodings = get_face_encodings(image)

        if len(face_encodings) > 0:
            encodings.extend(face_encodings)
            filenames.extend([file_path] * len(face_encodings))

    if len(encodings) > 0:
        encodings = np.array(encodings)

        # Perform face clustering using the selected algorithm
        clusters = perform_clustering(encodings, algorithm, num_clusters)

        return render_template('index.html', clusters=clusters, algorithm=algorithm)

    return render_template('index.html', error_message='No faces found in the uploaded images.')

if __name__ == '__main__':
    app.run(debug=True)