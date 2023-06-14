from flask import Flask, render_template, request
import os
import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient

app = Flask(__name__, template_folder="templateFiles")

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")
db = client["face_clustering"]
collection = db["image_clusters"]

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        folder_path = request.form.get("folder_path")
        if not os.path.exists(folder_path):
            return render_template("index.html", error="Invalid folder path. Please provide a valid folder path.")
        else:
            encodings, filenames = load_and_process_images(folder_path)
            if len(filenames) == 0:
                return render_template("index.html", error="No valid images found in the folder.")
            else:
                clusters = perform_face_clustering(encodings, filenames)
                save_clusters_to_mongodb(clusters)
                return render_template("clusters.html", clusters=clusters)
    return render_template("index.html")

# Function to load and process images
def load_and_process_images(folder_path):
    image_filenames = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]
    encodings = []
    filenames = []
    for filename in image_filenames:
        image = face_recognition.load_image_file(filename)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            encodings.append(encoding)
            filenames.append(filename)
    encodings = np.array(encodings)
    return encodings, filenames

# Function to perform face clustering
def perform_face_clustering(encodings, filenames):
    clustering = DBSCAN(metric="euclidean", n_jobs=-1)
    clustering.fit(encodings)
    labels = clustering.labels_
    clusters = {}
    for label, filename in zip(labels, filenames):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)
    return clusters

# Function to save clusters to MongoDB
def save_clusters_to_mongodb(clusters):
    collection.delete_many({})  # Clear previous clusters
    for label, filenames in clusters.items():
        cluster_data = {"label": label, "filenames": filenames}
        collection.insert_one(cluster_data)

# Run the Flask application
if __name__ == "__main__":
    app.run()
