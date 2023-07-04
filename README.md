# Face Clustering
This project focuses on clustering various faces according to their features. Face clustering is a computer vision technique that aims to group similar faces together based on their visual features. The goal is to automatically identify and group faces that belong to the same person or share common characteristics, such as facial expressions or poses.

In this project i have used 3 face clustering algorithms

 - **DBSCAN**: It is a density-based algorithm that groups together faces based on their proximity in the feature space. It is particularly useful when the number of clusters is not known in advance and when the clusters have irregular shapes or varying densities.
 - **KMEANS**: is a centroid-based algorithm that aims to minimize the sum of squared distances between the data points and their assigned cluster centroids. It requires specifying the number of clusters in advance and is suitable for cases where the clusters are well-separated and have similar sizes.
 - **Spectral Clustering**: graph-based algorithm that treats face clustering as a graph partitioning problem. It constructs a similarity graph based on pairwise distances between faces and performs clustering on the graph. Spectral Clustering is effective in capturing complex relationships between faces and can handle non-linear separations.

To deploy it, I have used flask web-framework.
Libraries and Dependencies:

-   The project imports several libraries, including `face_recognition`, `sklearn.cluster`, `numpy`, `matplotlib.pyplot`, `dlib`, and `Flask`.
-   These libraries provide functionalities for face detection, face recognition, clustering algorithms, array manipulation, visualization, web development, and more.
# Interface
![DBSCAN](https://github.com/Kush-2103/kush_project1/blob/main/images/db.png)
![Kmeans](https://github.com/Kush-2103/kush_project1/blob/main/images/km.png)

# Results
![enter image description here](https://github.com/Kush-2103/kush_project1/blob/main/images/1.png)
![enter image description here](https://github.com/Kush-2103/kush_project1/blob/main/images/2.png)
![enter image description here](https://github.com/Kush-2103/kush_project1/blob/main/images/3.png)
![enter image description here](https://github.com/Kush-2103/kush_project1/blob/main/images/4.png)
