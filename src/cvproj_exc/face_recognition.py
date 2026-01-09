import os
import pickle

import cv2
import numpy as np

from cvproj_exc.config import Config


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=1, max_distance=2.0, min_prob=0.0):
        # TODO: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()

        self.num_neighbours = int(num_neighbours)
        self.max_distance = float(max_distance)
        self.min_prob = float(min_prob)

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality))

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # TODO: Train face identification with a new face with labeled identity.
    def partial_fit(self, face, label):
        # face: aligned face image (224x224x3 BGR)
        # 1) color embedding (BGR)
        emb_color = self.facenet.predict(face).astype(np.float32)

        # 2) grayscale embedding (FaceNet expects 3 channels)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # still 3 channels
        emb_gray = self.facenet.predict(gray3).astype(np.float32)

        # store BOTH embeddings as two training samples with same label
        self.embeddings = np.vstack([self.embeddings, emb_color[None, :], emb_gray[None, :]])
        self.labels.append(str(label))
        self.labels.append(str(label))



    # TODO: Predict the identity for a new face.
    def predict(self, face) -> tuple[str, float, float]:
        """
        Closed-set kNN prediction using TWO query embeddings (color + grayscale).
        - label: majority vote among k nearest neighbors
        - posterior prob: p(Ci|x) = ki / k
        - distance to predicted class: d(Ci|x) = min distance among the ki neighbors of that class
        Distances are fused as: d_j = min(||q_color - e_j||2, ||q_gray - e_j||2)
        Returns: (pred_label, prob, dist_to_pred_class)
        """
        if self.embeddings.shape[0] == 0:
            return ("unknown", 0.0, float("inf"))

        # Query embeddings
        q_color = self.facenet.predict(face).astype(np.float32)  # (128,)

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        q_gray = self.facenet.predict(gray3).astype(np.float32)  # (128,)

        # Distances from BOTH query embeddings to all gallery embeddings
        d_color = np.linalg.norm(self.embeddings - q_color[None, :], axis=1)  # (N,)
        d_gray = np.linalg.norm(self.embeddings - q_gray[None, :], axis=1)  # (N,)

        # Fuse distances (use both embeddings)
        dists = np.minimum(d_color, d_gray)  # (N,)

        k = max(1, min(int(self.num_neighbours), dists.shape[0]))

        # k-NN indices (smallest distances)
        nn_idx = np.argpartition(dists, k - 1)[:k]
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]

        nn_labels = [self.labels[int(i)] for i in nn_idx]

        # Majority vote counts
        counts = {}
        for lab in nn_labels:
            counts[lab] = counts.get(lab, 0) + 1

        # Choose label with highest count; tie-break by smaller min fused distance within that class
        best_label = None
        best_count = -1
        best_class_min_dist = float("inf")

        for lab, cnt in counts.items():
            lab_dists = [float(dists[int(i)]) for i in nn_idx if self.labels[int(i)] == lab]
            lab_min_dist = min(lab_dists) if lab_dists else float("inf")

            if (cnt > best_count) or (cnt == best_count and lab_min_dist < best_class_min_dist):
                best_label = lab
                best_count = cnt
                best_class_min_dist = lab_min_dist

        # b3) posterior probability
        prob = float(best_count) / float(k)

        # b4) distance to predicted class
        dist_to_class = float(best_class_min_dist)

        # Open-set decision rule (c1)
        # tau_d: distance threshold, tau_p: probability threshold
        tau_d = getattr(self, "max_distance", float("inf"))
        tau_p = getattr(self, "min_prob", 0.0)

        if dist_to_class > tau_d or prob < tau_p:
            return ("unknown", prob, dist_to_class)

        return (best_label, prob, dist_to_class)



# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=200):
        # TODO: Prepare FaceNet.

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, FaceNet.embedding_dimensionality))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )

    # TODO
    def partial_fit(self, face):
        return None

    # TODO
    def fit(self):
        return None

    # TODO
    def predict(self, face) -> tuple[int, np.ndarray]:
        return None
