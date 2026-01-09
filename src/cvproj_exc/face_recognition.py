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
        emb = self.facenet.predict(face).astype(np.float32)  # (128,)
        self.embeddings = np.vstack([self.embeddings, emb[None, :]])
        self.labels.append(str(label))
        return None

    # TODO: Predict the identity for a new face.
    def predict(self, face) -> tuple[str, float, float]:
        """
        Returns: (predicted_label, posterior_probability, best_distance)
        """
        if self.embeddings.shape[0] == 0:
            return ("unknown", 0.0, float("inf"))

        q = self.facenet.predict(face).astype(np.float32)  # (128,)

        # L2 distances to all gallery embeddings
        diff = self.embeddings - q[None, :]
        dists = np.linalg.norm(diff, axis=1)  # (N,)

        k = max(1, min(self.num_neighbours, dists.shape[0]))
        nn_idx = np.argpartition(dists, k - 1)[:k]
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]  # sorted k-NN

        best_idx = int(nn_idx[0])
        best_dist = float(dists[best_idx])

        # Compute a simple "posterior" over the k neighbours via softmax(-dist)
        # (smaller distance -> higher weight)
        nn_d = dists[nn_idx]
        # numerical stability
        logits = -nn_d
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / (np.sum(weights) + 1e-12)

        # Aggregate probability mass per label
        label_scores = {}
        for w, idx in zip(weights, nn_idx):
            lab = self.labels[int(idx)]
            label_scores[lab] = label_scores.get(lab, 0.0) + float(w)

        pred_label = max(label_scores.items(), key=lambda x: x[1])[0]
        prob = float(label_scores[pred_label])

        # Apply rejection thresholds
        if best_dist > self.max_distance or prob < self.min_prob:
            return ("unknown", prob, best_dist)

        return (pred_label, prob, best_dist)


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
