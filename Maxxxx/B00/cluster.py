import numpy as np
import sklearn
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def findBestClusters(X, expected_n, range_r=3, random_state=42):
    best_kmeans = None
    best_labels = None
    best_score = -1
    for n in range(expected_n - range_r, expected_n + range_r):
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if silhouette_avg > best_score:
            best_kmeans = kmeans
            best_labels = cluster_labels
            best_score = silhouette_avg
    return best_kmeans, best_labels



class KNNClusterUMAPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        umap_reducer,
        n_neighbors=5,
        *,
        algorithm="auto",
        metric="minkowski",
        p=2
    ):
        self.reducer = umap_reducer
        
        self.n_neighbors = n_neighbors
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric, p=p)
        self.fitted = False
        
        self.y = None
        self.cluster = None
        self.cluster_prob = None

        
    def fit(self, X, y):
        """
        Only works for 0/1 labels right now!!!
        """
        if self.fitted:
            raise Exception("This classifier can only be fitted once.")
        self.fitted = True
        
        self.y = y
        embedding = self.reducer.fit_transform(X)
        
        
        # run kmeans clustering with silhouette
        # HARD CODED FOR NOW
        kmeans, cluster = findBestClusters(X, 9)
        
        # save indices->cluster map and cluster->probability map
        self.cluster = cluster
        cluster_prob = []
        for i in range(kmeans.n_clusters):
            p = np.sum(y[cluster==i]) / y[cluster==i].shape[0]
            cluster_prob.append(p)
        self.cluster_prob = np.array(cluster_prob) # probability of label 1
        
        self.nn.fit(embedding)
        
        return self
        
    def predict(self, X, n_neighbors=None):
        predicted_prob = self.predict_proba(X, n_neighbors)
        return (predicted_prob > 0.5).astype(int)
        
    def predict_proba(self, X, n_neighbors=None):
        embedding = self.reducer.transform(X)
        # n_neighbors=None lets the nn use self.n_neighbors
        # only specify n_neighbors when an override is desired
        neigh_dist, neigh_ind = self.nn.kneighbors(embedding, n_neighbors)
        
        
        # TODO: try different heuristics?
        neigh_clusters = self.cluster[neigh_ind]
        probs = self.cluster_prob[neigh_clusters]
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        return np.sum(probs, axis=1) / n_neighbors
        
    def predict_log_proba(self, X):
        raise Exception("Not supported")
        
        
class HeuristicKNNClusterClassifier(KNNClusterUMAPClassifier):
    def __init__(
        self,
        umap_reducer,
        n_neighbors=5,
        *,
        c=0.1,
        algorithm="auto",
        metric="minkowski",
        p=2
    ):
        super().__init__(umap_reducer,
                       n_neighbors=n_neighbors,
                       algorithm=algorithm,
                       metric=metric,
                       p=p)
        self.c = c
    
    def predict_proba(self, X, n_neighbors=None):
        embedding = self.reducer.transform(X)
        neigh_dist, neigh_ind = self.nn.kneighbors(embedding, n_neighbors)
        
        weights = 1 / (neigh_dist + self.c)
        
        neigh_clusters = self.cluster[neigh_ind]
        probs = self.cluster_prob[neigh_clusters]
        return np.sum(probs * weights, axis=1) / np.sum(weights, axis=1)

    
class KNNClusterLogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        knn,
        logreg,
        threshold=0.65
    ):
        self.knn = knn
        self.logreg = logreg
        # threshold must be >0.5!!
        self.threshold = threshold

        
    def fit(self, X, y):
        """
        Only works for 0/1 labels right now!!!
        """
        self.knn.fit(X, y)
        self.logreg.fit(X, y)
        return self
        
    def predict(self, X, n_neighbors=None):
        predicted_prob = self.predict_proba(X, n_neighbors)
        return (predicted_prob > 0.5).astype(int)
        
    def predict_proba(self, X, n_neighbors=None):
        knn_proba = self.knn.predict_proba(X, n_neighbors)
        logreg_proba = self.logreg.predict_proba(X)
        return np.where((knn_proba > self.threshold) | (knn_proba < (1 - self.threshold)), knn_proba, logreg_proba[:,1])
    
    def predict_log_proba(self, X):
        raise Exception("Not supported")

class PickyHeuristicKNNClusterClassifier(HeuristicKNNClusterClassifier):

    def predict(self, X, n_neighbors=None):
        raise Exception("Not supported")
        
    def predict_proba(self, X, threshold=0.65, n_neighbors=None):
        knn_proba = super().predict_proba(X, n_neighbors)
        return np.where((knn_proba > threshold) | (knn_proba < (1 - threshold)), knn_proba, 0.5)
    
    def predict_log_proba(self, X):
        raise Exception("Not supported")        
        
        
        
