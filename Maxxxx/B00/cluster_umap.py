import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd  # sure takes a long time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pseudo_label

LABEL="Label"

def run_clustering_on_umap(embedding, num_clusters, cdata_df, label_df, key="PatientSSN"):
    columns = list(cdata_df.columns)
    columns.remove(key)
    clust = KMeans(n_clusters=num_clusters)
    # clust = SpectralClustering(n_clusters=num_clusters)
    clust.fit(embedding)
    
    plt.subplots(figsize=(10,10))
    for i in range(num_clusters):
        plt.scatter(embedding[clust.labels_==i,0],embedding[clust.labels_==i,1])
    plt.legend([i for i in range(num_clusters)])
#     for i in range(num_clusters):
#         print("Cluster",i+1)
#         # print(cdata_df.iloc[np.where(clust.labels_==i)[0],:].label.value_counts())
#         print()
    
#     X = cdata_df.loc[:,columns].values.astype('float') # cut out only data we wish to use
#     scaler = StandardScaler() #TODO: fix
#     sX = scaler.fit_transform(X) # scale appropriately
    sX = cdata_df.loc[:,columns].values.astype('float')
    
    # find average levels of each column for each cluster to tell them apart
    clusters = pd.DataFrame(data=clust.labels_, columns=["cluster"])
    cdata_scaled_df = pd.DataFrame(data=sX, columns=columns)
    clusters_data_df = pd.concat([cdata_scaled_df, clusters], axis=1)
    
    labeled_df = pd.concat([cdata_df, clusters], axis=1).merge(label_df, how="outer", left_on=key, right_on=key)
    labeled_df = labeled_df.fillna("Unlabeled")
#     display(labeled_df)
    
    mean_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    median_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    std_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    
    labels = list(label_df[LABEL].unique()) + ["Unlabeled"]
    label_clusters = np.zeros((clust.n_clusters, len(labels)))
    
    for i in range(clust.n_clusters):
        mean_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].mean()[:-1] # put in mean for each cluster
        median_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].median()[:-1] # put in median for each cluster
        std_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].std()[:-1]/np.sqrt(clusters_data_df[clusters_data_df["cluster"] == i].count()[:-1]) # put in standard error for each cluster
        
        cluster_df = labeled_df[labeled_df["cluster"] == i]
        for j in range(len(labels)):
            label_clusters[i,j] = len(cluster_df[cluster_df[LABEL] == labels[j]])

    mean_clusters_df = pd.DataFrame(mean_clusters, columns=clusters_data_df.columns[:-1])
    median_clusters_df = pd.DataFrame(median_clusters, columns=clusters_data_df.columns[:-1])
    label_clusters_df = pd.DataFrame(label_clusters, columns=labels)
    # mean_clusters_df.T.plot(kind='bar',figsize=(20,8), yerr=std_clusters) # bar chart, only use if necessary
    # plt.legend(["Cluster "+str(i+1) for i in range(4)])
    
    print(label_clusters_df)
    
#     plt.figure(figsize=(20,3))
#     plt.title('Label Counts for each cluster')
#     plt.imshow(mean_clusters_df,cmap='jet')
#     plt.xticks(ticks=np.arange(label_clusters_df.shape[1]), labels=label_clusters_df.columns, rotation=90)
#     plt.colorbar(shrink=0.5)
#     plt.yticks(ticks=np.arange(label_clusters_df.shape[0]), labels=["Cluster "+str(i+1) for i in range(label_clusters_df.shape[0])])

    plt.figure(figsize=(20,3))
    plt.title('Means for each cluster')
    plt.imshow(mean_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(mean_clusters_df.shape[1]), labels=mean_clusters_df.columns, rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(mean_clusters_df.shape[0]), labels=["Cluster "+str(i) for i in range(mean_clusters_df.shape[0])])

    plt.figure(figsize=(20,3))
    plt.title('Medians for each cluster')
    plt.imshow(median_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(median_clusters_df.shape[1]), labels=median_clusters_df.columns, rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(median_clusters_df.shape[0]), labels=["Cluster "+str(i) for i in range(median_clusters_df.shape[0])])
    return labeled_df



class UMAPLabeler(pseudo_label.PseudoLabeler):
    def __init__(self, reducer, num_clusters):
        """
        The UMAPLabeler is an implementation of PseudoLabeler, which uses KMeans to assign probabilities to unlabeled points based on labeled points in the same cluster

        Parameters
        --------
        reducer : UMAP
            trained umap reducer used to pseudo label data
        num_clusters : int
            number of clusters in pretrained reducer
        """
        self.reducer = reducer
        self.num_clusters = num_clusters
    
    def label(self, X_train, y_train, X_unlabeled, cutoff=0.66):
        """
        :param y_train: (n,) dimension label vector
        :return: (n,x) dimension label vector where column and label_cols 
        """
        X_train = X_train.reset_index(inplace=False, drop=True)
        y_train = y_train.reset_index(inplace=False, drop=True)
        X_unlabeled = X_unlabeled.reset_index(inplace=False, drop=True)
        
        X = pd.concat([X_train, X_unlabeled], axis=0)#.values.astype('float')
        X.reset_index(inplace=True, drop=True)
#         display(X)
        
        label_cols = list(y_train.unique())
        
        embedding = self.reducer.transform(X)
        
        clust = KMeans(n_clusters=self.num_clusters)
        clust.fit(embedding)

        # find average levels of each column for each cluster to tell them apart
        clusters = pd.DataFrame(data=clust.labels_, columns=["cluster"])
        clusters_data_df = pd.concat([X, clusters], axis=1)
    
        labeled_df = pd.concat([clusters, y_train], axis=1)
        labeled_df.dropna(inplace=True) # drops the n/a since nas are unlabeled

        label_clusters = np.zeros((clust.n_clusters, len(label_cols)))

        for i in range(clust.n_clusters):
            cluster_df = labeled_df[labeled_df["cluster"] == i]
            s = 0
            for j in range(len(label_cols)):
                label_clusters[i,j] = len(cluster_df[cluster_df[LABEL] == label_cols[j]])
                s += len(cluster_df[cluster_df[LABEL] == label_cols[j]])
#             if s < 2:
#                 label_clusters[i,:] = 1 / len(label_cols)
#             else:
#                 label_clusters[i,:] /= s
        totals = np.sum(label_clusters, axis=1)
        label_clusters = (label_clusters.T / totals).T
#         display(label_clusters)
        throwout_clusters = list()
        for i in range(clust.n_clusters):
#             for j in range(len(label_cols)):
            if abs(label_clusters[i,0] - 0.5) < cutoff - 0.5:
                throwout_clusters.append(i)
        
#         pd.set_option('display.max_rows', len(clusters_data_df))
#         display(clusters_data_df)
#         pd.reset_option('display.max_rows')
        pseudo_labels = clusters_data_df.iloc[X_train.shape[0]:,:].apply(lambda r: pd.Series(label_clusters[int(r["cluster"]), :]) if r["cluster"] not in throwout_clusters else np.nan, axis=1)
        # TODO join
#         pseudo_labels.dropna(inplace=True)
        # TODO seperate
#         print(y_train.shape)
        y = None
        for i in range(len(label_cols)):
#             print(i, label_cols[i])
            yslice = np.reshape((y_train.to_numpy() == label_cols[i]).astype(float), (-1, 1))
            if y is None:
                y = yslice
            else:
                y = np.concatenate((y, yslice), axis=1)
        labels = pd.concat([pd.DataFrame(y), pseudo_labels], axis=0)
        
        temp = pd.concat([X, labels], axis=1)
        temp.dropna(inplace=True)
        
        
        return temp.iloc[:,:X.shape[1]].copy(), temp.iloc[:,X.shape[1]:].to_numpy(), label_cols

        
        

