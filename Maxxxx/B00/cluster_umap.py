import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd  # sure takes a long time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    display(labeled_df)
    
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
    return clust
  