import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd  # sure takes a long time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def run_clustering_on_umap(mapper, num_clusters, cdata_df, columns):
    clust = KMeans(n_clusters=num_clusters)
    # clust = SpectralClustering(n_clusters=num_clusters)
    clust.fit(mapper.embedding_)
    plt.subplots(figsize=(10,10))
    for i in range(num_clusters):
        plt.scatter(mapper.embedding_[clust.labels_==i,0],mapper.embedding_[clust.labels_==i,1])
    plt.legend([i+1 for i in range(num_clusters)])
    for i in range(num_clusters):
        print("Cluster",i+1)
        # print(cdata_df.iloc[np.where(clust.labels_==i)[0],:].label.value_counts())
        print()
    
    X = cdata_df.loc[:,columns].values.astype('float') # cut out only data we wish to use
    scaler = StandardScaler()
    sX = scaler.fit_transform(X) # scale appropriately
    
    # find average levels of each column for each cluster to tell them apart
    clusters = pd.DataFrame(data=clust.labels_, columns=["cluster"])
    cdata_scaled_df = pd.DataFrame(data=sX, columns=columns)
    clusters_data_df = pd.concat([cdata_scaled_df, clusters], axis=1)
    mean_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    median_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    std_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    for i in range(clust.n_clusters):
        mean_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].mean()[:-1] # put in mean for each cluster
        median_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].median()[:-1] # put in median for each cluster
        std_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].std()[:-1]/np.sqrt(clusters_data_df[clusters_data_df["cluster"] == i].count()[:-1]) # put in standard error for each cluster

    mean_clusters_df = pd.DataFrame(mean_clusters, columns=clusters_data_df.columns[:-1])
    median_clusters_df = pd.DataFrame(median_clusters, columns=clusters_data_df.columns[:-1])
    # mean_clusters_df.T.plot(kind='bar',figsize=(20,8), yerr=std_clusters) # bar chart, only use if necessary
    # plt.legend(["Cluster "+str(i+1) for i in range(4)])

    plt.figure(figsize=(20,3))
    plt.title('Means for each cluster')
    plt.imshow(mean_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(mean_clusters_df.shape[1]), labels=mean_clusters_df.columns, rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(mean_clusters_df.shape[0]), labels=["Cluster "+str(i+1) for i in range(mean_clusters_df.shape[0])])

    plt.figure(figsize=(20,3))
    plt.title('Medians for each cluster')
    plt.imshow(median_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(median_clusters_df.shape[1]), labels=median_clusters_df.columns, rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(median_clusters_df.shape[0]), labels=["Cluster "+str(i+1) for i in range(median_clusters_df.shape[0])])
    return clust
  