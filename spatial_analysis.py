import pandas as pd
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as hcluster
from epigraphhub.analysis.clustering import lag_ccf

def get_clusters_swiss(t=0.3, ini_date = None, end_date=None, 
                    columns =["georegion", "entries"],
                    drop_values=["CH", "FL", "CHFL"],
                    smooth = True, plot = True, path = 'data_article/cases_swiss.csv' ):
    """
    This function it was create to allow the reproduction of the results of the article.
    Params to get the list of clusters computed by the compute_cluster function.
    :params t: float. Thereshold used in the clusterization.
    :param end_date: string. Indicates the last day used to compute the cluster
    :returns: Array with the clusters computed.
    """
    
    df = pd.read_csv(path)
    df.set_index("datum", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    if ini_date:
        df = df.loc[ini_date:]

    if end_date != None:
        df = df.loc[:end_date]

    inc = df.pivot(columns=columns[0], values=columns[1])

    if smooth:
        inc = inc.rolling(7).mean().dropna()

    if drop_values != None:
        for i in drop_values:
            del inc[i]

    inc = inc.dropna()

    cm = lag_ccf(inc.values)[0]

    # Plotting the dendrogram
    linkage = hcluster.linkage(cm, method="complete")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
        hcluster.dendrogram(linkage, labels=inc.columns, color_threshold=0.18, ax=ax)
        #ax.set_title(
        #    "Result of the hierarchical clustering of the series",
        #    fontdict={"fontsize": 20},
        #)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig('plots/dendro.png', dpi = 300)

        plt.show()

    # computing the cluster
    ind = hcluster.fcluster(linkage, t, "distance")

    grouped = pd.DataFrame(list(zip(ind, inc.columns))).groupby(0)

    clusters = [group[1][1].values for group in grouped]

    return clusters



