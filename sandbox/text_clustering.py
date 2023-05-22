import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from typing import Dict, List, Tuple, Union, Any

from .topic_keyword_extraction import KeywordExtractor
from .config import *
from .preprocessing import *

pd.options.mode.chained_assignment = None


class TopicsClusterer:
    def __init__(self, text_data: TextData, embedding_data: EmbeddingData):
        """
        Args:
            text_df: dataframe with textual data to perform clustering on
            text_field: name of the column with the textual data in it
            embedding_model: either name of pretrained HF language model from sentence transformers or 'tfidf'
            min_words_len: int, minimal length (in words) of text to cluster
            max_words_len: int, maximal length (in words) of text to cluster
            embed_sentences: bool flag determining whether to split the original texts to sentences, to allow clustering
              of the sentences
            use_multiprocessing: bool flag determining whether to use multiprocessing for text embedding, currently
                implemented for sentence_transformers only
            sampling: bool flag determining whether to use only a sample of the data in case sample size exceed
                max_sample_size
            max_sample_size: upper limit on the sample size to work with
            clean_digits: bool, weather or not to remove digits when preprocessing the text
        """
        self.text_data = text_data
        self.text_field = text_data.text_field
        self.text_df, self.sentence_df = text_data.text_df, text_data.sentence_df
        self.embedding_data = embedding_data


    @classmethod
    def create_clusterer_from_config(
        cls,
        text_df: pd.DataFrame,
        text_field: str,
        text_processing_config: TextProcessingConfig = TextProcessingConfig(),
        embedding_config: EmbeddingConfig = EmbeddingConfig(),
    ):
        text_data = TextPreProcessor.process_text(text_df, text_field, text_processing_config)

        embedding_data = EmbeddingEngine.create_embeddings(text_data, embedding_config)
        return TopicsClusterer(text_data, embedding_data)

    @classmethod
    def create_clusterer(cls, text_data: TextData, embedding_data: EmbeddingData):
        return TopicsClusterer(text_data, embedding_data)

    def extract_clustering_info(
        self, clustered_df: pd.DataFrame(), embeddings: Any, use_sentences: bool, n_representing: int = 0
    ) -> pd.DataFrame:
        """_summary_
        This function creates an info table given a clustered dataframe
        Args:
            clustered_df (pd.DataFrame): a dataframe that contains both the textual data that was clustered and a
            column "cluster" containing the clustering assignment
            embeddings (Any): embeddings of the textual data
            use_sentences(bool): if this flag is on, it means the clustering was done on the sentences, and the result
                provides both representing sentences and representing comment
            n_representing (int): the number of representing comment in the info table created by the table
        Returns:
            pd.DataFrame: an info table with the following fields for each cluster:
            1. cluster_size
            2. variance
            3. cluster_percentage
            4. representing_instances
        """
        cluster_info = (
            clustered_df[[list(clustered_df)[0], "cluster"]]
            .groupby("cluster")
            .count()
            .rename(columns={list(clustered_df)[0]: "cluster_size"})
        )
        # not all cluster objects have cluster centers, this code finds centers generally for clusters.
        y_predict = clustered_df["cluster"].values
        clf = NearestCentroid()
        if len(cluster_info) > 1:
            clf.fit(embeddings, y_predict)
            cluster_centers = dict(zip(cluster_info.index, clf.centroids_))  # clf.centroids_
        else:
            cluster_centers = {cluster_info.index[0]: np.mean(embeddings, axis=0)}
        sample_size = len(self.sentence_df) if use_sentences else len(self.text_df)
        cluster_info.loc[:, "cluster_percentage"] = (cluster_info["cluster_size"] / sample_size) * 100
        cluster_info["themes"] = None
        for i in cluster_info.index:
            cluster_embeddings = embeddings[np.where(clustered_df["cluster"] == i)]
            cluster_var = (
                sum((euclidean_distances([cluster_centers[i].tolist()], cluster_embeddings)[0]) ** 2)
                / cluster_info.loc[i, "cluster_size"]
            )
            cluster_info.at[i, "variance"] = cluster_var
            if cluster_info["cluster_size"][i] > 5:
                key_ex_cluster = KeywordExtractor(
                    clustered_df[clustered_df["cluster"] == i],
                    self.text_field,
                    clean_digits=self.text_data.text_processing_config.clean_digits,
                )
                top_k_cluster = key_ex_cluster.get_top_keywords(3, method="yake")
                cluster_info.at[i, "themes"] = " ,".join(top_k_cluster.Keyword.values)
            if n_representing is not None:
                # find cluster representing comments/sentences:
                distances = euclidean_distances([cluster_centers[i].tolist()], cluster_embeddings)
                rank_in_cluster = np.argsort(np.argsort(distances[0])) + 1
                clustered_df.loc[clustered_df["cluster"] == i, "rank_in_cluster"] = rank_in_cluster
                representing_df = (
                    clustered_df[clustered_df["cluster"] == i]
                    .sort_values(by="rank_in_cluster")
                    .head(min(cluster_info.loc[i, "cluster_size"], n_representing))
                )
                for j, row in representing_df.reset_index().iterrows():
                    if use_sentences:
                        sentence_num = f"representing_sentence_{str(j + 1)}"
                        cluster_info.at[i, sentence_num] = row["sentence"]
                    comment_num = f"representing_instance_{str(j + 1)}"
                    cluster_info.at[i, comment_num] = row[self.text_field]

        return cluster_info.rename_axis("cluster").reset_index()  # TODO reset index and set column "cluster"

    def cluster_data(
        self,
        n_representing: int,
        n_clusters: Union[int, None],
        n_second_level_clusters: int = None,
        clustering_method: str = "mbk",
        use_sentences: bool = False,
    ) -> Dict:
        """
        The call to run a clustering of the data.
        Args:
            n_representing: number of representing data samples for each cluster
            n_clusters: number of high level clusters to search, required for 'mbk' and 'hierarchical' clustering methods
            n_second_level_clusters: number of clusters to search within each high level cluster. if None second level
            clustering is omitted. Default is None.
            clustering_method: name of the clustering method, either "mbk" or "hierarchical", default is "mbk"
            use_sentences: bool, should clustering be done on top of a sentence breakdown of the original comments.
            Default is False

        Returns: A dict with the following keys:
            cluster_info - a dataframe with the clustering metadata and representing samples
            cluster_assignment - a dataframe with the cluster assignment per each sample
            cluster_obj - either a single clustering object or a dictionary of second level cluster objects

        """
        if use_sentences:
            df_to_cluster = self.sentence_df
            embeddings = self.embedding_data.sentence_embeddings
            if embeddings is None:
                raise ValueError(
                    "Request to cluster sentences, but sentences were not embed at initialization. "
                    "Initialize TopicsClusterer with embed_sentences=True and re-run clustering"
                )
        else:
            df_to_cluster = self.text_df
            embeddings = self.embedding_data.text_embeddings

        if n_clusters is not None and len(df_to_cluster) <= n_clusters:
            raise ValueError(f"Not enough distinct samples to create {n_clusters} clusters")
        clustered_df, cluster_obj = self.single_cluster_iteration(
            clustering_method, df_to_cluster.copy(deep=True), embeddings, n_clusters
        )
        clustering_info = self.extract_clustering_info(clustered_df, embeddings, use_sentences, n_representing)

        if n_second_level_clusters is not None:
            clustering_info, cluster_obj = self.second_level_clustering(
                clustered_df, embeddings, clustering_method, n_second_level_clusters, use_sentences, n_representing
            )

        return {"cluster_info": clustering_info, "cluster_assignment": clustered_df, "cluster_obj": cluster_obj}

    def single_cluster_iteration(
        self, clustering_method: str, df: pd.DataFrame, embeddings: Any, n_clusters: Union[int, None]
    ) -> Tuple[pd.DataFrame, Any]:
        """

        Args:
            clustering_method: either "mbk", "hierarchical" or "dbscan"
            df: dataframe to attach the clustering assignment to
            embeddings: 2d array of embeddings (either numpay array for HF models or CSR matrix for tfidf)
            n_clusters: int, number of clusters to cluster to

        Returns:

        """
        if clustering_method == "mbk":
            if not n_clusters:
                raise ValueError("usage :'mbk' method requires a number clusters (n_clusters provided was None)")
            clustered_df, clusters_obj = self.mbk_cluster_texts(df, embeddings, n_clusters)

        elif clustering_method == "hierarchical":
            if not n_clusters:
                raise ValueError(
                    "usage :'hierarchical' method requires a number clusters" " (n_clusters provided was None)"
                )
            clustered_df, clusters_obj = self.hierarchical_cluster_texts(df, embeddings, n_clusters)

        elif clustering_method == "dbscan":
            clustered_df, clusters_obj = self.DBscan_cluster_texts(df, embeddings)
        else:
            raise ValueError(f'Unknown clustering method "{clustering_method}"')
        return clustered_df, clusters_obj

    def second_level_clustering(
        self,
        clustered_df: pd.DataFrame,
        embeddings: Any,
        clustering_method: str,
        n_second_level_clusters: int,
        use_sentences: bool,
        n_representing: int,
    ) -> Tuple[pd.DataFrame, Dict]:
        """

        Args:
            clustered_df: a dataframe with textual field already clustered (has a field name "cluster")
            embeddings: 2d array of embeddings (either numpay array for HF models or CSR matrix for tfidf)
            clustering_method: either "mbk", "hierarchical" or "dbscan"
            n_second_level_clusters:
            use_sentences: bool, should clustering be done on top of a sentence breakdown of the original comments.
            Default is False
            n_representing: number of representing data samples for each cluster

        Returns:

        """
        clustered_df.rename(columns={"cluster": "high_level_cluster"}, inplace=True)
        second_level_clusters_infos = []
        second_level_clusters_objs = {}

        for i in clustered_df.high_level_cluster.unique():
            cluster_abs_loc = np.where(clustered_df["high_level_cluster"] == i)[0]
            cluster_df_idx = clustered_df.iloc[cluster_abs_loc].index
            df = clustered_df.iloc[cluster_abs_loc]
            if self.embedding_data.embedding_model == "tfidf":
                cluster_embeddings = self.select_rows_csr(embeddings, cluster_abs_loc, "keep")
            else:
                cluster_embeddings = np.take(embeddings, cluster_abs_loc, 0)
            if len(cluster_abs_loc) <= n_second_level_clusters:
                # https://github.com/pandas-dev/pandas/issues/46036 -> .loc instead of .at
                clustered_df.loc[cluster_df_idx, "second_level_cluster"] = 0
                clustered_df_second_level = df
                clustered_df_second_level["cluster"] = 0
                second_level_clusters_obj = None
            else:
                clustered_df_second_level, second_level_clusters_obj = self.single_cluster_iteration(
                    clustering_method, df, cluster_embeddings, n_second_level_clusters
                )
            second_level_clusters_info = self.extract_clustering_info(
                clustered_df_second_level, cluster_embeddings, use_sentences, n_representing
            )
            second_level_clusters_info.rename(columns={"cluster": "second_level_cluster"}, inplace=True)
            second_level_clusters_info["first_level_cluster"] = i
            second_level_clusters_info["cluster_path"] = (
                second_level_clusters_info["first_level_cluster"].astype(str)
                + "-->"
                + second_level_clusters_info["second_level_cluster"].astype(str)
            )
            # https://github.com/pandas-dev/pandas/issues/46036 -> .loc instead of .at
            clustered_df.loc[cluster_df_idx, "second_level_cluster"] = clustered_df_second_level["cluster"]
            second_level_clusters_infos.append(second_level_clusters_info)
            second_level_clusters_objs[i] = second_level_clusters_obj

        return pd.concat(second_level_clusters_infos), second_level_clusters_objs

    @staticmethod
    def mbk_cluster_texts(
        text_df: pd.DataFrame, embeddings: Any, n_clusters: int
    ) -> Tuple[pd.DataFrame, MiniBatchKMeans]:
        """
        A method for calling sklearn minibatch clustering
        Args:
            text_df: dataframe with an embedding field to cluster upon
            embeddings: 2d array of embeddings (either numpy array for HF models or CSR matrix for tfidf)
            n_clusters: int, number of clusters to cluster to

        Returns:
            text_df: the original dataframe along with a new field with the assigned cluster
            mbk: the trained MiniBatchKMeans clustering object

        """
        mbk = MiniBatchKMeans(n_clusters, init="k-means++", n_init=10, reassignment_ratio=0.1)
        mbk.fit(embeddings)
        mbk_means_labels = mbk.labels_.astype(int)
        text_df.loc[:, "cluster"] = mbk_means_labels
        return text_df, mbk

    @staticmethod
    def hierarchical_cluster_texts(
        text_df: pd.DataFrame, embeddings: Any, n_clusters: int
    ) -> Tuple[pd.DataFrame, AgglomerativeClustering]:
        """
        A method for calling sklearn minibatch clustering
        Args:
            text_df: dataframe with an embedding field to cluster upon
            embeddings: 2d array of embeddings (either numpay array for HF models or CSR matrix for tfidf)
            n_clusters: int, number of clusters to cluster to

        Returns:
            text_df: the original dataframe along with a new field with the assigned cluster
            mbk: the trained MiniBatchKMeans clustering object

        """
        agg = AgglomerativeClustering(n_clusters)
        agg.fit(embeddings)
        agg_means_labels = agg.labels_.astype(int)
        text_df.loc[:, "cluster"] = agg_means_labels
        return text_df, agg

    @staticmethod
    def DBscan_cluster_texts(text_df: pd.DataFrame, embeddings: Any) -> Tuple[pd.DataFrame, DBSCAN]:
        """
        A method for calling sklearn minibatch clustering
        Args:
            text_df: dataframe with an embedding field to cluster upon
            embeddings: 2d array of embeddings (either numpay array for HF models or CSR matrix for tfidf)
            n_clusters: int, number of clusters to cluster to

        Returns:
            text_df: the original dataframe along with a new field with the assigned cluster
            mbk: the trained MiniBatchKMeans clustering object

        """
        clustering = DBSCAN(eps=0.5, min_samples=2)
        clustering.fit(embeddings)
        clustering_labels = clustering.labels_.astype(int)
        text_df.loc[:, "cluster"] = clustering_labels
        return text_df, clustering

    @staticmethod
    def select_rows_csr(mat: scipy.sparse.csr_matrix, indices: List, what: str) -> scipy.sparse.csr_matrix:
        """
        Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
        Args:
            mat: scipy CSR matrix
            indices: list of indices

        Returns: scipy CSR matrix, subset of the original mat

        """
        if not isinstance(mat, scipy.sparse.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")
        indices = list(indices)
        if what == "drop":
            mask = np.ones(mat.shape[0], dtype=bool)
            mask[indices] = False
        elif what == "keep":
            mask = np.zeros(mat.shape[0], dtype=bool)
            mask[indices] = True
        return mat[mask]


def get_clustering_predictions_for_new_data(
    cluster_obj,
    new_data: pd.Series,
    classified_data: pd.DataFrame = None,
    text_field: str = None,
    classified_data_embeddings: Any = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    use_multiprocessing: bool = False,
) -> List:
    """_summary_
    This function uses a clustering object that was created by the library to cluster new examples.
    Args:
        cluster_obj (_type_): the clustering object that was created when performing clustering for the original data
        new_data (pd.Series): the new data that we want to get predictions for
        text_field: name of the column with the textual data in it
        classified_data (pd.DataFrame): the data we previously classified
        classified_data_embeddings:
        embedding_model:
        use_multiprocessing

    Returns:
        List : assignments for the data
    """
    if len(new_data) == 0:
        raise ValueError(f"Not enough samples")
    new_data.dropna(inplace=True)
    if len(new_data) == 0:
        raise ValueError(f"All samples has Nan values")
    embedding_engine = EmbeddingEngine(embedding_model)
    embeddings = embedding_engine.embed_text(new_data.values, use_multiprocessing)
    # getting predictions
    if type(cluster_obj) == MiniBatchKMeans:
        predictions = list(cluster_obj.predict(embeddings))
    elif type(cluster_obj) == DBSCAN or type(cluster_obj) == AgglomerativeClustering:
        if classified_data is None:
            raise ValueError(
                f"To get predictions from {type(cluster_obj)} clusterer you must provide classified_data"
            )
        if classified_data_embeddings is None:
            classified_data_embeddings = embedding_engine.embed_text(
                classified_data[text_field].values, use_multiprocessing
            )
        neigh = KNeighborsClassifier(n_neighbors=classified_data.cluster.nunique())
        neigh.fit(classified_data_embeddings, classified_data["cluster"])
        predictions = list(neigh.predict(embeddings))
    else:
        raise ValueError(f"clustering object of type {type(cluster_obj)} is not supported")
    return predictions
