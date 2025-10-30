import faiss
import torch
import time
import numpy as np
import logging
import os
import pickle
import yaml
import pprint
import pathlib
import json
from typing import Union, Optional, Dict, List, Tuple


def faiss_index_to_gpu(cpu_index):
    """
    Convert a Faiss CPU index to a GPU index.
    """
    # Configure GPU cloner options
    cloner_options = faiss.GpuClonerOptions()
    cloner_options.useFloat16 = False
    cloner_options.usePrecomputed = False
    cloner_options.indicesOptions = faiss.INDICES_CPU

    # Configure Faiss GPU resources
    gpu_resources = faiss.StandardGpuResources()

    # Convert CPU index to GPU index
    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index, cloner_options)

    return gpu_index


def compute_centroids(
    data: Union[np.memmap, np.ndarray],
    ncentroids: int = 1000,
    niter: int = 100,
    seed: int = 1234,
    Kmeans_with_cos_dist: bool = False,
    save_folder: str = "",
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
):

    """
    Runs K-means clustering on the input data using "faiss" and saves the following output files:

          1)faiss k-means index object (pickle file).
          2)k-means centroids (numpy array).
          3)Distance to centroid for data points in <data> (numpy array).
          4)Nearest centroid for data points in <data> (numpy array).
    args:
        data: A float32 numpy memmap array or numpy array of shape [dataset_size x d], where d is the embedding vector size..
        ncentroids: number of kmeans clusters/centroids.
        niter: The number of iterations to run the K-means algorithm for.
        seed: The random seed to use for reproducibility.
        Kmeans_with_cos_dist: (boolean) when True, run spherical kmeans.
        save_folder: path to save/load output files.
        logger: A logger instance to use for logging.

    returns:
        faiss k-means object, distances to centroids, and nearest centroids
    """
    os.makedirs(save_folder, exist_ok=True)
    # -- Compute Kmeans centroids
    if logger:
        logger.info(
            f"Running Kmeans clustering using faiss on dataset of shape {data.shape} ...."
        )
        logger.info(f"Kmeans parameters: {locals()} ....")
    else:
        print(f"Running Kmeans clustering using faiss on dataset of shape {data.shape} ....")
    # pprint.pprint(locals(), width=1, indent=4)

    d = data.shape[1]
    # -- Use GPUs for clustering when available
    use_gpu = torch.cuda.is_available()

    device = "cuda" if use_gpu else "cpu"

    if logger:
        logger.info(f"Clustering on {device} ....")
    else:
        print(f"Clustering on {device} ....")

    spherical = (
        Kmeans_with_cos_dist  # -- spherical=True when Kmeans_with_cos_dist is True
    )

    ## -- Step 1) Train faiss kmeans
    kmeans = faiss.Kmeans(
        d,
        ncentroids,
        niter=niter,
        verbose=verbose,
        seed=seed,
        spherical=spherical,
        gpu=use_gpu,
    )  ## -- faiss.Kmeans "gpu" argument: bool or int, optional. False: don't use GPU, True: use all GPUs, number: use this many GPUs.

    # -- If kmeans centroids are not saved - > create and train faiss Kmeans clustering object
    kmeans_obj_file_loc = pathlib.Path(save_folder, "kmeans_index.pickle")

    if not os.path.exists(kmeans_obj_file_loc):
        start_time = time.time()
        kmeans.train(data)
        if logger:
            logger.info(f"Time for clustering (mins): {(time.time()-start_time)/(60):.2f}")
        else:
            print(f"Time for clustering (mins): {(time.time()-start_time)/(60):.2f}")

        # -- Move kmeans index to cpu to save it
        kmeans_index = faiss.index_gpu_to_cpu(kmeans.index)
        if logger:
            logger.info(f"faiss kmeans index to store: {type(kmeans_index)}")
        else:
            print(f"faiss kmeans index to store: {type(kmeans_index)}")
        ## -- Save faiss kmeans index object as pickle file
        with open(kmeans_obj_file_loc, "wb") as file:
            pickle.dump(kmeans_index, file)
        ## -- save faiss kmeans centroids as npy file
        if kmeans.centroids is not None:
            np.save(pathlib.Path(save_folder, "kmeans_centroids.npy"), kmeans.centroids)

        if logger:
            logger.info(f"Saved!")
        else:
            print(f"Saved!")

    else:
        # -- Else, load stored kmeans object
        if logger:
            logger.info(
                f"Loading faiss Kmeans index pickle file from {kmeans_obj_file_loc}"
            )
        else:
            print(f"Loading faiss Kmeans index pickle file from {kmeans_obj_file_loc}")
        with open(kmeans_obj_file_loc, "rb") as file:
            kmeans_index = pickle.load(file)
            if use_gpu:
                # -- move kmeans index to gpu
                kmeans_index = faiss_index_to_gpu(kmeans_index)
            kmeans.index = kmeans_index

    ## -- Step 2) Find the nearest centroid for each data point, l2 distance search
    ## -- nearest_cent: the nearest centroid for each example in data. dist_to_cent: contains the squared L2 distances.
    start_time = time.time()
    dist_to_cent, nearest_cent = kmeans.index.search(data, 1)
    dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)
    if logger:
        logger.info(
            f"Time for finding nearest centroid for each data point (mins): {(time.time()-start_time)/(60):.2f}"
        )
    else:
        print(f"Time for finding nearest centroid for each data point (mins): {(time.time()-start_time)/(60):.2f}")

    ## -- save faiss nearest_cent and dist_to_cent as .npy files
    dist_to_cent_file = pathlib.Path(save_folder, "dist_to_cent.npy")
    nearest_cent_file = pathlib.Path(save_folder, "nearest_cent.npy")
    np.save(dist_to_cent_file, dist_to_cent)
    np.save(nearest_cent_file, nearest_cent)

    return kmeans, dist_to_cent, nearest_cent


def compute_conversation_scores(
    embeddings_path: str,
    metadata_path: str,
    ncentroids: int = 1000,
    niter: int = 300,
    save_folder: str = "clustering_results",
    output_path: str = "conversation_scores.json",
    seed: int = 1234,
) -> Dict[str, float]:
    """
    Compute conversation scores based on clustering.
    
    Args:
        embeddings_path: Path to the embeddings file
        metadata_path: Path to the metadata file
        ncentroids: Number of clusters
        niter: Number of iterations for k-means
        save_folder: Folder to save clustering results
        output_path: Path to save the scores
        seed: Random seed for clustering
        
    Returns:
        Dictionary mapping unique_ids to scores
    """
    # Set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("clustering_scoring")
    
    # Load embeddings and metadata
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Run clustering
    os.makedirs(save_folder, exist_ok=True)
    print(f"Running clustering with {ncentroids} centroids")
    kmeans, dist_to_cent, nearest_cent = compute_centroids(
        data=embeddings,
        ncentroids=ncentroids,
        niter=niter,
        seed=seed,
        save_folder=save_folder,
        logger=logger,
        verbose=True
    )
    
    # Calculate similarity scores (inverse of distance)
    # Normalize distances to [0, 1] range where 1 means closest to center
    max_dist = np.max(dist_to_cent)
    similarity_scores = 1.0 - (dist_to_cent / max_dist)
    
    # Map embeddings back to unique_ids and aggregate scores
    unique_id_scores = {}
    unique_id_counts = {}
    
    # First pass: sum up scores for each unique_id
    for idx, meta in enumerate(metadata):
        unique_id = meta['unique_id']
        if unique_id not in unique_id_scores:
            unique_id_scores[unique_id] = 0.0
            unique_id_counts[unique_id] = 0
        
        unique_id_scores[unique_id] += similarity_scores[idx]
        unique_id_counts[unique_id] += 1
    
    # Second pass: compute averages
    conversation_scores = {}
    for unique_id, score_sum in unique_id_scores.items():
        count = unique_id_counts[unique_id]
        average_score = score_sum / count
        conversation_scores[unique_id] = float(average_score)  # Convert to float for JSON serialization
    
    # Save scores to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_scores, f, indent=2)
    
    print(f"Saved conversation scores for {len(conversation_scores)} conversations to {output_path}")
    
    # Return scores dictionary
    return conversation_scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute conversation scores based on clustering')
    parser.add_argument('--embeddings_path', type=str, default='/data2/jkx/LLaVA/output/bert_embeddings_embeddings.npy',
                        help='Path to the embeddings file')
    parser.add_argument('--metadata_path', type=str, default='/data2/jkx/LLaVA/output/bert_embeddings_metadata.json',
                        help='Path to the metadata file')
    parser.add_argument('--ncentroids', type=int, default=1000, help='Number of clusters')
    parser.add_argument('--niter', type=int, default=100, help='Number of iterations for k-means')
    parser.add_argument('--save_folder', type=str, default='/data2/jkx/LLaVA/output/clustering_results',
                        help='Folder to save clustering results')
    parser.add_argument('--output_path', type=str, default='/data2/jkx/LLaVA/output/conversation_scores.json',
                        help='Path to save the scores')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for clustering')
    
    args = parser.parse_args()
    
    compute_conversation_scores(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        ncentroids=args.ncentroids,
        niter=args.niter,
        save_folder=args.save_folder,
        output_path=args.output_path,
        seed=args.seed
    )