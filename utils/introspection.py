from umap import UMAP
import matplotlib.pyplot as plt
from pandas import read_csv
import torch
from os import listdir, makedirs
from tqdm import tqdm
import numpy as np
import matplotlib.patches as mpatches
from skimage import io
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.decomposition._pca import PCA
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy import stats

import seaborn as sns

from utils.models import Mode
import warnings

warnings.simplefilter('ignore')


def visualize_latent_space(test_data, img_folder, csv, vq_vae,
                           max_degree, network_dir, network_name,
                           num_emb,
                           batch_size=5,
                           mode=Mode.vq_vae,
                           size_latent_space=None,
                           device="cpu",
                           levels=5):
    """
    Visualizes the latent space of a VQ-VAE and also its embedded Space.
    """

    # TODO: Adapt the vq_vae2 mode
    jpg_list = listdir(img_folder)
    csv_df = read_csv(csv)
    data_size: int = len(jpg_list)

    path = f'{network_dir}/{network_name}.pth'
    torch.save(vq_vae.state_dict(), path)

    colors = ['navy', 'darkorange', 'red', 'limegreen', 'turquoise', 'firebrick',
              'indigo', 'darkgreen', 'cornflowerblue', 'sienna']
    colormap = np.array([colors[c] for c in range(levels)])

    disease_states = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

    patches = []
    for i in range(levels):
        patches.append(mpatches.Patch(color=colormap[i], label=f'cluster {i}'))

    makedirs(f'{network_dir}/visualizations/', exist_ok=True)
    makedirs(f'{network_dir}/histograms/', exist_ok=True)

    if mode == Mode.vq_vae_2:
        embeddings = vq_vae.emb_bottom.embedding.detach().cpu().numpy()
    else:
        embeddings = vq_vae.vector_quantization.embedding.detach().cpu().numpy()

    #################################################################
    # Visualize the embedded space.
    #################################################################

    # Therefore reduce the embedded space from shape (num_dim, emb_dim) to (num_dim, n_compontents)
    # using UMAP (, TSNE) or PCA where n_components should be 2 or 3 to be able to do a scatter plot
    num_components = 3
    pca = PCA(n_components=num_components).fit(embeddings)
    for i in range(num_components):
        print("Explained variance of component {0:d}: {1:.4f}".format(i + 1, pca.explained_variance_ratio_[i]))
    pca = pca.transform(embeddings)

    # Hierarchical clustering of the embedded Space
    print(
        "\nNote: the Cophenetic Correlation Coefficient compares (correlates) only very briefly the actual pairwise distances of all samples to those implied by the hierarchical clustering. "
        "The closer the value is to 1, the better the clustering preserves the original distances.\n ")

    # save best order of embedded vectors
    best_order, best_c = None, 0
    for method in ['single', 'average', 'weighted', 'complete', 'median', 'ward']:
        # methods single, median, centroid and complete seems to be unsuitable

        if method not in ['centroid', 'median', 'ward']:
            metrics = ['euclidean', 'seuclidean', 'cosine', 'correlation',
                       # 'cityblock', 'chebyshev', seems not suitable
                       # 'canberra',  --> bad for now
                       'mahalanobis']
        else:
            # 'centroid', 'median' and 'ward' can only be applied when using euclidean distances as metric
            metrics = ['euclidean']

        for metric in metrics:
            try :
                Z = hierarchy.linkage(embeddings, method=method, metric=metric, optimal_ordering=True)
                emb_distances_vector = pdist(embeddings, metric=metric)
                c, coph_dists = hierarchy.cophenet(Z, emb_distances_vector)
                print(f"The Cophenetic Correlation Coefficient of method: {method} and metric: {metric} is: %.4f" % c)

                if c > best_c:
                    best_order = hierarchy.leaves_list(Z)

                plt.figure(figsize=(10, 7))
                plt.title(f"dendogram - embedded space - {method} - {metric}")
                hierarchy.dendrogram(Z)
                plt.xlabel("index of embedded vector")
                plt.show(block=False)
                plt.savefig(f"{network_dir}/visualizations/dendrogram_embbeding_{method}_{metric}.png")
                plt.pause(4)
                plt.close()

                # Plot hierarchical clustering with (TSNE,) PCA or UMAP Plot
                cluster = hierarchy.fcluster(Z, t=5, criterion='maxclust') - 1

                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.set_title(f"PCA - embedded space - {method} - {metric} \n")
                ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=colormap[cluster], s=1, label=colormap)
                plt.legend(handles=patches)
                plt.show(block=False)
                plt.pause(3)
                plt.savefig(f"{network_dir}/visualizations/hierarchical_clustering_embedding_pca_{method}_{metric}.png")
                plt.close(fig)

                # Use outcome of optimal order of distances to plot the heatmap
                emb_distances_matrix = squareform(emb_distances_vector)

                plt.figure(figsize=(7, 7))
                mask = np.zeros_like(emb_distances_matrix)
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    sns.heatmap(emb_distances_matrix[hierarchy.leaves_list(Z)],
                                cmap="YlGnBu",
                                mask=mask,
                                linewidth=0.0,
                                square=True)  # sns.clustermap also possible
                    plt.title(f"Heatmap - Embedded Space - {method} - {metric}", fontsize=13, fontweight='bold')
                    plt.savefig(f"{network_dir}/visualizations/heatmap_embeddings_{method}_{metric}.png")
                    plt.xlabel("index of embedded vector")
                    plt.ylabel("index of embedded vector")
                    plt.show(block=False)
                    plt.pause(3)
                    plt.close()
            except ValueError:
                pass

    # Plot embedded space with original order
    # Using U-Map
    umap_emb = UMAP(n_neighbors=40,
                    min_dist=0.01,
                    n_components=3,
                    # random_state=42
                    ).fit_transform(embeddings)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("UMAP Clustering - Embedded Space - original order\n")
    scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=np.arange(0, num_emb), s=1)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Index", labelpad=+1)
    plt.show()
    plt.savefig(f"{network_dir}/visualizations/umap_clustering_embeddings_original_order.png")
    plt.close(fig)

    # Using PCA
    num_components = 3
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("PCA Clustering - Embedded Space - original order\n")
    scatter = ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], s=1,
                         c=np.arange(0, num_emb))
    cbar = fig.colorbar(scatter)
    cbar.set_label("Index of embedded vector", labelpad=+1)
    plt.savefig(f"{network_dir}/visualizations/pca_clustering_original_order.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

    # Plot embedded space with optimal order
    # Using U-Map
    umap_emb = umap_emb[best_order]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("UMAP Clustering - Embedded Space - original order\n")
    scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], s=1)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Index", labelpad=+1)
    plt.savefig(f"{network_dir}/visualizations/umap_clustering_embeddings_original_order.png")
    plt.show()
    plt.close(fig)

    # Using PCA
    pca = pca[best_order]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("PCA Clustering - Embedded Space - best order\n")
    scatter = ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], s=1,
                         c=np.arange(0, num_emb))
    cbar = fig.colorbar(scatter)
    cbar.set_label("index of embedded vector", labelpad=+1)
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

    ######################################################
    # Visualization of the latents of a VQ-VAE
    ######################################################

    targets = np.zeros(data_size, dtype=np.uint8)

    # Numbers in angles have to be in a specific order
    angles = [x for x in range(-max_degree, -9)]
    angles.extend([x for x in range(10, max_degree + 1)])
    angles.extend([x for x in range(-9, 10)])

    print("Generate targets...")
    try:
        jpg_list.remove(".snakemake_timestamp")
    except ValueError:
        pass

    for i, jpg in tqdm(enumerate(jpg_list)):
        jpg = jpg.replace("_flipped", "")
        jpg = jpg.replace(".jpeg", "")
        jpg = jpg.replace(".jpg", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
        level = csv_df.iloc[row_number].at['level']
        targets[i] = level

    encodings = torch.zeros((data_size, size_latent_space))
    print("Generate encodings...")
    with torch.no_grad():
        for i, (data,) in tqdm(enumerate(test_data)):
            data = data.to(device)
            if mode == Mode.vq_vae_2:
                indices_top, indices_bottom = vq_vae.encode((data))
                indices = torch.cat((indices_top, indices_bottom), dim=0).view(-1, size_latent_space)
            else:
                indices = vq_vae.encode((data))

            encodings[i:i + data.size(0)] = indices.reshape(data.size(0), size_latent_space)

    encodings = encodings.detach().numpy()

    # U-Map Visualization
    clusterable_embedding = UMAP(
        n_neighbors=80,
        min_dist=0.0,
        n_components=2,
        #random_state=42,
    ).fit_transform(encodings)

    threshold = 3
    z = np.abs(stats.zscore(clusterable_embedding))
    pos_to_delete, = np.where(np.amax(z, axis=1) > threshold)

    print(f"\n{pos_to_delete.shape[0]} outliers:")
    for i in pos_to_delete:
        print(jpg_list[i])

    encodings = np.delete(encodings, pos_to_delete, axis=0)
    targets = np.delete(targets, pos_to_delete, axis=0)

    clusterable_embedding = UMAP(
        n_neighbors=80,
        min_dist=0.0,
        n_components=2,
        #random_state=42,
    ).fit_transform(encodings)

    plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
                c=colormap[targets],
                s=0.5
                )
    plt.legend(handles=patches)
    plt.title(f"UMAP-Visualization - latents", fontsize=13, fontweight='bold')
    plt.savefig(f"{network_dir}/visualizations/umap_visualization_latents.png")
    plt.show()
    plt.close()

    data = [[] for _ in range(levels)]
    print("Generate data for histograms...")
    for i, jpg in tqdm(enumerate(jpg_list)):
        saved_jpg = jpg
        jpg = jpg.replace("_flipped", "")
        jpg = jpg.replace(".jpeg", "")
        jpg = jpg.replace(".jpg", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
        level = csv_df.iloc[row_number].at['level']
        data[level].append(io.imread(img_folder + saved_jpg))

    """
    Split test data in specific levels and build histograms over these groups.
    """
    histograms = np.zeros((levels, num_emb))
    print("Generate histograms...")
    for j, level_data in tqdm(enumerate(data)):
        level_data = DataLoader(torch.tensor(level_data), batch_size=batch_size)

        if mode == Mode.vq_vae_2:
            for i, d in enumerate(level_data):
                d = d.permute(0, 3, 1, 2).float().to(device)
                indices_bottom = vq_vae.encode(d)
                indices_bottom = indices_bottom.to("cpu").detach().numpy().astype(np.uint8)

                for index in indices_bottom.ravel():
                    histograms[j][index] += 1
        else:
            with torch.no_grad():
                for i, d in enumerate(level_data):
                    d = d.permute(0, 3, 1, 2).float().to(device)
                    indices = vq_vae.encode(d)

                    indices = indices.to("cpu").detach().numpy().astype(np.int32)

                    for index in indices.ravel():
                        histograms[j][index] += 1

                # Normalize histogram and sort indices of embedded vectors regarding best order
                histograms[j] = np.divide(histograms[j], histograms[j].sum())[best_order]

    hist_intersection = np.amin(histograms, axis=0)

    for j, hist in enumerate(histograms):
        plt.hist(hist, bins=num_emb)
        plt.title(f"percentaged frequencies - \'{disease_states[j]}\'",
                  fontsize=13,
                  fontweight='bold'
                  )
        plt.xlabel("index")
        plt.ylabel("ratio of embedded vector")
        plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}.png")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        plt.hist(np.subtract(hist, hist_intersection), bins=num_emb)
        plt.title(f"percentaged frequencies - \'{disease_states[j]}\' - difference",
                  fontsize=13,
                  fontweight='bold'
                  )
        plt.xlabel("index")
        plt.ylabel("ratio of embedded vector")
        plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}_diff.png")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    # Plot overlap of histograms
    plt.figure(figsize=(12, 12))
    for i, hist in enumerate(histograms):
        plt.hist(hist, bins=num_emb, alpha=0.5, label=disease_states[i])

    plt.title("multiple histograms of all disease states",
              fontsize=13,
              fontweight='bold')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(f"{network_dir}/histograms/overlap_of_histogram.png")
    plt.close()



