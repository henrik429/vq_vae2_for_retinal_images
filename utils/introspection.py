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
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.decomposition._pca import PCA
from scipy.cluster import hierarchy
from scipy import stats
from torchvision.utils import make_grid, save_image
import seaborn as sns

from utils.models import Mode
import warnings

warnings.simplefilter('ignore')


def visualize_latent_space(img_folder, csv, vae,
                           max_degree, network_dir,
                           num_emb,
                           emb_dim,
                           z_dim=32,
                           batch_size=5,
                           mode=Mode.vq_vae,
                           size_latent_space=None,
                           device="cpu",
                           levels=5):
    """
    Visualizes the latent space of a VQ-VAE.
    """
    jpg_list = listdir(img_folder)
    csv_df = read_csv(csv)
    try:
        jpg_list.remove(".snakemake_timestamp")
    except ValueError:
        pass
    data_size: int = len(jpg_list)

    colors = ['navy', 'darkorange', 'red', 'limegreen', 'turquoise', 'firebrick',
              'indigo', 'darkgreen', 'cornflowerblue', 'sienna']
    colormap = np.array([colors[c] for c in range(levels)])

    disease_states = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

    patches = []
    for i in range(levels):
        patches.append(mpatches.Patch(color=colormap[i], label=f'cluster {i}'))

    makedirs(f'{network_dir}/visualizations/', exist_ok=True)
    makedirs(f'{network_dir}/histograms/', exist_ok=True)

    # Numbers in angles have to be in a specific order
    angles = [x for x in range(-max_degree, -9)]
    angles.extend([x for x in range(10, max_degree + 1)])
    angles.extend([x for x in range(-9, 10)])

    data = []
    targets = []
    print("Load test data and generate their targets...")
    for idx, jpg in tqdm(enumerate(jpg_list)):
        original_jpg = jpg
        jpg = jpg.replace("_flipped", "")
        jpg = jpg.replace(".jpeg", "")
        jpg = jpg.replace(".jpg", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
        level = csv_df.iloc[row_number].at['level']
        targets.append(level)
        im = io.imread(img_folder + original_jpg)
        data.append(im)

    test_data = torch.tensor(data).permute(0, 3, 1, 2).float()
    targets = np.asarray(targets)
    data_size = targets.shape[0]
    n_batches = data_size//batch_size

    if mode == Mode.vq_vae_2:
        num_emb_bottom = num_emb["bottom"]
        emb_dim_bottom = emb_dim["bottom"]
        num_emb_top = num_emb["top"]
        emb_dim_top = emb_dim["top"]
        encodings_bottom = torch.zeros((data_size, size_latent_space["bottom"]*emb_dim_bottom))
        indices_bottom = torch.zeros((data_size, size_latent_space["bottom"]))
        encodings_top = torch.zeros((data_size, size_latent_space["top"]*emb_dim_top))
        indices_top = torch.zeros((data_size, size_latent_space["top"]))
    elif mode == Mode.vq_vae_2:
        encodings = torch.zeros((data_size, size_latent_space*emb_dim))
        indices = torch.zeros((data_size, size_latent_space))
    elif mode == Mode.vae:
        encodings = torch.zeros((data_size, z_dim))

    ######################################################
    # Visualization of the latents of a VQ-VAE
    ######################################################

    print("Generate encodings...")
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            data = test_data[i*batch_size:(i+1)*batch_size].to(device)
            if mode == Mode.vq_vae_2:
                z_q_bottom, z_q_top, indices_bottom_batch, indices_top_batch = vae.encode((data))

                encodings_bottom[i:i + data.size(0)] = z_q_bottom.reshape(data.size(0), size_latent_space["bottom"] * emb_dim_bottom)
                indices_bottom[i:i + data.size(0)] = indices_bottom_batch.reshape(data.size(0), size_latent_space["bottom"])
                encodings_top[i:i + data.size(0)] = z_q_top.reshape(data.size(0), size_latent_space["top"] * emb_dim_top)
                indices_top[i:i + data.size(0)] = indices_top_batch.reshape(data.size(0), size_latent_space["top"])
            elif mode == Mode.vq_vae:
                z_q, indices_batch = vae.encode((data))
                encodings[i:i + data.size(0)] = z_q.reshape(data.size(0), size_latent_space * emb_dim)
                indices[i:i + data.size(0)] = indices_batch.reshape(data.size(0), size_latent_space)
            elif mode == Mode.vae:
                encodings[i:i + data.size(0)] = vae.encode(data)

            if i < 1:
                reconstruction = vae(data).cpu().detach()
                grid = make_grid(torch.cat((data.cpu()[0:10], reconstruction[0:10]), dim=0), nrow=10)
                save_image(grid, f"{network_dir}/test_{i}.png", normalize=True)

    if mode == Mode.vq_vae_2:
        encodings_bottom = encodings_bottom.cpu().detach().numpy()
        indices_bottom = indices_bottom.cpu().detach().numpy()
        encodings_top = encodings_top.cpu().detach().numpy()
        indices_top = indices_top.cpu().detach().numpy()
    else:
        encodings = encodings.cpu().detach().numpy()

    patches = []
    for i in range(levels):
        patches.append(mpatches.Patch(color=colormap[i], label=f'{disease_states[i]}'))

    makedirs(f"{network_dir}/outlier", exist_ok=True)
    for m in range(2):
        """
        m == 0: encodings = z_q
        m == 1: encodings = indices
        """
        for n_neighbors in [20, 50, 150]:
            if mode == Mode.vq_vae_2:
                if m == 1:
                    encodings_bottom = indices_bottom
                    encodings_top = indices_top

                # U-Map Visualization
                clusterable_embedding = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0,
                    n_components=2,
                ).fit_transform(encodings_bottom)

                threshold = 3
                z = np.abs(stats.zscore(clusterable_embedding))
                pos_to_delete, = np.where(np.amax(z, axis=1) > threshold)

                print(f"\n{pos_to_delete.shape[0]} outliers:")
                for i in pos_to_delete:
                    os.system(f"cp {img_folder}/{jpg_list[i]} {network_dir}/outlier/{jpg_list[i]}")
                    print(jpg_list[i])

                encodings_bottom = np.delete(encodings_bottom, pos_to_delete, axis=0)
                encodings_top = np.delete(encodings_top, pos_to_delete, axis=0)
                indices_bottom = np.delete(indices_bottom, pos_to_delete, axis=0)
                indices_top = np.delete(indices_top, pos_to_delete, axis=0)
                targets = np.delete(targets, pos_to_delete, axis=0)

                clusterable_embedding = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0,
                    n_components=2,
                ).fit_transform(encodings_bottom)

                plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
                            c=colormap[targets],
                            s=0.4
                            )
                plt.legend(handles=patches)
                if m == 0:
                    plt.title(f"UMAP-Visualization - {n_neighbors} - encodings - bottom - latents",
                              fontsize=13,
                              fontweight='bold')
                    plt.savefig(
                        f"{network_dir}/visualizations/umap_visualization_encodings_bottom_latents_{n_neighbors}.png")
                else:
                    plt.title(f"UMAP-Visualization - {n_neighbors} - indices - bottom - latents",
                              fontsize=13,
                              fontweight='bold')
                    plt.savefig(
                        f"{network_dir}/visualizations/umap_visualization_indices_bottom_latents_{n_neighbors}.png")
                plt.close()

                # U-Map Visualization
                clusterable_embedding = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0,
                    n_components=2,
                ).fit_transform(encodings_top)

                threshold = 3
                z = np.abs(stats.zscore(clusterable_embedding))
                pos_to_delete, = np.where(np.amax(z, axis=1) > threshold)

                print(f"\n{pos_to_delete.shape[0]} outliers:")
                for i in pos_to_delete:
                    os.system(f"cp {img_folder}/{jpg_list[i]} {network_dir}/outlier/{jpg_list[i]}")
                    print(jpg_list[i])

                encodings_bottom = np.delete(encodings_bottom, pos_to_delete, axis=0)
                encodings_top = np.delete(encodings_top, pos_to_delete, axis=0)
                indices_bottom = np.delete(indices_bottom, pos_to_delete, axis=0)
                indices_top = np.delete(indices_top, pos_to_delete, axis=0)
                targets = np.delete(targets, pos_to_delete, axis=0)

                clusterable_embedding = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0,
                    n_components=2,
                ).fit_transform(encodings_top)

                plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
                            c=colormap[targets],
                            s=0.4
                            )
                plt.legend(handles=patches)
                if m == 0:
                    plt.title(f"UMAP-Visualization - {n_neighbors} - encodings - top - latents",
                              fontsize=13,
                              fontweight='bold')
                    plt.savefig(
                        f"{network_dir}/visualizations/umap_visualization_encodings_bottom_latents_{n_neighbors}.png")
                else:
                    plt.title(f"UMAP-Visualization - {n_neighbors} - indices - top - latents",
                              fontsize=13,
                              fontweight='bold')
                    plt.savefig(
                        f"{network_dir}/visualizations/umap_visualization_indices_bottom_latents_{n_neighbors}.png")
                plt.close()

            else:
                if m == 1 and mode == Mode.vae:
                    break
                elif m == 1 and mode == Mode.vq_vae:
                    encodings = indices

                # U-Map Visualization
                clusterable_embedding = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0,
                    n_components=2,
                ).fit_transform(encodings)

                threshold = 3
                z = np.abs(stats.zscore(clusterable_embedding))
                pos_to_delete, = np.where(np.amax(z, axis=1) > threshold)

                print(f"\n{pos_to_delete.shape[0]} outliers:")
                for i in pos_to_delete:
                    os.system(f"cp {img_folder}/{jpg_list[i]} {network_dir}/outlier/{jpg_list[i]}")
                    print(jpg_list[i])

                encodings = np.delete(encodings, pos_to_delete, axis=0)
                if mode == Mode.vq_vae:
                    indices = np.delete(indices, pos_to_delete, axis=0)
                targets = np.delete(targets, pos_to_delete, axis=0)

                clusterable_embedding = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=2,
                ).fit_transform(encodings)

                plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
                            c=colormap[targets],
                            s=0.4
                            )
                plt.legend(handles=patches)
                if m == 0:
                    plt.title(f"UMAP-Visualization - {n_neighbors} - encodings - latents",
                              fontsize=13,
                              fontweight='bold')
                    plt.savefig(f"{network_dir}/visualizations/umap_visualization_encodings_latents_{n_neighbors}.png")
                else:
                    plt.title(f"UMAP-Visualization - {n_neighbors} - indices - latents",
                              fontsize=13,
                              fontweight='bold')
                    plt.savefig(f"{network_dir}/visualizations/umap_visualization_indices_latents_{n_neighbors}.png")
                plt.show()
                plt.close()

    if not mode == Mode.vae:
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
    if mode == Mode.vq_vae_2:
        histograms = {"bottom": np.zeros((levels, num_emb_bottom), dtype=np.float64),
                      "top": np.zeros((levels, num_emb_top), dtype=np.float64)}

        for stage in ["bottom", "top"]:
            # Plot overlap of histograms
            plt.figure(figsize=(12, 12))
            plt.title(f"Overlap of Histograms of all Disease States - {stage}",
                      fontsize=13,
                      fontweight='bold')
            plt.xlabel("Index")
            plt.ylabel("Ratio of Embedded Vector")
            plt.legend(handles=patches, loc='upper right')
            plt.grid(alpha=0.4)

            print("Generate histograms...")
            for j, level_data in tqdm(enumerate(data)):
                level_data = torch.tensor(level_data)
                counts = np.zeros(num_emb[stage])
                level_data = DataLoader(level_data, batch_size=batch_size)
                with torch.no_grad():
                    for i, d in enumerate(level_data):
                        d = d.permute(0, 3, 1, 2).float().to(device)
                        _, _, indices_bottom, indices_top = vae.encode(d)
                        if stage == "bottom":
                            indices = indices_bottom.cpu().detach().numpy().astype(np.uint16).ravel()
                        else:
                            indices = indices_top.cpu().detach().numpy().astype(np.uint16).ravel()

                        counts_to_add = np.bincount(indices, minlength=num_emb[stage])
                        counts = np.add(counts, counts_to_add)

                # Normalize counts
                histograms[stage][j] = np.divide(counts, np.sum(counts))
                plt.bar(np.arange(num_emb[stage]), histograms[stage][j], color=colormap[j])

            plt.savefig(f"{network_dir}/histograms/overlap_of_histograms_{stage}.png")
            plt.close()

            hist_intersection = np.amin(histograms[stage], axis=0)

            for j, hist in enumerate(histograms[stage]):
                # Sort indices of embedded vectors regarding best order
                plt.bar(np.arange(num_emb[stage]), hist)
                plt.title(f"Percentaged Frequencies - \'{disease_states[j]}\' - {stage}",
                          fontsize=13,
                          fontweight='bold'
                          )
                plt.xlabel("Index")
                plt.ylabel("Ratio of Embedded Vector")
                plt.grid(alpha=0.4)
                plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}_{stage}.png")
                plt.show(block=False)
                plt.pause(2)
                plt.close()

                plt.bar(np.arange(num_emb[stage]), np.subtract(hist, hist_intersection))
                plt.title(f"Percentaged Frequencies - Difference - \'{disease_states[j]}\' - {stage}",
                          fontsize=13,
                          fontweight='bold'
                          )
                plt.grid(alpha=0.4)
                plt.xlabel("Index")
                plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}_diff_{stage}.png")
                plt.show(block=False)
                plt.pause(2)
                plt.close()

                if j != 0:
                    plt.bar(np.arange(num_emb[stage]), np.subtract(hist, histograms[stage][0]))
                    plt.title(f"Percentaged Frequencies - Difference to No DR group - \'{disease_states[j]}\' -  {stage}",
                              fontsize=13,
                              fontweight='bold'
                              )
                    plt.grid(alpha=0.4)
                    plt.xlabel("Index")
                    plt.ylabel("Ratio of Embedded Vector")
                    plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}_diff_to_no_DR_{stage}.png")
                    plt.show(block=False)
                    plt.pause(2)
                    plt.close()

    elif mode == Mode.vq_vae:
        histograms = np.zeros((levels, num_emb), dtype=np.float64)

        # Plot overlap of histograms
        plt.figure(figsize=(12, 12))
        plt.title("Overlap of Histograms of all Disease States",
                  fontsize=13,
                  fontweight='bold')
        plt.xlabel("Index")
        plt.ylabel("Ratio of Embedded Vector")
        plt.legend(handles=patches, loc='upper right')
        plt.grid(alpha=0.4)

        print("Generate histograms...")
        for j, level_data in tqdm(enumerate(data)):
            level_data = torch.tensor(level_data)
            counts = np.zeros(num_emb)
            level_data = DataLoader(level_data, batch_size=batch_size)
            with torch.no_grad():
                for i, d in enumerate(level_data):
                    d = d.permute(0, 3, 1, 2).float().to(device)

                    if mode == Mode.vq_vae_2:
                        _, _, indices_bottom, indices_top = vae.encode(d)
                        indices_bottom = indices_bottom.cpu().detach().numpy().astype(np.uint16).ravel()
                        counts_to_add = np.bincount(indices_bottom, minlength=num_emb)
                        counts = np.add(counts, counts_to_add)
                    else:
                        _, indices = vae.encode(d)
                        indices = indices.cpu().detach().numpy().astype(np.uint16).ravel()
                        counts_to_add = np.bincount(indices, minlength=num_emb)
                        counts = np.add(counts, counts_to_add)

            # Normalize counts
            histograms[j] = np.divide(counts, np.sum(counts))
            plt.bar(np.arange(num_emb), histograms[j])

        plt.savefig(f"{network_dir}/histograms/overlap_of_histograms.png")
        plt.close()

        hist_intersection = np.amin(histograms, axis=0)

        for j, hist in enumerate(histograms):
            plt.bar(np.arange(num_emb), hist)
            plt.title(f"Percentaged Frequencies - \'{disease_states[j]}\'",
                      fontsize=13,
                      fontweight='bold'
                      )
            plt.xlabel("Index")
            plt.ylabel("Ratio of Embedded Vector")
            plt.grid(alpha=0.4)
            plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}.png")
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            plt.bar(np.arange(num_emb), np.subtract(hist, hist_intersection))
            plt.title(f"Percentaged Frequencies - Difference - \'{disease_states[j]}\' ",
                      fontsize=13,
                      fontweight='bold'
                      )
            plt.grid(alpha=0.4)
            plt.xlabel("Index")
            plt.ylabel("Ratio of Embedded Vector")
            plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}_diff.png")
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            if j != 0:
                plt.bar(np.arange(num_emb), np.subtract(hist, histograms[0]))
                plt.title(f"Percentaged Frequencies - Difference to No DR group - \'{disease_states[j]}\'",
                          fontsize=13,
                          fontweight='bold'
                          )
                plt.grid(alpha=0.4)
                plt.xlabel("Index")
                plt.ylabel("Ratio of Embedded Vector")
                plt.savefig(f"{network_dir}/histograms/histogram_{disease_states[j]}_diff_to_no_DR.png")
                plt.show(block=False)
                plt.pause(2)
                plt.close()


