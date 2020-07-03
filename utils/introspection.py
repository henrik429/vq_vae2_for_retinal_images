#from umap import UMAP
from sklearn.manifold._t_sne import TSNE
import matplotlib.pyplot as plt
from pandas import read_csv
import torch
from os import listdir, makedirs
from tqdm import tqdm
import numpy as np
import matplotlib.patches as mpatches
from skimage import io
from torch.utils.data import DataLoader

from utils.models import Mode
import warnings
warnings.simplefilter('ignore')


def visualize_latent_space(test_data, img_folder, csv, vq_vae, writer,
                           max_degree, network_dir, network_name,
                           num_emb,
                           batch_size=5,
                           mode = Mode.vq_vae,
                           size_latent_space=None,
                           device="cpu",
                           levels=5):
    """
    Visualizes the latent space of a VQ-VAE using class method 'plot'.
    """
    jpg_list = listdir(img_folder)
    csv_df = read_csv(csv)
    data_size:int = len(jpg_list)

    path = f'{network_dir}/{network_name}.pth'
    torch.save(vq_vae.state_dict(), path)

    targets = np.zeros(data_size,  dtype=np.uint8)

    # Numbers in angles have to be in a specific order
    angles = [x for x in range(-max_degree, -9)]
    angles.extend([x for x in range(10, max_degree+1)])
    angles.extend([x for x in range(-9, 10)])

    print("Generate targets...")
    for i, jpg in tqdm(enumerate(jpg_list)):
        jpg = jpg.replace("_flipped", "")
        jpg = jpg.replace(".jpeg", "")
        jpg = jpg.replace(".jpg", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
        level = csv_df.iloc[row_number].at['level']

        targets[i] = level

    colors = ['navy', 'darkorange', 'red',  'limegreen', 'turquoise',  'firebrick',
              'indigo', 'darkgreen', 'cornflowerblue', 'sienna' ]
    colormap = np.array([colors[c] for c in range(levels)])

    disease_states = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

    makedirs(f'{network_dir}/visualizations/', exist_ok=True)

    if mode == Mode.vq_vae_2:
        size_latent_space = size_latent_space["top"] + size_latent_space["bottom"]

    encodings = torch.zeros((data_size, size_latent_space))
    print("Generate encodings...")
    for i, (data,) in tqdm(enumerate(test_data)):
        data = data.to(device)
        if mode == Mode.vq_vae_2:
            indices_top, indices_bottom = vq_vae.encode((data))
            indices = torch.cat((indices_top, indices_bottom), dim=0).view(-1, size_latent_space)
        else:
            indices = vq_vae.encode((data))

        encodings[i:i+data.size(0)] = indices.reshape(data.size(0), size_latent_space)

    writer.add_embedding(encodings, metadata=targets, global_step=".")

    # tSNE Visualization of the encoded latent vector
    tsne = TSNE(random_state=123).fit_transform(encodings.detach().numpy())

    plt.scatter(tsne[:, 0], tsne[:, 1],
                c=colormap[targets],
                s=10
    )

    patches = []
    for i, disease_state in enumerate(disease_states):
        patches.append(mpatches.Patch(color=colormap[i], label=disease_state))

    plt.legend(handles=patches)
    plt.title(f"tSNE-Visualization\n", fontsize=16, fontweight='bold')
    plt.savefig(f"{network_dir}/visualizations/tsne_visualization.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    try:
        #import umap
        # U-Map Visualization
        clusterable_embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(encodings)

        plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
                    c=colormap[targets],
                    s=1
                    )
        plt.legend(handles=patches)

        plt.title(f"UMAP-Visualization\n", fontsize=16, fontweight='bold')
        plt.savefig(f"{network_dir}/visualizations/umap_visualization.png")
        plt.show()
        plt.close()
    except:
        pass

    del test_data
    del targets
    del encodings

    data = [[] for _ in range(levels)]
    for i, jpg in tqdm(enumerate(jpg_list)):
        saved_jpg = jpg
        jpg = jpg.replace("_flipped", "")
        jpg = jpg.replace(".jpeg", "")
        jpg = jpg.replace(".jpg", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
        level = csv_df.iloc[row_number].at['level']
        data[level].append(io.imread(img_folder+saved_jpg))

    """
    Split test data in specific levels and build histograms over these groups.
    """
    for j, level_data in enumerate(data):
        level_data = DataLoader(torch.tensor(level_data), batch_size=batch_size)

        if mode == Mode.vq_vae_2:
            bins_top = np.zeros(num_emb["top"])
            bins_bottom = np.zeros(num_emb["bottom"])
            for i, d in enumerate(level_data):
                d = d.permute(0, 3, 1, 2).float().to(device)
                indices_top, indices_bottom = vq_vae.encode(d)
                indices_bottom = indices_bottom.detach().numpy().astype(np.uint8)
                indices_top = indices_top.detach().numpy().astype(np.uint8)

                for index in indices_top.ravel():
                    bins_top[index] += 1
                for index in indices_bottom.ravel():
                    bins_bottom[index] += 1

            plt.bar(np.arange(0, num_emb["top"]), bins_top)
            plt.title(f"Histogram - \'{disease_states[j]}\' - top level ",
                      fontsize=16,
                      fontweight='bold'
            )
            plt.savefig(f"{network_dir}/visualizations/histogram_{disease_states[j]}_top.png")
            plt.xlabel("Index")
            plt.ylabel("Number embedded vectors")
            plt.show(block=False)
            plt.pause(3)
            plt.close()

            plt.bar(np.arange(0, num_emb["bottom"]), bins_bottom)
            plt.title(f"Histogram - \'{disease_states[j]}\' - bottom level ",
                      fontsize=16,
                      fontweight='bold'
            )
            plt.savefig(f"{network_dir}/visualizations/histogram_{disease_states[j]}_bottom.png")
            plt.xlabel("Index")
            plt.ylabel("Number embedded vectors")
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        else:
            bins = np.zeros(num_emb)
            for i, d in enumerate(level_data):
                d = d.permute(0, 3, 1, 2).float().to(device)
                indices = vq_vae.encode(d)
                indices = indices.detach().numpy().astype(np.uint8)

                for index in indices.ravel():
                    bins[index] += 1

            plt.bar(np.arange(0, num_emb), bins)
            plt.title(f"Histogram - \'{disease_states[j]}\'",
                      fontsize=16,
                      fontweight='bold'
            )
            plt.xlabel("Index")
            plt.ylabel("Number embedded vectors")
            plt.savefig(f"{network_dir}/visualizations/histogram_{disease_states[j]}.png")
            plt.show(block=False)
            plt.pause(2)
            plt.close()




