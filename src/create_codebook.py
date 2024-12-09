
# imports for creating codebook
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import numpy as np
from matplotlib.pyplot import plot as plt

def get_n_batches(data_loader, num_batches):
    collected_batches = 0
    all_samples = []
    all_labels = []
    for samples, labels in data_loader:
        all_samples.append(samples)
        all_labels.append(labels)
        collected_batches += 1
        if collected_batches == num_batches:
            break
    all_samples_tensor = torch.cat(all_samples, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    return all_samples_tensor, all_labels_tensor


def add_figure_encoder(flatten_ze, cluster_centers_):
    # Initialize PCA with 2 components
    pca = PCA(n_components=2)
    reduced_data_encoder = pca.fit_transform(flatten_ze)
    reduced_data_codebook = pca.fit_transform(cluster_centers_)
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'magenta', 'cyan', 'yellow']
    color_index = int(np.log2(len(reduced_data_codebook)))
    x = np.array([elem[0] for elem in reduced_data_encoder])
    y = np.array([elem[1] for elem in reduced_data_encoder])
    name = str(np.log2(len(reduced_data_codebook)) + 'Bits Vectors')
    train_x_vals = np.array([elem[0] for elem in reduced_data_codebook])
    train_y_vals = np.array([elem[1] for elem in reduced_data_codebook])
    plt.scatter(train_x_vals, train_y_vals, s=10, alpha=0.1, label='Train Vectors')
    plt.scatter(x, y, s=250, alpha=1, label=name, c=colors[color_index])

    # Add axis labels and a title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Scatter Plot')
    plt.grid()
    plt.legend(loc='best')
    # Show the plot
    plt.show()
    codebook_size = len(cluster_centers_)
    plt.savefig(f'scatter_plot_codebook_{codebook_size}.png', dpi=300, bbox_inches='tight')


def create_codebook_command(encoder : nn.Sequential, input_data, cb_vec_dim, num_clusters):
    print(f'Perofrm Codebook generation for VQ-VAE architecture')

    # K-Means Quantization Setup
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 11,
        "max_iter": 300,
    }

    with torch.no_grad():
        z_e = encoder(input_data)
    z_e = z_e - z_e.mean()
    flatten_ze = z_e.view(-1, cb_vec_dim)

    kmeans = KMeans(num_clusters, **kmeans_kwargs)
    kmeans.fit(flatten_ze.cpu().numpy())

    codebook_vectors = torch.Tensor(kmeans.cluster_centers_)
    
    #add_figure_encoder(z_e, kmeans.cluster_centers_)
    return codebook_vectors


def init_weights_lbg(module, codebook):
    weight_tensor = torch.Tensor(codebook)
    module.weight = nn.Parameter(weight_tensor)
    return module.weight