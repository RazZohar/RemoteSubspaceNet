
# imports for creating codebook
import torch
import torch.nn as nn

from sklearn.cluster import KMeans

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
    return torch.Tensor(kmeans.cluster_centers_)


def init_weights_lbg(module, codebook):
    weight_tensor = torch.Tensor(codebook)
    module.weight = nn.Parameter(weight_tensor)
    return module.weight