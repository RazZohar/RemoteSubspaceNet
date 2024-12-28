import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FixedVectorQuantizer(nn.Module):
    """
    class FixedVectorQuantizer(nn.Module):
        This class implements a fixed vector quantization technique.
        its part of NN module and recieve the codebook_size parameter, dimesnsion of vector in codebook.
    """
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33):
        super(FixedVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # Initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)

        # Balancing parameter lambda for the commitment loss
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.d)

        # Use the entire codebook for quantization
        actives = self.codebook.weight

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(actives ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, actives.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.p, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

        if self.training:
            # Loss
            q_latent_loss = torch.nn.functional.mse_loss(quantized, inputs.detach())  # Commitment loss
            e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)  # Alignment loss
            cb_loss = q_latent_loss + self.lambda_c * e_latent_loss  # Codebook loss

            # Gradient copying for the straight-through estimator
            quantized = inputs + (quantized - inputs).detach()
        else:
            cb_loss = 0

        return quantized, cb_loss



class AdaptiveVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33):
        super(AdaptiveVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)

        # Balancing parameter lambda for the commintment loss
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

        self.first = True

    def forward(self, inputs, num_vectors, prev_vecs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.d)

        quant_vecs = []
        losses = []

        for num_actives in range(int(np.log2(num_vectors))):
            actives = self.codebook.weight[:pow(2, num_actives + 1)]

            # Calculate distances
            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(actives ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, actives.t()))

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
            quant_vecs.append(quantized)

        for num_actives in range(int(np.log2(num_vectors))):
            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quant_vecs[num_actives], inputs.detach())  # commitment loss

                if num_actives == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)  # alignment loss

                elif num_actives == 1:
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)
                    prox_loss = (num_actives * self.lambda_p) * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                                           actives[:pow(2, num_actives + 1) // 2])
                else:
                    e_latent_loss = 0
                    prox_loss = self.lambda_p * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                           actives[:pow(2, num_actives + 1) // 2])  # proximity_loss

                cb_loss = q_latent_loss + self.lambda_c * e_latent_loss + prox_loss  # codebook loss

                quant_vecs[num_actives] = inputs + (quant_vecs[num_actives] - inputs).detach()  # gradient copying

            else:
                cb_loss = 0

            losses.append(cb_loss)

        return quant_vecs, losses, actives
