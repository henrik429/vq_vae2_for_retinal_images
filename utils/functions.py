import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn


class Vector_Quantization(Function):
    """
    Applies the Vector Quantization Operation within a VQ-VAE
    :param z_e: output of the encoder
    :param emb: embedded space or codebook
    :return z_q: the quantized output of the encoder (and therefore the input of the decoder)
    :return z: indices of the embeddings which have the closest distance to the vectors of
             the output of the encoder.
    """
    @staticmethod
    def forward(ctx, z_e_flatten, emb, shape_z_e):
        # Compute the distances to the embedded vectors
        # Calculation of distances from z_e_flatten to embeddings e_j using the Euclidean Distance:
        # (z_e_flatten - emb)^2 = z_e_flatten^2 + emb^2 - 2 emb * z_e_flatten
        emb_sqr = torch.sum(emb.pow(2), dim=1)
        z_e_sqr = torch.sum(z_e_flatten.pow(2), dim=1, keepdim=True)

        distances = torch.addmm(torch.add(emb_sqr, z_e_sqr), z_e_flatten, emb.t(), alpha=-2.0, beta=1.0)
        indices = torch.argmin(distances, dim=1)

        z_q_flatten = torch.index_select(emb, dim=0, index=indices)
        z_q = z_q_flatten.view(shape_z_e)

        """
        If indices should be stored as integer values, use:   
        vide supra
                
        Else:   
        Save integer values as one-hot vectors and multiply them by the embedded weights
        
        z_q = torch.matmul(z, emb)
        z_q = z_q.view(shape_z_e)
        """

        z = F.one_hot(indices, num_classes=emb.size(0)).float()

        ctx.save_for_backward(indices, emb)
        ctx.mark_non_differentiable(indices)

        return z_q, indices.float(), z

    @staticmethod
    def backward(ctx, grad_output, grad_indices, _):
        grad_input, grad_emb = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_input = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the embedded space
            emb, indices, = ctx.saved_tensors
            emb_dim = emb.size(1)

            grad_output_flatten = (grad_output.contiguous().view(-1, emb_dim))
            grad_emb = torch.zeros_like(emb)
            grad_emb.index_add_(0, indices, grad_output_flatten)

        return grad_input, grad_emb, None


vector_quantization = Vector_Quantization.apply

if __name__ == '__main__':
    dim_size = 10
    num_emb = 30
    emb = nn.Embedding(num_emb, dim_size)

    z_e = torch.randn(10,dim_size, 32, 32)
    indices, z_q = vector_quantization(z_e, emb.weight.data)

