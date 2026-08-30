import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.device = device
    def Uniformity_loss(self,z, delta):
        class SGE:
            def __init__(self, delta):
                self.delta = delta
            def kernel(self, z):
                cosine_sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

                return torch.exp(cosine_sim/ self.delta)
            def score(self, z):
                kernel_matrix = self.kernel(z)
                grad_kernel = torch.autograd.grad(kernel_matrix.sum(), z, create_graph=True)[0]
                return grad_kernel
        sge = SGE(delta)
        score = sge.score(z).detach()
        return (score * z).sum(-1).mean()

    def alignment_loss(self, Z_1, Z_2):
        cos_sim = F.cosine_similarity(Z_1, Z_2)
        loss = cos_sim.mean()/self.batch_size
        return loss

