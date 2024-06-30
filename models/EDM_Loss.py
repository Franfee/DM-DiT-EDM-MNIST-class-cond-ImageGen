
import torch


#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img, condition):
        
        # noise
        rnd_normal = torch.randn([img.shape[0], 1, 1, 1], device=img.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(img) * sigma
        
        # denoised noised
        D_yn = net(img + n, sigma, condition)
        
        # reweight loss
        loss = weight * ((D_yn - img) ** 2)

        return loss