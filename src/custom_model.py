# custom_model.py

import torch
import torchvision.models as models
import torch.nn as nn

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create the encoders
        # Encoder q
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # Initialize the key encoder parameters to the same values as the query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, im_q, im_k=None):
        # Compute query features
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        if im_k is not None:
            # Compute key features
            with torch.no_grad():
                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)
            return q, k
        return q

def load_model(model_path, model_type='resnet18'):
    if model_type == 'resnet18':
        model = models.resnet18()
    elif model_type == 'moco':
        model = MoCo(base_encoder=models.resnet18, dim=128, K=65536, m=0.999, T=0.07)
    elif model_type == 'custom':
        model = MoCo(base_encoder=models.resnet18, dim=128, K=65536, m=0.999, T=0.07)  # Adjust this line if you have a different custom model

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model
