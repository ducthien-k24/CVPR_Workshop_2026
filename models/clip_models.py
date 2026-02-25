from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

# class CLIPModel(nn.Module):
#     def __init__(self, name, num_classes=1):
#         super(CLIPModel, self).__init__()

#         self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
#         self.fc = nn.Linear( CHANNELS[name], num_classes )
 

#     def forward(self, x, return_feature=False):
#         features = self.model.encode_image(x) 
#         if return_feature:
#             return features
#         return self.fc(features)

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes = 1, num_heads = 1, feature_dropout = 0.0):
        super(CLIPModel, self).__init__()
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.num_heads = num_heads
        self.feature_dropout = feature_dropout
        
        in_dims = CHANNELS[name]
        
        # N heads (each is a linear probe)
        self.fcs = nn.ModuleList([nn.Linear(in_dims, num_classes) for _ in range(self.num_heads)])
        
        # Feature dropout to diversify heads (applies during training)
        self._feat_drop = nn.Dropout(p = self.feature_dropout) if self.feature_dropout > 0 else None
        
    def forward(self, x, return_feature = False, return_all_logits = False):
        features = self.model.encode_image(x)
        
        if return_feature:
            return features
        
        if self._feat_drop is not None:
            features = self._feat_drop(features)

        # logits from each head: shape [M, B, C]
        logits = torch.stack([fc(features) for fc in self.fcs], dim=0)

        if return_all_logits:
            return logits  # useful for debugging/analysis

        # ensemble by averaging logits (recommended)
        return logits.mean(dim=0)
        
        