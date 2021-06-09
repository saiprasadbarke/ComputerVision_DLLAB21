import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
try:
    from deeplab_decoder import Decoder
except ModuleNotFoundError:
    from .deeplab_decoder import Decoder


class Segmentator(nn.Module):

    def __init__(self, num_classes, encoder, img_size=(512, 512), shallow_decoder=False):
        super().__init__()
        self.low_feat = IntermediateLayerGetter(encoder, {"layer1": "layer1"}).cuda()
        self.encoder = IntermediateLayerGetter(encoder, {"layer4": "out"}).cuda()

        # n_classes, encoder_dim, img_size, low_level_dim, rates
        self.decoder = Decoder(num_classes, 512, img_size, low_level_dim=64, rates=[1, 6, 12, 18])
        self.num_classes = num_classes

    def forward(self, x):
        self.low_feat.eval()
        self.encoder.eval()
        with torch.no_grad():
            # This is possible since gradients are not being updated
            low_level_feat = self.low_feat(x)['layer1']
            enc_feat = self.encoder(x)['out']

        segmentation = self.decoder(enc_feat, low_level_feat)

        if self.num_classes==1:
            segmentation = torch.sigmoid(segmentation)
        return segmentation

