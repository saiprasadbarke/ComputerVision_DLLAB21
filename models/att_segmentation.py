import random
import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
try:
    from deeplab_decoder import Decoder
    from attention_layer import Attention
except ModuleNotFoundError:
    from .deeplab_decoder import Decoder
    from .attention_layer import Attention


class AttSegmentator(nn.Module):

    def __init__(self, num_classes, encoder, att_type='additive', img_size=(512, 512)):
        super().__init__()
        self.low_feat = IntermediateLayerGetter(encoder, {"layer1": "layer1"}).cuda()
        self.encoder = IntermediateLayerGetter(encoder, {"layer4": "out"}).cuda()
        # For resnet18
        encoder_dim = 512
        low_level_dim = 64
        self.num_classes = num_classes

        self.class_encoder = nn.Linear(num_classes, 512)

        self.attention_enc = Attention(encoder_dim, att_type)

        self.decoder = Decoder(2, encoder_dim, img_size, low_level_dim=low_level_dim, rates=[1, 6, 12, 18])

    def forward(self, x, v_class, out_att=False):

        raise NotImplementedError("TODO: Implement the attention-based segmentation network")
        # Write the forward pass of the model.
        # Base the model on the segmentation model and add the attention layer.
        # Be aware of the dimentions.
        x_enc, attention = self.attention_enc(x_enc, class_vec)

        if out_att:
            return segmentation, attention
        return segmentation

if __name__ == "__main__":
    from torchvision.models.resnet import resnet18
    pretrained_model = resnet18(num_classes=4).cuda()
    model = AttSegmentator(10, pretrained_model, att_type='dotprod', double_att=True).cuda()
    model.eval()
    print(model)
    image = torch.randn(1, 3, 512, 512).cuda()
    v_class = torch.randn(1, 10).cuda()
    with torch.no_grad():
        output = model.forward(image, v_class)
    print(output.size())
