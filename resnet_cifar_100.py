import torch
import torch.nn as nn

# TODOS:
# rename output_channels to "intermediate channels"
# switch to doc comments.
# split out to separate files.
# perform unit testing.
# fix the awkward input_channels situation.


def ResNet50(img_channels=3, num_classes=100):
    return ResNet([3, 4, 6, 3], img_channels, num_classes)

def test():
    net = ResNet50()
    inp = torch.randn(2, 3, 224, 224)
    output = net(inp).to('cuda')
    print(output.shape())

test()
