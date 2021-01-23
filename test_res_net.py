import unittest
import torch
from res_net import ResNet

def BasicResNet(layers, input_channels = 3, num_classes = 10):
    return ResNet([3, 4, 6, 3], input_channels = input_channels, num_classes = 10, layers=layers)

INPUT_TENSOR = torch.randn(4, 3, 224, 224)
class TestResNet(unittest.TestCase):
    def test_layerdef_checked(self):
        with self.assertRaises(AssertionError):
            ResNet([1, 2, 3], input_channels = 3, num_classes = 10)

    def _assertSize(self, tensor, sizespec):
        self.assertEqual(tensor.size(), torch.Size(sizespec))

    def test_layer_1(self):
        net = BasicResNet(1)
        # layer 1 takes the input tensor and changes it to have 64 channels.
        # it also strides over the image and downsizes it by a factor of 2.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 112, 112])

    def test_layer_2(self):
        net = BasicResNet(2)
        # layer 2 doesn't change the output size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 112, 112])

    def test_layer_3(self):
        net = BasicResNet(3)
        # layer 3 doesn't change the output size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 112, 112])

    def test_layer_4(self):
        net = BasicResNet(4)
        # layer 4 performs another stride-2 shrink.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 56, 56])

    def test_layer_5(self):
        net = BasicResNet(5)
        # layer 5 performs an expansion.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 256, 56, 56])

    def test_layer_6(self):
        net = BasicResNet(6)
        # layer 6 performs another stride-2 shrink.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 512, 28, 28])

    def test_layer_7(self):
        net = BasicResNet(7)
        # layer 7 performs another stride-2 shrink.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 1024, 14, 14])

    def test_layer_8(self):
        net = BasicResNet(8)
        # layer 8 performs another stride-2 shrink.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 2048, 7, 7])

    def test_layer_9(self):
        net = BasicResNet(9)
        # layer 9 average-pools all of the activations.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 2048, 1, 1])

    def test_layer_10(self):
        net = BasicResNet(10)
        # layer 10 collapses the last dimension
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 2048])

    def test_layer_11(self):
        net = BasicResNet(11)
        # layer 11 gives a final fully connected codebook
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 10])