import unittest
import torch
from res_net_block import ResNetBlock

def BasicBlock(layers):
    return ResNetBlock(64, 64, layers = layers)

def StaticBlock(layers):
    return ResNetBlock(64, 16, layers = layers)

def StridedBlock(layers):
    return ResNetBlock(64, 16, stride = 2, layers = layers)

# torch tensors are row-major.  Who the fuck thought that was a good idea?
# the order is as follows:  batch index, channels, image column, image row.

INPUT_TENSOR = torch.randn(4, 64, 224, 224)
class TestResNetBlock(unittest.TestCase):
    # general formula:
    # (imgs, chan, w, h) -> (imgs, chan * 4, w / stride, h / stride)
    # internal layer count is... whatever you want it to be!

    def _assertSize(self, tensor, sizespec):
        self.assertEqual(tensor.size(), torch.Size(sizespec))

    def test_layer_1(self):
        net = BasicBlock(1)
        # layer 1 (conv1) doesn't change the tensor size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    def test_layer_2(self):
        net = BasicBlock(2)
        # layer 2 (batchnorm1) doesn't change the tensor size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    def test_layer_3(self):
        net = BasicBlock(3)
        # layer 3 (relu) doesn't change the tensor size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    def test_layer_4(self):
        net = BasicBlock(4)
        # layer 4 (conv2) doesn't change the tensor size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    def test_layer_5(self):
        net = BasicBlock(5)
        # layer 5 (batchnorm2) doesn't change the tensor size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    def test_layer_6(self):
        net = BasicBlock(6)
        # layer 6 (relu) doesn't change the tensor size.
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    def test_layer_7(self):
        net = BasicBlock(7)
        # layer 7 (conv3) makes there be 4x more channels
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 256, 224, 224])

    def test_layer_8(self):
        # layer 8 (batchnorm3) doesn't change the tensor size.
        net = BasicBlock(8)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 256, 224, 224])

    def test_layer_9(self):
        # layer 9 (addition origin) in the basic block case, must
        # have a supplied downsampler
        net = BasicBlock(9)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 256, 224, 224])

    def test_layer_10(self):
        # layer 10 (addition origin) in the basic block case, must
        # have a supplied downsampler
        net = BasicBlock(10)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 256, 224, 224])

    # tests against the case where you use a static block.  A static
    # block has # of internal layers equal to input_layers / 4.
    def test_static_layer_1(self):
        net = StaticBlock(1)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 16, 224, 224])

    def test_static_layer_6(self):
        net = StaticBlock(6)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 16, 224, 224])

    # between layers 6 and 7 is when the channel expansion happens.

    def test_static_layer_7(self):
        net = StaticBlock(7)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    def test_static_layer_10(self):
        net = StaticBlock(10)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 224, 224])

    # tests against the case where you use a strided block.  A strided
    # block should naturally shrink the size of the image.
    def test_strided_layer_1(self):
        net = StridedBlock(1)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 16, 224, 224])

    def test_strided_layer_3(self):
        net = StridedBlock(3)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 16, 224, 224])

    # between layers 3 and 4 is when the size shrinks.

    def test_strided_layer_4(self):
        net = StridedBlock(4)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 16, 112, 112])

    def test_strided_layer_6(self):
        net = StridedBlock(6)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 16, 112, 112])

    # between layers 6 and 7 is when the channels are increased.

    def test_strided_layer_7(self):
        net = StridedBlock(7)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 112, 112])

    def test_strided_layer_10(self):
        net = StridedBlock(10)
        output = net(INPUT_TENSOR).to("cpu")
        self._assertSize(output, [4, 64, 112, 112])