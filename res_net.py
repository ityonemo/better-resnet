import torch
import torch.nn as nn
from res_net_block import ResNetBlock
from res_net_block import EXPANSION_FACTOR

class ResNet(nn.Module):
    """
    Initialization Parameters for a ResNet

    A ResNet has the following layouts:

    image tensor
    |> convnet |> batchnorm |> relu |> maxpool # preshaping
    # resnet blocks
    |> block layer 1
    |> block layer 2
    |> block layer 3
    |> block layer 4
    # encoding layers
    |> avgpool |> fcnn

    It ingests tensors of the form (samples x channels x w x h)
    and returns tensors of the form (samples x num_classes)

    This treatment does not perform any classification operations (such as softmax)
    on the output vectors.

    ## Block Layers
    a block layer is n `ResNetBlock` layers (see the corresponding class)
    of which the first layer is used to expand the number of channels and
    possibly perform a stride, and the remaining n-1 do not change the channels or
    the image tensor dimensions.

    The following parameters are used:
    - block_counts: how many ResNetBlocks are in each layer.  Must be a list of 4
      integers; Typically of the form `[3, N, M, 3]` where N and M are variable and
      determine the "size" of the ResNet.  ResNet 50 is [3, 4, 6, 3].  ResNet 101
      is [3, 4, 23, 3].  ResNet152 is [3, 8, 36, 3].

    - input_channels:  how many channels in your input tensor.
    - num_classes:  how many classes the result vector will encode.
    - internal_channels (default 64):  Scales how many internal channels you want
      your ResNet to pass between block layers.  The scaling will be as follows:
      - block layer 1: x4
      - block layer 2: x2
      - block layer 3: x2
      - block layer 4: x2
      so your final layer will have 32x as many channels as the input.

    - layers: for testing.
      - 'all' (default): return a full ResNet.
      - (int): return a ResNet with only the first (int) layers.
    """
    def __init__(self, block_counts, input_channels, num_classes, internal_channels = 64, layers = 'all'):
        # make sure that our block counts is a list with 4 parameters.
        assert len(block_counts) == 4
        super(ResNet, self).__init__()

        channels = list(map(lambda x: x * 64, [1, 4, 8, 16, 32]))

        # expand "color" channels to "the main channel"
        self.conv1      = self._option(1, layers, nn.Conv2d(input_channels, internal_channels, kernel_size=7, stride=2, padding=3))
        self.batchnorm1 = self._option(2, layers, nn.BatchNorm2d(internal_channels))
        self.relu       = self._option(3, layers, nn.ReLU())
        self.maxpool    = self._option(4, layers, nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        ## ResNet blocklayers
        self.blocklayer1 = self._option(5, layers, self._make_blocklayer(block_counts[0], channels[0], channels[1]))
        self.blocklayer2 = self._option(6, layers, self._make_blocklayer(block_counts[1], channels[1], channels[2], stride=2))
        self.blocklayer3 = self._option(7, layers, self._make_blocklayer(block_counts[2], channels[2], channels[3], stride=2))
        self.blocklayer4 = self._option(8, layers, self._make_blocklayer(block_counts[3], channels[3], channels[4], stride=2))

        ## cleanup and postprocessing
        self.avgpool =        self._option(9, layers, nn.AdaptiveAvgPool2d((1, 1)))
        self.reshape =        self._option(10, layers, self._final_reshape)
        self.fullyconnected = self._option(11, layers, nn.Linear(2048, num_classes))
        self.softmax =        nn.LogSoftmax(dim=1)
        self.nll_loss =       nn.NLLLoss()

    def forward(self, input, targets = None):
        ## apply the pretreatment layers, generating activations.
        activation = self.conv1(input)
        activation = self.batchnorm1(activation)
        activation = self.relu(activation)
        activation = self.maxpool(activation)
        ## apply the resnet block layers
        activation = self.blocklayer1(activation)
        activation = self.blocklayer2(activation)
        activation = self.blocklayer3(activation)
        activation = self.blocklayer4(activation)
        ## apply the postprocessing layers
        activation = self.avgpool(activation)
        activation = self.reshape(activation)
        activation = self.fullyconnected(activation)

        if targets is not None:
            logits = self.softmax(activation)
            loss = self.nll_loss(logits, targets)
            return logits, loss
        else:
            return activation

    def _make_blocklayer(self, block_count, input_channels, output_channels, stride = 1):
        # build up a layers list
        layers = []
        # first layer reconciles the size difference between the input and the output.
        layers.append(ResNetBlock(input_channels, output_channels, stride))
        for i in range(block_count - 1):
            layers.append(ResNetBlock(output_channels, output_channels))
        # splat the layers list.
        return nn.Sequential(*layers)

    def _final_reshape(_self, activation):
        return activation.reshape(activation.shape[0], -1)

    # required, from Andrej Karpathy's minGPT
    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # tools for running tests:
    def _option(self, layer_id, use_layers, layer, identity = None):
        if use_layers == 'all' or layer_id <= use_layers:
            return layer
        elif identity == None:
            return self._identity_layer
        else:
            return identity

    def _identity_layer(_self, activation):
        return activation
    def _identity_layer2(_self, activation, _):
        return activation