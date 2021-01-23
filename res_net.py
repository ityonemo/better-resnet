import torch
import torch.nn as nn
from res_net_block import ResNetBlock
from res_net_block import EXPANSION_FACTOR

class ResNet(nn.Module):
    def __init__(self, block_counts, input_channels, num_classes, layers = 'all'):
        # make sure that our block counts is a list with 4 parameters.
        assert len(block_counts) == 4

        super(ResNet, self).__init__()

        # how many channels we'll start out with in the first convnet layer.
        # TODO: figure out how to pass these explicitly.  Also this needs to be renamed.
        self.input_channels = 64
        # the first "big" channel
        self.conv1      = self._option(1, layers, nn.Conv2d(input_channels, self.input_channels, kernel_size=7, stride=2, padding=3))
        self.batchnorm1 = self._option(2, layers, nn.BatchNorm2d(self.input_channels))
        self.relu       = self._option(3, layers, nn.ReLU())
        self.maxpool    = self._option(4, layers, nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        ## ResNet blocklayers
        self.blocklayer1 = self._option(5, layers, self._make_blocklayer(block_counts[0], input_channels=64, internal_channels=64, stride=1))
        self.blocklayer2 = self._option(6, layers, self._make_blocklayer(block_counts[1], input_channels=256, internal_channels=128, stride=2))
        self.blocklayer3 = self._option(7, layers, self._make_blocklayer(block_counts[2], input_channels=512, internal_channels=256, stride=2))
        self.blocklayer4 = self._option(8, layers, self._make_blocklayer(block_counts[3], input_channels=1024, internal_channels=512, stride=2))

        ## cleanup and postprocessing
        self.avgpool =        self._option(9, layers, nn.AdaptiveAvgPool2d((1, 1)))
        self.reshape =        self._option(10, layers, self._final_reshape)
        self.fullyconnected = self._option(11, layers, nn.Linear(512 * EXPANSION_FACTOR, num_classes))

    def forward(self, input):
        # apply the pretreatment layers, generating activations.
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
        return activation

    def _make_blocklayer(self, block_count, input_channels, internal_channels, stride):
        downsampler = None

        # were going to be building up a layers list.
        layers = []

        # note that ResNetBlock will return size input_channels * 4!!  this is super confusing!
        layers.append(ResNetBlock(input_channels, internal_channels, stride))
        for i in range(block_count - 1):
            layers.append(ResNetBlock(internal_channels * EXPANSION_FACTOR, internal_channels))

        # splat the layers list.
        return nn.Sequential(*layers)

    def _final_reshape(_self, activation):
        return activation.reshape(activation.shape[0], -1)

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