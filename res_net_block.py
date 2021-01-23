import torch
import torch.nn as nn

EXPANSION_FACTOR = 4

class ResNetBlock(nn.Module):
    """
    Initialization Parametrs for a ResNetBlock

    input_channel_count:  An image tensor is a (k x w x h) tensor where k represents
       how many channels, naively, groups of features discovered by the ml model.
       the shape of the input tensor for this block of layers is
       (input_channel x w x h).
    internal_channel_count:
       Internal activations are all image tensors (see above), with tensor shape
       (internal_channel_count x w x h).  For a ResNet, the internal channels don't
       increase within the layers of a block, but are increased on the last step of
       processing.

    downsampler:  A shape-changing identity layer that is necessary if we change
       the input size or if we change the number of channels.
    stride: unless 1, skips (stride - 1) pixels in the image as a downsizing strategy.
    """
    def __init__(self, input_channel_count, internal_channel_count, stride=1, layers = 'all'):
        super(ResNetBlock, self).__init__()

        # a parameter which expresses the multiplicity of the channels coming out relative
        # the channels coming in.
        self.conv1 =       self._option(1, layers, nn.Conv2d(input_channel_count, internal_channel_count, kernel_size=1, stride=1, padding=0))
        self.batchnorm1 =  self._option(2, layers, nn.BatchNorm2d(internal_channel_count))
        self.relu1 =       self._option(3, layers, nn.ReLU())
        self.conv2 =       self._option(4, layers, nn.Conv2d(internal_channel_count, internal_channel_count, kernel_size=3, stride=stride, padding=1))
        self.batchnorm2 =  self._option(5, layers, nn.BatchNorm2d(internal_channel_count))
        self.relu2 =       self._option(6, layers, nn.ReLU())
        self.conv3 =       self._option(7, layers, nn.Conv2d(internal_channel_count, internal_channel_count * EXPANSION_FACTOR, kernel_size=1, stride=1, padding=0))
        self.batchnorm3 =  self._option(8, layers, nn.BatchNorm2d(internal_channel_count * EXPANSION_FACTOR))
        self.relu3 =       self._option(9, layers, nn.ReLU())
        self.downsampler = ResNetBlock.downsampler_fn(input_channel_count, internal_channel_count, stride)
        self.skiplayer =   self._option(10, layers, self._skiplayer, self._identity_layer2)
        self.relu3 =       nn.ReLU()

    def forward(self, activation):
        identity = activation.clone()

        # apply the predefined layers, generating activations as you go along.
        activation = self.conv1(activation)
        activation = self.batchnorm1(activation)
        activation = self.relu1(activation)
        activation = self.conv2(activation)
        activation = self.batchnorm2(activation)
        activation = self.relu2(activation)
        activation = self.conv3(activation)
        activation = self.batchnorm3(activation)

        # we may have to alter the shape of the initial input
        if self.downsampler is not None:
            identity = self.downsampler(identity)

        # add the original layer in in "skipping" fashion.  This means all of the
        # previous layers are calculating "residuals", are closer to gaussian random
        # initialization and generally are better at being found.
        activation = self.skiplayer(activation, identity)
        output =     self.relu3(activation)
        return output

    """
    provides a downsampler neural net for general use.  This can be called,
    because the calling method is overloaded.

    Note this is a static function.
    """
    def downsampler_fn(input_channel_count, internal_channel_count, stride):
        if stride == 1 and input_channel_count == internal_channel_count * EXPANSION_FACTOR:
            return None
        return nn.Sequential(
                    nn.Conv2d(
                        input_channel_count,
                        internal_channel_count * EXPANSION_FACTOR,
                        kernel_size=1,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(internal_channel_count * EXPANSION_FACTOR))

    """
    provides a "skipping" layer that is the primary feature of ResNets:
    convergence of weights to zero and allowance of activations to be
    residuals.
    """
    def _skiplayer(_self, activation, identity):
        return activation + identity

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