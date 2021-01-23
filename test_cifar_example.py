import unittest
import torch
from cifar_example import CifarDataset

class TestCifarDataset(unittest.TestCase):

    def _dummy_input(_self):
        return torch.tensor([0 for _ in range(3072)], dtype=torch.uint8)

    def test_labels_single(self):
        dataset = CifarDataset([self._dummy_input()], [47])
        _, label = dataset.__getitem__(0)
        # check validity of the one-hot encoding.
        self.assertEqual(label[47], 1.0)
        # nothing else is encoded.
        for idx in range(100):
            if idx != 47:
                self.assertEqual(label[idx], 0.0)

    def test_labels_multi(self):
        dataset = CifarDataset([self._dummy_input(), self._dummy_input()], [47, 35])
        _, label = dataset.__getitem__(1)
        # check validity of the one-hot encoding.
        self.assertEqual(label[35], 1.0)
        # nothing else is encoded.
        for idx in range(100):
            if idx != 35:
                self.assertEqual(label[idx], 0.0)

    def _array1024(_self, v):
        return [v for _ in range(1024)]

    def test_data_single(self):
        # dataset provides a flat Torch array, which is a 3072-entry
        # 1-d Tensor.
        # The first 1024 entries contain the red channel values,
        # the next 1024 the green, and the final 1024 the blue.
        # values are 0..255.

        # should be encoded into a (channels x h x w) tensor.  We sort
        # of don't care about orientation, so just make sure we are
        # encoding channels correctly.  Fill red channel with 0,
        # fill green channel with 1, fill blue channel with 2
        sublists = [self._array1024(i) for i in range(3)]
        flat_python_list = [px for sublist in sublists for px in sublist]
        # double check we have made this correctly.

        self.assertEqual(flat_python_list[0], 0)
        self.assertEqual(flat_python_list[1024], 1)
        self.assertEqual(flat_python_list[2048], 2)

        torch_input = torch.tensor(flat_python_list, dtype=torch.uint8)

        self.assertEqual(torch_input[0], 0)
        self.assertEqual(torch_input[1024], 1)
        self.assertEqual(torch_input[2048], 2)

        dataset = CifarDataset([torch_input], [47])
        data, _ = dataset.__getitem__(0)

        # make sure the tensor shape is correct.
        self.assertEqual(data.size(), torch.Size([3, 32, 32]))

        # validate that our concept of the torch tensors is correct
        for channel in range(3):
            image = data[channel, :, :]
            self.assertEqual(image.size(), torch.Size([32, 32]))
            for i in range(32):
                for j in range(32):
                    self.assertEqual(image[i, j], channel)


