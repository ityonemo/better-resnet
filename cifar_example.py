# runnable as python3 cifar_example.py <path_to_cifar>
import os
import torch
from torch.utils.data import Dataset

# make sure the cifar directory exists.
if (not os.path.isdir("cifar-100-python")):
    print("you should run `download_cifar_100.sh` first")
    exit(1)

# unpickling (see guidelines in https://www.cs.toronto.edu/~kriz/cifar.html)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

class CifarDataset(Dataset):
    """
    creates a cifar 100-dataset.

    data are a flat list of numpy arrays of dimension 3072
    labels are a flat list of labels (integers 1..100)
    """
    def __init__(self, data, labels):
        # data validation.  Make sure that all labels are between 1 and 100
        # Make sure that all entries are the expected sort of torch arrays.
        assert len(data) == len(labels)
        for label in labels:
            assert 0 <= label <= 100
        for entry in data:
            assert entry.size() == torch.Size([3072])

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # reshape as a 3x32x32 tensor.
        x = torch.reshape(self.data[idx], [3, 32, 32])
        # encode as a one-hot tensor of 100 values
        y = torch.tensor([(1.0 if onehotidx == self.labels[idx] else 0.0) for onehotidx in range(100)], dtype=torch.float)
        return x, y

