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
    def __init__(self, data = None, labels = None):
        # get the data from the cifar dataset.
        if data == None or labels == None:
            cifar_dataset = unpickle("cifar-100-python/train")
            data = cifar_dataset[b"data"] if data == None else data
            labels = cifar_dataset[b"fine_labels"] if labels == None else data

        # data validation.  Make sure that all labels are between 1 and 100
        # Make sure that all entries are the expected sort of torch arrays.
        assert len(data) == len(labels)
        for label in labels:
            assert 0 <= label <= 100
        for entry in data:
            assert len(entry) == 3072

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # make a correctly dimensioned tensor (from flat 3072-list) -> 3x32x32 tensor.
        x = torch.reshape(torch.tensor(self.data[idx], dtype=torch.float), [3, 32, 32])
        # encode as a single value
        y = self.labels[idx] #torch.tensor([], dtype=torch.float)
        return x, y

from trainer import Trainer, TrainerConfig
from res_net import ResNet

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", type="string",
                      help="input checkpoint name")
    parser.add_option("-o", "--output", dest="checkpoint", type="string",
                      help="output checkpoint name")
    parser.add_option("-e", "--epochs", dest="epochs", type="int",
                      default=100, help="number of epochs to run")
    parser.add_option("-l", "--learning-rate", dest="lr", type="float",
                      default=6e-4, help="learining rate for the run")
    (options, args) = parser.parse_args()

    # initialize a trainer instance and kick off training
    train_dataset = CifarDataset()

    model = ResNet([3, 4, 6, 3], input_channels=3, num_classes=100)

    if os.path.isfile(options.input):
        checkpoint = torch.load(options.input)
        model.load_state_dict(checkpoint)

    tconf = TrainerConfig(max_epochs=options.epochs, batch_size=512,
      learning_rate=options.lr, num_workers=4, ckpt_path = options.checkpoint)
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()
