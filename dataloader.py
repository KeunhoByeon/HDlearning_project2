import struct

import os
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import torch


def read_choirdat(dataset_path):
    with open(dataset_path, 'rb') as f:
        # reads meta information
        features = struct.unpack('i', f.read(4))[0]
        classes = struct.unpack('i', f.read(4))[0]

        # lists containing all samples and labels to be returned
        samples = list()
        labels = list()

        while True:
            # load a new sample
            sample = list()

            # load sample's features
            for i in range(features):
                val = f.read(4)
                if val is None or not len(val):
                    return (samples, labels), features, classes
                sample.append(struct.unpack('f', val)[0])

            # add the new sample and its label
            label = struct.unpack('i', f.read(4))[0]
            samples.append(sample)
            labels.append(label)

    return (samples, labels), features, classes


def load_isolet(data_dir='./data'):
    trainset_path = os.path.join(data_dir, 'isolet_train.choir_dat')
    testset_path = os.path.join(data_dir, 'isolet_test.choir_dat')

    # Load trainset
    train_samples, features, classes = read_choirdat(trainset_path)
    x, y = train_samples

    # Load testset
    test_samples, _, _ = read_choirdat(testset_path)
    x_test, y_test = test_samples

    x = np.array(x).astype(np.float64)
    y = np.array(y).astype(np.float64)
    x_test = np.array(x_test).astype(np.float64)
    y_test = np.array(y_test).astype(np.float64)

    # Normalize
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # Changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    # Print dataset info
    print('Trainset    size: {}    data range: {:4f} ~ {:4f}'
          .format(list(x.size()), x.min().item(), x.max().item()))
    print('Testset     size: {}    data range: {:4f} ~ {:4f}'
          .format(list(x_test.size()), x_test.min().item(), x_test.max().item()))

    return x, x_test, y, y_test
