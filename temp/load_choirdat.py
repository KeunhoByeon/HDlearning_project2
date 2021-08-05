## Loads .choirdat file's information
# returns a tuple: ((samples, labels), features, classes); where features and
# classes is a number and samples and labels is a list of vectors
def load_choirdat(dataset_path, train_data=None, train_label=None):
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
            if train_data==None:
                samples.append(sample)
                labels.append(label)
            else:
                train_data.append(sample)
                train_label.append(label)
    return (samples, labels), features, classes