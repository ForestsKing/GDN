from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, data, label, windows_size):
        self.data = data
        self.label = label
        self.windows_size = windows_size

    def __getitem__(self, index):
        X = self.data[index:index + self.windows_size, :]
        y = self.data[index + self.windows_size, :]
        label = self.label[index + self.windows_size]
        return X, y, label

    def __len__(self):
        return len(self.data) - self.windows_size
