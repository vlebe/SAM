from torch.utils.data import Dataset

class Dataset(Datset):
    def __init__(self, data):
        self.data = data

    def __get_item__(self, index):
        return self.data[index]