import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
        super(CustomDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self._custom_collate_fn)

    def _custom_collate_fn(self, batch):
        # Sort the batch based on the input dimension (assuming the input is a tensor)
        labels, inputs = zip(*batch)

        max_length = max(input.size(0) for input in inputs)

        padded_batch = torch.zeros(len(batch), max_length, 1, 25, 64)
        for i, input in enumerate(inputs) :
            num_zeros_needed = max_length - input.size(0)
            zero_tensors = torch.zeros(num_zeros_needed, 1, 25, 64)
            padded_batch[i] = torch.cat([input, zero_tensors], dim=0)
        
        return torch.tensor(labels), padded_batch.squeeze(0)
