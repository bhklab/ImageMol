import torch
from torch.utils.data import Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    """
    Samples equal numbers of positive and negative samples for each batch.
    Assumes binary classification (labels are 0 or 1).
    """
    def __init__(self, labels, batch_size, sampler_ratio=0.5):
        self.labels = np.array(labels).flatten()
        self.batch_size = batch_size
        self.sampler_ratio = sampler_ratio
        self.num_neg = int(self.batch_size * self.sampler_ratio)
        self.num_pos = self.batch_size - self.num_neg
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        assert self.num_pos > 0 and self.num_neg > 0, "Batch size and sampler_ratio must result in at least 1 positive and 1 negative."

    def __iter__(self):
        pos = np.random.permutation(self.pos_indices)
        neg = np.random.permutation(self.neg_indices)
        max_batches_pos = len(pos) // self.num_pos
        max_batches_neg = len(neg) // self.num_neg
        num_batches = min(max_batches_pos, max_batches_neg)
        for i in range(num_batches):
            pos_batch = pos[i*self.num_pos:(i+1)*self.num_pos]
            neg_batch = neg[i*self.num_neg:(i+1)*self.num_neg]
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        max_batches_pos = len(self.pos_indices) // self.num_pos
        max_batches_neg = len(self.neg_indices) // self.num_neg
        return min(max_batches_pos, max_batches_neg) * self.batch_size
