from torch.utils.data import Sampler
import random
import torch

class CustomSampler(Sampler[int]):
    def __init__(self, dataset, batch_size, train=True, positive_ratio=0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.train = train

        self.targets = dataset.tensors[1]

        self.positive_indices = torch.where(self.targets == 1)[0]
        self.negative_indices = torch.where(self.targets == 0)[0]

        if train:
            self.num_positive_per_batch = max(1, int(batch_size * self.positive_ratio))
            self.num_negative_per_batch = batch_size - self.num_positive_per_batch
            self.num_batches = len(self.positive_indices) // self.num_positive_per_batch
            self.remaining_positives = len(self.positive_indices) % self.num_positive_per_batch
        else:
            self.num_positive_samples = len(self.positive_indices)
            self.num_negative_samples = len(self.negative_indices)
            self.num_batches = (self.num_positive_samples+self.num_negative_samples) // self.batch_size
            # self.num_batches = (self.num_positive_samples*2) // self.batch_size

    def shuffle_indices(self):
        self.positive_indices = self.positive_indices[torch.randperm(len(self.positive_indices))]
        self.negative_indices = self.negative_indices[torch.randperm(len(self.negative_indices))]
        
    def __iter__(self):
        if self.train:
            self.shuffle_indices()
            selected_negative_indices = self.negative_indices[:self.num_batches * self.num_negative_per_batch]
        
            for i in range(self.num_batches):
                pos_batch_indices = self.positive_indices[i * self.num_positive_per_batch: (i + 1) * self.num_positive_per_batch]
                neg_batch_indices = selected_negative_indices[i * self.num_negative_per_batch: (i + 1) * self.num_negative_per_batch]

                # Add remaining positive samples to the last few batches
                if self.remaining_positives > 0 and i >= (self.num_batches - self.remaining_positives):
                    remaining_pos_index = self.positive_indices[-(self.remaining_positives - (i - (self.num_batches - self.remaining_positives)))]
                    pos_batch_indices = torch.cat((pos_batch_indices, remaining_pos_index.unsqueeze(0)))
                    neg_batch_indices = neg_batch_indices[:-1]

                batch_indices = torch.cat((pos_batch_indices, neg_batch_indices))
                batch_indices = batch_indices[torch.randperm(len(batch_indices))]
                yield batch_indices
        
        else:
            # self.negative_indices = self.negative_indices[torch.randperm(len(self.negative_indices))]
            # selected_negative_indices = self.negative_indices[:self.num_positive_samples]
            selected_negative_indices = self.negative_indices

            all_indices = torch.cat((self.positive_indices, selected_negative_indices))
            all_indices = all_indices[torch.randperm(len(all_indices))]

            # Yield batches
            for i in range(self.num_batches):
                batch_indices = all_indices[i * self.batch_size: (i + 1) * self.batch_size]
                yield batch_indices

            # Handle remaining samples if any
            if len(all_indices) % self.batch_size != 0:
                yield all_indices[self.num_batches * self.batch_size:]
    
    def __len__(self) -> int:
        if self.train:
            return self.num_batches
        else:
            return (self.num_positive_samples+self.num_negative_samples + self.batch_size - 1) // self.batch_size