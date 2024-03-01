import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import product

from boolean_tokenizer import BooleanTokenizer


class MaxBooleanFunctionDataset(Dataset):
    def __init__(self, tokenizer, num_bits, num_examples, label2id):
        self.tokenizer = tokenizer
        self.num_bits = num_bits
        self.num_examples = num_examples
        self.label2id = label2id

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Generate a sample on the fly,
        # accepting the low probability of a generating the same sample twice or in both train and test
        bits = np.random.choice([-1, 1], size=self.num_bits)
        label = np.max(bits)

        # prepare the input for the model
        bits_tokens = torch.tensor(self.tokenizer.encode(bits), dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)
        label = self.label2id[str(label)]
        label = torch.tensor(label).long()
        return {
            'input_ids': bits_tokens,
            'attention_mask': attention_mask,
            'labels': label
        }


class MajorityBooleanFunctionDataset(Dataset):
    def __init__(self, tokenizer, num_bits, num_examples, label2id):
        self.tokenizer = tokenizer
        self.num_bits = num_bits
        self.num_examples = num_examples
        self.label2id = label2id

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Generate a sample on the fly,
        # accepting the low probability of a generating the same sample twice or in both train and test
        bits = np.random.choice([-1, 1], size=self.num_bits)
        label = 1 if np.sum(bits) >= 0 else -1

        # prepare the input for the model
        bits_tokens = torch.tensor(self.tokenizer.encode(bits), dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)
        label = self.label2id[str(label)]
        label = torch.tensor(label).long()
        return {
            'input_ids': bits_tokens,
            'attention_mask': attention_mask,
            'labels': label
        }


class KSparseBooleanFunctionDataset(Dataset):
    def __init__(self, tokenizer, num_bits, num_examples, k, label2id):
        self.tokenizer = tokenizer
        self.num_bits = num_bits
        self.num_examples = num_examples
        self.k = k
        # create a fixed random boolean function based on k bits
        mapping = [(perm, np.random.choice([-1, 1])) for perm in list(product([-1, 1], repeat=k))]
        self.ksparse_function = lambda x: mapping[x][1]
        self.label2id = label2id

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Generate a sample on the fly,
        # accepting the low probability of a generating the same sample twice or in both train and test
        bits = np.random.choice([-1, 1], size=self.num_bits)
        label = self.ksparse_function(bits[:self.k])

        # prepare the input for the model
        bits_tokens = torch.tensor(self.tokenizer.encode(bits), dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)
        label = self.label2id[str(label)]
        label = torch.tensor(label).long()
        return {
            'input_ids': bits_tokens,
            'attention_mask': attention_mask,
            'labels': label
        }


class RandomBooleanFunctionDataset(Dataset):
    def __init__(self, tokenizer, num_bits, num_examples, label2id):
        self.tokenizer = tokenizer
        self.num_bits = num_bits
        self.num_examples = num_examples
        self.label2id = label2id

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Generate a sample on the fly,
        # accepting the low probability of a generating the same sample twice or in both train and test
        bits = np.random.choice([-1, 1], size=self.num_bits)
        perturbed_bits = bits.copy()
        perturbed_bits[np.random.choice(self.num_bits)] *= -1

        # prepare the input for the model
        bits_tokens = torch.tensor(self.tokenizer.encode(bits, add_special_tokens=False), dtype=torch.long)
        perturbed_bits_tokens = torch.tensor(self.tokenizer.encode(perturbed_bits, add_special_tokens=False),
                                             dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)
        return {
            'input_ids': bits_tokens,
            'perturbed_input_ids': perturbed_bits_tokens,
            'attention_mask': attention_mask,
        }


class ParityFunctionDataset(Dataset):
    def __init__(self, tokenizer, num_bits, parity_length, num_examples, label2id):
        assert num_bits >= parity_length, "The parity length should be less than or equal to the number of bits."
        self.tokenizer = tokenizer
        self.num_bits = num_bits
        self.num_examples = num_examples
        self.parity_length = parity_length
        self.label2id = label2id

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Generate a sample on the fly,
        # accepting the low probability of a generating the same sample twice or in both train and test
        bits = np.random.choice([-1, 1], size=self.num_bits)
        label = np.prod(bits[:self.parity_length])

        # prepare the input for the model
        bits_tokens = torch.tensor(self.tokenizer.encode(bits), dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)
        label = self.label2id[str(label)]
        label = torch.tensor(label).long()
        return {
            'input_ids': bits_tokens,
            'attention_mask': attention_mask,
            'labels': label
        }


class StaircaseFunctionDataset(Dataset):
    def __init__(self, tokenizer, num_bits, staircase_length, num_examples, label2id):
        self.tokenizer = tokenizer
        self.num_bits = num_bits
        self.staircase_length = staircase_length
        self.num_examples = num_examples
        self.label2id = label2id

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Generate a sample on the fly,
        # accepting the low probability of a generating the same sample twice or in both train and test
        bits = np.random.choice([-1, 1], size=self.num_bits)
        label = 1 if np.sum(np.cumprod(bits[:self.staircase_length])) >= 0 else -1

        # prepare the input for the model
        bits_tokens = torch.tensor(self.tokenizer.encode(bits), dtype=torch.long)
        attention_mask = torch.ones_like(bits_tokens)
        label = self.label2id[str(label)]
        label = torch.tensor(label).long()
        return {
            'input_ids': bits_tokens,
            'attention_mask': attention_mask,
            'labels': label
        }


if __name__ == "__main__":
    tokenizer = BooleanTokenizer()
    # dataset = ParityFunctionDataset(tokenizer, 10, 1, 1000)
    dataset = StaircaseFunctionDataset(tokenizer, 4, 1000, {'-1': 0, '1': 1})
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for batch in dataloader:
        print(batch)
