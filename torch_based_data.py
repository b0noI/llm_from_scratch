import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):

    #TODO: why not to split by sentances and NOT in full full like we are doing now?
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.traget_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.traget_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.traget_ids[idx]
    

def create_dataloader_v1(txt, batch_size=1, max_length=4, stride=1, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


def load_verdict_txt_dataloader(*, batch_size=8, max_length=4, stride=4, shuffle=True):
    verdict_txt = None
    with open("the-verdict.txt", "r") as file:
        verdict_txt = file.read()

    return create_dataloader_v1(verdict_txt, batch_size=batch_size, max_length=max_length, shuffle=shuffle, stride=stride)
