import torch
import torch.nn as nn
import torch.utils.data as data

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from collections import defaultdict
from os.path import isdir


auto_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)


def filterDataset(file_path, destination_path):
  finalLines = []
  with open(file_path, "r", encoding="utf-8") as f:
    for l in f:
      l = l.split("\t")
      isValid = True
      for c in l[0]:
        if ord(c) >= 128:
          isValid = False
          break
      if isValid:
        finalLines.append("\t".join(l))
  with open(destination_path, "w", encoding="utf-8") as f:
    for l in finalLines:
      f.write(l)

def read_data(datapath, indices, uniqueLabels):
    inputs, labels = [], []
    with open(datapath, "r") as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(label_list)
                    inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                    labels.append([pad_token_label_id] + label_list + [pad_token_label_id])
                
                token_list, label_list = [], []
                continue
            
            splits = line.split("\t")
            if len(splits) < 2:
              continue
            token, label = [splits[i] for i in indices]


            subs_ = auto_tokenizer.tokenize(token)
            if len(subs_) > 0 and label in uniqueLabels:
                label_list.extend([uniqueLabels.index(label)] + [pad_token_label_id] * (len(subs_) - 1))
                token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))

    return inputs, labels


def collate_fn(data):
    inputs, labels = zip(*data)
    lengths = [len(l) for l in inputs]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(inputs), max_lengths).fill_(auto_tokenizer.pad_token_id)
    padded_labels = torch.LongTensor(len(inputs), max_lengths).fill_(pad_token_label_id)
    for i, (seq, y_) in enumerate(zip(inputs, labels)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_labels[i, :length] = torch.LongTensor(y_)

    return padded_seqs, padded_labels


def getDataset(datapath, uniqueLabels, indices):
  inputs, labels = read_data(datapath, indices, uniqueLabels)
  return CustomDataset(inputs, labels)
