import torch
import torch.nn as nn

from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel  

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.base = BertModel.from_pretrained('bert-base-cased')
        self.nerLayers = nn.Sequential(
            nn.Linear(self.base.pooler.dense.out_features, 7),
            nn.LogSoftmax(dim=2)
        )
        self.posLayers = nn.Sequential(
            nn.Linear(self.base.pooler.dense.out_features, 39),
            nn.LogSoftmax(dim=2)
        )
    
    def forward(self, ids, _type):
        output= self.base(ids)[0]
        if _type == "NER":
          output = self.nerLayers(output)
        elif _type == "POS":
          output = self.posLayers(output)
        return output