import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import spacy
import os

from torch.utils.data import Dataset

spacy_eng = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sent_list):
        freqs = {}
        idx = 4
        for sent in sent_list:
            for word in self.tokenize(sent):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, sents):
        tokens = self.tokenize(sents)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokens]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, csv_file, transforms=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms

        self.img_pts = self.df.iloc[:, 0]
        self.caps = self.df.iloc[:, 1]
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.caps.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        captions = self.caps[idx]
        img_pt = self.img_pts[idx]

        img = Image.open(os.path.join(self.root_dir, img_pt)).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        encoded_cap = []
        encoded_cap += [self.vocab.stoi["<SOS>"]]  # stoi string to index
        encoded_cap += self.vocab.numericalize(captions)
        encoded_cap += [self.vocab.stoi["<EOS>"]]
        encoded_cap = torch.LongTensor(encoded_cap)

        return img, encoded_cap


class CapsCollate:
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        return imgs, targets
