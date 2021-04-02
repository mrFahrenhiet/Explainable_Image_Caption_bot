import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import *
from model import *
from utils import *

state_checkpoint = torch.load("/content/attention_model_state.pth")  # change paths

# model params
vocab = state_checkpoint['vocab']
embed_size = 300
embed_wts = None
vocab_size = state_checkpoint['vocab_size']
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
fc_dims = 256
learning_rate = 5e-4

model = EncoderDecoder(embed_size,
                       vocab_size,
                       attention_dim,
                       encoder_dim,
                       decoder_dim,
                       fc_dims,
                       p=0.3,
                       embeddings=embed_wts).to(device)

model.load_state_dict(state_checkpoint['state_dict'])
