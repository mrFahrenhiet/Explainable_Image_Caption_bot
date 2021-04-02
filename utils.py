from matplotlib import pyplot as plt
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


def show_img(img, caption):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    img = img.permute(1, 2, 0)
    img = img.to('cpu').numpy()
    plt.imshow(img)
    plt.title(caption)
    plt.show()


def load_embeding(embed_file, vocab):
    with open(embed_file, 'r') as f:
        embed_dims = len(f.readline().split(' ')) - 1

    words = set(vocab.stoi.keys())
    embeddings = torch.FloatTensor(len(words), embed_dims)
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)
    print("\nLoading embeddings...")
    for line in open(embed_file, 'r'):
        line = line.split(' ')
        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        # Ignore word if not in train_vocab
        if emb_word not in words:
            continue
        embeddings[vocab.stoi[emb_word]] = torch.FloatTensor(embedding)
    print("\nEmbeddings loaded!")
    return embeddings, embed_dims


def save_model(model, num_epochs, optimizer, train_loss, val_loss, vocab):
    model_state = {
        'num_epochs': num_epochs,
        'vocab': vocab,
        'vocab_size': len(vocab.stoi),
        'state_dict': model.state_dict(),
        'optimizer_denoise_state_dict': optimizer,
        'training_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(model_state, 'attention_model_state.pth')
