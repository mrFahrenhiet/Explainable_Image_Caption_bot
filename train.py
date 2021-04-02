import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import *
from model import *
from utils import *

spacy_eng = spacy.load('en')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# init seed
seed = torch.randint(100, (1,))
torch.manual_seed(seed)
shuffle = True
# src folders
root_folder = "/content/flickr8k/Images"  # change this
csv_file = "/content/flickr8k/captions.txt"  # change this

# image transforms and augmentation
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# define dataset
dataset = FlickrDataset(root_folder, csv_file, transforms)

# split dataset
val_size = 512
test_size = 256
train_size = len(dataset) - val_size - test_size
train_ds, val_ds, test_ds = random_split(dataset,
                                         [train_size, val_size, test_size])

# Define data loader parameters
num_workers = 4
pin_memory = True
batch_size_train = 256
batch_size_val_test = 128
pad_idx = dataset.vocab.stoi["<PAD>"]

# define loaders
dataloader_train = DataLoader(train_ds,
                              batch_size=batch_size_train,
                              pin_memory=pin_memory,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True))
dataloader_validation = DataLoader(val_ds,
                                   batch_size=batch_size_val_test,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                   collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True))
dataloader_test = DataLoader(test_ds,
                             batch_size=batch_size_val_test,
                             pin_memory=pin_memory,
                             num_workers=num_workers,
                             shuffle=shuffle,
                             collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True))

# model parameters
embed_wts, embed_size = load_embeding("/content/glove.42B.300d.txt", dataset.vocab)  # change path
vocab_size = len(dataset.vocab)
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
loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# training parmeters
num_epochs = 35
train_loss_arr = []
val_loss_arr = []


def training(dataset, dataloader, loss_criteria, optimize, grad_clip=5.):
    total_loss = 0
    for i, (img, cap) in enumerate(tqdm(dataloader, total=len(dataloader))):
        img, cap = img.to(device), cap.to(device)
        optimize.zero_grad()
        output, attention = model(img, cap)
        targets = cap[:, 1:]
        loss = loss_criteria(output.view(-1, vocab_size), targets.reshape(-1))
        total_loss += (loss.item())
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimize.step()

    total_loss = total_loss / len(dataloader)

    return total_loss


@torch.no_grad()
def validate(dataset, dataloader, loss_cr):
    total_loss = 0
    for val_img, val_cap in tqdm(dataloader, total=len(dataloader)):
        val_img, val_cap = val_img.to(device), val_cap.to(device)
        output, attention = model(val_img, val_cap)
        targets = val_cap[:, 1:]
        loss = loss_cr(output.view(-1, vocab_size), targets.reshape(-1))
        total_loss += (loss.item())

    total_loss /= len(dataloader)
    return total_loss


# for see results while training
@torch.no_grad()
def test_on_img(data, dataloader):
    dataiter = iter(dataloader)
    img, cap = next(dataiter)
    features = model.EncoderCNN(img[0:1].to(device))
    caps, alphas = model.DecoderLSTM.gen_captions(features, vocab=data.vocab)
    caption = ' '.join(caps)
    show_img(img[0], caption)


def main():
    best_val_loss = 6.0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = training(dataset, dataloader_train, loss_fn, optimizer)
        train_loss_arr.append(train_loss)

        model.eval()
        val_loss = validate(dataset, dataloader_validation, loss_fn)
        val_loss_arr.append(val_loss)
        print(f"train_loss: {train_loss} validation_loss: {val_loss}")
        test_on_img(dataset, dataloader_validation)
        if len(val_loss_arr) == 1 or val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, epoch, optimizer, train_loss, val_loss, vocab=dataset.vocab)
            print("best model saved successfully")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
