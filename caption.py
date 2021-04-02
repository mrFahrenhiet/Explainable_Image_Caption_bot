from tqdm import tqdm

from model import *
from utils import *


# Will only work for batch size 1
def get_bleu_score(dataset, img, target):
    features = model.EncoderCNN(img[0:1].to(device))
    caps, alphas = model.DecoderLSTM.gen_captions(features, vocab=dataset.vocab)
    target_ = [[dataset.vocab.itos[i] for i in target[0].tolist() if dataset.vocab.itos[i] not in "<SOS>"]]
    return sentence_bleu(target_, caps)


def get_batched_bleu(dataloader_eval, dataset):
    bleu = 0
    for batch in tqdm(dataloader_eval, total=len(dataloader_eval)):
        img, cap = batch
        img, cap = img.to(device), cap.to(device)
        bleu += get_bleu_score(dataset, img, cap)

    bleu /= len(dataloader_eval)
    print("\nBlue Score: ", bleu * 100)


def get_caps_from(features_tensors):
    model.eval()
    with torch.no_grad():
        features = model.EncoderCNN(features_tensors[0:1].to(device))
        caps, alphas = model.DecoderLSTM.gen_captions(features, vocab=vocab)
        caption = ' '.join(caps)
        show_img(features_tensors[0], caption)

    return caps, alphas


def plot_attention(img, target, attention_plot):
    img = img.to('cpu').numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))
    len_caps = len(target)
    for i in range(len_caps):
        temp_att = attention_plot[i].reshape(7, 7)
        ax = fig.add_subplot(len_caps // 2, len_caps // 2, i + 1)
        ax.set_title(target[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.5, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
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
