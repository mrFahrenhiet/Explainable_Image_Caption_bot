from tqdm import tqdm
from dataset import Vocabulary
from model import *
from utils import *
import torchvision.transforms as T
from PIL import Image
import argparse


# Will only work for batch size 1
def get_bleu_score(dataset, img, target, model, vocab=None):
    features = model.EncoderCNN(img[0:1].to(device))
    caps, alphas = model.DecoderLSTM.gen_captions(features, vocab=vocab)
    target_ = [[dataset.vocab.itos[i] for i in target[0].tolist() if dataset.vocab.itos[i] not in "<SOS>"]]
    return sentence_bleu(target_, caps)


def get_batched_bleu(dataloader_eval, dataset, model, vocab=None):
    bleu = 0
    for batch in tqdm(dataloader_eval, total=len(dataloader_eval)):
        img, cap = batch
        img, cap = img.to(device), cap.to(device)
        bleu += get_bleu_score(dataset, img, cap, model, vocab)

    bleu /= len(dataloader_eval)
    print("\nBlue Score: ", bleu * 100)


def get_caps_from(features_tensors, model, vocab=None):
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


def plot_caption_with_attention(img_pth, model, transforms_=None, vocab=None):
    img = Image.open(img_pth)
    img = transforms_(img)
    img.unsqueeze_(0)
    caps, attention = get_caps_from(img, model, vocab)
    plot_attention(img[0], caps, attention)


def main(arguments):
    state_checkpoint = torch.load(arguments.state_chechpoint, map_location=device)  # change paths
    # model params
    vocab = state_checkpoint['vocab']
    embed_size = arguments.embed_size
    embed_wts = None
    vocab_size = state_checkpoint['vocab_size']
    attention_dim = arguments.attention_dim
    encoder_dim = arguments.encoder_dim
    decoder_dim = arguments.decoder_dim
    fc_dims = arguments.fc_dims

    model = EncoderDecoder(embed_size,
                           vocab_size,
                           attention_dim,
                           encoder_dim,
                           decoder_dim,
                           fc_dims,
                           p=0.3,
                           embeddings=embed_wts).to(device)

    model.load_state_dict(state_checkpoint['state_dict'])

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img_path = arguments.image
    plot_caption_with_attention(img_path, model, transforms, vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--state_checkpoint', type=str, required=True, help='path for state checkpoint')
    parser.add_argument('--embed_size', type=int, default=300, help='dimension of word embedding vectors')
    parser.add_argument('--attention_dim', type=int, default=256, help='dimension of attention layer')
    parser.add_argument('--encoder_dim', type=int, default=2048, help='dimension of encoder layer')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder layer')
    parser.add_argument('--fc_dims', type=int, default=256, help='dimension of fully connected layer')
    args = parser.parse_args()
    main(args)
