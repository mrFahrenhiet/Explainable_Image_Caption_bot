import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image, ImageOps
import io
import streamlit as st
import requests
import os
from io import BytesIO
from download_files import *
from matplotlib import pyplot as plt

from model import *
from dataset import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Explainable Image Caption Bot"
)


def transform_img(img):
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transforms(img)


@st.cache()
def download_checkpoints():
    path = "/attention_model_state.pth"

    if not os.path.exists(path):
        with st.spinner('Downloading state checkpoint'):
            check_pt_url = "1-AIlZp28kvn13sEGJpD1vJY10aBuBg2a"
            download_file_from_google_drive(check_pt_url, path)

    print("Model Downloaded")


@st.cache()
def load_model():
    state_checkpoint = torch.load("/attention_model_state.pth", map_location=device)  # change paths
    # model params
    vocab = state_checkpoint['vocab']
    embed_size = 300
    embed_wts = None
    vocab_size = state_checkpoint['vocab_size']
    attention_dim = 256
    encoder_dim = 2048
    decoder_dim = 512
    fc_dims = 256

    model = EncoderDecoder(embed_size,
                           vocab_size,
                           attention_dim,
                           encoder_dim,
                           decoder_dim,
                           fc_dims,
                           p=0.3,
                           embeddings=embed_wts).to(device)

    model.load_state_dict(state_checkpoint['state_dict'])
    return model, vocab


def get_caps_from(features_tensors, model, vocab=None):
    model.eval()
    with torch.no_grad():
        features = model.EncoderCNN(features_tensors[0:1].to(device))
        caps, alphas = model.DecoderLSTM.gen_captions(features, vocab=vocab)
        caption = ' '.join(caps)

    return caption, caps, alphas


def plot_attention(img, target, attention_plot):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    img = img.to('cpu').numpy().transpose((1, 2, 0))

    temp_image = img

    fig = plt.figure(figsize=(10, 10))
    len_caps = len(target)
    for i in range(len_caps):
        temp_att = attention_plot[i].reshape(7, 7)
        ax = fig.add_subplot(len_caps // 2, len_caps // 2, i + 1)
        ax.set_axis_off()
        ax.set_title(target[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.5, extent=img.get_extent())

    plt.tight_layout()
    st.pyplot(fig)


def plot_caption_with_attention(img_pth, model, transforms_=None, vocab=None):
    img = Image.open(img_pth)
    img = transforms_(img)
    img.unsqueeze_(0)
    caption, caps, attention = get_caps_from(img, model, vocab)
    st.markdown(f"## Image Caption:\n"
                f" #### {caption[:-5]}\n\n")
    plot_attention(img[0], caps, attention)


@st.cache(ttl=3600, max_entries=10)
def load_output_image(img):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        img_bytes = img.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    image = ImageOps.exif_transpose(image)
    return image


if __name__ == '__main__':

    download_checkpoints()
    model, vocab = load_model()

    st.title("The Explainable Image Captioning Bot")
    st.text("")
    st.text("")
    st.success("Welcome! Please upload an image!"
               )

    img_upload = st.file_uploader(label='Upload Image', type=['png', 'jpg', 'jpeg', 'webp'])
    img_pt = "imgs/test2.jpeg" if img_upload is None else img_upload

    image = load_output_image(img_pt)

    st.sidebar.markdown('''
    Hello! :hand: and Welcome,
    This is a Image caption bot:\n
     Its main job is to give captions :speech_balloon: or description for your
     input image.\n
     But we have tried something different here \n
     This app gives 2 outputs
    - The Caption for your image duh? :upside_down_face:
    - Explaination as in a image grid i.e. the parts of the image
    where the AI looks when trying to caption your image :nerd_face: \n
    If you are getting random captions, then try :-
    - Using a PC
    - Try images of bicycles or motarbikes
    - Try images with children in it
    - Try images of dogs
    ''')

    st.sidebar.markdown('''Check the model details [here](https://github.com/mrFahrenhiet/Explainable_Image_Caption_bot)
    \n Liked it? Give a :star:  on GitHub ''')

    st.image(image, use_column_width=True)

    if st.button('Generate captions!'):
        plot_caption_with_attention(img_pt, model, transform_img, vocab)
        st.success("Try a different image by uploading")
        st.balloons()
