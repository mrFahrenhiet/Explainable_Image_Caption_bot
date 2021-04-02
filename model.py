import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]  # extracting the last conv layer from the model
        self.resnet = nn.Sequential(*modules)

    def forward(self, imgs):
        features = self.resnet(imgs)
        features = features.permute(0, 2, 3, 1)  # batch x 7 x 7 x 2048
        features = features.view(features.size(0), -1, features.size(-1))  # batch x 49 x 2048
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, attention_dims):
        super(Attention, self).__init__()
        self.attention_dims = attention_dims  # size of attention network
        self.U = nn.Linear(encoder_dims, attention_dims)  # a^(t)
        self.W = nn.Linear(decoder_dims, attention_dims)  # s^(t` - 1)
        self.A = nn.Linear(attention_dims, 1)  # cvt the attention dims back to 1

    def forward(self, features, hidden):
        u_as = self.U(features)
        w_as = self.W(hidden)
        combined_state = torch.tanh(u_as + w_as.unsqueeze(1))
        attention_score = self.A(combined_state)
        attention_score = attention_score.squeeze(2)
        alpha = F.softmax(attention_score, dim=1)
        attention_weights = features * alpha.unsqueeze(2)  # batch x num_timesteps (49) x features
        attention_weights = attention_weights.sum(dim=1)
        return alpha, attention_weights


class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, fc_dims, p=0.3,
                 embeddings=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm = nn.LSTMCell(encoder_dim + embed_size, decoder_dim, bias=True)
        self.fcn1 = nn.Linear(decoder_dim, vocab_size)
        self.fcn2 = nn.Linear(fc_dims, vocab_size)
        self.drop = nn.Dropout(p)

        if embeddings is not None:
            self.load_pretrained_embed(embeddings)

    def forward(self, features, captions):

        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_timesteps = features.size(1)

        embed = self.embedding(captions)
        h, c = self.init_hidden_state(features)  # initialize h and c for LSTM

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_timesteps).to(device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_inp = torch.cat((embed[:, s], context), dim=1)
            h, c = self.lstm(lstm_inp, (h, c))
            out = self.drop(self.fcn1(h))
            preds[:, s] = out
            alphas[:, s] = alpha

        return preds, alphas

    def gen_captions(self, features, max_len=20, vocab=None):
        h, c = self.init_hidden_state(features)
        alphas = []
        captions = []
        word = torch.tensor(vocab.stoi["<SOS>"]).view(1, -1).to(device)
        embed = self.embedding(word)
        for i in range(max_len):
            alpha, context = self.attention(features, h)
            alphas.append(alpha.cpu().detach().numpy())

            lstm_inp = torch.cat((embed[:, 0], context), dim=1)
            h, c = self.lstm(lstm_inp, (h, c))
            out = self.drop(self.fcn1(h))
            word_out_idx = torch.argmax(out, dim=1)
            captions.append(word_out_idx.item())
            if vocab.itos[word_out_idx.item()] == "<EOS>":
                break
            embed = self.embedding(word_out_idx.unsqueeze(0))

        return [vocab.itos[word] for word in captions], alphas

    def load_pretrained_embed(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)
        for p in self.embedding.parameters():
            p.requires_grad = True

    def init_hidden_state(self, encoder_output):
        mean_encoder_out = encoder_output.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, fc_dims, p=0.3,
                 embeddings=None):
        super().__init__()
        self.EncoderCNN = Encoder()
        self.DecoderLSTM = Decoder(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, fc_dims, p,
                                   embeddings)

    def forward(self, imgs, caps):
        features = self.EncoderCNN(imgs)
        out = self.DecoderLSTM(features, caps)
        return out
