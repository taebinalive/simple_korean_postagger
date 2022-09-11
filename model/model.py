import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention

from utils import init_weight

torch.manual_seed(999)
torch.cuda.manual_seed(999)


class Tagger:
    def __init__(self, config):
        super(Tagger, self).__init__()
        self.vocab = json.load(open(config.tokenizer_path, 'r', encoding='utf-8'))
        self.tag2idx = json.load(open(os.path.join(config.dic_path, "tag_to_idx.json"), 'r', encoding='utf-8'))
        self.idx2tag = json.load(open(os.path.join(config.dic_path, "idx_to_tag.json"), 'r', encoding='utf-8'))
        self.label_size = len(self.tag2idx)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLSTM(Tagger, nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__(config)
        self.token_embedding = nn.Embedding(len(self.vocab), config.emb_dim, padding_idx=self.vocab["[PAD]"])

        self.inner_conf = config.BiLSTM
        self.lstm = getattr(nn, self.inner_conf["rnn_base"])(input_size=config.emb_dim,
                                                             hidden_size=config.hidden_dim // 2,
                                                             num_layers=self.inner_conf["num_layers"],
                                                             bidirectional=True, batch_first=True)
        init_weight(self.lstm)
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.fc = nn.Linear(config.hidden_dim, len(self.idx2tag))

        init_weight(self.fc)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        # x: [batch_size, max_seq_len]
        out = self.token_embedding(x)
        # out: [batch_size, max_seq_len, emb_dim]

        out, *_ = self.lstm(out)
        out = self.dropout(out)

        out = self.fc(self.layer_norm(self.dropout(F.relu(out))))

        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BLAN(nn.Module):
    def __init__(self, config, inner_conf):
        super(BLAN, self).__init__()
        self.lstm = getattr(nn, inner_conf["rnn_base"])(input_size=2 * config.hidden_dim,
                                                        hidden_size=config.hidden_dim // 2, num_layers=1,
                                                        bidirectional=True, batch_first=True)

        init_weight(self.lstm)
        self.label_attention = Attention(inner_conf["num_heads"], config.hidden_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, label_emb, mask):
        out, *_ = self.lstm(x)
        out = self.dropout(out)
        label_attention = self.label_attention(out, label_emb, label_emb, mask)

        out = torch.cat([out, label_attention], dim=-1)

        return out


class BiLSTMLAN(Tagger, nn.Module):
    def __init__(self, config):
        super(BiLSTMLAN, self).__init__(config)
        self.token_embedding = nn.Embedding(len(self.vocab), config.emb_dim, padding_idx=self.vocab["[PAD]"])

        # set label embedding dim to hidden_dim
        self.label_embedding = nn.Embedding(self.label_size, config.hidden_dim, padding_idx=self.tag2idx["[PAD]"])
        self.inner_conf = config.BiLSTMLAN

        self.first_lstm = getattr(nn, self.inner_conf["rnn_base"])(
            input_size=config.emb_dim,
            hidden_size=config.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True)
        self.last_lstm = getattr(nn, self.inner_conf["rnn_base"])(
            input_size=2 * config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True)

        init_weight(self.first_lstm)
        init_weight(self.last_lstm)

        self.first_attention = Attention(self.inner_conf["num_heads"], config.hidden_dim, config.dropout)
        self.last_attention = Attention(self.inner_conf["num_heads"], config.hidden_dim, config.dropout)
        self.lan_layers = nn.ModuleList(
            [BLAN(config, self.inner_conf) for _ in range(self.inner_conf["num_layers"] - 2)])

        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.fc = nn.Linear(config.hidden_dim, len(self.idx2tag))

        init_weight(self.fc)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        # x: [batch_size, max_seq_len]
        batch_size = x.shape[0]

        out = self.token_embedding(x)
        labels = torch.arange(0, self.label_size).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        label_embeddings = self.label_embedding(labels)
        out, *_ = self.first_lstm(out)
        out = self.dropout(out)

        label_attention = self.first_attention(out, label_embeddings, label_embeddings, mask)
        out = torch.cat([out, label_attention], dim=-1)

        for ll in self.lan_layers:
            out = ll(out, label_embeddings, mask)

        out, *_ = self.last_lstm(out)
        out = self.last_attention(out, label_embeddings, label_embeddings, mask)

        # out: [batch_size, max_seq_len, 2 * hidden_dim]

        """
        activation list: relu, selu, LeakyReLU, tanh, elu, silu
        """
        out = self.dropout(F.relu(out))
        # out = self.fc(out)
        out = self.fc(self.layer_norm(out))

        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNLayer(nn.Module):
    def __init__(self, config, inner_conf):
        super(CNNLayer, self).__init__()
        self.inner_conf = inner_conf
        self.max_len = config.max_seq_len
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(config.emb_dim, config.emb_dim, kernel_size=kernel_size) for
             kernel_size in range(2, self.inner_conf["max_kernel_size"] + 1)])

        init_weight(self.conv_layers)

        #self.fc = nn.Linear(len(self.conv_layers) * config.emb_dim, config.emb_dim)
        #init_weight(self.fc)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        conv_outputs = []

        for layer in self.conv_layers:
            conv = layer(x)
            conv = F.relu(conv)
            max_pooled = F.max_pool1d(conv, conv.shape[2])
            conv_outputs.append(max_pooled)

        outs = torch.cat(conv_outputs, dim=1).permute(0, 2, 1).squeeze(1)
        outs = self.dropout(outs)

        return outs


class CNN(Tagger, nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        self.token_embedding = nn.Embedding(len(self.vocab), config.emb_dim, padding_idx=self.vocab["[PAD]"])
        self.inner_conf = config.CNN
        self.max_len = config.max_seq_len

        self.cnn_layer = nn.ModuleList(
            [CNNLayer(config, self.inner_conf) for _ in range(self.inner_conf["num_layers"])])

        #self.layer_norm = nn.LayerNorm(4 * config.emb_dim, eps=config.layer_norm_epsilon)
        self.hidden_layer = nn.Linear((self.inner_conf["max_kernel_size"]-1) * config.emb_dim, (len(self.idx2tag) + (self.inner_conf["max_kernel_size"]-1) * config.emb_dim) // 2)
        self.fc = nn.Linear((len(self.idx2tag) + (self.inner_conf["max_kernel_size"]-1) * config.emb_dim) // 2, len(self.idx2tag))
        init_weight(self.hidden_layer)
        init_weight(self.fc)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        # out: (batch_size, max_seq_len, hidden_dim)
        out = self.token_embedding(x)

        for layer in self.cnn_layer:
            out = layer(out)

        out = self.hidden_layer(out)
        out = self.dropout(F.relu(out))
        #out = self.fc(self.layer_norm(out))
        out = self.fc(out)

        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
