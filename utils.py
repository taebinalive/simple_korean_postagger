import json
import os
import torch
import re

from pathlib import Path


def create_mask(input, pad_idx):
    # input: (batch_size, max_seq_len)
    max_len = input.shape[-1]
    # (batch_size, max_seq_len, max_seq_len)
    # out = (input != pad_idx).unsqueeze(1).repeat(1, max_len, 1)

    # (batch_size, max_seq_len)
    out = (input != pad_idx)

    return out


def encode(text, vocab, max_len):
    return [Syllable("[BOS]", vocab)] + \
           [Syllable(char, vocab) for char in text] + \
           [Syllable("[EOS]", vocab)] + \
           [Syllable("[PAD]", vocab)] * (max_len - len(text) - 2)


def truncate(tokens, max_len):
    while True:
        if len(tokens) > max_len - 2:
            tokens.pop()
        else:
            break


def init_weight(layer):
    for name, param in layer.named_parameters():
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0.0)


def make_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def get_tag_list(words):
    tags = []
    for word in words:
        tags.append(word.split("\t")[1])
    return " [BLK] ".join(tags).split(" ")


class RNNDataLoader:
    def __init__(self, config, mode='train'):
        self.tag2idx = json.load(open(os.path.join(config.dic_path, "tag_to_idx.json")))
        self.idx2tag = json.load(open(os.path.join(config.dic_path, "idx_to_tag.json")))
        self.contexts = [sent for sent in re.split("\n\n+", open(os.path.join(config.data_path, config.file_suffix + "." + mode), 'r', encoding='utf-8').read())]
        self.tokenizer = Tokenizer(config.tokenizer_path)
        self.max_seq_len = config.max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = config.batch_size
        self.idx = 0

    def get_label(self, tag):
        return self.tag2idx[tag]

    def __iter__(self):
        while True:
            text = []
            label = []
            raw_sentences = []
            for _ in range(self.batch_size):
                try:
                    sentence = self.contexts[self.idx].strip()
                except IndexError:
                    break

                raw_sen, *words = sentence.split("\n")
                raw_sen = list(raw_sen)

                tag_list = get_tag_list(words)
                truncate(raw_sen, self.max_seq_len)
                truncate(tag_list, self.max_seq_len)

                input_sen = Sentence(raw_sen, tag_list, self.tokenizer.vocab, self.tag2idx, self.max_seq_len)

                text.append([syl.id for syl in input_sen.syllables])
                label.append(input_sen.labels)
                raw_sentences.append(input_sen.sent)
                self.idx += 1

            text_tensors = torch.as_tensor(text, dtype=torch.long) if len(text) > 0 else None
            label_tensors = torch.as_tensor(label, dtype=torch.long) if len(label) > 0 else None
            mask_tensors = create_mask(text_tensors, self.tokenizer.vocab["[PAD]"]) if len(text) > 0 else None

            if text_tensors is not None and label_tensors is not None:
                yield text_tensors, label_tensors, mask_tensors, raw_sentences
            else:
                self.idx = 0
                return


class CNNDataLoader:
    def __init__(self, config, mode="train"):
        self.tag2idx = json.load(open(os.path.join(config.dic_path, "tag_to_idx.json")))
        self.idx2tag = json.load(open(os.path.join(config.dic_path, "idx_to_tag.json")))
        self.ws = config.CNN["window_size"]
        self.sentences = [sent for sent in re.split("[\r\n]\n+", open(
            os.path.join(config.data_path, config.file_suffix + "." + mode), 'r', encoding='utf-8').read())]
        self.contexts = []

        self.tokenizer = Tokenizer(config.tokenizer_path)
        self.make_context()
        self.max_seq_len = config.max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.batch_size
        self.idx = 0

    def make_context(self):
        for sent in self.sentences:
            sent, *eojuls = sent.split("\n")
            raw_sen = self.ws * ["[PAD]"] + list(sent.strip()) + self.ws * ["[PAD]"]
            # if eojuls[0] == "그\tI-MM":
            #     a = 1
            try:
                tag_list = self.ws * ["[PAD]"] + get_tag_list(eojuls) + self.ws * ["[PAD]"]
            except IndexError:
                continue

            for i in range(self.ws, len(tag_list) - self.ws):
                try:
                    context = [self.tokenizer.vocab[ch if ch != ' ' else "[BLK]"] for ch in raw_sen[i-self.ws:i+self.ws+1]]
                    labels = [self.tag2idx[tag_list[i]]]

                    self.contexts.append((context, labels))
                except KeyError:
                    continue

    def get_label(self, tag):
        return self.tag2idx[tag]

    def __iter__(self):
        while True:
            text = []
            label = []
            for _ in range(self.batch_size):
                try:
                    context = self.contexts[self.idx]
                except IndexError:
                    break

                text.append(context[0])
                label.append(context[1])
                self.idx += 1

            text_tensors = torch.as_tensor(text, dtype=torch.long) if len(text) > 0 else None
            label_tensors = torch.as_tensor(label, dtype=torch.long) if len(label) > 0 else None
            mask_tensors = create_mask(text_tensors, self.tokenizer.vocab["[PAD]"]) if len(text) > 0 else None

            if text_tensors is not None and label_tensors is not None:
                yield text_tensors, label_tensors, mask_tensors
            else:
                self.idx = 0
                return


class Sentence:
    def __init__(self, sent, tags, vocab, tag_map, max_len):
        self.sent = sent
        self.syllables = encode(sent, vocab, max_len)
        self.labels = [tag_map["[BOS]"]] + [tag_map[tag] for tag in tags] + [tag_map["[EOS]"]] + \
                      [tag_map["[PAD]"]] * (max_len - len(sent) - 2)

    def __repr__(self):
        return f'sentence: {self.sent}' + '\n' + '\n'.join([str(syl) + ', ' + label for syl, label in zip(self.syllables, self.labels)])


class Syllable:
    def __init__(self, char, vocab):
        self.syl = "[UNK]"
        self.id = vocab[self.syl]

        if char == ' ':
            self.syl = "[BLK]"
            self.id = vocab[self.syl]
        else:
            if char in vocab:
                self.syl = char
                self.id = vocab[self.syl]

    def __repr__(self):
        return f'syllable: {self.syl}, vocab_id: {self.id}'


class Tokenizer:
    def __init__(self, path):
        self.vocab = json.load(open(path, 'r', encoding='utf-8'))


    def get_tokens(self, input):
        tokens = []

        return tokens
