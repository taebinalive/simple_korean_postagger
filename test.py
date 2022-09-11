import argparse
import re
import time
import torch
import os

from model import model as m
from config import Config
from utils import truncate, create_mask, encode


def decode_eojul(morphs):
    """
    :param morphs: ['예/I-NNG', '방/I-NNG', '접/B-NNG', '종/I-NNG', '률/I-XSN', '이/I-JKS']
    :return: 예방/NNG + 접종/NNG + 률/XSN + 이/JKS
    """

    curr_syl = ''
    curr_pos = ''
    res = []

    for idx, morph in enumerate(morphs):
        syl_, tag = morph.rsplit("/", 1)
        iob_, pos_ = tag.split("-")
        if idx == 0:
            curr_syl += syl_
            curr_pos = pos_

            if len(morphs) == 1:
                res.append(curr_syl + '/' + pos_)
            continue

        if iob_ == 'B':
            res.append(curr_syl + '/' + curr_pos)
            curr_syl = syl_
            curr_pos = pos_
            if idx == len(morphs) - 1:
                res.append(curr_syl + '/' + pos_)
            continue

        if curr_pos == pos_:
            curr_syl += syl_
        else:
            res.append(curr_syl + '/' + curr_pos)
            curr_syl = syl_

        if idx == len(morphs) - 1:
            res.append(curr_syl + '/' + pos_)

        curr_pos = pos_

    return res



def decode(input_sent, tags, restore_dic, tag_map, blk_idx):
    res = []
    inputs = ''.join(input_sent).split(" ")
    out_tags = []
    tmp_tags = []
    for tag in tags:
        if tag != blk_idx:
            tmp_tags.append(tag)
        else:
            out_tags.append(tmp_tags)
            tmp_tags = []
    if tmp_tags:
        out_tags.append(tmp_tags)

    for idx, (eojul, eojul_tags) in enumerate(zip(inputs, out_tags)):
        morphs = []
        for syl, tag_idx in zip(eojul, eojul_tags):
            tag = tag_map[str(tag_idx)]
            if ':' not in tag:
                morphs.append(syl + '/' + tag)
            else:
                key = syl + '/' + tag
                if key in restore_dic:
                    morphs.extend(restore_dic[key].split(" "))
                else:
                    morphs.append(syl + '/I-XXX')

        eojul_res = decode_eojul(morphs)
        res.append(' + '.join(eojul_res))

    return inputs, res


def load_restoration(path):
    res = {}
    with open(os.path.join(path, 'restore.dic'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            entry, val = line.strip().split("\t")
            res[entry] = val
    return res


def main(args):
    config = Config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert args.model in ("CNN", "BiLSTM", "BiLSTMLAN"), "model_type should be 'CNN' or 'BiLSTM' or 'BiLSTMLAN'"

    if args.model == "CNN":
        config.emb_dim = 150
    elif args.model == "BiLSTM":
        config.emb_dim = 100
        config.hidden_dim = 600
    else:
        config.emb_dim = 100
        config.hidden_dim = 400

    m_type = args.model
    model = getattr(m, m_type)(config)
    checkpoint = torch.load(os.path.join('./checkpoint', m_type, 'model.pt'), map_location='cuda' if torch.cuda.is_available() else 'cpu')
    # checkpoint = torch.load('./checkpoint/config_lstm_l4_hidden_200_epoch_19_0.286.pt', map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'model loaded')

    restore_dic = load_restoration(config.dic_path)

    while True:
        print('input >> ')
        input_sent = input()
        # input_sent = "우리 집에선 회사까지 1시간이 걸렸다."
        # input_sent = "냄비에 끓여서 먹어라."
        start = time.time()
        input_sent = re.sub("\s+", " ", input_sent)

        if not input_sent:
            continue

        input_sent = list(input_sent)

        if m_type == "CNN":
            ws = getattr(config, m_type)["window_size"]
            sentence = ws * ["[PAD]"] + input_sent + ws * ["[PAD]"]
            contexts = []

            for i in range(ws, len(sentence) - ws):
                contexts.append([model.vocab[c if c != ' ' else "[BLK]"] for c in sentence[i - ws:i + ws + 1]])

            input_tensor = torch.as_tensor(contexts, dtype=torch.long)
            logits = model(input_tensor, None)
            logits = logits.view(-1, logits.shape[-1])
            logits = logits.argmax(-1)

            tag_indices = logits.tolist()

            result = decode(input_sent, tag_indices, restore_dic, model.idx2tag, model.tag2idx["[BLK]"])
            end = time.time()
            for eojul, morphs in zip(*result):
                print(eojul + '\t' + morphs)
            print(f"elapsed: {end - start:.2f} s")
        else:
            truncate(input_sent, config.max_seq_len)
            sentence = encode(input_sent, model.vocab, config.max_seq_len)
            input_tensor = torch.as_tensor([[syl.id for syl in sentence]], dtype=torch.long)
            mask = create_mask(input_tensor, model.vocab["[PAD]"])
            logits = model(input_tensor, mask).squeeze(0)
            # logits = logits.view(-1, logits.shape[-1])
            logits = logits.argmax(-1)

            tag_indices = logits.tolist()
            tag_indices = tag_indices[1:tag_indices.index(model.tag2idx["[EOS]"])]

            result = decode(input_sent, tag_indices, restore_dic, model.idx2tag, model.tag2idx["[BLK]"])
            end = time.time()
            for eojul, morphs in zip(*result):
                print(eojul + '\t' + morphs)
            print(f'elapsed: {end-start:.2f} s')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path", required=True)
    parser.add_argument("--model", type=str, help="model type", required=True)

    args = parser.parse_args()
    main(args)