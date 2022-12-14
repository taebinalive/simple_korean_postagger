import io
import json
import os
from collections import defaultdict
from argparse import ArgumentParser

from exception import AlignError
from jamo import decompose, compose
from corpus import Corpus


def load_gold_align(path):
    res = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, val = line.strip().split('\t')
            res[key].add(val)
    return res


def gold_align(src, dst):
    aligned = False
    if len(src) == 0:
        return False
    for _ in src:
        if aligned:
            break
        aligned = True
        if len(_) != len(dst):
            aligned = False
            continue
        for c0, (c1, tag) in zip(_, dst):
            if c0 != c1:
                aligned = False

    return aligned


def _decompose(word):
    return ''.join([e for ch in word for e in decompose(ch)]).replace(" ", "")


def get_raw_morph(morph_list):
    return ''.join([morph.lexicon for morph in morph_list])


def is_only_cho(jamo):
    return jamo[1] == '' and jamo[2] == ' '


def is_only_noun(morphs):
    for morph in morphs:
        if morph[1].split("-")[1] not in ("NNG", "NNP", "XR"):
            return False
    return True


class Aligner:
    def __init__(self, args):
        self.args = args
        self.corpus = Corpus(args.data_dir)
        self.unmapped = []
        self.aligned = open(args.aligned, 'w', encoding='utf-8')
        self.gold_align = load_gold_align(os.path.join(args.dic_path, 'gold_align.map'))
        self.output_path = open(args.output, 'w', encoding='utf-8')
        self.restore_path = open(os.path.join(args.dic_path, 'restore.dic'), 'w', encoding='utf-8')
        self.vocab_path = open(os.path.join(args.vocab_path, 'vocab.json'), 'w', encoding='utf-8')
        self.restore_map = defaultdict(list)
        self.tag_set = set()
        self.vocab = {
            "[UNK]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[PAD]": 3,
            "[BLK]": 4,
        }
        self.char_count_map = {}
        self.vocab_idx = len(self.vocab)
        self.tag_to_idx = {}
        self.idx_to_tag = {}

    def close_all_files(self):
        for e in self.__dict__:
            attr = getattr(self, e)
            if isinstance(attr, io.TextIOWrapper):
                attr.close()

    def is_gold_align(self, wc, mc):
        gold_mc = self.gold_align[wc]
        return gold_align(gold_mc, mc)

    def convert(self, sent, aligned_morphs):
        self.tag_set |= {"[UNK]", "[BOS]", "[EOS]", "[PAD]", "[BLK]"}
        res_str = sent.sentence.strip() + '\n'

        # make morpheme restoration dic
        for word, morphs in zip(sent.sentence.split(" "), aligned_morphs):
            res_str += word + '\t'
            tags = []
            for char, align in zip(word, morphs):
                if len(align) == 1 and char == align[0][0]:
                    tags.append(align[0][1])
                    self.tag_set.add(align[0][1])
                    continue

                tag = ':'.join([e[1] for e in align])
                val = self.restore_map[(char, tag)]

                if val:
                    if align in val:
                        tag = f'{tag}:{val.index(align)}'
                    else:
                        tag = f'{tag}:{len(val)}'
                        val.append(align)
                else:
                    val.append(align)
                    tag = f'{tag}:0'

                tags.append(tag)

            res_str += ' '.join(tags) + '\n'
            self.tag_set |= set(tags)

        # return tagged sentence str
        return res_str

    def write_unmapped(self):
        with open(self.args.unmapped, 'w', encoding='utf-8') as f:
            f.write("\n".join(sorted(self.unmapped)))

    def write_restore(self):
        for k, v in list(self.restore_map.items()):
            for i in range(len(v)):
                self.restore_path.write(k[0] + '/' + k[1] + ':' + str(i) + '\t' +
                                        ' '.join(['/'.join(e) for e in v[i]]) + '\n')

    def write_tag_map(self):
        for i, tag in enumerate(self.tag_set):
            self.tag_to_idx[tag] = i
            self.idx_to_tag[i] = tag

        with open(os.path.join(self.args.dic_path, 'tag_to_idx.json'), 'w', encoding='utf-8') as f:
            json.dump(self.tag_to_idx, f, indent=4, ensure_ascii=False)
        with open(os.path.join(self.args.dic_path, 'idx_to_tag.json'), 'w', encoding='utf-8') as f:
            json.dump(self.idx_to_tag, f, indent=4, ensure_ascii=False)

    def _update(self, key):
        val = self.char_count_map.get(key, 0)
        self.char_count_map[key] = val + 1

        if key not in self.vocab.keys():
            self.vocab[key] = self.vocab_idx
            self.vocab_idx += 1

    def update_vocab(self, sent):
        for eojul in sent.eojuls:
            for char in eojul.eojul:
                self._update(char)

            for morph in eojul.morphs:
                for char in morph.lexicon:
                    self._update(char)

    def write_vocab(self):
        json.dump(self.vocab, self.vocab_path, indent=4, ensure_ascii=False)

    def align(self):
        print(f'total # of sentences in corpus: {len(self.corpus.sentences)}')
        print(f'total # of eojuls in corpus: {self.corpus.total_eojuls}')
        aligned_sentences = 0
        aligned_eojuls = 0
        for sentence in self.corpus.sentences:
            res = []

            try:
                for eojul in sentence.eojuls:
                    res.append(self._align(sentence.sentence, eojul))
            except AlignError:
                continue
            except IndexError:
                continue
            else:
                instance_str = self.convert(sentence, res)
                aligned_sentences += 1
                aligned_eojuls += len(instance_str.strip().split('\n')) - 1
                self.output_path.write(instance_str + '\n')
                self.update_vocab(sentence)

        print(f'total # of aligned sentences: {aligned_sentences}')
        print(f'total # of aligned eojuls: {aligned_eojuls}')

        self.write_unmapped()
        self.write_restore()
        self.write_tag_map()
        self.write_vocab()

        ####
        a = dict(sorted(self.char_count_map.items(), key=lambda x: -x[1]))
        with open('char_count.json', 'w', encoding='utf-8') as f:
            json.dump(a, f, indent=4, ensure_ascii=False)
        ####

        # self.write_plain_text()

    def _align(self, sent, eojul):
        res = []
        for e_idx, morph in enumerate(eojul.morphs):
            for m_idx, char in enumerate(morph.lexicon):
                iob = 'I'
                if m_idx == 0 and e_idx > 0 and eojul.morphs[e_idx - 1].tag == morph.tag:
                    iob = 'B'
                res.append((char, f'{iob}-{morph.tag}'))

        raw_morph_str = get_raw_morph(eojul.morphs)

        # ???????????? <-> ?????????/NNG + ???/JKG
        if eojul.eojul == raw_morph_str:
            return [[e] for e in res]

        eojul_jamo_str = _decompose(eojul.eojul)
        morph_jamo_str = _decompose(raw_morph_str)

        # ???????????? <-> ??????/VX + ?????????/EC, ????????? <-> ??????/NNG + ???/XSA + ???/ETM
        if eojul_jamo_str == morph_jamo_str:
            res = self._align_norm(eojul.eojul, res)
        else:
            # ?????? ??????????????? ????????? ??? ?????? ????????? ?????? ???
            res = self._align_conjugation(sent, eojul, res)
        if len(res) != len(eojul.eojul):
            raise AlignError("length of eojul and aligned morpheme is not same")

        return res

    def _align_norm(self, raw_word, morph_list):
        """
        :param raw_word: ?????????
        :param morph_list: [(???, I-NNG), (???, I-NNG), (???, I-XSV), (???, I-ETM)]
        :return: new aligned list
                 [[(???, I-NNG)], [(???, I-NNG)], [(???, I-XSV), (???, I-ETM)]]
        """

        new_char_aligned = []
        morph_idx = 0

        for word_char in raw_word:
            if word_char == morph_list[morph_idx][0]:
                new_char_aligned.append([morph_list[morph_idx]])
            else:
                morph_first_jamo = decompose(morph_list[morph_idx][0])
                morph_second_jamo = decompose(morph_list[morph_idx + 1][0]) if morph_idx + 1 < len(morph_list) else (
                    '', '', '')
                composed = compose(morph_first_jamo[0], morph_first_jamo[1], morph_second_jamo[0])
                if word_char == composed:
                    new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                    morph_idx += 1
                else:
                    raise AlignError("align normalized error")

            morph_idx += 1

        self.aligned.write(raw_word + "\t" + "  ###  ".join([str(ch) for ch in new_char_aligned]) + "\n")

        return new_char_aligned

    def _align_conjugation(self, sent, eojul, morph_list):
        """
        :param eojul: ?????????
        :param morph_list: [(???, I-VV), (???, I-EC), (???, I-EC)]
        :return: new aligned list
                 [[(???, I-VV)], [(???, I-EC)], [(???, I-EC)]]
        """

        sent_no = eojul.no
        raw_word = eojul.eojul
        new_char_aligned = []

        morph_idx = 0
        word_idx = 0
        loop_cnt = 0
        try:
            while word_idx < len(raw_word):
                loop_cnt += 1
                if loop_cnt > 30:
                    raise AlignError("loop count limit exceed in 'align_conjugation'")

                curr_word_char = raw_word[word_idx]
                next_word_char = raw_word[word_idx + 1] if word_idx + 1 < len(raw_word) else ''
                curr_morph_char = morph_list[morph_idx][0]
                next_morph_char = morph_list[morph_idx+1][0] if morph_idx + 1 < len(morph_list) else ''
                curr_morph_tag = morph_list[morph_idx][1].split("-")[1]
                next_morph_tag = morph_list[morph_idx+1][1].split("-")[1] if morph_idx + 1 < len(morph_list) else ''
                curr_word_jamo = decompose(curr_word_char)
                curr_morph_jamo = decompose(curr_morph_char)

                ##########################
                # check if (curr_word_char, curr_morph_char) exists in gold align
                try:
                    pre_result, w_count, m_count = self._pre_match(raw_word, morph_list, word_idx, morph_idx)
                except ValueError:
                    pre_result, w_count, m_count = [], 0, 0
                    print(sent, raw_word)
                if pre_result:
                    new_char_aligned.extend(pre_result)
                    word_idx += w_count
                    morph_idx += m_count
                    continue
                ##########################

                if word_idx >= len(raw_word):
                    self.aligned.write(raw_word + "\t" +
                                       "  ###  ".join([str(char) for char in new_char_aligned]) + "\n")
                    break

                if curr_word_char == curr_morph_char:
                    if morph_idx + 1 < len(morph_list):
                        # ??? -> ??? + ???, ??? -> ??? + ???
                        if next_morph_tag in ["ETM", "EC", "EF"]:
                            if curr_morph_tag in ["VV", "VA", "VX"]:

                                if next_morph_char in ['???', '???'] and next_word_char != '???':
                                    if word_idx + 1 < len(raw_word):
                                        if morph_idx + 2 == len(morph_list):
                                            if next_word_char == next_morph_char or \
                                                    (next_morph_char == '???' and next_word_char == '???'):
                                                # ????????? -> ?????? + ???
                                                new_char_aligned.append([morph_list[morph_idx]])
                                                word_idx += 1
                                                morph_idx += 1
                                            else:
                                                new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                                                word_idx += 1
                                                morph_idx += 2
                                        elif curr_word_char == curr_morph_char and next_word_char == next_morph_char:
                                            new_char_aligned.append([morph_list[morph_idx]])
                                            word_idx += 1
                                            morph_idx += 1
                                        elif decompose(next_word_char)[0] == '???':
                                            new_char_aligned.append([morph_list[morph_idx]])
                                            word_idx += 1
                                            morph_idx += 1
                                        elif next_word_char == '???' and next_morph_char == '???':
                                            new_char_aligned.append([morph_list[morph_idx]])
                                            word_idx += 1
                                            morph_idx += 1
                                        else:
                                            new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                                            word_idx += 1
                                            morph_idx += 2
                                    else:
                                        new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                                        word_idx += 1
                                        morph_idx += 2
                                elif curr_word_char == curr_morph_char and next_morph_char in ['???', '???', '???']:
                                    # ??? -> ??? + ???
                                    new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                                    word_idx += 1
                                    morph_idx += 2
                                else:
                                    new_char_aligned.append([morph_list[morph_idx]])
                                    word_idx += 1
                                    morph_idx += 1
                            else:
                                # ??? + ???/EC + ???/EC -> [??????] [???] [???]
                                new_char_aligned.append([morph_list[morph_idx]])
                                word_idx += 1
                                morph_idx += 1

                        else:
                            new_char_aligned.append([morph_list[morph_idx]])
                            word_idx += 1
                            morph_idx += 1
                    else:
                        new_char_aligned.append([morph_list[morph_idx]])
                        word_idx += 1
                        morph_idx += 1
                elif curr_morph_tag == "VA" and curr_word_jamo[2] == ' ' and \
                        compose(curr_word_jamo[0], curr_word_jamo[1], '???') == curr_morph_char:
                    # ????????? -> ?????? + ???
                    # ?????????????????? -> ????????? + ?????????
                    new_char_aligned.append([morph_list[morph_idx]])
                    word_idx += 1
                    morph_idx += 1

                elif morph_idx + 2 < len(morph_list) and next_morph_tag == "VCP" and \
                        (morph_list[morph_idx+2][1].split("-")[1] == "ETM" or
                         morph_list[morph_idx+2][1].split("-")[1] == "EC"):
                    # '???' ?????? + VCP + [???, ???]
                    # ????????? -> [???] [?????????] [???]
                    new_char_aligned.append(morph_list[morph_idx:morph_idx + 3])
                    morph_idx += 3
                    word_idx += 1
                elif morph_idx + 1 < len(morph_list) and curr_word_jamo[2] == '???' and next_morph_tag == "EP":
                    # '???' ?????? + EP
                    # ??? -> [???] + [???], ?????? -> ??? + [??????], ??? -> [???] [???]
                    if curr_morph_tag in ("VV", "VA", "VX", "VCP", "XSV", "XSA", "EP") or \
                            curr_morph_tag.startswith("S"):
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1

                elif morph_idx + 1 < len(morph_list) and curr_morph_jamo[2] == '???' and curr_word_jamo[2] == '???':
                    # '???' ?????????
                    if next_morph_tag.startswith("E"):
                        new_char_aligned.append([morph_list[morph_idx]])
                        morph_idx += 1
                        word_idx += 1

                elif curr_word_jamo[2] == ' ' and curr_morph_jamo[2] == '???':
                    # '???' ?????????
                    if morph_idx + 1 < len(morph_list):
                        if next_morph_tag == "ETM":
                            # ?????? -> ?????? + ???
                            next_morph_jamo = decompose(next_morph_char)
                            if next_morph_jamo[1] == '':
                                new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                                morph_idx += 2
                                word_idx += 1
                            else:
                                new_char_aligned.append([morph_list[morph_idx]])
                                morph_idx += 1
                                word_idx += 1
                        elif next_morph_tag in ["EF", "EC"] and next_morph_char == '???':
                            # ??? -> ??? + ???
                            # ?????? -> ??? + ??????
                            next_word_cho = decompose(next_word_char)[0]
                            if '???' <= next_word_cho <= '???':
                                new_char_aligned.append([morph_list[morph_idx]])
                                morph_idx += 1
                                word_idx += 1
                            else:
                                new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                                morph_idx += 2
                                word_idx += 1
                        else:
                            new_char_aligned.append([morph_list[morph_idx]])
                            morph_idx += 1
                            word_idx += 1
                elif morph_idx + 2 < len(morph_list) and next_morph_char == '???' and \
                    (curr_morph_tag.startswith("N") or curr_morph_tag in ("ETN", "MAG")) and \
                        curr_word_char == compose(curr_morph_jamo[0],
                                                  curr_morph_jamo[1],
                                                  decompose(morph_list[morph_idx+2][0])[0]):
                    # N + VCP + EF
                    # ???????????? -> ???[???+???+???]??????
                    # ????????? -> [???+???+???]??????
                    # ?????????????????? -> ?????????[???+???+???]??????
                    new_char_aligned.append(morph_list[morph_idx:morph_idx+3])
                    morph_idx += 3
                    word_idx += 1

                elif curr_word_jamo[2] == '???' and word_idx + 1 < len(raw_word):
                    # '???' ?????????
                    raw_second_jamo = decompose(raw_word[word_idx + 1])
                    next_jamo = decompose(next_morph_char) if morph_idx + 1 < len(morph_list) else ('', '', '')

                    if morph_idx + 1 < len(morph_list) and raw_second_jamo[0] == '???' and decompose(next_morph_char)[2] == ' ':
                        # if decompose(next_morph_char)[1] != '':
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        new_char_aligned.append(morph_list[morph_idx+2:morph_idx+3])
                        word_idx += 2
                        morph_idx += 3
                    elif morph_idx + 1 < len(morph_list) and curr_morph_jamo[2] == ' ' and next_jamo[0] in ['???', '???']:
                        if [e[0] for e in decompose(raw_word[word_idx])] == [e for e in decompose(curr_morph_char) + decompose(next_morph_char) if e != '' and e != ' ']:
                            # ?????? ??? + ??? ??? -> [??????]
                            new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                            word_idx += 1
                            morph_idx += 2
                    elif morph_idx + 1 < len(morph_list) and curr_morph_jamo[2] == '???' and \
                        next_jamo[0] == '???':
                        # ?????? -> ?????? + ???, ?????? -> ?????? + ???
                        new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                        morph_idx += 2
                        word_idx += 1

                elif curr_word_jamo[2] == ' ' and curr_morph_jamo[2] == '???' and word_idx + 1 < len(raw_word):
                    # '???' ?????????
                    raw_second_jamo = decompose(raw_word[word_idx + 1])
                    if morph_idx + 1 < len(morph_list):
                        # ????????? (?????? -> ??? + ???)
                        if raw_second_jamo[2] != ' ' and raw_second_jamo[2] == next_morph_char:
                            # 1:1 align
                            new_char_aligned.append(morph_list[morph_idx:morph_idx + 1])
                            new_char_aligned.append(morph_list[morph_idx + 1:morph_idx + 2])
                            word_idx += 2
                            morph_idx += 2
                        elif raw_second_jamo[2] != ' ' or (raw_second_jamo[1] == '???' and next_morph_char == '???') or ((raw_second_jamo[1] == '???' or raw_second_jamo[1] == '???') and next_morph_char == '???') or (raw_second_jamo[1] == '???' and next_morph_char == '???'):
                            new_char_aligned.append(morph_list[morph_idx:morph_idx+1])
                            new_char_aligned.append(morph_list[morph_idx+1:morph_idx+2])
                            # new_char_aligned.append(morph_list[morph_idx+1:morph_idx+2])
                            word_idx += 2
                            morph_idx += 2

                elif word_idx + 1 < len(raw_word) and curr_word_jamo[2] == ' ' and curr_morph_jamo[2] == '???' and next_morph_tag.startswith("E"):
                    # '???' ?????????
                    new_char_aligned.append([morph_list[morph_idx]])
                    word_idx += 1
                    morph_idx += 1

                elif curr_morph_tag == "NNB" and next_morph_tag.startswith("J"):
                    if morph_idx + 1 < len(morph_list):
                        # ???????????? + ??????
                        # ??? -> ??? + ???
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1
                elif (curr_morph_tag == "VV" or curr_morph_tag == "VA") and \
                    (next_morph_tag == "ETM" and decompose(next_morph_char)[1] == '' and decompose(next_morph_char)[2] == ' '):
                    if morph_idx + 1 < len(morph_list):
                        # ?????? + ETM
                        # ??? -> ??? + ???
                        new_char_aligned.append(morph_list[morph_idx:morph_idx + 2])
                        morph_idx += 2
                        word_idx += 1
                elif curr_morph_tag == "VCP":
                    if curr_word_char == next_morph_char[0] or curr_word_jamo[2] == next_morph_char[0]:
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1
                    elif morph_idx + 2 < len(morph_list):
                        # ???????????? -> ????????? + [???+???+???]
                        second_morph_jamo = decompose(morph_list[morph_idx+2][0])
                        next_morph_jamo = decompose(next_morph_char)
                        if is_only_cho(second_morph_jamo) and compose(next_morph_jamo[0], next_morph_jamo[1], second_morph_jamo[0]) == curr_word_char:
                            new_char_aligned.append(morph_list[morph_idx:morph_idx+3])
                            morph_idx += 3
                            word_idx += 1

                elif curr_morph_tag in ["VV", "VA", "VX", "XSV", "XSA"]:
                    next_morph_jamo = decompose(next_morph_char) if morph_idx + 1 < len(morph_list) else ('', ' ', ' ')
                    if next_morph_tag == "EF" or next_morph_tag == "EC":
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1
                    elif next_morph_jamo[1] == '':
                        # ?????? -> ??? + ??????
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1
                elif curr_morph_jamo[0] == '???' and curr_word_jamo[0] == '???' and curr_morph_tag in ["EC", "EP"]:
                    # ????????? -> ??????, ???????????? -> ???????????? ??????
                    new_char_aligned.append([morph_list[morph_idx]])
                    morph_idx += 1
                    word_idx += 1
                elif curr_morph_tag == "EP":
                    next_morph_jamo = decompose(next_morph_char) if morph_idx + 1 < len(morph_list) else ('', '', '')
                    # EP + ETM
                    # ?????? -> [???] [???+???]
                    if (curr_word_char == '???' and curr_morph_char == '???') or (curr_word_char == '???' and curr_morph_char == '???'):
                        new_char_aligned.append([morph_list[morph_idx]])
                        morph_idx += 1
                        word_idx += 1
                    elif next_morph_tag.startswith("ET"):
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1
                    elif morph_idx + 1 < len(morph_list) and '???' <= next_morph_jamo[0] <= '???' and curr_word_char == compose(curr_morph_jamo[0], curr_morph_jamo[1], next_morph_jamo[0]):
                        # ????????? -> [???+???] ??????
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1
                elif morph_idx + 1 < len(morph_list) and curr_morph_jamo[2] == '???':
                    # ?????? -> ?????? + ???, ?????? -> ?????? + ???
                    if next_morph_tag == "ETM" and decompose(next_morph_char)[2] in ['???', '???']:
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1

                elif decompose(curr_word_char)[2] == '???' and curr_morph_jamo[2] == ' ':
                    # ?????????: ??? <-> ???/NNG
                    # ?????????: ??? <-> ???/NNG
                    # ??????: ??? <-> ???/MAG
                    if curr_morph_tag in ("NNG", "MAG") and compose(curr_morph_jamo[0], curr_morph_jamo[1], '???') == curr_word_char:
                        new_char_aligned.append([morph_list[morph_idx]])
                        morph_idx += 1
                        word_idx += 1
                elif morph_idx + 1 < len(morph_list) and decompose(next_morph_char)[1] == '' and decompose(next_morph_char)[2] == ' ':
                    try:
                        self._align_norm(curr_word_char, morph_list[morph_idx:morph_idx+2])
                    except AlignError:
                        pass
                    else:
                        new_char_aligned.append(morph_list[morph_idx:morph_idx+2])
                        morph_idx += 2
                        word_idx += 1

                else:
                    # loop_cnt += 1
                    # if loop_cnt > 20:
                    raise AlignError
                #################################
                # TODO: '???' ?????????, gold_align ??????
                # ?????????????????? -> ????????? + ?????????
                #################################

            # if aligned:
            #   self.aligned.write(raw_word + "\t" + "  ###  ".join([str(char) for char in new_char_aligned]) + "\n")

            if [] in new_char_aligned:
                print(sent)
                raise AlignError

        except AlignError:
            self.unmapped.append(sent_no + "\t" + raw_word + "\t" + str(morph_list))
        except IndexError:
            self.unmapped.append(sent_no + "\t" + raw_word + "\t" + str(morph_list))
        else:
            self.aligned.write(raw_word + "\t" + "  ###  ".join([str(char) for char in new_char_aligned]) + "\n")

        return new_char_aligned

    def _pre_match(self, word, morph_list, w_idx, m_idx):
        res = []
        m = 0
        w = 0
        word_char = word[w_idx]

        #######################
        # TODO: next word syl == next morph syl ??? ?????? pre match ?????? ?????? (????????? pos??? NNG??? ???????)
        # - ?????????????????? ???????????? + ??? + ??????
        # gold_align: ??? - ??????
        # line: BTEO0340-00018720

        # - ?????????????????? ????????? + ?????? + ??????
        # gold_align: ??? - ??????
        # line: BTAA0011-0014993
        #######################

        if m_idx + 3 < len(morph_list) and self.is_gold_align(word_char, tuple(morph_list[
                                                                               m_idx:m_idx + 4])):
            res.append(morph_list[m_idx:m_idx + 4])
            m += 4
            w += 1

        elif m_idx + 2 < len(morph_list) and self.is_gold_align(word_char, tuple(morph_list[
                                                                                 m_idx:m_idx + 3])):
            res.append(morph_list[m_idx:m_idx + 3])
            m += 3
            w += 1
        elif m_idx + 1 < len(morph_list) and not is_only_noun(morph_list[m_idx:m_idx + 2]) and self.is_gold_align(word_char, tuple(
            morph_list[m_idx:m_idx + 2])):
            res.append(morph_list[m_idx:m_idx + 2])
            m += 2
            w += 1
        elif self.is_gold_align(word_char, [e for e in morph_list[m_idx:m_idx + 1]]):
            res.append([morph_list[m_idx]])
            m += 1
            w += 1

        return res, w, m


def run(args):
    aligner = Aligner(args)
    # aligner.is_gold_align('???', tuple(['???', '???']))
    aligner.align()
    aligner.close_all_files()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', required=True)
    parser.add_argument('--dic_path', help='dictionary & knowledge path\n'
                                           '(align map, restore, tag map)', required=True)
    parser.add_argument('--vocab_path', help='vocabulary path', required=True)
    parser.add_argument('--unmapped', help='unmapped alignment file path', required=True)
    parser.add_argument('--aligned', help='aligned file path', required=True)
    parser.add_argument('--output', help='new corpus output directory', required=True)
    args = parser.parse_args()
    run(args)
