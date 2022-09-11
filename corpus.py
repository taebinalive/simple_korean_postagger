import glob
import os

import tqdm


class Corpus:
    def __init__(self, corpus_path):

        # TODO: add extra corpus parameter

        files = glob.glob(corpus_path + '/**/*.txt', recursive=True)
        # files = ["./data/BTEO0340.txt"]

        self.sentences = []
        self.total_eojuls = 0

        # sejong corpus
        for file in tqdm.tqdm(files, desc='Files'):
            line_cnt = 0
            eojul_list = []
            sentence = ''
            is_content = False

            for line in open(file, 'r', encoding='utf-8'):
                line_cnt += 1
                line = line.strip()
                # line = normalize('NFKD', line).strip()  # ㄱ (0x3131) -> ㄱ (0x1100)
                if line.startswith('<p>') or line.startswith('<head>'):
                    is_content = True
                elif line.startswith('</p>') or line.startswith('</head>'):
                    # make sentence object
                    if is_content:
                        try:
                            sent_obj = Sentence(sentence.strip(), eojul_list)
                            self.sentences.append(sent_obj)
                            sentence = ''
                            eojul_list = []
                            is_content = False
                        except:
                            print(os.path.basename(file), sentence)
                else:
                    # add to eojul list
                    if is_content:
                        try:
                            sent_no, eojul, morphs = line.split('\t')
                            eojul_list.append((sent_no, eojul, morphs))
                            sentence += eojul + ' '
                        except ValueError:
                            # discard current sentence
                            sentence = ''
                            eojul_list = []
                            is_content = False

        # filter sentences which has syllable less than 2
        self.sentences = [e for e in list(set(self.sentences)) if len(e.sentence) > 1]
        self.total_eojuls = sum([len(sen.eojuls) for sen in self.sentences])

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return "\n\n".join([str(sentence) for sentence in self.sentences])

    def __getitem__(self, idx):
        return self.sentences[idx]


class Sentence:
    def __init__(self, sentence, eojuls):
        self.sentence = sentence
        self.eojuls = [Eojul(*eojul) for eojul in eojuls]

    def __eq__(self, other):
        if len(self.eojuls) != len(other.eojuls):
            return False
        for src, dst in zip(self.eojuls, other.eojuls):
            if src.eojul != dst.eojul:
                return False
        return True

    def __hash__(self):
        return hash(self.sentence)# + ''.join([e.eojul for e in self.eojuls]))

    def __repr__(self):
        return f'sentence: {self.sentence}' + "\n" + "\n".join([str(eojul) for eojul in self.eojuls])


class Eojul:
    def __init__(self, no, eojul, morphs):
        self.no = no
        self.eojul = eojul
        self.morphs = [Morph(*morph.rsplit('/', 1)) for morph in morphs.split(" + ")]

    def __repr__(self):
        return self.eojul + "\t" + ", ".join([str(morph) for morph in self.morphs])


class Morph:
    def __init__(self, lexicon, tag):
        self.lexicon = lexicon
        self.tag = tag

    def __repr__(self):
        return f'({self.lexicon}, {self.tag})'
