import math
import os
from tqdm import tqdm
import pickle
from config import *


class WordSequence:
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    EOS_TAG = 'EOS'
    SOS_TAG = 'SOS'
    PAD = 0
    UNK = 1
    EOS = 2
    SOS = 3

    def __init__(self):
        self.dict = {
            self.PAD_TAG: self.PAD,
            self.UNK_TAG: self.UNK,
            self.EOS_TAG: self.EOS,
            self.SOS_TAG: self.SOS,
        }
        self.counts = {}
        self.inverse_dict = {}

    def fit(self, sequence):
        for w in sequence:
            self.counts[w] = self.counts.get(w, 0) + 1

    def build_vocab(self, min_count=0, max_count=math.inf):
        self.counts = dict([(k, v) for k, v in self.counts.items() if min_count <= v <= max_count])
        for w in self.counts.keys():
            self.dict[w] = len(self.dict)

        self.inverse_dict = dict([(v, k) for (k, v) in self.dict.items()])

    def transform(self, sentence, max_len=None, eos=False):

        sen_len = len(sentence)

        if max_len is not None:
            if eos:
                max_len = max_len - 1
            if sen_len > max_len:
                sentence = sentence[:max_len]

        if eos:
            sentence += [self.EOS_TAG]
        if max_len is not None and sen_len < max_len:
            sentence += [self.PAD_TAG] * (max_len - sen_len)

        return [self.dict.get(x, self.UNK) for x in sentence]

    def inverse_transform(self, seq_list):
        sentence = []
        for x in seq_list:
            if x == self.EOS:
                break
            if x == self.PAD:
                continue
            if x == self.SOS:
                continue
            sentence.append(self.inverse_dict.get(int(x), self.UNK_TAG))
        return sentence

    def vocab_size(self):
        return len(self.dict)


if not os.path.exists(WORD_SEQUENCE_VOCAB):
    ws = WordSequence()
    for line in tqdm(open(CHATBOT_XHJ_INPUT, 'r', encoding='utf-8').readlines(), desc='inputs'):
        ws.fit(line.split())
    for line in tqdm(open(CHATBOT_XHJ_TARGET, 'r', encoding='utf-8').readlines(), desc='targets'):
        ws.fit(line.split())
    ws.build_vocab(min_count=2)
    pickle.dump(ws, open(WORD_SEQUENCE_VOCAB, 'wb'))

    ws = WordSequence()
    for line in tqdm(open(CHATBOT_XHJ_INPUT_BY_WORD, 'r', encoding='utf-8').readlines(), desc='inputs by word'):
        ws.fit(line.split())
    for line in tqdm(open(CHATBOT_XHJ_TARGET_BY_WORD, 'r', encoding='utf-8').readlines(), desc='targets by word'):
        ws.fit(line.split())
    ws.build_vocab(min_count=5)
    pickle.dump(ws, open(WORD_SEQUENCE_BY_WORD_VOCAB, 'wb'))

if not os.path.exists(CHAT_SEQUENCE_VOCAB):
    ws = WordSequence()
    for line in tqdm(open(CHATBOT_CHATTER_INPUT, 'r', encoding='utf-8').readlines(), desc='inputs'):
        ws.fit(line.split())
    for line in tqdm(open(CHATBOT_CHATTER_TARGET, 'r', encoding='utf-8').readlines(), desc='targets'):
        ws.fit(line.split())
    ws.build_vocab(min_count=1)
    pickle.dump(ws, open(CHAT_SEQUENCE_VOCAB, 'wb'))

    ws = WordSequence()
    for line in tqdm(open(CHATBOT_CHATTER_INPUT_BY_WORD, 'r', encoding='utf-8').readlines(), desc='inputs by word'):
        ws.fit(line.split())
    for line in tqdm(open(CHATBOT_CHATTER_TARGET_BY_WORD, 'r', encoding='utf-8').readlines(), desc='targets by word'):
        ws.fit(line.split())
    ws.build_vocab(min_count=1)
    pickle.dump(ws, open(CHAT_SEQUENCE_BY_WORD_VOCAB, 'wb'))

if not os.path.exists(WEIBO_SEQUENCE_VOCAB):
    ws = WordSequence()
    for line in tqdm(open(CHATBOT_WEIBO_INPUT, 'r', encoding='utf-8').readlines(), desc='inputs'):
        ws.fit(line.split())
    for line in tqdm(open(CHATBOT_WEIBO_TARGET, 'r', encoding='utf-8').readlines(), desc='targets'):
        ws.fit(line.split())
    ws.build_vocab(min_count=3)
    pickle.dump(ws, open(WEIBO_SEQUENCE_VOCAB, 'wb'))

    ws = WordSequence()
    for line in tqdm(open(CHATBOT_WEIBO_INPUT_BY_WORD, 'r', encoding='utf-8').readlines(), desc='inputs by word'):
        ws.fit(line.split())
    for line in tqdm(open(CHATBOT_WEIBO_TARGET_BY_WORD, 'r', encoding='utf-8').readlines(), desc='targets by word'):
        ws.fit(line.split())
    ws.build_vocab(min_count=5)
    pickle.dump(ws, open(WEIBO_SEQUENCE_BY_WORD_VOCAB, 'wb'))


ws = pickle.load(open(WORD_SEQUENCE_VOCAB, 'rb'))
ws_vocab_size = ws.vocab_size()
ws_by_word = pickle.load(open(WORD_SEQUENCE_BY_WORD_VOCAB, 'rb'))
ws_vocab_size_by_word = ws_by_word.vocab_size()

# ws = pickle.load(open(CHAT_SEQUENCE_VOCAB, 'rb'))
# ws_vocab_size = ws.vocab_size()
# ws_by_word = pickle.load(open(CHAT_SEQUENCE_BY_WORD_VOCAB, 'rb'))
# ws_vocab_size_by_word = ws_by_word.vocab_size()

# ws = pickle.load(open(WEIBO_SEQUENCE_VOCAB, 'rb'))
# ws_vocab_size = ws.vocab_size()
# ws_by_word = pickle.load(open(WEIBO_SEQUENCE_BY_WORD_VOCAB, 'rb'))
# ws_vocab_size_by_word = ws_by_word.vocab_size()


def get_ws(by_word=True):
    return ws_by_word if by_word else ws


if __name__ == '__main__':
    print(ws_vocab_size_by_word, ws_vocab_size)
