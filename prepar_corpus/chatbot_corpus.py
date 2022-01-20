from config import *
from tqdm import tqdm
import re
import jieba


def cut_sequence(str):
    l = list()
    tmp = ''
    for x in list(str):
        if re.match(r'[a-zA-Z]', x):
            tmp += x
        else:
            if tmp != '':
                l.append(tmp)
                tmp = ''
            if x != ' ':
                l.append(x)
    if tmp != '':
        l.append(tmp)
    return l


def filter_corpus(x, y):
    if len(x) > 1 and y.count('=') < 2:
        return True
    return False


def prepare_xiaohuangji(by_word=False):
    n = 0
    pair = []
    f_input = open(CHATBOT_XHJ_INPUT_BY_WORD if by_word else CHATBOT_XHJ_INPUT, 'w', encoding='utf-8')
    f_target = open(CHATBOT_XHJ_TARGET_BY_WORD if by_word else CHATBOT_XHJ_TARGET, 'w', encoding='utf-8')

    with open(CHATBOT_XHJ_CORPUS, 'r', encoding='utf-8') as f:

        for line in tqdm(f.readlines(), ascii=True):
            line = line.strip()
            if line == 'E':
                pair = []
            else:
                line = line[1:].strip()
                line = re.sub(r'/', '', line)
                if by_word:
                    pair.append(' '.join(jieba.lcut(line)))
                else:
                    pair.append(' '.join(cut_sequence(line)))

                if len(pair) == 2:
                    if filter_corpus(*pair):
                        f_input.write(pair[0] + '\n')
                        f_target.write(pair[1] + '\n')
                        n += 1
                    pair = []

    f_input.close()
    f_target.close()
    return n


def prepare_chatter(by_word=False):
    n = 0
    files = os.listdir(CHATBOT_CHATTER_CORPUS)
    f_input = open(CHATBOT_CHATTER_INPUT_BY_WORD if by_word else CHATBOT_CHATTER_INPUT, 'w', encoding='utf-8')
    f_target = open(CHATBOT_CHATTER_TARGET_BY_WORD if by_word else CHATBOT_CHATTER_TARGET, 'w', encoding='utf-8')
    for file in files:
        if file[-3:] != 'yml':
            continue
        with open(os.path.join(CHATBOT_CHATTER_CORPUS, file), 'r', encoding='utf-8') as f:
            c = 0
            pre = ''
            for line in tqdm(f.readlines(), ascii=True):
                if c >= 3:
                    seq = line[4:].strip()
                    if by_word:
                        seq = ' '.join(jieba.lcut(seq))
                    else:
                        seq = ' '.join(cut_sequence(seq))
                    if line[:4] == '- - ':
                        pre = seq
                    elif seq != '':
                        f_input.write(pre + '\n')
                        f_target.write(seq + '\n')
                        n += 1
                c += 1
    f_target.close()
    f_input.close()
    return n


def prepare_weibo(by_word=False):
    n = 2000000
    f_input = open(CHATBOT_WEIBO_INPUT_BY_WORD if by_word else CHATBOT_WEIBO_INPUT, 'w', encoding='utf-8')
    f_target = open(CHATBOT_WEIBO_TARGET_BY_WORD if by_word else CHATBOT_WEIBO_TARGET, 'w', encoding='utf-8')

    fp = open(CHATBOT_WEIBO_CORPUS + '/stc_weibo_train_post', 'r', encoding='utf-8')
    fr = open(CHATBOT_WEIBO_CORPUS + '/stc_weibo_train_response', 'r', encoding='utf-8')

    fw = f_input
    for f in [fp, fr]:
        count = 0
        for line in tqdm(f.readlines(), ascii=True):
            line = ''.join(line.strip().split())
            if by_word:
                line = ' '.join(jieba.lcut(line))
            else:
                line = ' '.join(cut_sequence(line))

            fw.write(line + '\n')
            count += 1
            if count >= n:
                break
        fw = f_target

    f_input.close()
    f_target.close()
    fp.close()
    fr.close()
    return n


if __name__ == '__main__':
    # n1 = prepare_xiaohuangji()
    # n2 = prepare_xiaohuangji(True)
    # print('By character: %d, By word: %d' % (n1, n2))

    prepare_weibo(True)
