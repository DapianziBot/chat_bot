import jieba
import config

jieba.load_userdict(config.USER_DICT_PATH)

if __name__ == '__main__':
    sentance = 'python和c语言wang zhaohui哪个难'
    ret = jieba.lcut(sentance)
    print(ret)