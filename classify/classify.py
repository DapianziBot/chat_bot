import fasttext

import config


class Classify:
    def __init__(self):
        """
        加载训练好的模型
        """
        self.model = fasttext.load_model(config.CLASSIFY_MODEL)
        self.model_by_word = fasttext.load_model(config.CLASSIFY_MODEL_BY_WORD)

    def predict(self, sentence: dict):

        result1 = self.model.predict(sentence['cut'])
        result2 = self.model_by_word.predict(sentence['cut_by_word'])

        for label, acc, label_by_word, acc_by_word in zip(*result1, *result2):
            # 将二分类转变成单值
            if label == '__label__chat':
                label = '__label_QA'
                acc = 1 - acc
            if label_by_word == '__label__chat':
                label_by_word = '__label_QA'
                acc_by_word = 1 - acc_by_word

            if acc > 0.95 or acc_by_word > 0.98:
                return 'QA', max(acc, acc)
            else:
                return 'chat', max(1-acc, 1-acc_by_word)
