import fasttext

from config import CLASSIFY_MODEL, CLASSIFY_CORPUS


def build_classify_model():
    model = fasttext.train_supervised(
        CLASSIFY_CORPUS,
        epoch=20,
        wordNgrams=1,
        minCount=5
    )
    model.save_model(CLASSIFY_MODEL)


def get_classify_model():
    return fasttext.load_model(CLASSIFY_MODEL)
