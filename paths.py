import os


ROOT = os.path.abspath(os.path.dirname(__file__))


def real_path(path):
    return os.path.join(ROOT, path)


# corpus 语料相关
USER_DICT_PATH = real_path('corpus/user_dict/user_dict.txt')

CHATBOT_XHJ_CORPUS = real_path('corpus/xiaohuangji50w_fenciA.conv')
CHATBOT_XHJ_INPUT = real_path('prepar_corpus/chatbot/xhj_input.txt')
CHATBOT_XHJ_TARGET = real_path('prepar_corpus/chatbot/xhj_target.txt')
CHATBOT_XHJ_INPUT_BY_WORD = real_path('prepar_corpus/chatbot/xhj_input_by_word.txt')
CHATBOT_XHJ_TARGET_BY_WORD = real_path('prepar_corpus/chatbot/xhj_target_by_word.txt')

CHATBOT_CHATTER_CORPUS = real_path('corpus/chatter')
CHATBOT_CHATTER_INPUT = real_path('prepar_corpus/chatbot/chatter_inputs.txt')
CHATBOT_CHATTER_TARGET = real_path('prepar_corpus/chatbot/chatter_targets.txt')
CHATBOT_CHATTER_INPUT_BY_WORD = real_path('prepar_corpus/chatbot/chatter_inputs_by_word.txt')
CHATBOT_CHATTER_TARGET_BY_WORD = real_path('prepar_corpus/chatbot/chatter_targets_by_word.txt')

CHATBOT_WEIBO_CORPUS = real_path('corpus/weibo-400w')
CHATBOT_WEIBO_INPUT = real_path('prepar_corpus/chatbot/weibo_inputs.txt')
CHATBOT_WEIBO_TARGET = real_path('prepar_corpus/chatbot/weibo_targets.txt')
CHATBOT_WEIBO_INPUT_BY_WORD = real_path('prepar_corpus/chatbot/weibo_inputs_by_word.txt')
CHATBOT_WEIBO_TARGET_BY_WORD = real_path('prepar_corpus/chatbot/weibo_target_by_word.txt')

QA_BAIKE_CORPUS = real_path('corpus/baike_qa2019/baike_qa_train.json')


# chatbot
WORD_SEQUENCE_VOCAB = real_path('dataset/chatbot/word_sequence.pkl')
WORD_SEQUENCE_BY_WORD_VOCAB = real_path('dataset/chatbot/word_sequence_by_word.pkl')

CHAT_SEQUENCE_VOCAB = real_path('dataset/chatbot/chatter.pkl')
CHAT_SEQUENCE_BY_WORD_VOCAB = real_path('dataset/chatbot/chatter_words.pkl')

WEIBO_SEQUENCE_VOCAB = real_path('dataset/chatbot/weibo.pkl')
WEIBO_SEQUENCE_BY_WORD_VOCAB = real_path('dataset/chatbot/weibo_words.pkl')

# QA
QA_BAIKE_DICT = real_path('dataset/qa/baike.json')
QA_BAIKE_CLUSTER_INDEX = real_path('dataset/qa/baike_cluster_index.pkl')
QA_BAIKE_TFIDF_VECTORS = real_path('dataset/qa/baike_tfidf_vectors.pkl')
QA_BAIKE_SEQUENCE_VOCAB = real_path('dataset/qa/baike_vocab.pkl')
QA_BAIKE_SEQUENCE_BY_WORD_VOCAB = real_path('dataset/qa/baike_vocab_by_words.pkl')

QA_BAIKE_QUESTIONS = real_path('dataset/qa/baike_questions.pkl')
QA_BAIKE_QUESTIONS_BY_WORD = real_path('dataset/qa/baike_questions_by_word.pkl')
QA_BAIKE_SIM_QUESTIONS = real_path('dataset/qa/baike_sim_questions.pkl')
QA_BAIKE_SIM_QUESTIONS_BY_WORD = real_path('dataset/qa/baike_sim_questions_by_word.pkl')

# 分类相关
CLASSIFY_CORPUS = os.path.join(ROOT, 'corpus/classify/classify.txt')
CLASSIFY_MODEL = os.path.join(ROOT, 'model/classify/ft_classify.ftm')
CLASSIFY_MODEL_BY_WORD = os.path.join(ROOT, 'model/classify/ft_classify_by_word.ftm')