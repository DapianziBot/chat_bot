from models import Seq2Seq
from model.Siamese import SiameseNet
from classify.build_model import get_classify_model
from config import *
from dataset.chatbot.word_sequence import get_ws

ws = get_ws(BY_WORD)


def chat_answer(sentence):
    # 1. load model
    # 2. transform sentence
    # 3. model predict
    # 4. return answer
    model = Seq2Seq(batch_size=1)
    model = model.to(DEVICE)
    params = ROOT + '/logs/chatbot-seq2seq.pkl'
    if not os.path.exists(params):
        print('run training before you do prediction')
        return
    model.load_state_dict(torch.load(params))
    model.eval()

    # sentence = jieba.lcut(sentence) if BY_WORD else list(sentence)
    x = ws.transform(sentence, MAX_SEQ_LEN)
    len_x = [len(x)]
    x = torch.tensor([x]).to(DEVICE)
    out = model.evaluate(x, len_x)
    print('小黄鸡: ' + ''.join(ws.inverse_transform(out)))


def qa_answer(sentence):
    # 1. load recall model
    # recall = get_recall_model()
    # 2. recall top k questions [k, seq_len]
    # topk = recall.predict(sentence)
    # 3. load sort model
    # siamese = SiameseNet()
    # siamese.load_state_dict(STATE_PATH)
    # 4. prepare inputs for SiameseNet
    # x1 = [ws.transform(sentence, max_len)] * len(topk)
    # x2 = [ws.transform(s, max_len) for s in topk]
    # 5. predict sorted questions
    # outputs = siamese(x1, x2) # [batch_size, 2]
    # best = torch.argmax(outputs[:,-1])
    # return qa_dict[best].answer
    pass




