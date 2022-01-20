import jieba
import torch
from tqdm import tqdm
from torch import nn
from matplotlib import pyplot as plt

from config import *
from dataset.chatbot.chats import get_data_loader as get_chats_loader
from dataset.chatbot.word_sequence import get_ws
from models import Seq2Seq

train_data = get_chats_loader(batch_size=BATCH_SIZE)

ws = get_ws(BY_WORD)
print('load dictionary: vocab size = ', ws.vocab_size())


def init_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


loss_fn = nn.NLLLoss(ignore_index=ws.PAD)


def training():

    model = Seq2Seq()
    model = model.to(DEVICE)
    model.train()
    optimizer = init_optimizer(model)

    params = ROOT + '/logs/chatbot-seq2seq-image.pkl'
    if os.path.exists(params):
        print('load image state')
        model.load_state_dict(torch.load(params))
    avg_loss = []
    for epoch in range(EPOCHES):
        losses = train(model, optimizer, epoch)
        avg_loss.append(sum(losses) / len(losses))
        torch.save(model.state_dict(), ROOT + ('/logs/seq2seq-epoch-%d-loss-%.4f.pkl' % (epoch, losses[-1])))
        print('avg loss: %.4f' % (avg_loss[-1]))

    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(avg_loss)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train(model, optimizer, epoch):
    losses = []
    pbar = tqdm(train_data)
    for x, y, len_x, _ in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # input_len = input_len.to(DEVICE)
        optimizer.zero_grad()
        out, h = model(x, y, len_x)
        loss = loss_fn(out.view(-1, ws.vocab_size()), y.view(-1))
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        pbar.set_description('Epoch %s: loss=%.4f' % (epoch, loss.item()))
        losses.append(loss.item())
    return losses


def predict():
    model = Seq2Seq(batch_size=1)
    model = model.to(DEVICE)
    params = ROOT + '/logs/chatbot-seq2seq.pkl'
    if not os.path.exists(params):
        print('run training before you do prediction')
        return

    model.load_state_dict(torch.load(params))
    model.train(False)
    while True:
        sentence = input('You: ')
        if sentence == 'bye' or sentence == 'quit':
            print('bye')
            break

        # sentence = jieba.lcut(sentence) if BY_WORD else list(sentence)
        x = ws.transform(sentence, MAX_SEQ_LEN)
        len_x = [len(x)]
        x = torch.tensor([x]).to(DEVICE)
        out = model.evaluate(x, len_x)
        print('小黄鸡: ' + ''.join(ws.inverse_transform(out)))


if __name__ == '__main__':
    cmd = input('Train or Predict (P for predict, T for train) ?\n').strip().lower()
    if cmd == 'p' or cmd == 'predict':
        predict()
    elif cmd == 't' or cmd == 'train':
        training()
