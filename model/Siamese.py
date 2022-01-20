from torch import nn
from torch.nn import functional as F

from config import *
from dataset.chatbot.word_sequence import get_ws

ws = get_ws(BY_WORD)


class SiameseNet(nn.Module):
    """孪生网络"""

    def __init__(self):
        super(SiameseNet, self).__init__()

        self.emb = nn.Embedding(ws.vocab_size(), embedding_dim=QA_EMBEDDING_DIM, padding_idx=ws.PAD)
        self.gru1 = nn.GRU(input_size=QA_EMBEDDING_DIM,
                           hidden_size=QA_HIDDEN_SIZE,
                           num_layers=QA_NUM_LAYERS,
                           batch_first=True,
                           bidirectional=False)
        self.gru2 = nn.GRU(input_size=4 * QA_HIDDEN_SIZE,
                           hidden_size=QA_HIDDEN_SIZE,
                           num_layers=QA_NUM_LAYERS,
                           batch_first=True,
                           bidirectional=False)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(QA_HIDDEN_SIZE * 4),
            nn.Linear(QA_HIDDEN_SIZE * 4, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, *inputs):
        sent1, sent2 = inputs
        # 使用mask在后面计算attention的时候忽略pad的位置
        mask1, mask2 = sent1.eq(ws.PAD), sent2.eq(ws.PAD)

        # embeds: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x1 = self.emb(sent1)
        x2 = self.emb(sent2)
        #
        output1, _ = self.gru1(x1)  # [batch_size, seq_len_1, hidden_size]
        output2, _ = self.gru1(x2)

        # attention
        q1_align, q2_align = self._soft_attn_align(output1, output2, mask1, mask2)

        # submal

        # 再进行一轮lstm处理
        q1_conbined = torch.cat([output1, q1_align, self._submul(output1, q1_align)], -1)
        q2_conbined = torch.cat([output2, q1_align, self._submul(output2, q2_align)], -1)

        # [batch_size, seq_len, 4*hidden_size] -> [batch_size, seq_len, hidden_size]
        q1_compose, _ = self.gru2(q1_conbined)
        q2_compose, _ = self.gru2(q2_conbined)

        # pooling [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, 2*hidden_size]
        q1_rep = self.apply_pooling(q1_compose)
        q2_rep = self.apply_pooling(q2_compose)
        # [batch_size, seq_len, 2 * hidden_size] -> [batch_size, seq_len, 4*hidden_size]
        x = torch.cat([q1_rep, q2_rep], -1)
        return self.fc(x)

    def apply_pooling(self, x):
        inputs = x.transpose(1, 2)
        size = x.size(1)
        pool1 = F.max_pool1d(inputs, kernel_size=size, stride=size).squeeze(-1)
        pool2 = F.avg_pool1d(inputs, kernel_size=size, stride=size).squeeze(-1)
        return torch.cat([pool1, pool2], -1)

    def _submul(self, x1, x2):
        return torch.cat((x1 * x2, x1 - x2), -1)

    def _soft_attn_align(self, x1, x2, mask1, mask2):
        """
        self attention with x1, x2
        :param x1: [batch_size, seq_len_1, hidden_size]
        :param x2: [batch_size, seq_len_2, hidden_size]
        :param mask1: [batch_size, seq_len_1]
        :param mask2: [batch_size, seq_len_2]
        :return:
        """

        attn_weight = torch.bmm(x1, x2.transpose(1, 2))  # [batch_size, seq_len_1, seq_len_2]

        # tensor([[0., 0., -inf, 0., 0., 0., 0., -inf, -inf, -inf]])
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight1
        weight1 = F.softmax(attn_weight + mask2.unsqueeze(1), dim=-1)  # [batch_size, seq_len_1, seq_len_2]
        x1_align = torch.matmul(weight1, x2)  # [batch_size, seq_len1, hidden_size]

        # weight2
        weight2 = F.softmax(attn_weight.transpose(1, 2) + mask1.unsqueeze(1),
                            dim=-1)  # [batch_size, seq_len_2, seq_len_1]
        x2_align = torch.matmul(weight2, x1)  # [batch_size, seq_len_2, hidden_size]

        return x1_align, x2_align


if __name__ == '__main__':
    model = SiameseNet()
    data = [
        ([['python', '怎', '么', '入', '门'], ['你', '是', '谁']],
         [['如', '何', '从', '零', '开', '始', '学', 'python'], ['你', '叫', '什', '么', '名', '字']]),
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        for x1, x2 in data:
            x1 = torch.tensor([ws.transform(x, 10) for x in x1])
            x2 = torch.tensor([ws.transform(x, 10) for x in x2])
            optimizer.zero_grad()
            y_ = model(x1, x2)

            loss = loss_fn(y_, torch.tensor([1, 1]))
            loss.backward()
            optimizer.step()
            print(loss.detach().numpy())
