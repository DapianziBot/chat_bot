import torch
import heapq
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import *
from dataset.chatbot.word_sequence import get_ws, WordSequence


ws = get_ws(BY_WORD)
vocab_size = ws.vocab_size()


class NumEncoder(nn.Module):

    def __init__(self):
        super(NumEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.dropout = ENCODER_DROP_OUT

        self.emb = nn.Embedding(self.vocab_size, EMBEDDING_DIM, padding_idx=ws.PAD)
        self.gru = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=ENCODER_HIDDEN_SIZE, num_layers=ENCODER_NUM_LAYERS,
                          batch_first=True)

    def forward(self, input, input_len):
        embeded = self.emb(input)

        x = pack_padded_sequence(embeded, batch_first=True, lengths=input_len)

        out, h = self.gru(x)

        out, out_len = pad_packed_sequence(out, total_length=MAX_SEQ_LEN, batch_first=True, padding_value=ws.PAD)

        return out, h


class NumDecoder(nn.Module):
    def __init__(self, batch_size=BATCH_SIZE):
        super(NumDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.dropout = DECODER_DROP_OUT

        self.emb = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=ws.PAD)
        self.gru = nn.GRU(EMBEDDING_DIM, DECODER_HIDDEN_SIZE, num_layers=DECODER_NUM_LAYERS, dropout=DECODER_DROP_OUT,
                          batch_first=True)

        # Luong Attention
        self.attn = Attention(method=ATTENTION_FN, batch_size=batch_size)
        self.wa = nn.Linear(ENCODER_HIDDEN_SIZE + DECODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, bias=False)

        self.fc = nn.Linear(DECODER_HIDDEN_SIZE, vocab_size)
        self.act = nn.LogSoftmax(-1)

    def forward(self, target, encoder_hidden, encoder_outputs):
        # 获取batch size
        batch_size = encoder_hidden.size(1)
        # 1. 获取 Encoder 的输出，作为第一次的 hidden_state
        decoder_hidden = encoder_hidden
        # 2. 准备 Decoder 的第一个输入： [batch_size, 1] * SOS
        decoder_input = torch.ones((batch_size, 1), dtype=torch.long) * ws.SOS
        decoder_input = decoder_input.to(DEVICE)
        # 3. 循环时间步
        decoder_output = torch.zeros((batch_size, MAX_SEQ_LEN, self.vocab_size)).to(DEVICE)
        # use teacher forcing
        use_teacher_forcing = np.random.random() < TEACHER_FORCING_RATIO
        for i in range(MAX_SEQ_LEN):
            output_step, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            #
            decoder_output[:, i, :] = output_step
            # teacher forcing
            if use_teacher_forcing:
                decoder_input = target[:, i].unsqueeze(1)
            else:
                value, index = torch.topk(output_step, 1)
                decoder_input = index

        return decoder_output, decoder_hidden

    def forward_step(self, x0, h0, hs):
        """
        计算每个时间步
        :param x0: decoder 上一个step的输出
        :param h0 decoder 上一个step的hidden_state
        :param hs encoder_outputs Encoder 的输出序列
        :return:
        """
        embedded = self.emb(x0)
        # out   --> [batch_size, hidden_size, embedding_size]
        # h     --> [num_layer, batch_size, hidden_size]
        out, h = self.gru(embedded, h0)

        # attention
        attention_weight = self.attn(h0, hs)  # [batch_size, 1, seq_len]
        context_vector = attention_weight.bmm(hs)  # [batch_size, 1, hidden_size]
        concat = torch.cat((out, context_vector), dim=-1).squeeze(1)  # [batch_size, hidden_size*2]
        out = torch.tanh(self.wa(concat))
        out = self.fc(out)
        out = self.act(out)

        return out, h

    def evaluate(self, encoder_hidden, encoder_outputs):
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.ones((batch_size, 1), dtype=torch.long) * ws.SOS
        decoder_input = decoder_input.to(DEVICE)

        indices = []
        for i in range(MAX_SEQ_LEN + 1):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            # 逆 embed
            value, index = torch.topk(decoder_output, 1)
            decoder_input = index
            indices.append(index.item())
        return indices

    def evaluation_beam_search(self, encoder_hidden, encoder_outputs):
        batch_size = encoder_hidden.size(1)
        # 1. 构造第一次需要的输入数据，保存在堆中
        decoder_input = torch.tensor([[ws.SOS] * batch_size], dtype=torch.long).to(DEVICE)
        decoder_hidden = encoder_hidden  # 需要输入的hidden

        prev_beam = Beam()
        prev_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden)
        while True:
            cur_beam = Beam()
            # 2. 取出堆中的数据，进行forward_step的操作，获得当前时间步的output，hidden
            # 这里使用下划线进行区分
            for _probility, _complete, _seq, _decoder_input, _decoder_hidden in prev_beam:
                # 判断前一次的_complete是否为True，如果是，则不需要forward
                # 有可能为True，但是概率并不是最大
                if _complete:
                    cur_beam.add(_probility, _complete, _seq, _decoder_input, _decoder_hidden)
                else:
                    decoder_output, decoder_hidden = self.forward_step(_decoder_input, _decoder_hidden, encoder_outputs)
                    value, index = torch.topk(decoder_output, BEAM_WIDTH)  # [batch_size, beam_widht]
                    # 3. 从output中选择topk（k=beam width）个输出，作为下一次的input
                    for m, n in zip(value[0], index[0]):
                        decoder_input = torch.tensor([[n]], dtype=torch.long).to(DEVICE)
                        seq = _seq + [n]
                        probility = _probility * m
                        if n.item() == ws.EOS:
                            complete = True
                        else:
                            complete = False

                        # 4. 把下一个实践步骤需要的输入等数据保存在一个新的堆中
                        cur_beam.add(probility, complete, seq, decoder_input, decoder_hidden)
            # 5. 获取新的堆中的优先级最高（概率最大）的数据，判断数据是否是EOS结尾或者是否达到最大长度，如果是，停止迭代
            best_prob, best_complete, best_seq, _, _ = max(cur_beam)
            if best_complete or len(best_seq) == MAX_SEQ_LEN:  # 减去sos
                return best_seq
            else:
                # 6. 则重新遍历新的堆中的数据
                prev_beam = cur_beam


class Seq2Seq(nn.Module):
    def __init__(self, batch_size=BATCH_SIZE):
        super(Seq2Seq, self).__init__()
        self.encoder = NumEncoder()
        self.decoder = NumDecoder(batch_size=batch_size)

    def forward(self, input, label, input_len):
        encoder_out, encoder_h = self.encoder(input, input_len)
        decoder_out, decoder_h = self.decoder(label, encoder_h, encoder_out)

        return decoder_out, decoder_h

    def evaluate(self, input, input_len):
        encoder_out, encoder_h = self.encoder(input, input_len)
        # 使用topk贪心
        # decoder_seq = self.decoder.evaluate(encoder_h, encoder_out)
        # 使用Beam search
        decoder_seq = self.decoder.evaluation_beam_search(encoder_h, encoder_out)
        return decoder_seq


# encoder_hidden_size == decoder_hidden_size
class Attention(nn.Module):
    def __init__(self, method='general', batch_size=BATCH_SIZE):
        super(Attention, self).__init__()
        assert method in ['dot', 'general', 'concat'], 'method must be on of "general", "dot", "concat"'
        self.method = method
        if self.method == 'dot':
            pass
            # self.wh = nn.Linear(ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, bias=False)
        elif self.method == 'general':
            self.wa = nn.Linear(DECODER_HIDDEN_SIZE, ENCODER_HIDDEN_SIZE, bias=False)
        elif self.method == 'concat':
            # 1. concat with Parameter
            self.wa = nn.Linear(ENCODER_HIDDEN_SIZE + DECODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, bias=False)
            self.va = nn.Parameter(torch.tensor((batch_size, DECODER_HIDDEN_SIZE), dtype=torch.float32))
            # 2. concat with Linear
            # self.wa = nn.Linear(ENCODER_HIDDEN_SIZE + DECODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, bias=False)
            # self.va = nn.Linear(DECODER_HIDDEN_SIZE, 1, bias=False)
            # 3. add
            # self.wa1 = nn.Linear(ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, bias=False)
            # self.wa2 = nn.Linear(DECODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, bias=False)
            # self.va = nn.Parameter(torch.tensor((batch_size, DECODER_HIDDEN_SIZE), dtype=torch.float32))

    def forward(self, hidden_state, encoder_output):
        """
        :param hidden_state: [batch_size, decoder_hidden_size]
        :param encoder_output: [batch_size, seq_len, encoder_hidden_size]
        :return:
        """

        # 获取sequence length
        seq_len = encoder_output.size(1)
        # 仅使用最后一层 hidden_state, 调整形状
        hidden_state = hidden_state[-1, :, :]  # [batch_size, decoder_hidden_size]

        # 1. dot
        # 仅当 ENCODER_HIDDEN_SIZE === DECODER_HIDDEN_SIZE
        if self.method == 'dot':
            # bmm for batch matrix multiple, 批量矩阵乘法
            attention_weight = encoder_output.bmm(hidden_state).squeeze(-1)  # [batch_size, seq_len, 1]

        # 2. general
        # 适用 ENCODER_HIDDEN_SIZE !== DECODER_HIDDEN_SIZE
        elif self.method == 'general':
            hidden_state = self.wa(hidden_state)  # [batch_size, encoder_hidden_size]
            hidden_state = hidden_state.unsqueeze(-1)  # [batch_size, encoder_hidden_size, 1]
            attention_weight = encoder_output.bmm(hidden_state).squeeze(-1)  # [batch_size, seq_len, 1]
        # 3. concat
        # 适用 ENCODER_HIDDEN_SIZE !== DECODER_HIDDEN_SIZE
        else:
            hidden_state = hidden_state.repeat(1, seq_len, 1)  # [batch_size, seq_len, decoder_hidden_size]
            concated = torch.cat((hidden_state, encoder_output), dim=-1)  # [batch_size, seq_len, both_hidden_size]

            concated = torch.tanh(self.wa(concated))  # [batch_size, seq_len, decoder_hidden_size]
            attention_weight = concated.bmm(self.va.unsqueeze(-1))  # [batch_size, seq_len, 1]

        return F.softmax(attention_weight.squeeze(-1), dim=-1).unsqueeze(1)  # [batch_size, 1, seq_len]


class Beam:
    def __init__(self):
        self.heap = list()
        # 保存数据的总数
        self.beam_width = BEAM_WIDTH

    def add(self, probability, complete, seq, decoder_input, decoder_hidden):
        """
        添加数据，同时判断总的数据个数，多则删除
        :param probability: 概率乘积
        :param complete: 最后一个是否为EOS
        :param seq: list，所有token的列表
        :param decoder_input: 下一次进行解码的输入，通过前一次获得
        :param decoder_hidden: 下一次进行解码的hidden，通过前一次获得
        :return:
        """
        heapq.heappush(self.heap, [probability, complete, seq, decoder_input, decoder_hidden])
        # 判断数据的个数，如果大，则弹出。保证数据总个数小于等于 beam width
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)



