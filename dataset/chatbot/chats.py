from dataset.chatbot.word_sequence import get_ws
from torch.utils.data import DataLoader, Dataset
from config import *

ws = get_ws(BY_WORD)


class ChatsSets(Dataset):

    def __init__(self):
        super(ChatsSets, self).__init__()

        self.inputs = open(CHATBOT_INPUTS, encoding='utf-8').readlines()
        self.targets = open(CHATBOT_TARGETS, encoding='utf-8').readlines()
        # total = len(inputs)
        # train_num = int(total * 0.85)
        # self.inputs = inputs[:train_num] if train else inputs[train_num:]
        # self.targets = targets[:train_num] if train else targets[train_num:]
        self.length = len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx].strip().split()
        y = self.targets[idx].strip().split()
        return x, y, len(x), len(y)

    def __len__(self):
        return self.length


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[2], reverse=True)

    x, y, len_x, len_y = zip(*batch)
    x = torch.tensor([ws.transform(x, MAX_SEQ_LEN) for x in x], dtype=torch.int64)
    y = torch.tensor([ws.transform(i, MAX_SEQ_LEN, True) for i in y], dtype=torch.int64)

    len_x = [x if x < MAX_SEQ_LEN else MAX_SEQ_LEN for x in len_x]
    len_y = [y if y < MAX_SEQ_LEN else MAX_SEQ_LEN for y in len_y]
    return x, y, torch.tensor(len_x), torch.tensor(len_y)


def get_data_loader(batch_size=BATCH_SIZE):

    chats_data = ChatsSets()
    return DataLoader(chats_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)


if __name__ == '__main__':
    pass
