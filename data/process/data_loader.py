from data.process.ed_reader import load_dataset
from utils import config
import torch
import os
import torch.utils.data as data
import logging
import pprint
pp = pprint.PrettyPrinter(indent=1)
from models.CAB.others_layer import write_config

class GDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        self.vocab = vocab
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6,
            'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12,
            'hopeful': 13, 'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19,
            'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25,
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31, 'null': 32
        }
        self.act_map = {
            'agreeing': 0,
            'acknowledging': 1,
            'encouraging': 2,
            'consoling': 3,
            'sympathizing': 4,
            'suggesting': 5,
            'questioning': 6,
            'wishing': 7
        }


    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_g_text"] = self.data["key_concepts"][index]
        item["triple"] = self.data["triples"][index]
        item["key_tokens"] = self.data["key_tokens"][index]
        item["dia_id"] = self.data["dia_id"][index]
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["context_s_text"] = self.data["context_s"][index]
        item["context_l_text"] = self.data["context_l"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_s_text"] = self.data["emotion_s"][index]
        item["emotion_l_text"] = self.data["emotion_l"][index]
        item["act_text"] = self.data["act"][index]

        item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)
        item["emotion_s_label"] = self.preprocess_emo(item["emotion_s_text"], self.emo_map)
        item["emotion_l_label"] = self.preprocess_emo(item["emotion_l_text"], self.emo_map)
        item["act_label"] = self.preprocess_act(item["act_text"], self.act_map)

        item["context"], item["context_mask"], item["context_ext_batch"], item["oovs"] = self.preprocess(item["context_text"], item["triple"])#
        item["context_s"], item["context_s_mask"] = self.preprocess(item["context_s_text"],cls = True,cls_s= True)
        item["context_l"], item["context_l_mask"] = self.preprocess(item["context_l_text"],cls = True,cls_l= True)

        item["posterior"], item["posterior_mask"] = self.preprocess(arr=[item["target_text"]], cls = True, posterior=True)

        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["target_ext"] = self.target_oovs(item["target_text"], item["oovs"])

        item["key_concepts"] = self.preprocess(item["context_g_text"], key = True)
        return item

    def target_oovs(self, target, oovs):
        ids = []
        for w in target:
            if w not in self.vocab.word2index:
                if w in oovs:
                    ids.append(len(self.vocab.word2index) + oovs.index(w))
                else:
                    ids.append(config.UNK_idx)
            else:
                ids.append(self.vocab.word2index[w])
        ids.append(config.EOS_idx)
        return torch.LongTensor(ids)

    def process_oov(self, context, concept):  #
        ids = []
        oovs = []
        for si, sentence in enumerate(context):
            for w in sentence:
                if w in self.vocab.word2index:
                    i = self.vocab.word2index[w]
                    ids.append(i)
                else:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(self.vocab.word2index) + oov_num)
            if len(context) != 1:
                ids.append(config.SEP_idx)
        ids.append(config.EOS_idx)

        # for group_triple in concept:
        #     for triple in group_triple:
        #         if triple[-1] not in oovs and triple[-1] not in self.vocab.word2index:
        #             oovs.append(triple[-1])
        return ids, oovs

    def preprocess(self, arr, triple = None, key = False, anw=False, cls = False, cls_s = False, cls_l = False,meta=None, posterior=False):
        """Converts words to ids."""
        if(key):
            X_dial = []
            for i, word in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx]
            return torch.LongTensor(X_dial)
        elif (anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                        arr] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        elif cls:
            if posterior:
                X_dial = [config.CLSp_idx]
                X_mask = [config.CLSp_idx]
            elif cls_s:
                X_dial = [config.CLSs_idx]
                X_mask = [config.CLSs_idx]
            else:
                X_dial = [config.CLSl_idx]
                X_mask = [config.CLSl_idx]
            for i, sentence in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                           sentence]
                spk = self.vocab.word2index["USR"] if cls_s else self.vocab.word2index["SYS"]
                if posterior: spk = self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))]
                if len(arr) != 1:
                    X_dial += [config.SEP_idx]
                    X_mask += [config.SEP_idx]
            X_dial += [config.EOS_idx]
            X_mask += [config.EOS_idx]
            assert len(X_dial) == len(X_mask)
            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)
        else:
            X_dial = [config.CLS_idx]
            X_dial_ext = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            X_ext, X_oovs = self.process_oov(arr, triple)
            X_dial_ext += X_ext
            for i, sentence in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                           sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                if posterior: spk = self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))]
                if len(arr) != 1:
                    X_dial += [config.SEP_idx]
                    X_mask += [config.SEP_idx]
            X_dial += [config.EOS_idx]
            X_mask += [config.EOS_idx]
            assert len(X_dial) == len(X_mask)

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask), torch.LongTensor(X_dial_ext), torch.LongTensor(X_oovs)

    def preprocess_emo(self, emotion, emo_map):
        # program = [0]*len(emo_map)
        # program[emo_map[emotion]] = 1
        if emotion == []:
            return emo_map['null']
        else:
            return emo_map[emotion[0]]
    def preprocess_act(self, act, act_map):
        # program = [0]*len(emo_map)
        # program[emo_map[emotion]] = 1
        return act_map[act]


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1 [32,69]
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)
    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info['context'])
    input_ext_batch, _ = merge(item_info['context_ext_batch'])
    input_s_batch, input_s_lengths = merge(item_info['context_s'])
    input_l_batch, input_l_lengths = merge(item_info['context_l'])
    # input_g_batch, input_g_lengths = merge(item_info['context_g'])
    # input_g_mask, input_g_mask_lengths = merge(item_info['context_g_mask'])
    # input_tr_batch, input_tr_lengths = merge(item_info['node'])
    posterior_batch, posterior_lengths = merge(item_info['posterior'])
    input_mask, input_mask_lengths = merge(item_info['context_mask'])
    input_s_mask, input_s_mask_lengths = merge(item_info['context_s_mask'])
    input_l_mask, input_l_mask_lengths = merge(item_info['context_l_mask'])
    posterior_mask, posterior_mask_lengths = merge(item_info['posterior_mask'])
    ## Target
    target_batch, target_lengths = merge(item_info['target'])
    target_ext_batch, _ = merge(item_info['target_ext'])
    # GPU
    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        input_ext_batch = input_ext_batch.cuda()
        input_s_batch = input_s_batch.cuda()
        input_l_batch = input_l_batch.cuda()
        posterior_batch = posterior_batch.cuda()
        posterior_mask = posterior_mask.cuda()
        input_mask = input_mask.cuda()
        input_s_mask = input_s_mask.cuda()
        input_l_mask = input_l_mask.cuda()
        target_batch = target_batch.cuda()
        target_ext_batch = target_ext_batch.cuda()

    d = {}
    d["input_batch"] = input_batch
    d["input_ext_batch"] = input_ext_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["input_mask"] = input_mask

    d["input_s_batch"] = input_s_batch
    d["input_s_lengths"] = torch.LongTensor(input_s_lengths)
    d["input_s_mask"] = input_s_mask

    d["input_l_batch"] = input_l_batch
    d["input_l_lengths"] = torch.LongTensor(input_l_lengths)
    d["input_l_mask"] = input_l_mask

    d["posterior_batch"] = posterior_batch
    d["posterior_lengths"] = torch.LongTensor(posterior_lengths)
    d["posterior_mask"] = posterior_mask

    d["target_batch"] = target_batch
    d["target_ext_batch"] = target_ext_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)

    d["concepts_batch"] = item_info['key_concepts']
    ##program
    d["act_label"] = torch.LongTensor(item_info['act_label'])
    d["emotion_label"] = torch.LongTensor(item_info['emotion_label'])
    d["emotion_s_label"] = torch.LongTensor(item_info['emotion_s_label'])
    d["emotion_l_label"] = torch.LongTensor(item_info['emotion_l_label'])
    if config.USE_CUDA:
        d["act_label"] = d["act_label"].cuda()
        d["emotion_label"] = d["emotion_label"].cuda()
        d["emotion_s_label"] = d["emotion_s_label"].cuda()
        d["emotion_l_label"] = d["emotion_l_label"].cuda()

        for i in range(len(d["concepts_batch"])):
            d["concepts_batch"][i] = d["concepts_batch"][i].cuda()
    ##text
    d["input_txt"] = item_info['context_text']
    d["situation_txt"] = item_info['situation_text']
    d["oovs"] = item_info['oovs']
    d["input_s_txt"] = item_info['context_s_text']
    d["input_l_txt"] = item_info['context_l_text']
    d["key_tokens"] = item_info["key_tokens"]
    d["target_txt"] = item_info['target_text']
    d["act_txt"] = item_info['act_text']
    d["emotion_s_txt"] = item_info['emotion_s_text']
    d["emotion_l_txt"] = item_info['emotion_l_text']
    d["emotion_txt"] = item_info['emotion_text']
    d["triples"] = item_info['triple']
    d["dia_id"] = item_info['dia_id']
    return d


def prepare_data_seq(batch_size=32, kn = False):

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))
    if kn:
        pairs = {}
        for key, value in (pairs_tra.items()):
            pairs[key] = pairs_tra[key] + pairs_val[key] + pairs_tst[key]

        datasets = GDataset(pairs, vocab)
        data_loader = torch.utils.data.DataLoader(dataset=datasets,
                                                 batch_size=batch_size,
                                                 shuffle=False, collate_fn=collate_fn)
        return data_loader, vocab

    dataset_train = GDataset(pairs_tra, vocab)
    # posterior_mask = dataset_train.preprocess(pairs_tra["context"][1])
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)

    dataset_valid = GDataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
    #print('val len:',len(dataset_valid))
    dataset_test = GDataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=collate_fn)
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map), len(dataset_train.act_map)
