import os
import pickle
import json
from utils import config
import numpy as np
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=1)
from collections import deque
from nltk.tokenize import word_tokenize
class GData:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def drop_comma_(df):
    for i in range(len(df)):
        df['utterance'][i] = df['utterance'][i].replace('_comma_',' ,').replace('.',' .').replace('?',' ?').replace('!',' !')
        df['prompt'][i] = df['prompt'][i].replace('_comma_',' ,').replace('.',' .').replace('?',' ?').replace('!',' !')
    return df


def create_dialog(df_orig,df_his_context,df_target,df_his_emotion,df_act,df_situation):
    for i in range(len(df_orig)):
        if df_orig['utterance_idx'][i] % 2 != 0:
            if df_orig['utterance_idx'][i] == 1:
                df_his_context.append([df_orig['utterance'][i]])
                df_his_emotion.append([df_orig['emotion_label'][i]])
                df_situation.append(df_orig['prompt'][i])
                j = i
            else:
                df_his_context.append(df_orig.iloc[j:i + 1, 5].tolist())
                df_his_emotion.append(df_orig.iloc[j:i + 1, 8].tolist())
                df_situation.append(df_orig['prompt'][i])
        else:
            if df_orig['actor'][i] == 'listener':
                df_target.append(df_orig['utterance'][i])
                df_act.append(df_orig['act_label'][i])
    return df_his_context,df_target,df_his_emotion,df_act,df_situation

def create_3_sentence_dialog(df_orig,df_his_context,df_target,df_his_emotion,df_act,df_situation):
    for i in range(len(df_orig)):
        if df_orig['utterance_idx'][i] % 2 != 0:
            if df_orig['utterance_idx'][i] == 1:
                df_his_context.append([df_orig['utterance'][i]])
                df_his_emotion.append([df_orig['emotion_label'][i]])
                df_situation.append(df_orig['prompt'][i])
            else:
                df_his_context.append(df_orig.iloc[i - 2:i + 1, 5].tolist())
                df_his_emotion.append(df_orig.iloc[i - 2:i + 1, 8].tolist())
                df_situation.append(df_orig['prompt'][i])
        else:
            if df_orig['actor'][i] == 'listener':
                df_target.append(df_orig['utterance'][i])
                df_act.append(df_orig['act_label'][i])
    return df_his_context,df_target,df_his_emotion,df_act,df_situation

def clean(sentence,word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v)
    sentence = word_tokenize(sentence)
    # sentence = sentence.split()
    return sentence

def read_data(vocab):

    word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "shouldn't": "should not", "shouldn've": "should have", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not", "haven't": "have not", "hadn't": "had not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}


    train_df = pd.read_json('./data/ed_data/train.json')
    valid_df = pd.read_json('./data/ed_data/valid.json')
    test_df = pd.read_json('./data/ed_data/test.json')

    train_df = drop_comma_(train_df)
    valid_df = drop_comma_(valid_df)
    test_df = drop_comma_(test_df)

    dia_his_context_train, dia_his_context_valid, dia_his_context_test = [], [], []
    dia_target_train, dia_target_valid, dia_target_test = [], [], []
    dia_his_emotion_train, dia_his_emotion_valid, dia_his_emotion_test = [], [], []
    dia_act_train, dia_act_valid, dia_act_test = [], [], []
    dia_situation_train, dia_situation_valid, dia_situation_test = [], [], []
    dia_his_context_train,dia_target_train,dia_his_emotion_train,dia_act_train,dia_situation_train= \
        create_dialog(train_df,dia_his_context_train,dia_target_train,dia_his_emotion_train,dia_act_train,dia_situation_train)
    dia_his_context_valid,dia_target_valid,dia_his_emotion_valid,dia_act_valid,dia_situation_valid = \
        create_dialog(valid_df,dia_his_context_valid,dia_target_valid,dia_his_emotion_valid,dia_act_valid,dia_situation_valid)
    dia_his_context_test,dia_target_test,dia_his_emotion_test,dia_act_test,dia_situation_test = \
        create_dialog(test_df,dia_his_context_test,dia_target_test,dia_his_emotion_test,dia_act_test,dia_situation_test)

    data_train = {'context': [], 'context_s': [], 'context_l': [], 'target': [], 'emotion': [], 'emotion_s': [], 'emotion_l': [], 'act': [], 'situation': [], 'id': []}
    data_dev = {'context': [], 'context_s': [], 'context_l': [], 'target': [], 'emotion': [], 'emotion_s': [], 'emotion_l': [], 'act': [], 'situation': [], 'id': []}
    data_test = {'context': [], 'context_s': [], 'context_l': [], 'target': [], 'emotion': [], 'emotion_s': [], 'emotion_l': [], 'act': [], 'situation': [], 'id': []}

    for context in dia_his_context_train:
        u_list = []#u_list = deque([], maxlen=3)
        s_list = []
        l_list = []
        for i,u in enumerate(context):
            if len(context) == 1:
                s_list.append(clean(u,word_pairs))
            elif i%2 == 0:
                s_list.append(clean(u,word_pairs))
            else:
                l_list.append(clean(u,word_pairs))
            u = clean(u,word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_train['context'].append(list(u_list))
        data_train['context_s'].append(list(s_list))
        if l_list == []:
            data_train['context_l'].append(list(l_list))
        else:
            data_train['context_l'].append([l_list[-1]])
    for target in dia_target_train:
        target = clean(target,word_pairs)
        data_train['target'].append(target)
        vocab.index_words(target)
    for situation in dia_situation_train:
        situation = clean(situation,word_pairs)
        data_train['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in dia_his_emotion_train:
        e_list = []#u_list = []
        se_list = []
        le_list = []
        for i,u in enumerate(emotion):
            if len(emotion) == 1:
                se_list.append(u)
            elif i%2 == 0:
                se_list.append(u)
            else:
                le_list.append(u)
            e_list.append(u)
        data_train['emotion'].append([e_list[0]])
        data_train['emotion_s'].append(list(se_list))
        if le_list == []:
            data_train['emotion_l'].append(list(le_list))
        else:
            data_train['emotion_l'].append([le_list[-1]])
    for act in dia_act_train:
        data_train['act'].append(act)
    assert len(data_train['context']) == len(data_train['context_s']) == len(data_train['context_l']) == len(data_train['target']) \
           == len(data_train['emotion']) == len(data_train['emotion_s']) == len(data_train['emotion_l']) == len(data_train['situation']) == len(data_train['act'])

    for context in dia_his_context_valid:
        u_list = []#u_list = []
        s_list = []
        l_list = []
        for i,u in enumerate(context):
            if len(context) == 1:
                s_list.append(clean(u,word_pairs))
            elif i%2 == 0:
                s_list.append(clean(u,word_pairs))
            else:
                l_list.append(clean(u,word_pairs))
            u = clean(u,word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_dev['context'].append(list(u_list))
        data_dev['context_s'].append(list(s_list))
        # data_dev['context_l'].append(list(l_list))
        if l_list == []:
            data_dev['context_l'].append(list(l_list))
        else:
            data_dev['context_l'].append([l_list[-1]])

    for target in dia_target_valid:
        target = clean(target,word_pairs)
        data_dev['target'].append(target)
        vocab.index_words(target)
    for situation in dia_situation_valid:
        situation = clean(situation,word_pairs)
        data_dev['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in dia_his_emotion_valid:
        e_list = []#u_list = []
        se_list = []
        le_list = []
        for i,u in enumerate(emotion):
            if len(emotion) == 1:
                se_list.append(u)
            elif i%2 == 0:
                se_list.append(u)
            else:
                le_list.append(u)
            e_list.append(u)
        data_dev['emotion'].append([e_list[0]])
        data_dev['emotion_s'].append(list(se_list))
        if le_list == []:
            data_dev['emotion_l'].append(list(le_list))
        else:
            data_dev['emotion_l'].append([le_list[-1]])
    for act in dia_act_valid:
        data_dev['act'].append(act)

    assert len(data_dev['context']) == len(data_dev['context_s']) == len(data_dev['context_l']) == len(data_dev['target']) \
           == len(data_dev['emotion']) == len(data_dev['emotion_s']) == len(data_dev['emotion_l']) == len(data_dev['situation']) == len(data_dev['act'])

    for context in dia_his_context_test:
        u_list = []#u_list = []
        s_list = []
        l_list = []
        for i,u in enumerate(context):
            if len(context) == 1:
                s_list.append(clean(u,word_pairs))
            elif i%2 == 0:
                s_list.append(clean(u,word_pairs))
            else:
                l_list.append(clean(u,word_pairs))
            u = clean(u,word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_test['context'].append(list(u_list))
        data_test['context_s'].append(list(s_list))
        if l_list == []:
            data_test['context_l'].append(list(l_list))
        else:
            data_test['context_l'].append([l_list[-1]])

    for target in dia_target_test:
        target = clean(target,word_pairs)
        data_test['target'].append(target)
        vocab.index_words(target)
    for situation in dia_situation_test:
        situation = clean(situation,word_pairs)
        data_test['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in dia_his_emotion_test:
        e_list = []#u_list = []
        se_list = []
        le_list = []
        for i,u in enumerate(emotion):
            if len(emotion) == 1:
                se_list.append(u)
            elif i%2 == 0:
                se_list.append(u)
            else:
                le_list.append(u)
            e_list.append(u)
        data_test['emotion'].append([e_list[0]])
        data_test['emotion_s'].append(list(se_list))
        if le_list == []:
            data_test['emotion_l'].append(list(le_list))
        else:
            data_test['emotion_l'].append([le_list[-1]])
    for act in dia_act_test:
        data_test['act'].append(act)

    assert len(data_test['context']) == len(data_test['context_s']) == len(data_test['context_l']) == len(data_test['target']) \
           == len(data_test['emotion']) == len(data_test['emotion_s']) == len(data_test['emotion_l']) == len(data_test['situation']) == len(data_test['act'])

    return data_train, data_dev, data_test, vocab


def load_dataset():
    print(os.getcwd())
    if(os.path.exists('data/all_ed_concept_dataset_preproc.p')):
        print("LOADING ed data")
        with open('data/all_ed_concept_dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab  = read_data(
            vocab = GData({config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS", config.SEP_idx: "SEP",
                        config.USR_idx: "USR", config.SYS_idx: "SYS", config.KG_idx: "KG", config.CLS_idx: "CLS", config.CLSs_idx: "CLSs", config.CLSl_idx: "CLSl", config.CLSp_idx: "CLSp",
                        config.Y_idx: "Y"}))

        # data_tra, data_val, data_tst, vocab = read_langs_persona(data_tra, data_val, data_tst, vocab)

        with open('./data/dataset_preproc.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    for i in range(3):
        print('[situation]:', ' '.join(data_tra['situation'][i]))
        print('[emotion]:', data_tra['emotion'][i])
        print('[emotion_s]:', data_tra['emotion_s'][i])
        print('[emotion_l]:', data_tra['emotion_l'][i])
        print('[act]:', data_tra['act'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[context_s]:', [' '.join(u) for u in data_tra['context_s'][i]])
        print('[context_l]:', [' '.join(u) for u in data_tra['context_l'][i]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab
