import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import math
import scipy.sparse as sp
import os
from utils import config
from  utils.config import args
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import numpy as np
from nltk.util import ngrams
from functools import reduce
import operator
import random
import nltk
from nltk.tokenize import word_tokenize
from utils.common_layer import PositionwiseFeedForward
from utils.config import args
from utils.get_metrics import get_dist, calc_bleu

def write_config():
    if(not config.test):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        with open(config.save_path+'config.txt', 'w') as the_file:
            for k, v in config.args.__dict__.items():
                if("False" in str(v)):
                    pass
                elif("True" in str(v)):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k,v))

def learning_rate_decay(self,config):
    self.learning_rate = self.learning_rate * config.lr_decay
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.learning_rate

def set_seed(seed: int) -> None:
    """Set random seed to a fixed value.

    Set everything to be deterministic
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def comput_bleu(reference, hypothesis):
    s1, s2, s3, s4 = [], [], [], []
    assert len(reference) == len(hypothesis)
    for i,j in zip(reference, hypothesis):
        score1 = nltk.translate.bleu_score.sentence_bleu([word_tokenize(i)], word_tokenize(j), weights=(1, 0, 0, 0))
        score2 = nltk.translate.bleu_score.sentence_bleu([word_tokenize(i)], word_tokenize(j), weights=(0, 1, 0, 0))
        score3 = nltk.translate.bleu_score.sentence_bleu([word_tokenize(i)], word_tokenize(j), weights=(0, 0, 1, 0))
        score4 = nltk.translate.bleu_score.sentence_bleu([word_tokenize(i)], word_tokenize(j), weights=(0, 0, 0, 1))
        s1.append(score1)
        s2.append(score2)
        s3.append(score3)
        s4.append(score4)
    s1_avg = np.mean(s1)
    s2_avg = np.mean(s2)
    s3_avg = np.mean(s3)
    s4_avg = np.mean(s4)
    return s1_avg, s2_avg, s3_avg, s4_avg

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file,encoding= 'utf-8').readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

class Embeddings(nn.Module):
    def __init__(self,vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        # init.xavier_uniform(self.lut.weight)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def glove_embedding(vocab, pretrain=True):
    #import nn.Embedding
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)

    #update glove
    if(pretrain):
        #glove embedding（vocab.n_words,config.emb_dim）
        pre_embedding = gen_embeddings(vocab)
        embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.lut.weight.data.requires_grad = True
    return embedding

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=-1)
    return kld

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def get_input_from_batch(batch):
    enc_batch = batch['input_batch']
    enc_lens = batch['input_lengths']
    batch_size, max_enc_len = enc_batch.size()
    assert len(enc_lens) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len = max_enc_len).float()

    if config.USE_CUDA:
        enc_padding_mask = enc_padding_mask.cuda()
    return enc_batch

def get_output_from_batch(batch):
    dec_batch = batch['target_batch']
    target_batch = dec_batch

    dec_lens = batch['target_lengths']
    max_dec_len = max(dec_lens)
    assert max_dec_len == target_batch.size(1)
    dec_padding_mask = sequence_mask(dec_lens, max_len=max_dec_len).float()

    return dec_batch

def print_sample(emotion_s,emotion_l,knowledge,act,history,ref,hyp_g,hyp_b,hyp_t):
    print("History:{}".format(history))
    print("Knowledge:{}".format(knowledge))
    print("Emotion_s:{}".format(emotion_s))
    print("Emotion_l:{}".format(emotion_l))
    print("Act:{}".format(act))

    print("Greedy:{}".format(hyp_g))
    print("Beam Search:{}".format(hyp_b))
    print("Top-k/Top-p:{}".format(hyp_t))
    print("Ref:{}".format(ref))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

def print_emotion(emotion_s,pred_s, true_s):
    print("Emotion_s:{}".format(emotion_s))
    # print("Emotion_l:{}".format(emotion_l))
    print("Pre_s:{}".format(pred_s))
    # print("Pre_l:{}".format(pred_l))
    print("True_s:{}".format(true_s))
    # print("True_l:{}".format(true_l))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

def distinct_k(sentences):
    unigram = []
    bigram = []
    trigram = []
    for sent in sentences:
        s = sent.split()
        unigram.append(s)
        bigram.append(list(ngrams(s,2)))
        trigram.append(list(ngrams(s,3)))
    unigram = reduce(operator.concat, unigram)
    bigram = reduce(operator.concat, bigram)
    trigram = reduce(operator.concat, trigram)
    epss = 0.0000000000001
    d1 = len(set(unigram))/(len(unigram) +epss)
    d2 = len(set(bigram))/(len(bigram)+epss)
    d3 = len(set(trigram))/(len(trigram)+epss)
    return d1, d2, d3

def read_list(list):
    result = []
    for i in list:
        for j in i:
            result.append(j)
    return result

def get_losses_weights(losses:[list, np.ndarray, torch.Tensor]):
	if type(losses) != torch.Tensor:
		losses = torch.tensor(losses)
	weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
	return weights

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 2, bidirectional = False):
        super(GRU,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional= bidirectional, batch_first= True)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
    def __init__hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size)
        return hidden.cuda()
    def forward(self, input, seq_lengths):
        batch_size = input.size(0)
        hidden = self.__init__hidden(batch_size)
        gru_input = pack_padded_sequence(input, seq_lengths, batch_first= True, enforce_sorted=False)

        output, hidden = self.gru(input, hidden)
        # output = self.linear(output)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        return output, hidden_cat

class Node_attention_layer(nn.Module):
    def __init__(self, hid, d_model, embedding):
        super(Node_attention_layer, self).__init__()
        self.attention_layer = nn.Sequential(nn.Linear(d_model*2, d_model), nn.Tanh())
        self.attention_v = nn.Linear(d_model, 1, bias=False)
        self.hidden_layer = nn.Sequential(nn.Linear(d_model*2, hid), nn.Tanh())
        self.softmax = nn.Softmax(dim=-1)
        self.emb = embedding

    def forward(self, enc_outputs, x, key_concepts, mask_enc):
        # enc_outputs.shape == (batch_size,seq,hid)
        # x.shape == (batch_size, seq_len, d_model)
        # mask.shape == (batch_size, 1, 1, seq_len)
        # caculate attention score
        scale = enc_outputs.size(-1) ** -0.5
        concat_enc_outputs = torch.ones(enc_outputs.size(0), enc_outputs.size(1), enc_outputs.size(2)).cuda()
        k_scores_batch = torch.ones(enc_outputs.size(0), enc_outputs.size(1)).cuda()
        for batch in range(enc_outputs.size(0)):
            if len(x[batch]) != 0:
                projected = self.attention_layer(x[batch])  # (batch_size, seq_len, d_model)
                logits = torch.matmul(enc_outputs[batch], projected.transpose(0, 1))  # (batch_size, seq, seq_len)
                if len(key_concepts[batch]) != 0:
                    k_logits = torch.matmul(enc_outputs[batch] * scale, self.emb(key_concepts[batch]).transpose(0, 1))
                else:
                    k_logits = torch.tensor([]).cuda()
                mask = mask_enc[batch].squeeze().unsqueeze(1)
                if mask is not None:
                    logits = logits.masked_fill(mask, -1e18)  # (batch_size, seq_len)
                    k_logits = k_logits.masked_fill(mask, -1e18)

                scores = self.softmax(logits)# (batch_size, seq, seq_len)
                k_scores = self.softmax(k_logits)
                if len(key_concepts[batch]) != 0:
                    k_scores, _ = torch.max(k_scores, dim=-1) #(batch,seq_len)
                    k_scores_batch[batch] = k_scores

                dot_x = torch.matmul(scores, projected) # x.shape == (batch_size, seq, d_model)
                temp = torch.cat((enc_outputs[batch], dot_x), dim = -1)
                concat_enc_outputs[batch] = self.hidden_layer(temp)
        return concat_enc_outputs, k_scores_batch

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Latent(nn.Module):
    # get kl loss和 sampling z
    def __init__(self,is_eval):
        super(Latent,self).__init__()
        #Linear + RELU + Linear+ RELU + Linear
        self.mean = PositionwiseFeedForward(config.hidden_dim, config.filter, config.latent_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.var = PositionwiseFeedForward(config.hidden_dim, config.filter, config.latent_dim,
                                           layer_config='lll', padding='left',
                                           dropout=0)
        #posterior
        self.mean_p = PositionwiseFeedForward(config.hidden_dim*2, config.filter, config.latent_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.var_p = PositionwiseFeedForward(config.hidden_dim*2, config.filter, config.latent_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.is_eval = is_eval
    def forward(self,x,x_p, train=True):
        # mean
        mean = self.mean(x)
        # variance
        log_var = self.var(x)
        # Sampling eps from standard normal distribution
        eps = torch.randn(mean.size())
        # standard deviation
        std = torch.exp(0.5 * log_var)
        if config.USE_CUDA: eps = eps.cuda()
        # Sampling hidden variables from a prior distribution (reparameterization method)
        z = eps * std + mean
        if x_p is not None:
            mean_p = self.mean_p(torch.cat((x_p,x),dim=-1))
            log_var_p = self.var_p(torch.cat((x_p,x),dim=-1))
            kld_loss = gaussian_kld(mean_p,log_var_p,mean,log_var) # KL
            kld_loss = torch.mean(kld_loss)
        # If it is training, z samples from the posterior distribution
        if train:
            std = torch.exp(0.5 * log_var_p)
            if config.USE_CUDA: eps = eps.cuda()
            z = eps * std + mean_p
        return kld_loss, z

class ActLatent(nn.Module):
    def __init__(self,is_eval):
        super(ActLatent,self).__init__()
        #Linear + RELU + Linear+ RELU + Linear
        self.mean = PositionwiseFeedForward(config.hidden_dim, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.var = PositionwiseFeedForward(config.hidden_dim, config.filter, config.hidden_dim,
                                           layer_config='lll', padding='left',
                                           dropout=0)
        # posterior
        self.mean_p = PositionwiseFeedForward(config.hidden_dim*2+config.act_dim, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.var_p = PositionwiseFeedForward(config.hidden_dim*2+config.act_dim, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.is_eval = is_eval
    def forward(self,x,x_p,act_emb, train=True):
        # mean
        mean = self.mean(x)
        # variance
        log_var = self.var(x)
        # Sampling eps from standard normal distribution
        eps = torch.randn(x.size())
        # standard deviation
        std = torch.exp(0.5 * log_var)
        if config.USE_CUDA: eps = eps.cuda()
        # Sampling hidden variables from a prior distribution (reparameterization method)
        z = eps * std + mean
        if x_p is not None:
            mean_p = self.mean_p(torch.cat((x_p,x,act_emb),dim=-1))
            log_var_p = self.var_p(torch.cat((x_p,x,act_emb),dim=-1))
            kld_loss = gaussian_kld(mean_p,log_var_p,mean,log_var) # KL
            kld_loss = torch.mean(kld_loss)
        # If it is training, z samples from the posterior distribution
        if train:
            std = torch.exp(0.5 * log_var_p)
            if config.USE_CUDA: eps = eps.cuda()
            z = eps * std + mean_p
        return kld_loss, z

class OutputLayer(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(d_model, config.hidden_dim)
        # init.xavier_normal_(self.linear.weight)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(config.hidden_dim, vocab, bias= False)
        # init.xavier_normal_(self.proj.weight)

        self.relu = nn.ReLU()
    def forward(self, x, train = False):
        logit = self.relu(self.linear(x))
        if train:
            logit = self.dropout(logit)
        logit = self.proj(logit)
        # return F.log_softmax(logit,dim=-1)
        return logit

class SoftmaxOutputLayer(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(SoftmaxOutputLayer, self).__init__()
        self.single_proj = nn.Linear(d_model, d_model)
        init.xavier_normal_(self.single_proj.weight)

        self.both_proj = nn.Linear(d_model*2, d_model)
        init.xavier_normal_(self.both_proj.weight)

        self.fc = nn.Linear(d_model,vocab)
        init.xavier_normal_(self.fc.weight)

    def forward(self, x, both = False):
        if both:
            logit = self.fc(F.relu(self.both_proj(x)))
        else:
            logit = self.fc(F.relu(self.single_proj(x)))

        return F.log_softmax(logit,dim=-1)
        # return logit

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        # init.xavier_normal_(self.proj.weight)

        self.p_gen_linear = nn.Linear(args.hidden_dim, 1)
        init.xavier_normal_(self.p_gen_linear.weight)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, beam_search = False, temp=1):
        if args.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if args.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq

            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                if beam_search:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros.repeat(5,1,1)], 2)
                else:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18)
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


def evaluate(model, data, ty = 'val', max_dec_step = 50, print_file = None, val_result=None):
    if ty=="test":
        print("testing generation:")
    l = [] # total loss
    l_rec = [] #PPL
    kl = [] #KL loss
    bow = [] # bag of word loss
    elbo = [] # evidence lower bound(loss)
    act_l = []
    emotion_s_acc = []
    emotion_l_acc = []
    act_acc = []

    greedy_cands = []
    greedy_sentences = []

    pred_save_path = os.path.join(print_file, 'prediction')
    if os.path.exists(pred_save_path) is False: os.makedirs(pred_save_path)
    outputs = open(os.path.join(pred_save_path, 'output.txt'), 'w', encoding= 'utf-8')
    pbar = tqdm(enumerate(data),total=len(data))
    # t = Translator(model, vocab)
    for j, batch in pbar:
        if(j==235):
            print('stop')
        if ty == 'val':
            loss, loss_rec, elb, kld, aux_bow, act_loss, emo_s_acc, emo_l_acc, a_acc = model.train_one_batch(batch, 1, train=False)
        else:
            loss, loss_rec, elb, kld, aux_bow, act_loss, emo_s_acc, emo_l_acc, a_acc = model.train_one_batch(batch, 1, train=False)
        if j%5 == 0:
            torch.cuda.empty_cache()
        l.append(loss)
        l_rec.append(loss_rec)
        elbo.append(elb)
        act_l.append(act_loss)
        emotion_s_acc.append(emo_s_acc)
        emotion_l_acc.append(emo_l_acc)
        act_acc.append(a_acc)
        kl.append(kld)
        bow.append(aux_bow)

        if ty == 'test':
            sample_g = model.decoder_greedy(batch, train=False, max_dec_step = max_dec_step)
            # sample_b = t.beam_search(batch, ty, max_dec_step = max_dec_step)
            # for a,b in enumerate(sample_b):
            #     if 'EOS' in sample_b[a]:
            #         index = sample_b[a].split().index('EOS')
            #         sample_b[a] = ' '.join(sample_b[a].split()[:index])
            # sample_t = model.decoder_topk(batch, ty, train=False, max_dec_step = max_dec_step)
            for i, greedy_sent in enumerate(sample_g):
                rf = " ".join(batch["target_txt"][i])
                greedy_cands.append(greedy_sent)
                greedy_sentences.append([greedy_sent.strip().split(), batch["target_txt"][i]])
                if i < 1 and j < 1000:
                    print_sample(history = [" ".join(s) for s in batch['input_txt'][i]],
                                 knowledge = batch["triples"][i],
                                 emotion_s=batch["emotion_s_txt"][i],
                                 emotion_l = batch["emotion_l_txt"][i],
                                 act = batch["act_txt"][i],
                                 hyp_g = greedy_sent,
                                 hyp_b = [],
                                 hyp_t = [],
                                 ref = rf)
                outputs.write("Context:{} \n".format([" ".join(s) for s in batch['input_txt'][i]]))
                outputs.write("Situation:{} \n".format([" ".join(batch['situation_txt'][i])]))
                outputs.write("Knowledge:{} \n".format(batch['triples'][i]))
                outputs.write("Emotion_s:{} \n".format(batch["emotion_txt"][i]))
                outputs.write("Emotion_l:{} \n".format(batch["emotion_l_txt"][i]))
                outputs.write("Emotion:{} \n".format(batch["emotion_l_txt"][i]))
                outputs.write("Act:{} \n".format(batch["act_txt"][i]))
                outputs.write("Greedy:{} \n".format(greedy_sent))
                outputs.write("Ref:{} \n".format(rf))
                outputs.write("----------------------------------------------------------------------------\n")

        pbar.set_description(
            "loss:{:.4f} ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l_rec)))
        )

    loss = np.mean(l)
    loss_rec = np.mean(l_rec)
    act_loss = np.mean(act_l)
    emotion_s_acc = np.mean(emotion_s_acc)
    emotion_l_acc = np.mean(emotion_l_acc)
    act_acc = np.mean(act_acc)
    kld = np.mean(kl)
    bow = np.mean(bow)
    elbo = np.mean(elbo)

    if ty == 'test':
        _, _, mi_dist1, mi_dist2, avg_len = get_dist(greedy_cands)
        avg_bleu, bleu1, bleu2, bleu3, bleu4 = calc_bleu(greedy_sentences)
        print('{} the decode type is greedy'.format(ty), file=val_result)
        print("mi_dist1\tmi_dist2\tavg_len", file=val_result)
        print(
            "{:.4f}\t{:.4f}\t{:.4f}".format(mi_dist1, mi_dist2, avg_len),
            file=val_result)

        print("avg_bleu\tbleu1\tbleu2\tbleu3\tbleu4", file=val_result)
        print(
            "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_bleu, bleu1, bleu2, bleu3, bleu4),
            file=val_result)

    print("EVAL\tLoss\tPPL\tact_loss\tsacc\tlacc\tactacc", file=val_result)
    print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(ty, loss, math.exp(loss_rec), act_loss, emotion_s_acc, emotion_l_acc, act_acc), file = val_result)
    return loss, math.exp(loss_rec), elbo, kld, bow, act_loss, emotion_s_acc, emotion_l_acc, act_acc












