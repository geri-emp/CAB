import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from utils import config
from utils.config import args
import pprint
pp = pprint.PrettyPrinter(indent=1)
import os
import torch.nn.init as init
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from models.CAB.hgt_layer import Node_attention_layer
from models.CAB.others_layer import GRU, glove_embedding, Latent, SoftmaxOutputLayer, Generator, NoamOpt, print_sample, OutputLayer
from utils.common_layer import TransEncoder, Encoder, Decoder, EmoAttention, top_k_top_p_filtering

class EmoActCvae_woAct(nn.Module):
    def __init__(self,vocab,emo_number,act_number, model_file_path = None, is_eval = False, load_optim = False):
        super(EmoActCvae_woAct,self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.emo_number = emo_number
        self.act_number = act_number
        self.relation_dict = {
            "TokenToToken": "is before", "Reverse-TokenToToken": "is after",
            "Reverse-IsA": "is a", "IsA": "is a",
            "Reverse-HasProperty": "is an attribute of", "HasProperty": "can",
            "Reverse-Desires": "is desired by", "Desires": "desires",
            "Reverse-HasA": "is owned by", "HasA": "has",
            "Reverse-RelatedTo": "is related to", "RelatedTo": "is related to",
            "Reverse-ReceivesAction": 'is', "ReceivesAction": "can be",
            "Reverse-Causes": "is because of", "Causes": "causes",
            "Reverse-HasSubevent": "before", "HasSubevent": "then",
            "Reverse-UsedFor": "needs", "UsedFor": "is used for",
            "Reverse-PartOf": "includes", "PartOf": "is part of",
            "Reverse-HasPrerequisite": "is the condition of", "HasPrerequisite": "has prerequisite",
            "Reverse-HasContext": "has meaning of", "HasContext": "has meaning of",
            "Reverse-MannerOf": "is the result of", "MannerOf": "is one manner of",
            "Reverse-SimilarTo": "is similar to", "SimilarTo": "is similar to",
            "Reverse-CapableOf": "benefit from", "CapableOf": "can",
            "Reverse-MotivatedByGoal": "desires", "MotivatedByGoal": "becauses",
            "Reverse-CausesDesire": "is desired by", "CausesDesire": "desires",
            "Reverse-LocatedNear": "is located near", "LocatedNear": "is located near",
            "Reverse-Entails": "is part of", "Entails": "entails",
            "Reverse-HasLastSubevent": "before", "HasLastSubevent": "then",
            "Reverse-HasFirstSubevent": "before", "HasFirstSubevent": "then",
            "self": "is"
        }

        self.embedding = glove_embedding(self.vocab,config.pretrain_emb)
        self.dropout = nn.Dropout(0.2)

        self.gru = GRU(config.emb_dim, config.hidden_dim, bidirectional = True)
        self.emo_encoder = TransEncoder(config.emb_dim, config.hidden_dim, num_layers = config.hop, num_heads = config.heads, total_key_depth = config.depth,total_value_depth = config.depth, filter_size = config.filter, universal=config.universal, layer_dropout=args.dropout)
        self.emo_attention = nn.Sequential(*[EmoAttention(config.hidden_dim, config.depth, config.depth, config.hidden_dim, dropout=args.dropout) for _ in range(config.hop)])
        self.act_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers = config.hop, num_heads = config.heads, total_key_depth = config.depth,total_value_depth = config.depth, filter_size = config.filter, universal=config.universal, layer_dropout=args.dropout)
        self.add_path = Node_attention_layer(args.d_model, args.d_model, self.embedding)
        self.r_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers = config.hop, num_heads = config.heads, total_key_depth = config.depth,total_value_depth = config.depth, filter_size = config.filter, universal=config.universal, layer_dropout=args.dropout)

        self.emo_s_latent_layer = Latent(is_eval)
        self.emo_l_latent_layer = Latent(is_eval)

        self.trans_input = nn.Linear(args.hidden_dim + args.act_dim, args.hidden_dim)
        self.trans_output = nn.Linear(args.hidden_dim + args.latent_dim *2, args.hidden_dim)
        self.trans_z = nn.Linear(args.hidden_dim + args.latent_dim *2, args.hidden_dim)
        self.decoder = Decoder(config.emb_dim, config.hidden_dim, num_layers = config.hop, num_heads = config.heads, total_key_depth = config.depth,total_value_depth = config.depth, filter_size = config.filter, layer_dropout=args.dropout)

        self.bow_softmax = SoftmaxOutputLayer(config.latent_dim,self.vocab_size)
        self.generator = Generator(config.hidden_dim, self.vocab_size)


        self.emo_embedding = nn.Embedding(self.emo_number, config.emo_dim)
        init.xavier_uniform_(self.emo_embedding.weight)
        self.emo_s_fc = OutputLayer(config.hidden_dim, emo_number)
        self.emo_l_fc = OutputLayer(config.hidden_dim, emo_number)
        self.emo_criterion = nn.CrossEntropyLoss()

        self.act_embedding = nn.Embedding(self.act_number,config.act_dim)
        init.xavier_uniform_(self.act_embedding.weight)
        self.act_fc = OutputLayer(config.hidden_dim, act_number)
        self.act_criterion = nn.CrossEntropyLoss()

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)

        if (config.USE_CUDA):
            self.cuda()
        if is_eval:
            self.eval()
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr = config.lr)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
            if (config.noam):
                self.optimizer = NoamOpt(300, 1, 8000,
                                         torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            if (load_optim):
                #  self.optimizer.load_state_dict(state['optimizer'])
                if config.USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
        self.model_dir = model_file_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, model, epoch, loss, ppl, kld, bow, s_acc, l_acc, act_acc, model_name = None):

        state = model.state_dict()
        model_save_path = os.path.join(self.model_dir,
                                       '{}_{:.4f}_{:.4f}'.format(model_name, epoch, ppl))
        self.best_path = model_save_path
        torch.save({"model": state,
                    "result": [loss, ppl, kld, bow, s_acc, l_acc, act_acc]},
                    model_save_path)

    def get_know_paths(self, batch_data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1 [32,69]
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            for i, leng in enumerate(lengths):
                if leng == 0:
                    lengths[i] = 1
            return padded_seqs, lengths

        #change the triple to path
        dia_path = []
        #get the key token
        key_tokens = []
        for i, dia_id in enumerate(batch_data['dia_id']):
            key_tokens.append(batch_data['key_tokens'][i])
            if batch_data['triples'][i] == []:
                dia_path.append([])
            else:
                paths = []
                for d_path in batch_data['triples'][i]:
                    path = []
                    for j in d_path[1:]:
                        for k in j:
                            path += [k]
                        path += '.'
                    # re_path = list(set(path))
                    # re_path.sort(key= path.index)
                    str_path = ' '.join(path)
                    for k, v in self.relation_dict.items():
                        str_path = str_path.replace(k,v)
                    paths.append(str_path.split())
                dia_path.append(paths)
        dial_sent = [i for i in range(len(dia_path))]
        dial_token = [i for i in range(len(dia_path))]
        for i, (s_path, token) in enumerate(zip(dia_path, key_tokens)):
            sentence_dial = []
            token_dial = []
            for j, sentence in enumerate(s_path):
                sentence_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                       sentence]
            token_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                       token]

            sentence_dial = torch.LongTensor(sentence_dial)
            token_dial = torch.LongTensor(token_dial)
            dial_sent[i] = sentence_dial
            dial_token[i] = token_dial.cuda()
        kn_batch, kn_lengths = merge(dial_sent)
        if kn_batch.size(-1) != 0:
            kn_batch, kn_hidden = self.gru(self.embedding(kn_batch.cuda()), kn_lengths)
            return kn_batch, dial_token
        else:
            return None, None

    def forward(self, batch):
        # enc_batch：context
        enc_batch = batch['input_batch']
        enc_batch_extend_vocab = batch["input_ext_batch"]
        enc_s_batch = batch['input_s_batch']
        enc_l_batch = batch['input_l_batch']

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).cuda()

        # dec_batch：torch.Size([batch_size, length])
        dec_batch = batch['target_batch']
        dec_ext_batch = batch["target_ext_batch"]

        update_paths, key_concepts = self.get_know_paths(batch)

        #r_encode
        # torch.Size([32, 1, 23])
        mask_res = batch['posterior_batch'].data.eq(config.PAD_idx).unsqueeze(1)
        posterior_mask = self.embedding(batch["posterior_mask"])
        r_encoder_outputs = self.r_encoder(self.embedding(batch["posterior_batch"]) + posterior_mask, mask_res)
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_s = enc_s_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_l = enc_l_batch.data.eq(config.PAD_idx).unsqueeze(1)

        input_mask = self.embedding(batch['input_mask'])
        input_s_mask = self.embedding(batch['input_s_mask'])
        input_l_mask = self.embedding(batch['input_l_mask'])

        s_encoder_outputs = self.emo_encoder(self.embedding(enc_batch)+input_mask, self.embedding(enc_s_batch)+input_s_mask, mask_src, mask_s)
        l_encoder_outputs = self.emo_encoder(self.embedding(enc_batch)+input_mask, self.embedding(enc_l_batch)+input_l_mask, mask_src, mask_l)
        act_encoder_outputs = self.act_encoder(self.embedding(enc_batch)+input_mask, mask_src)


        if update_paths != None:
            act_encoder_outputs, k_scores = self.add_path(act_encoder_outputs, update_paths, key_concepts, mask_src)
        else:
            k_scores = torch.tensor([])
        return enc_batch, dec_batch, enc_batch_extend_vocab, extra_zeros, dec_ext_batch, mask_src, mask_s, mask_l, s_encoder_outputs, l_encoder_outputs, r_encoder_outputs, act_encoder_outputs, k_scores


    def train_one_batch(self, batch, iter, train = True):
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        enc_batch, dec_batch, enc_batch_extend_vocab, extra_zeros, dec_ext_batch, \
        mask_src, mask_s, mask_l, s_encoder_outputs, l_encoder_outputs,\
        r_encoder_outputs, act_encoder_outputs, k_scores = self.forward(batch)

        #torch.Size([16, 33])
        emotion_s_logit_prob = self.emo_s_fc(s_encoder_outputs[:, 0], train = train)
        emotion_s_loss = self.emo_criterion(emotion_s_logit_prob, batch['emotion_s_label'].cuda())
        softmax_s_logit = F.log_softmax(emotion_s_logit_prob, dim=-1)
        pred_emotion_s_embedding = torch.matmul(softmax_s_logit, self.emo_embedding.weight)
        s_encoder_outputs, _, _, _, _ = self.emo_attention((s_encoder_outputs, pred_emotion_s_embedding.unsqueeze(1).expand(pred_emotion_s_embedding.size(0),s_encoder_outputs.size(1),pred_emotion_s_embedding.size(1)), s_encoder_outputs, s_encoder_outputs, mask_s))

        emotion_l_logit_prob = self.emo_l_fc(l_encoder_outputs[:, 0], train = train)
        emotion_l_loss = self.emo_criterion(emotion_l_logit_prob,batch['emotion_l_label'].cuda())
        softmax_l_logit = F.log_softmax(emotion_l_logit_prob, dim=-1)
        pred_emotion_l_embedding = torch.matmul(softmax_l_logit, self.emo_embedding.weight)
        l_encoder_outputs, _, _, _, _ = self.emo_attention((l_encoder_outputs, pred_emotion_l_embedding.unsqueeze(1).expand(pred_emotion_l_embedding.size(0),l_encoder_outputs.size(1),pred_emotion_l_embedding.size(1)), l_encoder_outputs, l_encoder_outputs, mask_l))

        if train:
            emotion_s_kld_loss, z_s = self.emo_s_latent_layer(s_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train = True)
            emotion_l_kld_loss, z_l = self.emo_l_latent_layer(l_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train = True)
        else:
            emotion_s_kld_loss, z_s = self.emo_s_latent_layer(s_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train = False)
            emotion_l_kld_loss, z_l = self.emo_l_latent_layer(l_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train = False)

        if not train:
            pred_emotion_s = np.argmax(softmax_s_logit.detach().cpu().numpy(), axis=1)
            emotion_s_acc = accuracy_score(batch['emotion_s_label'].cpu(), pred_emotion_s)

            pred_emotion_l = np.argmax(softmax_l_logit.detach().cpu().numpy(),axis=1)
            emotion_l_acc = accuracy_score(batch['emotion_l_label'].cpu(), pred_emotion_l)

        #get weight for z_s and z_l according to the emotion polarity
        weight = torch.cosine_similarity(pred_emotion_s_embedding, pred_emotion_l_embedding, dim = 1)
        abs_pos_weight = abs(weight)
        #Decode
        # torch.Size([32, 1]) [[3],[3]……[3]]得到batch_size大小的sos的tensor
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()

        # (batch, len)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        mask_trg_bow = mask_trg.data.eq(False)
        input_vector = self.embedding(dec_batch_shift)

        weight_z_s = torch.mul(abs_pos_weight.unsqueeze(1), z_s)
        weight_z_l = torch.mul((1-abs_pos_weight).unsqueeze(1), z_l)

        pre_logit, attn_dist = self.decoder(input_vector, act_encoder_outputs, k_scores, (mask_src, mask_trg))

        z_s_expand = weight_z_s.unsqueeze(1).expand(z_s.size(0), pre_logit.size(1), z_s.size(1))
        z_l_expand = weight_z_l.unsqueeze(1).expand(z_l.size(0), pre_logit.size(1), z_l.size(1))
        pre_logit = torch.cat([pre_logit, z_s_expand, z_l_expand], dim=-1)
        pre_logit = self.trans_z(pre_logit)
        logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if args.pointer_gen else None, extra_zeros)

        loss_rec = self.criterion(logit.contiguous().view(-1,logit.size(-1)), dec_batch.contiguous().view(-1) if args.pointer_gen else dec_ext_batch.contiguous().view(-1))
        # [batch_size, vocab_size]

        z = torch.cat((weight_z_s, weight_z_l),dim = -1)
        z_logit = self.bow_softmax(z, both = True)
        bow_loss = -F.log_softmax(z_logit, dim=1).gather(1, dec_batch) * mask_trg_bow
        bow_loss = torch.sum(bow_loss, 1)
        loss_aux = torch.mean(bow_loss)

        kl_weight = min(iter / config.full_kl_step, 1.0)
        kld_loss = emotion_s_kld_loss + emotion_l_kld_loss
        weighted_kld_loss = kl_weight * kld_loss

        elbo = loss_rec + weighted_kld_loss
        loss = loss_rec + weighted_kld_loss + 0.1*loss_aux + 0.1*emotion_l_loss + 0.03*emotion_s_loss

        act_loss = torch.tensor([0])
        act_acc = 0
        if(train):
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()
            return loss.item(), loss_rec.item(), emotion_s_loss, emotion_l_loss, act_loss.item(), kld_loss.item(), loss_aux.item(), elbo.item()
        else:
            return loss.item(), loss_rec.item(), elbo.item(), kld_loss.item(), loss_aux.item(), act_loss.item(), emotion_s_acc, emotion_l_acc, act_acc

    def decoder_greedy(self, batch, train=False, max_dec_step = 50):
        enc_batch, dec_batch, enc_batch_extend_vocab, extra_zeros, dec_ext_batch, \
        mask_src, mask_s, mask_l, s_encoder_outputs, l_encoder_outputs,\
        r_encoder_outputs, act_encoder_outputs, k_scores = self.forward(batch)

        # torch.Size([32, 32])
        emotion_s_logit_prob = self.emo_s_fc(s_encoder_outputs[:, 0], train = train)
        softmax_s_logit = F.log_softmax(emotion_s_logit_prob, dim=-1)
        pred_emotion_s_embedding = torch.matmul(softmax_s_logit, self.emo_embedding.weight)
        s_encoder_outputs, _, _, _, _ = self.emo_attention((s_encoder_outputs, pred_emotion_s_embedding.unsqueeze(1).expand(pred_emotion_s_embedding.size(0), s_encoder_outputs.size(1), pred_emotion_s_embedding.size(1)), s_encoder_outputs, s_encoder_outputs, mask_s))

        emotion_l_logit_prob = self.emo_l_fc(l_encoder_outputs[:, 0], train = train)
        softmax_l_logit = F.log_softmax(emotion_l_logit_prob, dim=-1)
        pred_emotion_l_embedding = torch.matmul(softmax_l_logit, self.emo_embedding.weight)
        l_encoder_outputs, _, _, _, _ = self.emo_attention((l_encoder_outputs, pred_emotion_l_embedding.unsqueeze(1).expand(pred_emotion_l_embedding.size(0), l_encoder_outputs.size(1), pred_emotion_l_embedding.size(1)), l_encoder_outputs, l_encoder_outputs, mask_l))

        emotion_s_kld_loss, z_s = self.emo_s_latent_layer(s_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train=False)
        emotion_l_kld_loss, z_l = self.emo_l_latent_layer(l_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train=False)

        weight = torch.cosine_similarity(pred_emotion_s_embedding, pred_emotion_l_embedding, dim = 1)
        abs_pos_weight = abs(weight)

        ys = torch.ones(enc_batch.shape[0], 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        decoded_words = []
        for i in range(max_dec_step + 1):
            input_vector = self.embedding(ys)

            weight_z_s = torch.mul(abs_pos_weight.unsqueeze(1), z_s)
            weight_z_l = torch.mul((1 - abs_pos_weight).unsqueeze(1), z_l)

            out, attn_dist = self.decoder(input_vector, act_encoder_outputs, k_scores, (mask_src, mask_trg))

            z_s_expand = weight_z_s.unsqueeze(1).expand(z_s.size(0), out.size(1), z_s.size(1))
            z_l_expand = weight_z_l.unsqueeze(1).expand(z_l.size(0), out.size(1), z_l.size(1))
            out = torch.cat([out, z_s_expand, z_l_expand], dim=-1)
            out = self.trans_z(out)

            prob = self.generator(out, attn_dist, enc_batch_extend_vocab if args.pointer_gen else None,
                                   extra_zeros)

            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])

            if config.USE_CUDA:
                # ys:[32,1] next_word:[32]——>ys:torch.Size([32, 2])
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+=e+' '
            sent.append(st)
        return sent






















