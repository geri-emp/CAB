import os
import pickle
import torch
import json
from tqdm import tqdm
from models.CAB.others_layer import glove_embedding
from utils.common_layer import TransKnowEncoder
from utils.config import args
from utils import config
from data.process.util import wordCate, Stack, lemmatize_all
import csv
from textrank4zh import TextRank4Keyword
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from ast import literal_eval

REMOVE_RELATIONS = ["ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom", "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo", "SymbolOf", "FormOf", "AtLocation", "DerivedFrom", "CreatedBy", "Synonym", "MadeOf",
                    "Reverse-Antonym", "Reverse-ExternalURL", "Reverse-NotDesires", "Reverse-NotHasProperty", "Reverse-NotCapableOf", "Reverse-dbpedia", "Reverse-DistinctFrom", "Reverse-EtymologicallyDerivedFrom",
                    "Reverse-EtymologicallyRelatedTo", "Reverse-SymbolOf", "Reverse-FormOf", "Reverse-AtLocation", "Reverse-DerivedFrom",
                    "Reverse-CreatedBy", "Reverse-Synonym", "Reverse-MadeOf"
                    ]


if (os.path.exists('../dataset_preproc.p')):
    print("LOADING ed data for create context concept")
    with open('../dataset_preproc.p', "rb") as f:
        [data_tra, data_val, data_tst, vocab] = pickle.load(f)

embeddings = glove_embedding(vocab)
encoder = TransKnowEncoder(args.emb_dim, args.hidden_dim, num_layers=1, num_heads=args.heads, total_key_depth=args.depth,
                       total_value_depth=args.depth, filter_size=args.filter, universal=args.universal)


def get_concept_dict():
    '''
    Retrieve concepts from ConceptNet using the EmpatheticDialogue tokens as queries
    :return:
    '''
    if(os.path.exists('../dataset_preproc.p')):
        print("LOADING ed data for get concept dict")
        with open('../dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
            word2index = vocab.word2index

    embeddings = glove_embedding(vocab)
    CN = csv.reader(open("../assertions.csv", "r", encoding="utf-8"))

    concept_dict = {}
    concept_file = open("../ConceptNet.json", "w", encoding="utf-8")

    relation_dict = {}
    rd = open("../relation.json", "w", encoding="utf-8")

    for i, row in enumerate(CN):
        if i%1000000 == 0:
            print("Processed {} rows".format(i))

        items = "".join(row).split("\t")

        language = items[2].split("/")[2]
        if language == "en":
            if len(items) != 5:
                print("concept error")

            relation = items[1].split("/")[2]
            c1 = items[2].split("/")[3]
            c2 = items[3].split("/")[3]

            c1 = wnl.lemmatize(c1)
            c2 = wnl.lemmatize(c2)

            weight = literal_eval("{" + row[-1].strip())["weight"]
            score = (weight - 1) / (10.0 - 1.0) 
            if weight < 1.0:  # filter tuples where confidence score is smaller than 1.0
                continue
            if c1 in word2index and c2 in word2index and c1 != c2 and c1.isalpha() and c2.isalpha():
                if relation not in word2index:
                    if relation in relation_dict:
                        relation_dict[relation] += 1
                    else:
                        relation_dict[relation] = 0
                #shape:300 
                c1_vector = torch.Tensor(embeddings(torch.tensor([[word2index[c1]]]))[0][0])
                c2_vector = torch.Tensor(embeddings(torch.tensor([[word2index[c2]]]))[0][0])
                c1_c2_sim = torch.cosine_similarity(c1_vector, c2_vector, dim=0).item()
                # <c1 relation c2>
                if c2 not in stop_words:
                    if c1 in concept_dict:
                        concept_dict[c1][c2] = [relation, c1_c2_sim, score]
                    else:
                        concept_dict[c1] = {}
                        concept_dict[c1][c2] = [relation, c1_c2_sim, score]
                # reverse relation  <c2 relation c1>
                if c1 not in stop_words:
                    if c2 in concept_dict:
                        concept_dict[c2][c1] = ['Reverse-' +relation, c1_c2_sim, score]
                    else:
                        concept_dict[c2] = {}
                        concept_dict[c2][c1] = ['Reverse-' +relation, c1_c2_sim, score]


    print("concept num: ", len(concept_dict))
    json.dump(concept_dict, concept_file)

    relation_dict = sorted(relation_dict.items(), key=lambda x: x[1], reverse=True)
    json.dump(relation_dict, rd)

    rank_concept_file = open('../ConceptNet_ranked_dict.json', 'w', encoding='utf-8')
    rank_concept = {}
    for i in concept_dict:

        # [relation, c1_c2_sim, score]
        rank_concept[i] = dict(sorted(concept_dict[i].items(), key= lambda x: x[1][2], reverse= True))

        rank_concept[i] = [[l, rank_concept[i][l][0], rank_concept[i][l][1], rank_concept[i][l][2]] for l in rank_concept[i]]
    json.dump(rank_concept, rank_concept_file, indent=4)

def get_encode(token, sentence):
    word2index = vocab.word2index
    sentence = sentence.clone().detach().unsqueeze(0)
    mask = sentence.data.eq(config.PAD_idx).unsqueeze(1)

    token_emb = embeddings(torch.tensor([[word2index[token]]]))
    sentence_emb = embeddings(sentence)

    token_hidden = encoder(sentence_emb, token_emb, mask)

    return token_hidden[0][0]

def first_extract_triples(data):
    context = data['context_s']
    context_id = {}
    for i, sentence in enumerate(context):
        for num, sgl in enumerate(sentence):
            if num==0:
                context_id[i] = [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                   sgl]
            else:
                context_id[i] += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                   sgl]  
        if config.USE_CUDA:
            context_id[i] = torch.tensor(context_id[i])
    assert len(context) == len(context_id)

    context_dict = {}
    for i, text in enumerate(context):
        for j, sgl_text in enumerate(text):
            if j == 0:
                context_dict[i] = ' '.join(sgl_text)
            else:
                context_dict[i] = context_dict[i] + ' ' + ' '.join(sgl_text)
    
    concept = json.load(open("../knowledge_data/ConceptNet_ranked_dict.json", "r", encoding="utf-8"))
    word2index = vocab.word2index
    data['key_concepts'] = []
    data['triples'] = []
    for i, (sample, id) in tqdm(enumerate(zip(context_dict.values(), context_id.values())),total= len(context_dict)):
        concepts = {}  # concepts of each sample
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=sample, lower=True, window=2)  
        if len(sample) >= 20:
            for item in tr4w.get_keywords(10, word_min_len=1):  
                key = lemmatize_all(item.word)
                if key in concept and item.word in word2index:
                    concepts[key] = word2index[item.word]
                    # print(item.word, item.weight)
        elif len(sample) < 20 and len(sample) > 10:
            for item in tr4w.get_keywords(5, word_min_len=1):  
                key = lemmatize_all(item.word)
                if key in concept and item.word in word2index:
                    concepts[key] = word2index[item.word]
        else:
            for item in tr4w.get_keywords(3, word_min_len=1):  
                key = lemmatize_all(item.word)
                if key in concept and item.word in word2index:
                    concepts[key] = word2index[item.word]
        concepts = dict(sorted(concepts.items(), key=lambda x: x[1], reverse=False))
        words_pos = nltk.pos_tag(concepts.keys())
        dialog_concepts = [
            word if word in word2index and word not in stop_words and wordCate(words_pos[wi]) else ''
            for wi, word in enumerate(concepts)]

        while '' in dialog_concepts:
            dialog_concepts.remove('')
        data['key_concepts'].append(dialog_concepts)


        G = [] 
 
        J = Stack() 
        for con in dialog_concepts:
            tail = set(dialog_concepts) - {con}
            tail = list(tail)

            for ta in tail:
                h = 0 
                J.push((' ', ' ', con))
                g = []
                while not J.is_empty():
                    head = J.pop(J.peek())
                    h +=1
                    g.append(head)


                    candi_c = [concept[head[2]]]

                    triples = []  
                    # head_vector = embeddings(torch.tensor([[word2index[head[2]]]]).cuda())[0][0]
                    head_vector = get_encode(head[2], id)
                    if candi_c != []:
                        for x, c in enumerate(candi_c[0]):
                            if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                                candi_vector = embeddings(torch.tensor([[word2index[c[0]]]]))[0][0]
                                head_cos_candi = torch.cosine_similarity(head_vector, candi_vector, dim=0).item()
                                socre = head_cos_candi + c[3]
                                triples.append([head[2], c[1], c[0], socre])
                    triples = sorted(triples, key=lambda x: x[3], reverse=True)  
                    triples = triples[:args.K_num]

                    # ta_vector = embeddings(torch.tensor([[word2index[ta]]]).cuda())[0][0]
                    ta_vector = get_encode(ta, id)
                    for tr in triples:
                        tr_vector = embeddings(torch.tensor([[word2index[tr[2]]]]))[0][0]
                        ta_cos_tr = torch.cosine_similarity(ta_vector, tr_vector, dim=0).item()
                        tr.append(ta_cos_tr)
                    triples = sorted(triples, key= lambda x: x[4], reverse= True)
                    triples = triples[: args.k_num]
                    triples = triples[::-1]

                    for tr in triples:
                        if tr[2] == ta:
                            g.append((tr[0],tr[1],tr[2]))
                            G.append(g)
                            J.clean()
                            break
                        else:
                            if h < args.multi_hop:
                                J.push((tr[0],tr[1],tr[2]))
                   
                    if h == args.multi_hop:
                        h -= 1
                        g = g[:-1]

                        if not J.is_empty() and J.peek()[0] != head[0]:
                            g = g[:-1]
        G = sorted(G, key= lambda x: len(x), reverse= True)
        # G = G[: args.path_num]
        data['triples'].append(G)
    return data

def second_extract_triples(data):
    concept = json.load(open("../knowledge_data/ConceptNet_ranked_dict.json", "r", encoding="utf-8"))
    word2index = vocab.word2index

    for q, three_tri in tqdm(enumerate(data['triples']), total= len(data['context_s'])):
        if three_tri == []:
            context = data['context_s'][q]
            context_id = []
            for i, sentence in enumerate(context):
                context_id += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                       sentence]  
            if config.USE_CUDA:
                context_id = torch.tensor(context_id).cuda()
            else:
                context_id = torch.tensor(context_id)


            G = [] 

            J = Stack() 
            key_concepts = [] 
            for word in data['key_concepts'][q]:
                if word in word2index:
                    key_concepts.append(word)
            for con in key_concepts:
                tail = set(key_concepts) - {con}
                tail = list(tail)

                for ta in tail:
                    h = 0 
                    J.push((' ', ' ', con))
                    g = []
                    while not J.is_empty():
                        head = J.pop(J.peek())
                        h += 1
                        g.append(head)
                       
                        candi_c = [concept[head[2]]]
                        
                        triples = []  

                        head_vector = get_encode(head[2], context_id)
                        if candi_c != []:
                            for x, c in enumerate(candi_c[0]):
                                if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                                    candi_vector = embeddings(torch.tensor([[word2index[c[0]]]]))[0][0]
                                    head_cos_candi = torch.cosine_similarity(head_vector, candi_vector, dim=0).item()
                                    socre = abs(head_cos_candi) + c[3]
                                    triples.append([head[2], c[1], c[0], socre])
                        triples = sorted(triples, key=lambda x: x[3], reverse=True)  
                        triples = triples[:args.K_num]

                        ta_vector = get_encode(ta, context_id)
                        for tr in triples:
                            tr_vector = embeddings(torch.tensor([[word2index[tr[2]]]]))[0][0]
                            ta_cos_tr = torch.cosine_similarity(ta_vector, tr_vector, dim=0).item()
                            tr.append(ta_cos_tr)
                        triples = sorted(triples, key=lambda x: x[4], reverse=True)  
                        triples = triples[: args.k_num]
                        triples = triples[::-1] 

                        for tr in triples:
                            if tr[2] == ta:
                                g.append((tr[0], tr[1], tr[2]))
                                G.append(g)
                                J.clean()
                                break
                            else:
                                if h < args.multi_hop:  
                                    J.push((tr[0], tr[1], tr[2]))
                            
                        if h == args.multi_hop:
                            h -= 1
                            g = g[:-1]

                            if not J.is_empty() and J.peek()[0] != head[0]:  
                                g = g[:-1]

            # G = [n for n in G if len(n) > 2]
            G = sorted(G, key= lambda x: len(x), reverse= True)
            # G = G[: args.path_num]
            data['triples'][q] = G
    return data

def third_extract_triples(data):
    concept = json.load(open("../knowledge_data/ConceptNet_ranked_dict.json", "r", encoding="utf-8"))
    word2index = vocab.word2index

    for q, three_tri in tqdm(enumerate(data['triples']), total= len(data['triples'])):
        concepts = {}
        if three_tri == []:
            context = data['context_s'][q]
            key_concepts = []  
            ori_key_concepts = data['key_concepts'][q]
            if len(ori_key_concepts) != 0:
                key_concepts = ori_key_concepts
            else:
                for i in context:
                    for j in i:
                        concepts[j] = word2index[j]
                concepts = dict(sorted(concepts.items(), key=lambda x: x[1], reverse=False))
                words_pos = nltk.pos_tag(concepts.keys())
                dialog_concepts = [
                    word if word not in stop_words and wordCate(words_pos[wi]) else ''
                    for wi, word in enumerate(concepts)]
                for word in dialog_concepts:
                    if word in concept:
                        key_concepts.append(word)
                if len(key_concepts)>5:
                    key_concepts.sort(key=lambda i: len(i), reverse=True)
                    key_concepts = key_concepts[:5]
            data['key_concepts'][q] = key_concepts
            context_id = []
            for i, sentence in enumerate(context):
                context_id += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                       sentence]  
            if config.USE_CUDA:
                context_id = torch.tensor(context_id).cuda()
            else:
                context_id = torch.tensor(context_id)
            
            G = [] 

            J = Stack() 

            for con in key_concepts:
                J.push(['third ', ' ', con])
                g = []
                while not J.is_empty():
                    head = J.pop(J.peek())
                    g.append(head)
                    
                    candi_c = [concept[head[2]]]
                    
                    triples = []  
                    # head_vector = embeddings(torch.tensor([[word2index[head[2]]]]).cuda())[0][0]
                    head_vector = get_encode(head[2], context_id)
                    if candi_c != []:
                        for x, c in enumerate(candi_c[0]):
                            if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                                candi_vector = embeddings(torch.tensor([[word2index[c[0]]]]))[0][0]
                                head_cos_candi = torch.cosine_similarity(head_vector, candi_vector, dim=0).item()
                                socre = abs(head_cos_candi) + c[3]
                                triples.append([head[2], c[1], c[0], socre])
                    triples = sorted(triples, key=lambda x: x[3], reverse=True)  
                    if triples == []:
                        G.append(g)
                        break
                    else:
                        triples = triples[:args.K_num]
                    for tr in triples:
                        g.append([tr[0],tr[1],tr[2]])
                    G.append(g)
                    J .clean()

            data['triples'][q] = G
    return data

def get_sample_data(data):
    for key, value in data.items():
            data[key] = value[0:10]
    return data

def first_create_context_concept(data):


    data_tra = get_sample_data(data)

    tra = first_extract_triples(data_tra)
    val = first_extract_triples(data_val, vocab)
    tst = first_extract_triples(data_tst, vocab)

    fp = open('../prepare_data/tra_preproc.json', "w")
    json.dump(tra, fp)
    print("Saved JSON")


def second_create_context_concept(data):

    # data = get_sample_data(data)
    sec_pro_data = second_extract_triples(data)

    fp = open('../prepare_data/tra_preproc.json', "w", encoding= 'utf-8')
    json.dump(sec_pro_data, fp)
    print("Saved JSON")

  def third_create_context_concept(data):

  thi_pro_data = third_extract_triples(data)

  fp = open('../prepare_data/tra_preproc.json', "w", encoding='utf-8')
  json.dump(thi_pro_data, fp)
  print("Saved JSON")

if __name__ == '__main__':

    get_concept_dict()

    first_create_context_concept(data_tra) #origin --multi_hop=6 --K_num=5 --k_num=3
    second_create_context_concept(data_tra) #change para --multi_hop=8 --K_num=7 or 10(change twice) --k_num=5 or 8(change twice)
    third_create_context_concept(data_tra) #find one_hop kn K_num=3
    
    first_create_context_concept(data_val)
    second_create_context_concept(data_val)
    third_create_context_concept(data_val)
    
    first_create_context_concept(data_tst)
    second_create_context_concept(data_tst)
    third_create_context_concept(data_tst)
    
    #after get all json file you can merge them to get final file which is ed_concept_dataset_preproc.p in our project

