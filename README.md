## 	CAB: Empathetic Dialogue Generation with Cognition, Affection and Behavior

This is the official implementation for paper [CAB: Empathetic Dialogue Generation with Cognition, Affection and Behavior] (DASFAA 2023).

## Model Architecture

![Image of KEMP](/model.jpg)


## Setup
- Check the packages needed or simply run the command:
```console
pip install -r requirements.txt
```
- Download GloVe vectors from [**here (glove.840B.300d.txt)**](https://drive.google.com/file/d/15ZEUyHCZ0f0mg0ecAFIbGInlkIUkOudY/view?usp=sharing) and put it into `/data/`.

- Download the completely processed dataset data from [**Google Drive**](https://drive.google.com/file/d/125ODBDGy2VMNvCiKBPJdmd8hwY9poJj0/view?usp=sharing) and place it into `/data/` for experiments.

- If you want to see the annotated dataset with dialogue act label and both interlector's emotion lables, then you could:

    - Download the processed EmpatheticDialogues dataset with dialogue act label and both interlector's emotion lables from [**here**](https://drive.google.com/drive/folders/1Pvgh5PZE_svSna3A_yhHf_Wngb4T5tcl?usp=sharing) and place processed dataset `train.json, valid.json and test.json` into `/data/ed_data/`.

- If you want to reconstruct knowledge paths, then you could:

    - Download the processed [**ConceptNet data**](https://drive.google.com/file/d/1pURkucLpa0SAWfiwba_J28kM5NQuo0qD/view?usp=sharing) and place processed data `ConceptNet_ranked_dict.json` into `/data/knowledge_data/`,  meanwhile, download [**dataset_preproc.p**](https://drive.google.com/file/d/1si4hznX37dValdmKCgf38v4mMF07n3Z0/view?usp=sharing) and place it into `/data/`.

- For reproducibility purposes, we place the model checkpoints at [**Google Drive**](https://drive.google.com/drive/folders/1g0MCmSClzM3VQoFy7LMhoEyBjOjiHTaJ?usp=sharing). You could download and move it under `/save/final/`.

- To skip training, please check folder `/result/CAB/output.txt/`.



## Data preprocessing

The dataset (EmpatheticDialogue) is preprocessed and stored under `data` in pickle format
```bash
python preprocess.py
```
You can skip the data processing and directly use the processed file `kemp_dataset_preproc.json`.

## Training
#### KEMP (Our)
```bash
python main.py \
--cuda \
--label_smoothing \
--noam \
--emb_dim 300 \
--hidden_dim 300 \
--hop 1 \
--heads 2 \
--pretrain_emb \
--model KEMP \
--device_id 0 \
--concept_num 1 \
--total_concept_num 10 \
--attn_loss \
--pointer_gen \
--save_path result/KEMP/ \
--emb_file data/glove.6B.300d.txt
```

#### KEMP w/o ECE

This model does not consider the emotional context graph of **E**motional **C**ontext **E**ncoder (ECE). 

In ECE, we enrich the dialogue history with external knowledge into an emotional context graph. Then, the emotional signals of context are distilled based on the embeddings and emotion intensity values from the emotional context graph.
```bash
python main.py \
--cuda \
--label_smoothing \
--noam \
--emb_dim 300 \
--hidden_dim 300 \
--hop 1 \
--heads 2 \
--pretrain_emb \
--model wo_ECE \
--device_id 0 \
--concept_num 1 \
--total_concept_num 10 \
--pointer_gen \
--save_path result/wo_ECE/ \
--emb_file data/glove.6B.300d.txt
```

#### KEMP w/o EDD

This model does not consider the emotional dependency strategies of **E**motion-**D**ependency **D**ecoder (EDD). 

In EDD, given emotional signal and emotional context graph, we incorporate an emotional cross-attention mechanism to selectively learn the emotional dependencies. 
```bash
python main.py \
--cuda \
--label_smoothing \
--noam \
--emb_dim 300 \
--hidden_dim 300 \
--hop 1 \
--heads 2 \
--pretrain_emb \
--model wo_EDD \
--device_id 0 \
--concept_num 1 \
--total_concept_num 10 \
--pointer_gen \
--save_path result/wo_EDD/ \
--emb_file data/glove.6B.300d.txt
```

## Testing
> Add `--test` into above commands.

You can directly run `/result/cal_metrics.py` script to evaluate the model predictions.


## Citation
If you find our work useful, please cite our paper as follows:

```bibtex
@article{li-etal-2022-kemp,
  title={Knowledge Bridging for Empathetic Dialogue Generation},
  author={Qintong Li and Piji Li and Zhaochun Ren and Pengjie Ren and Zhumin Chen},
  booktitle={AAAI},
  year={2022},
}
```

