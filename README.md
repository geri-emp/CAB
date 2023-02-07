## 	CAB: Empathetic Dialogue Generation with Cognition, Affection and Behavior

This is the official implementation for paper [**CAB: Empathetic Dialogue Generation with Cognition, Affection and Behavior**](https://arxiv.org/abs/2302.01935) (DASFAA 2023).

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

- For reproducibility purposes, we place the model [**checkpoints**](https://drive.google.com/drive/folders/1g0MCmSClzM3VQoFy7LMhoEyBjOjiHTaJ?usp=sharing). You could download and move it under `/save/final/`.

- To skip training, please check folder `/result/CAB/output.txt/`.


## Training
#### CAB (Our)
```bash
python main.py \
--cuda \
--batch_size 16 \
--lr 1e-4 \
--hidden_dim 300 \
--emo_dim 300 \
--act_dim 300 \
--latent_dim 200\
--hop 1 \
--heads 2 \
--pretrain_emb \
--model CAB \
--multi_hop 5 \
--K_num 5 \
--k_num 3 \
--path_num 15 \
--pointer_gen \
--emb_file data/glove.840B.300d.txt
```


## Testing
> Add `--test` into above commands.

You can directly run `/evaluate_result.py` script to evaluate the model predictions.


## Citation
If you find our work useful, please cite our paper as follows:

```bibtex

```

