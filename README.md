# Concept Perception - Target-dependent Sentiment Analysis (cope-tsa)

based on https://github.com/songyouwei/ABSA-PyTorch

## Requirements

```bash
conda create --yes -n cope-tsa python=3.7
conda activate cope-tsa
conda install --yes pandas tqdm scikit-learn
conda install --yes pytorch torchvision cudatoolkit=9.2 -c pytorch # w/o cuda: conda install --yes pytorch torchvision -c pytorch
conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate
conda install --yes -c anaconda requests gensim openpyxl
pip install pytorch-transformers # previously: conda install --yes  -c conda-forge transformers (see https://github.com/songyouwei/ABSA-PyTorch/issues/27#issuecomment-551058509)
```

second try
```bash
conda create --yes -n ctsa python=3.7
conda activate ctsa
conda install --yes pandas tqdm scikit-learn
conda install --yes pytorch torchvision cudatoolkit=9.2 -c pytorch # w/o cuda: conda install --yes pytorch torchvision -c pytorch
conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate
conda install --yes -c anaconda requests gensim openpyxl
pip install pytorch-transformers # previously: conda install --yes  -c conda-forge transformers (see https://github.com/songyouwei/ABSA-PyTorch/issues/27#issuecomment-551058509)
```

To setup GloVe:
```
cd embeddings/glove/data
wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip
rm -f glove.42B.300d.zip
python gensimconvert.py
```

## development notes

scp -r scc:/home/scc/fhamborg/code/cope-tsa/statistics .

## Usage

### Training

```sh
python train.py --model_name aen_distilbert --dataset_name poltsanews
```

See [train.py](./train.py) for more training arguments.

Refer to [train_k_fold_cross_val.py](./train_k_fold_cross_val.py) for k-fold cross validation support.

### Inference

Please refer to [infer_example.py](./infer_example.py).
