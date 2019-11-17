# Concept Perception - Target-dependent Sentiment Analysis (cope-tsa)



## Requirements

```bash
conda create --yes -n ctsacuda python=3.7
conda activate ctsacuda
conda install --yes pandas tqdm scikit-learn
conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch # w/o cuda: conda install --yes pytorch torchvision -c pytorch
conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate imbalanced-learn
conda install --yes -c anaconda requests gensim openpyxl
pip install pytorch-transformers # previously: conda install --yes  -c conda-forge transformers (see https://github.com/songyouwei/ABSA-PyTorch/issues/27#issuecomment-551058509)

# oneliner
conda create --yes -n ctsacuda python=3.7 && conda activate ctsacuda && conda install --yes pandas tqdm scikit-learn && conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch && conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate imbalanced-learn && conda install --yes -c anaconda requests gensim openpyxl && pip install pytorch-transformers  
```

To setup GloVe:
```
cd embeddings/glove/data
wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip
rm -f glove.42B.300d.zip
python gensimconvert.py
```


# Acknowledgements
based on https://github.com/songyouwei/ABSA-PyTorch