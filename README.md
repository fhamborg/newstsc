# Target-dependent Sentiment Classification in News Articles (NewsTSC)
Code for our paper submitted to the ACL 2020. Note that some files had to be changed temporarily due to the 
double-blind requirements of ACL 2020.

# Installation
```bash
conda create --yes -n ctsacuda python=3.7
conda activate ctsacuda
conda install --yes pandas tqdm scikit-learn

# with CUDA 10.0
conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch 
# w/o cuda: conda install --yes pytorch torchvision -c pytorch

conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate imbalanced-learn
conda install --yes -c anaconda requests gensim openpyxl
pip install pytorch-transformers
```

## Domain adapated 
As mentioned in the paper, we fine-tuned BERT on 10M sentences randomly sampled from the Common Crawl News Crawl. To use
it, place the folder `downloaded-folder/bert/bert_news_ccnc_10mio_3ep` into 
`downloaded-folder/code/cope-tsa/pretrained_models/bert_news_ccnc_10mio_3ep`.

## GloVe:
```
cd embeddings/glove/data
wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip
rm -f glove.42B.300d.zip
python gensimconvert.py
```

# Training 
There are two entry points to the system. `train.py` is used to train and evaluate a specific model on a specific dataset using 
specific hyperparameters. We call a single run an _experiment_. `controller.py` is used to run multiple experiments 
automatically. This is for example useful for model selection and evaluating hundreds or thousands of combinations of 
models, hyperparameters, and datasets.

## Running a single experiment 
`train.py` allows fine-grained control over the training and evaluation process, yet for most command line arguments
we provide useful defaults. Important arguments include `--model_name` (which model is used, e.g., `LCF_BERT`) and 
`--dataset_name` (which dataset is used, e.g., `newstsc`). For more information refer to `train.py` and 
`combinations_absadata_0.py`. If you just want to test the system, the command below should work out of the box.

```
python train.py --model_name lcf_bert --optimizer adam --initializer xavier_uniform_ --learning_rate 2e-5 --batch_size 16 --balancing None --num_epoch 3 --lsr True --use_tp_placeholders False --eval_only_after_last_epoch True --devmode False --local_context_focus cdm --SRD 3 --pretrained_model_name bert_news_ccnc_10mio_3ep --snem recall_avg --dataset_name newstsc --experiment_path ./experiments/newstsc_20191126-115759/0/ --crossval 0 --task_format newstsc
```

## Running multiple experiments
`controller.py` takes a set of values for each argument, creates combinations of arguments, applies conditions to remove
unnecessary combinations (e.g., some arguments may only be used for a specific model), and creates a multiprocessing 
pool to run experiments of these argument combinations in parallel. After completion, `controller.py` creates a summary,
which contains detailed results, including evaluation performance, of all experiments. By using `createoverview.py`, you
can export this summary into an Excel spreadsheet.   

# Acknowledgements
The core functionality of this repository is strongly based on 
[ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch). We thank Song et al. for making their excellent repository
open source.

# License
An open-source license will be added after review. 

Copyright by the authors.
 