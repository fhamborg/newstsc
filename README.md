# Target-dependent Sentiment Classification in News Articles (NewsTSC)
Code for our paper submitted to the ACL 2020. Note that some files had to be changed temporarily due to the 
double-blind requirements of ACL 2020.

# Installation
We use Anaconda for setting up all requirements. If you do not have it yet, follow Anaconda's [installation instructions](https://docs.anaconda.com/anaconda/install/) - it's easy :-) cope-tsa was tested on MacOS.

## Minimum installation
```bash
conda create --yes -n ctsacuda python=3.7
conda activate ctsacuda

# choose either, first is recommended if you have an NVIDIA GPU that supports CUDA)
# with CUDA 10.0
conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch 
# without CUDA (calculations will be performed on your CPU)
conda install --yes pytorch torchvision -c pytorch

conda install --yes pandas tqdm scikit-learn
conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate imbalanced-learn
conda install --yes -c anaconda requests gensim openpyxl

pip install pytorch-transformers
```

If you want to train your own models, that's it! Still, see below for optional things to setup that may improve the performance.
If you want to perform target-dependent sentiment classification on your own data, see `Use our model` (or, of course, train your own).

## Use our model
If you just want to classify sentiment in sentences and do not want to train your own model: we provide the model that performed best in our evaluation. 
You can download the model that performed best during our evaluation. Download it [here](https://github.com/fhamborg/cope-tsa/releases/download/news_v1.0/lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip), extract it, and place the folder `lcf_bert_newstsc_val_recall_avg_0.5954_epoch3` into `pretrained_models/state_dicts/`.

Terminal friends may instead use (when in the project's root directory):
```
wget https://github.com/fhamborg/cope-tsa/releases/download/news_v1.0/lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip
unzip lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip
rm -f lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip
mv lcf_bert_newstsc_val_recall_avg_0.5954_epoch3 pretrained_models/state_dicts
```

For optimal classification performance, we recommend using our [news-adapted BERT language model](https://github.com/fhamborg/cope-tsa/releases/tag/bert_news_v1.0_3e).
See instructions below for setting it up.

Note that we currently use `pytorch-transformers` for increased performance. You can also use the more recent `transformers` package, but it will lead to a [performance drop](https://github.com/songyouwei/ABSA-PyTorch/issues/27#issuecomment-551058509).

## News-adapted BERT (recommended, optional)
We fine-tuned BERT on 10M sentences randomly sampled from news articles from the [Common Crawl News Crawl](https://commoncrawl.org/2016/10/news-dataset-available/). To use
it, download the [model](https://github.com/fhamborg/cope-tsa/releases/download/bert_news_v1.0_3e/bert_news_ccnc_10mio_3ep.zip), 
extract it, and place the folder `bert_news_ccnc_10mio_3ep` into 
`pretrained_models/`.

Terminal friends may instead use (when in the project's root directory):
```
wget https://github.com/fhamborg/cope-tsa/releases/download/bert_news_v1.0_3e/bert_news_ccnc_10mio_3ep.zip
unzip bert_news_ccnc_10mio_3ep.zip
rm -f bert_news_ccnc_10mio_3ep.zip
mv bert_news_ccnc_10mio_3ep pretrained_models/
```

## GloVe (optional)
BERT-based models yield higher performance, but cope-tsa also supports GloVe for TSC. You can install GloVe embeddings as follows.
```
cd embeddings/glove/data
wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip
rm -f glove.42B.300d.zip
python gensimconvert.py
```

# Target-dependent Sentiment Classification
Target-dependent sentiment classification works out-of-the-box if you setup our state_dict (you may also train your own, see below). Have a look at infer.py or give it a try:
```
python infer.py
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
[MIT License](LICENSE).
