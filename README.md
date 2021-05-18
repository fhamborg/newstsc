# Announcement
We recommend using our newer, improved [model and dataset](https://github.com/fhamborg/NewsMTSC), which have various advantages, most importantly: 

1. larger dataset (11k sentences rather than 3k),
2. improved labels in the dataset, e.g., including more realistic cases of sentiment in news articles, such as through subtle word choice, and
3. improved model that achieves better classification performance.

The repository you are currently viewing is not maintained any longer.

# [Click here to go to NewsMTSC](https://github.com/fhamborg/NewsMTSC)

# Target-dependent Sentiment Classification in News Articles (NewsTSC)
Code and dataset for our [paper](https://www.gipp.com/wp-content/papercite-data/pdf/hamborg2021.pdf) "Towards Target-dependent Sentiment Classification in News Articles" accepted at the iConference 2021.

This readme consists of two main parts: installation of NewsTSC and how to use it. For both, there are instructions describing two use cases of the system: 

1. How to easily classify your own data (using our best performing model).
2. How to train your own models (using the NewsTSC dataset or any other dataset).

If you only want to download the dataset, please click [here](https://github.com/fhamborg/newstsc/raw/master/newstsc-dataset.zip).

# Installation
To setup the required environment, we use Anaconda. If you haven't installed Anaconda yet, first follow their [installation instructions](https://docs.anaconda.com/anaconda/install/). NewsTSC was tested on MacOS and Ubuntu. Other OS may work, too. Let us know :-)

## Core installation
```bash
conda create --yes -n ctsacuda python=3.7
conda activate ctsacuda

# choose either of both: the first is recommended if you have an NVIDIA GPU that supports CUDA
# with CUDA 10.0
conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch 
# without CUDA (calculations will be performed on your CPU, not recommended for training your own model but should be okay if you only classify sentiment in news articles)
conda install --yes pytorch torchvision -c pytorch

conda install --yes pandas tqdm scikit-learn
conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate imbalanced-learn
conda install --yes -c anaconda requests gensim openpyxl networkx
  
pip install pytorch-transformers

pip install spacy==2.1.0
python -m spacy download en_core_web_lg
pip install neuralcoref --no-binary neuralcoref
```

If you want to *train your own models*, that's it! Still, see [below](#train-your-own-models) for optional things to setup that may improve the performance.

If you want to perform *target-dependent sentiment classification on your own news articles*, see [Use NewsTSC for classification](#use-newstsc-for-classification) (or, of course, train your own).

## Use NewsTSC for classification
If you want to classify sentiment in sentences and do not want to train your own model, follow these instructions to download our news-adapted BERT language model and our fine-tuned weights for the model.

### Download news-adapted BERT
We fine-tuned BERT on 10M sentences randomly sampled from news articles from the [Common Crawl News Crawl](https://commoncrawl.org/2016/10/news-dataset-available/). To use
it, download the [model](https://github.com/fhamborg/newstsc/releases/download/bert_news_v1.0_3e/bert_news_ccnc_10mio_3ep.zip), 
extract it, and place the folder `bert_news_ccnc_10mio_3ep` into 
`pretrained_models/`.

Terminal friends may instead use (when in the project's root directory):
```
wget https://github.com/fhamborg/newstsc/releases/download/bert_news_v1.0_3e/bert_news_ccnc_10mio_3ep.zip
unzip bert_news_ccnc_10mio_3ep.zip
rm -f bert_news_ccnc_10mio_3ep.zip
mv bert_news_ccnc_10mio_3ep pretrained_models/
```

### Download fine-tuned weights
You can download the model that performed best during our evaluation. Download it [here](https://github.com/fhamborg/newstsc/releases/download/news_v1.0/lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip), extract it, and place the folder `lcf_bert_newstsc_val_recall_avg_0.5954_epoch3` into `pretrained_models/state_dicts/`.

Alternatively, execute the following:
```
wget https://github.com/fhamborg/newstsc/releases/download/news_v1.0/lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip
unzip lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip
rm -f lcf_bert_newstsc_val_recall_avg_0.5954_epoch3.zip
mkdir pretrained_models/state_dicts
mv lcf_bert_newstsc_val_recall_avg_0.5954_epoch3 pretrained_models/state_dicts
```

## Train your own models
You can start training right away after completing the [core installation](#core-installation). However, for improved performance we recommend to use the news-adapted BERT language model (for download instructions, see [here](#download-news-adapted-bert).

### GloVe (optional)
BERT-based models yield higher performance, but NewsTSC also supports GloVe for TSC. You can install GloVe embeddings as follows.
```
cd embeddings/glove/data
wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip
rm -f glove.42B.300d.zip
python gensimconvert.py
```

# How to use NewsTSC
## Target-dependent Sentiment Classification
Target-dependent sentiment classification works out-of-the-box if you setup our state_dict (you may also train your own, see below). Have a look at infer.py or give it a try:
```
python infer.py
```

## Training 
There are two entry points to the system. `train.py` is used to train and evaluate a specific model on a specific dataset using 
specific hyperparameters. We call a single run an _experiment_. `controller.py` is used to run multiple experiments 
automatically. This is for example useful for model selection and evaluating hundreds or thousands of combinations of 
models, hyperparameters, and datasets.

### Running a single experiment 
`train.py` allows fine-grained control over the training and evaluation process, yet for most command line arguments
we provide useful defaults. Important arguments include `--model_name` (which model is used, e.g., `LCF_BERT`) and 
`--dataset_name` (which dataset is used, e.g., `newstsc`). For more information refer to `train.py` and 
`combinations_absadata_0.py`. If you just want to test the system, the command below should work out of the box.

```
python train.py --model_name lcf_bert --optimizer adam --initializer xavier_uniform_ --learning_rate 2e-5 --batch_size 16 --balancing None --num_epoch 3 --lsr True --use_tp_placeholders False --eval_only_after_last_epoch True --devmode False --local_context_focus cdm --SRD 3 --pretrained_model_name bert_news_ccnc_10mio_3ep --snem recall_avg --dataset_name newstsc --experiment_path ./experiments/newstsc_20191126-115759/0/ --crossval 0 --task_format newstsc
```

### Running multiple experiments
`controller.py` takes a set of values for each argument, creates combinations of arguments, applies conditions to remove
unnecessary combinations (e.g., some arguments may only be used for a specific model), and creates a multiprocessing 
pool to run experiments of these argument combinations in parallel. After completion, `controller.py` creates a summary,
which contains detailed results, including evaluation performance, of all experiments. By using `createoverview.py`, you
can export this summary into an Excel spreadsheet.   

# Additional notes
Note that we currently use `pytorch-transformers` for increased performance. You can also use the more recent `transformers` package, but it will lead to a [performance drop](https://github.com/songyouwei/ABSA-PyTorch/issues/27#issuecomment-551058509).

# Acknowledgements
This repository is strongly based on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch). We thank Song et al. for making their excellent repository open source. We also wish to thank J. Dix, M. Luettecke, and T. Spinde for their support on creating the dataset.

# How to cite
If you are using the dataset, please cite our [paper](http://www.gipp.com/wp-content/papercite-data/pdf/hamborg2021.pdf):
```
@InProceedings{Hamborg2021,
  author    = {Hamborg, Felix and Donnay, Karsten and Gipp, Bela},
  title     = {Towards Target-dependent Sentiment Classification in News Articles},
  booktitle = {Proceedings of the iConference 2021},
  year      = {2021},
  month     = {Mar.},
  location  = {Beijing, China (Virtual Event)},
}
```
You can find more information on this and other news projects on our [website](https://felix.hamborg.eu/).

# License
[MIT License](LICENSE).
