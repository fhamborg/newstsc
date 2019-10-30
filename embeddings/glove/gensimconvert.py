# taken from https://stackoverflow.com/a/47465278

from gensim.scripts.glove2word2vec import glove2word2vec

from embeddings.glove import original_path, gensim_path

glove2word2vec(glove_input_file=original_path, word2vec_output_file=gensim_path)
