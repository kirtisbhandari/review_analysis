# -*- coding: utf-8 -*-
import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, vocab_size=20000, encoding=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        input_file = os.path.join(data_dir, "yelp_review_data")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, encoding)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()



    def basic_tokenizer(self, line, normalize_digits=True):
        """ A basic tokenizer to tokenize text into tokens.
        Feel free to change this to suit your need. """
        """
        line = re.sub('<u>', '', line)
        line = re.sub('</u>', '', line)
        line = re.sub('\[', '', line)
        line = re.sub('\]', '', line)"""
        words = []
        _WORD_SPLIT = re.compile(b"([.,!?\"'-:;)(])")
        _DIGIT_RE = re.compile(r"\d")
        for fragment in line.strip().lower().split():
            for token in re.split(_WORD_SPLIT, fragment):
                if not token:
                    continue
                if normalize_digits:
                    token = re.sub(_DIGIT_RE, b'#', token)
                words.append(token)
        return words

    def build_vocab(self, all_words):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        """
        all_words = []
        for sentence in sentences:
            all_words.extend(self.basic_tokenizer(sentence))
        """
        # Build vocabulary
        word_counts = collections.Counter(all_words)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common(self.vocab_size -1)]
        vocabulary_inv.append("<UNK>")
        vocabulary_inv = list(sorted(vocabulary_inv))
        
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file, encoding):
        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        #data = self.clean_str(data)
        x_text = data.split('\n')

        all_words = []
        for sentence in x_text:
            all_words.extend(self.basic_tokenizer(sentence))
        
        self.vocab, self.words = self.build_vocab(all_words)
        #self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        # fix it here
        self.tensor = np.array([self.vocab.get(word, self.vocab['<UNK>']) for word in all_words])

        #self.tensor = np.array(list(map(self.vocab.get, x_text)))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


    def get_embeddings(self, data_dir, glove_filename): 

        glove_file_path = os.path.join(data_dir, ("../glove/"+ glove_filename))

        embedding = {}
        glove_vocab = []

        print("glove path", glove_file_path)
        file = codecs.open(glove_file_path,'r', 'utf8')
        #print("glove_vocab_size, glove_vector_size ", glove_vocab_size, glove_vector_size )
        
        for line in file.readlines():
            row = line.strip().split(' ')
            glove_vocab.append(row[0])
            coefs = np.asarray(row[1:], dtype='float32')
            embedding[row[0]] = coefs
        
        print('Loaded GloVe!')
        self.embedding_dim = len(embedding[glove_vocab[0]])
        print("Embedding size %d", self.embedding_dim)
        self.W = np.random.uniform(-1.0, 1.0, (self.vocab_size, self.embedding_dim))
        
        for (word, id) in self.vocab.iteritems():
            try:
                vector = None
                                
                wordnet_lemmatizer = WordNetLemmatizer()
                nnWord = wordnet_lemmatizer.lemmatize(word)
                vector = embedding[nnWord]
            except KeyError as e:
                print("Word not found", word)
                vector = np.random.uniform(-1.0, 1.0, self.embedding_dim)
            self.W[id] = vector
        file.close()
        return self.W

"""
def main():
    t = TextLoader("data/", 1,1, vocab_size=20000 ) 

if __name__ == '__main__':
    main()
"""
