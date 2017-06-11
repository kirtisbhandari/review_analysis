from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/yelp',
                       help='data directory containing input')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--data_size', type=int, default=100000,
                       help='size of data to train on') 
    parser.add_argument('--vocab_size', type=int, default=40000,
                       help='size of vocabulary') 
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                       help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)
    gc.collect()

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.data_size, args.vocab_size, args.input_encoding)
    print_config(args)
    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
        del saved_words
        del saved_vocab

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)
    data_loader.words = None
    data_loader.vocab = None

    model = Model(args)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)

    with fix_gpu_memory()  as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)
            if args.init_from is not None:
                data_loader.pointer = model.batch_pointer.eval()
                args.init_from = None
            for b in range(data_loader.pointer, data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                train_writer.add_summary(summary, e * data_loader.num_batches + b)
                speed = time.time() - start
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("learning rate =  {:.5f},{}/{} (epoch {}), train_loss = {:.4f}, time/batch = {:.3f}" \
                        .format(model.lr.eval(), e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
        train_writer.close()
        gc.collect()

def fix_gpu_memory():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    return sess

def print_config(args):

    print("args.data_dir %s, args.rnn_size %d, num_layers %d, args.batch_size %d, args.seq_length %d, args.data_size %d, args.num_epochs %d, args.vocab_size %d, args.input_encoding %s, args.learning_rate %f" 
           %( args.data_dir, args.rnn_size, args.num_layers, args.batch_size, args.seq_length, args.data_size,
            args.num_epochs, args.vocab_size, args.input_encoding, args.learning_rate))


if __name__ == '__main__':
    main()
