from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import ChatBotModel
import config
import data
import gc
import heapq
import numpy as np

def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in xrange(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_masks), decoder_size))

def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in xrange(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in xrange(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in xrange(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
    gc.collect()

def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us 
    choose a random bucket later on.
    """
    test_buckets = data.load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = data.load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in xrange(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale

def _get_skip_step(iteration):
    """ How many steps should the model train before it saves all the weights. """
    if iteration < 100:
        return 30
    return 100

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")

def _eval_test_set(sess, model, test_buckets):
    """ Evaluate on the test set. """
    for bucket_id in xrange(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(test_buckets[bucket_id], 
                                                                        bucket_id,
                                                                        batch_size=config.BATCH_SIZE)
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, 
                                   decoder_masks, bucket_id, True)
        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))

def train():
    """ Train the bot """
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()
    # in train mode, we need to create the backward path, so forwrad_only is False
    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with fix_gpu_memory() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(data_buckets[bucket_id], 
                                                                           bucket_id,
                                                                           batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % skip_step == 0:
                print('Iter {}: LR {}, loss {}, time {}'.format(iteration, model.learning_rate.eval(), total_loss/skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    _eval_test_set(sess, model, test_buckets)
                    start = time.time()
                sys.stdout.flush()
            gc.collect()

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in xrange(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])

"""
def _construct_response(output_logits, inv_dec_vocab):
    Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    
    print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    #if config.EOS_ID in outputs:
    #    outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])
"""


def greedy_decoder(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, input_token_ids, inv_dec_vocab):
    _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                         decoder_masks, bucket_id, True)    
    
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """

    top_k_indices = []
    print(output_logits[0])
    for logit in output_logits:
        val, indices = tf.nn.top_k(logit, k=10)
        indices = tf.unstack(indices, axis=1)
        top_k_word_choices = [tf.compat.as_str(inv_dec_vocab[index.eval()[0]]) for index in indices]
        print(top_k_word_choices)
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    #if config.EOS_ID in outputs:
    #    outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

def beam_search_decoder(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, input_token_ids, inv_dec_vocab):

    debug = True
    return_raw = True

    # Get output logits for the sentence.
    beams, new_beams, results = [(1, 0, {'eos': 0, 'dec_inp': decoder_inputs, 'prob': 1, 'prob_ts': 1, 'prob_t': 1})], [], [] # initialize beams as (log_prob, empty_string, eos)
    dummy_encoder_inputs = [np.array([config.PAD_ID]) for _ in range(len(encoder_inputs))]
    
    for dptr in range(len(decoder_inputs)-1):
      if dptr > 0: 
        decoder_masks[dptr] = [1.]
        beams, new_beams = new_beams[:config.BEAM_SIZE], []
      if debug: print("=====[beams]=====", beams)
      heapq.heapify(beams)  # since we will remove something
      for prob, _, cand in beams:
        #if cand['eos']: 
         # results += [(prob, 0, cand)]
          #continue


        if len(cand['dec_inp']) == 40:
          results += [(prob, 0, cand)]
          continue

        # normal seq2seq
        if debug: print(cand['prob'], " ".join([tf.compat.as_str(inv_dec_vocab[w[0]]) for w in cand['dec_inp']]))



        _, _, output_logits = run_step(sess, model, encoder_inputs, cand['dec_inp'], decoder_masks, bucket_id, True)
        all_prob_ts = softmax(output_logits[dptr][0])

        all_prob_t  = [0]*len(all_prob_ts)
        all_prob    = all_prob_ts

        # suppress copy-cat (respond the same as input)
        #if dptr < len(input_token_ids):
        #  all_prob[input_token_ids[dptr]] = all_prob[input_token_ids[dptr]] * 0.01

        # for debug use
        #if return_raw: print("probs ", all_prob, all_prob_ts, all_prob_t)
        
        # beam search  
        for c in np.argsort(all_prob)[::-1][:config.BEAM_SIZE]:
          new_cand = {
            'eos'     : (config.EOS_ID),
            'dec_inp' : [(np.array([c]) if i == (dptr+1) else k) for i, k in enumerate(cand['dec_inp'])],
            'prob_ts' : cand['prob_ts'] * all_prob_ts[c],
            'prob_t'  : cand['prob_t'] * all_prob_t[c],
            'prob'    : cand['prob'] * all_prob[c],
          }
          new_cand = (new_cand['prob'], random.random(), new_cand) # stuff a random to prevent comparing new_cand
          
          try:
            if (len(new_beams) < config.BEAM_SIZE):
              heapq.heappush(new_beams, new_cand)
            elif (new_cand[0] > new_beams[0][0]):
              heapq.heapreplace(new_beams, new_cand)
          except Exception as e:
            print("[Error]", e)
            print("-----[new_beams]-----\n", new_beams)
            print("-----[new_cand]-----\n", new_cand)
    
    results += new_beams  # flush last cands
    # post-process results
    res_cands = []
    for prob, _, cand in sorted(results, reverse=True):
      cand['dec_inp'] = " ".join([tf.compat.as_str(inv_dec_vocab[w[0]]) for w in cand['dec_inp']])
      res_cands.append(cand)

    return res_cands[:config.BEAM_SIZE]


def generate():
    """ in test mode, we don't to create the backward path
    """
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with fix_gpu_memory() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Welcome. Say something. Enter to exit. Max length is', max_length)
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            # Get token-ids for the input sentence.
            token_ids = data.sentence2id(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Max length I can handle is:', max_length)
                line = _get_user_input()
                continue
            # Which bucket does it belong to?
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])], 
                                                                            bucket_id,
                                                                            batch_size=1)
            # Get output logits for the sentence.
            decoder_responses = []        
            if config.BEAM_SEARCH:
                cand_decoder = beam_search_decoder(sess, model, encoder_inputs, decoder_inputs, 
                                            decoder_masks, bucket_id, token_ids, inv_dec_vocab)
                 
                for cand in cand_decoder:
                    decoder_responses.append(cand['dec_inp'])
            elif config.GREEDY:
                decoder_responses = greedy_decoder(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, token_ids, inv_dec_vocab)
            else:
                decoder_responses = temp_decoder(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, token_ids, inv_dec_vocab)
            
            print("responses")
            print(decoder_responses)

        output_file.write('=============================================\n')
        output_file.close()


def fix_gpu_memory():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
        #init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
        #sess = tf.Session()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
        #K.set_session(sess)
    return sess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'generate'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()
    """
    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepare_raw_data()
        data.process_data()
    """
    print('Data ready!')
    print("BUCKETS %s, NUM_LAYERS %d, HIDDEN_SIZE %d, BATCH_SIZE %d, LR %0.3f, MAX_GRAD_NORM %0.2f, ENC_VOCAB %d, DEC_VOCAB %d, DECAY_FACTOR %0.3f"
       %( config.BUCKETS , config.NUM_LAYERS , config.HIDDEN_SIZE , config.BATCH_SIZE , config.LR , config.MAX_GRAD_NORM , config.ENC_VOCAB , config.DEC_VOCAB, config.DECAY_FACTOR))
    # create checkpoints folder if there isn't one already
    data.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'generate':
        generate()

if __name__ == '__main__':
    main()
