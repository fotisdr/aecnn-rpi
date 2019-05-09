"""
Script for testing AECNN models. Edit the frontend variable for Tensorflow models.

Written by Fotis Drakopoulos, UGent, Jan 2019
Based on the training scipt by Deepak Baby, UGent, Oct 2018.
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
import numpy as np
from data_ops import *
from file_ops import *
from models import *

import time
from tqdm import *
import h5py
import os,sys
import scipy.io.wavfile as wavfile
#import shutil

def slice_1dsignal(signal, window_size, minlength, stride=0.5):
    """ 
    Return windows of the given signal by sweeping in stride fractions
    of window
    Slices that are less than minlength are omitted
    """
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    num_slices = (n_samples)
    slices = np.array([]).reshape(0, window_size) # initialize empty array
    for beg_i in range(0, n_samples, offset):
        end_i = beg_i + window_size
        if n_samples - beg_i < minlength :
            break
        if end_i <= n_samples :
            slice_ = np.array([signal[beg_i:end_i]])
        else :
            slice_ = np.concatenate((np.array([signal[beg_i:]]), np.zeros((1, end_i - n_samples))), axis=1)
        slices = np.concatenate((slices, slice_), axis=0)
    return slices.astype('float32')

def read_and_slice1d(wavfilename, window_size, minlength, stride=0.5):
    """
      Reads and slices the wavfile into windowed chunks
    """
    fs, signal =  wavfile.read(wavfilename)
    if fs != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    sliced = slice_1dsignal(signal, window_size, minlength, stride=stride)
    return sliced

if __name__ == '__main__':

    frontend = 'keras' #'tensorflow' for a converted model in .pb format
    fs = 16000

    wav_txt = open('test_wav.txt','r')
    wav_filenames = wav_txt.readlines()
    wav_txt.close()
    wav_filenames = [x.strip() for x in wav_filenames]

    if frontend == 'keras':
        import keras
        from keras.layers import Input, Dense, Conv1D, Conv2D, Conv2DTranspose, BatchNormalization
        from keras.layers import LeakyReLU, PReLU, Reshape, Concatenate, Flatten, Add, Lambda
        from keras.models import Sequential, Model, model_from_json
        from keras.optimizers import Adam
        from keras.callbacks import TensorBoard
        keras_backend = tf.keras.backend
        keras_initializers = tf.keras.initializers
        import keras.backend as K

    for model_name in os.listdir('.'):
        if model_name.startswith('AECNN_'):

            opts = {}
            opts['preemph']=0

            modeldir = model_name
            if int(model_name[6])==1:
                if int(model_name[7])==0:
                    opts ['window_length'] = 1024
                    buf_j = 3
                else:
                    opts ['window_length'] = 128
                    buf_j = 2
            if int(model_name[6])==2:
                opts ['window_length'] = 256
                buf_j = 2
            if int(model_name[6])==5:
                opts ['window_length'] = 512
                buf_j = 3

            ## Set the matfiles
            clean_train_matfile = "./data/clean_train_segan1d_%s.mat" % opts['window_length']
            noisy_train_matfile = "./data/noisy_train_segan1d_%s.mat" % opts['window_length']
            noisy_test_matfile = "./data/noisy_test_segan1d_%s.mat" % opts['window_length']

            print ("Loading model from " + modeldir + "/Gmodel")
            if frontend = 'tensorflow':
                sess = tf.Session()
                graph_def = tf.GraphDef()
                with tf.gfile.FastGFile("./converted/" + modeldir + "/Gmodel.pb", 'rb') as f:
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def,name='')
                for n in graph_def.node:
                    if n.op == 'Placeholder':
                        input_node = n.name + ':0'
                try:
                    output_layer = 'g_output/Reshape:0'
                    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
                except:
                    output_layer = 'model_1/g_output/Reshape:0'
                    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            else:
                json_file = open(modeldir + "/Gmodel.json", "r")
                loaded_model_json = json_file.read()
                json_file.close()
                G_loaded = model_from_json(loaded_model_json)
                G_loaded.compile(loss='mean_squared_error', optimizer=g_opt)
                G_loaded.load_weights(modeldir + "/Gmodel.h5")

            print ("********************************************")
            print ("               SEGAN TESTING                ")
            print ("********************************************")

            resultsdir = "./converted/" + modeldir + "/test_results_tf"
            if not os.path.exists(resultsdir):
                os.makedirs(resultsdir)
            else:
                #shutil.rmtree(resultsdir)
                #os.makedirs(resultsdir)
                if frontend == 'tensorflow':
                    sess.close()
                    tf.reset_default_graph()
                else:
                    K.clear_session()
                continue

            print ("Saving Results to " + resultsdir)

            buffersize=0.
            for buf_i in range(0,buf_j):
                if buf_i != 0:
                    buffersize += 0.5/buf_i
                for num_i in range(0,2):
                    overlap = 0.5*num_i

                    opts['stride']=(1-overlap) * (1-buffersize)
                    opts['minlength']= int(overlap * opts['window_length'])

                    for test_filenum in tqdm(range(len(wav_filenames))):
                        wav_file = './data/noisy_testset_wav_16kHz/' + wav_filenames[test_filenum]
                        noisy_test = read_and_slice1d(wav_file, opts['window_length'], opts['minlength'], stride=opts['stride'])

                        fst, signal =  wavfile.read(wav_file)
                        stride_length = int(opts['stride']*opts['window_length'])
                        buffer_length = int((1-buffersize)*opts['window_length'])
                        if opts['stride'] != 0:
                            ke = int(1/opts['stride'] - 1)
                            for k in range(0,ke):
                                signalt=np.concatenate((np.zeros((k+1)*stride_length),signal[0:(ke-k)*stride_length]))
                                signalt=np.reshape(signalt,(1,opts['window_length']))
                                noisy_test=np.concatenate((signalt,noisy_test))
                        cleanwavs = np.zeros((noisy_test.shape[0],buffer_length))
                        #print ("Number of test files: " +  str(noisy_test.shape) )

                        if frontend == 'tensorflow':
                            for test_num in (range(noisy_test.shape[0])) :
                                noisywavs = noisy_test[test_num,:] #T not necessary here
                                noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                                noisywavs.shape = (1,noisy_test.shape[1])
                                noisywavs = np.expand_dims(noisywavs, axis = 2)
                                cleaned_wavs = sess.run(prob_tensor, {input_node: noisywavs})
                                cleaned_wavs = np.reshape(cleaned_wavs, (1, noisywavs.shape[1]))
                                cleaned_wavs = np.reshape(cleaned_wavs, (-1,)) # make it to 1d by dropping the extra dimension
                                cleaned_wavs=cleaned_wavs[opts['window_length']-buffer_length:cleaned_wavs.shape[0]]
                                cleanwavs[test_num,:]=cleaned_wavs
                        else:
                            for test_num in (range(noisy_test.shape[0])) :
                                noisywavs = noisy_test[test_num,:] #T not necessary here
                                noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                                noisywavs.shape = (1,noisy_test.shape[1])
                                noisywavs = np.expand_dims(noisywavs, axis = 2)
                                cleaned_wavs = G_loaded.predict(noisywavs)
                                cleaned_wavs = np.reshape(cleaned_wavs, (1, noisywavs.shape[1]))
                                cleaned_wavs = np.reshape(cleaned_wavs, (-1,)) # make it to 1d by dropping the extra dimension
                                cleaned_wavs=cleaned_wavs[opts['window_length']-buffer_length:cleaned_wavs.shape[0]]
                                cleanwavs[test_num,:]=cleaned_wavs

                        if overlap == 0.5:
                            cleanwavs = np.delete(cleanwavs,(0),axis=0)

                        cleanwav = reconstruct_wav(cleanwavs, 1-overlap)
                        cleanwav = np.reshape(cleanwav,(-1,))

                        if opts['preemph'] > 0:
                            cleanwav = de_emph(cleanwav, coeff=opts['preemph'])

                        destfilename = resultsdir +  "/testwav_%d_%d_%d.wav" % ((test_filenum), int(100*buffersize), int(10*overlap))
                        wavfile.write(destfilename, fs, cleanwav)

            if frontend == 'tensorflow':
                sess.close()
                tf.reset_default_graph()
            else:
                K.clear_session()

