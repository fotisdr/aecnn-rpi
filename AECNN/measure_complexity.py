#!/usr/bin/env python

"""
Script to measure the complexity (number of parameters and floating-point operations) 
of a Keras model stored in .h5 format using Tensorflow

Fotis Drakopoulos, UGent
"""

import tensorflow as tf
import keras.backend as K
import keras.models
#from os import listdir

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--directory", help="Path to the folder of a pre-trained model", required=True, type=str)

    return parser


run_meta = tf.RunMetadata()

args = build_argparser().parse_args()
model_name = args.directory

with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)

    net = keras.models.load_model(model_name + '/Gmodel_full.h5')

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    # write in txt 
    #f = open('flops.txt','a')
    #f.write('\n' + model_name + '\n')
    #f.write('Flops = ' + str(flops.total_float_ops) + '\n')
    #f.write('Params = ' + str(params.total_parameters) + '\n')
    #f.close()

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

    sess.close()
