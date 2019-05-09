#!/usr/bin/env python

"""
Script to convert a Keras model to protobuf format (.pb) for inferencing in Tensorflow

Fotis Drakopoulos, UGent
"""

import tempfile
import tensorflow as tf
from tensorflow.python.framework import graph_io
import keras.models
from keras import backend as K
import os
from argparse import ArgumentParser

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = None
        output_names = output_names or []
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--directory", help="Path to the folder of a pre-trained model", required=True, type=str)

    return parser


args = build_argparser().parse_args()
modeldir = args.directory
print(modeldir)

pb_name = 'Gmodel.pb'

K.set_learning_phase(0)
net_model = keras.models.load_model(modeldir + '/Gmodel_full.h5')

frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
graph_io.write_graph(frozen_graph, modeldir, pb_name, as_text=False)
tf.reset_default_graph()
K.clear_session()

