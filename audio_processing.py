#!/usr/bin/env python

"""
Real-time single-channel denoising using AECNN speech enhancement model and jackd audio server

The AECNN model needs to be able to execute within the required processing time (framesize/fs), 
otherwise xrun errors are produced by the jackd audio server and the output gets filled with random values.

If you are unsure of your model's capabilities, you can run the benchmark script and measure its execution time.

Fotis Drakopoulos, UGent
"""

from __future__ import division, print_function
from time import time
import jack
import sys
import numpy as np
from argparse import ArgumentParser
from threading import Event
try:
    import queue  # Python 3.x
except ImportError:
    import Queue as queue  # Python 2.x
from subprocess import check_call

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to the folder of a pre-trained model - The model needs to be able to execute within the required processing time (framesize/fs)", required=True, type=str)
    parser.add_argument("-f", "--frontend", help="keras (for .h5, .json files) or tensorflow (for .pb files)", default='tensorflow', type=str)
    parser.add_argument("-n", "--framesize", help="Size of the input/output frames of the model", required=True, type=int)
    parser.add_argument("-b", "--buffersize", help="Percentage of buffering in the input/output frames for reducing latency - can be 0, 0.5 or 0.75 (0, 1 or 3 buffers)", default=0, type=float)
    parser.add_argument("-o", "--overlap", help="Overlap percentage of the audio frames - can be 0 or 0.5", default=0, type=float)
    parser.add_argument("-q", "--queuesize", help="Size of the input/output queues in buffers", default=4, type=int)
    parser.add_argument("-p", "--precision", help="Float precision of the model", default='float32', type=str)
    parser.add_argument("-fs", "--sampling_rate", help="16 kHz sampling rate is used for AECNN models by default", default=16000, type=int)
    parser.add_argument("-s", "--summary", help="Print summary of the model", default=0, type=bool)

    return parser

def print_error(*args):
    print(*args, file=sys.stderr)

def xrun(delay):
    print_error("An xrun occured, increase JACK's period size?")

def shutdown(status, reason):
    print_error('JACK shutdown!')
    print_error('status:', status)
    print_error('reason:', reason)
    event.set()

def stop_callback(msg=''):
    if msg:
        print_error(msg)
    for port in client.outports:
        port.get_array().fill(0)
    event.set()

def process(frames):
    if frames != blocksize:
        stop_callback('blocksize must not be changed, I quit!')
    try:
        datain=client.inports[0].get_array()
        qin.put(datain)
        data = qout.get_nowait()
        client.outports[0].get_array()[:] = data
    except queue.Empty:
        stop_callback('Buffer is empty: increase queuesize?')


args = build_argparser().parse_args() #parse arguments

# Import necessary modules for the specified frontend
frontend = args.frontend
if frontend == 'tensorflow' or frontend == 'Tensorflow' or frontend == 'tf':
    frontend = 'tensorflow'
    from tensorflow import Session, GraphDef, gfile, import_graph_def
elif frontend == 'keras' or frontend == 'Keras' or frontend == 'k':
    from keras.models import model_from_json
    from keras.optimizers import Adam
else:
    print('The frontend argument must be either "tensorflow" or "keras" - Tensorflow will be used')
    frontend = 'tensorflow'
    from tensorflow import Session, GraphDef, gfile, import_graph_def

# Start jackd server
overlap = args.overlap
buffersize = args.buffersize
fs = args.sampling_rate
blocksize = int((1-overlap) * (1-buffersize) * args.framesize)
command = './start_jackd.sh %d %d' % (blocksize,fs)
check_call(command.split()) # calls start_jackd script to start jackd server

# Use queues to pass data to/from the audio backend
if args.queuesize < 1:
    print('Queuesize must be at least 1')
    queuesize = 1
else:
    queuesize = args.queuesize
qout = queue.Queue(maxsize=queuesize)
qin = queue.Queue(maxsize=queuesize)
event = Event()

# Initialise variables
model_blocksize = args.framesize
if buffersize != 0 or overlap != 0:
    buffer_blocksize = int(model_blocksize - blocksize)
    if overlap != 0:
        cleanb=np.zeros((blocksize,),dtype='float32')
noisy=np.zeros((1,model_blocksize,1),dtype=precision)
data=np.zeros((blocksize,),dtype=precision)

# Load DNN model
precision = args.precision #not used at the moment
modeldir = args.model
print ("Loading model from " + modeldir + "/Gmodel")
if frontend == 'tensorflow':
    sess=Session()
    graph_def = GraphDef()
    with gfile.FastGFile(modeldir + '/Gmodel.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
        import_graph_def(graph_def, name='')
    output_layer = 'g_output/Reshape:0'
    for n in graph_def.node:
        if n.op == 'Placeholder':
            input_node = n.name + ':0'
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    clean = sess.run(prob_tensor, {input_node: noisy })
    del n, graph_def, output_layer
else:
    g_opt = Adam(lr=0.0002) # Define optimizers
    json_file = open(modeldir + "/Gmodel.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    G_loaded = model_from_json(loaded_model_json)
    G_loaded.compile(loss='mean_squared_error', optimizer=g_opt)
    G_loaded.load_weights(modeldir + "/Gmodel.h5")
    if args.summary:
        G_loaded.summary()
    clean = G_loaded.predict(noisy)
    del loaded_model_json, json_file

try:
    # Initialise jackd client
    client = jack.Client("thru_client")
    blocksize = client.blocksize
    samplerate = client.samplerate
    client.set_xrun_callback(xrun)
    client.set_shutdown_callback(shutdown)
    client.set_process_callback(process)

    client.inports.register('in_{0}'.format(1))
    client.outports.register('out_{0}'.format(1))
    i=client.inports[0]
    capture = client.get_ports(is_physical=True, is_output=True)
    playback = client.get_ports(is_physical=True, is_input=True, is_audio=True)
    o=client.outports[0]

    timeout = blocksize / samplerate
    print("Processing input in %d ms frames" % (int(round(1000 * timeout))))

    # Pre-fill queues
    #qin.put_nowait(data)
    qout.put_nowait(data) # the output queue needs to be pre-filled

    with client:
        i.connect(capture[0])
        # Connect mono file to stereo output
        o.connect(playback[0])
        o.connect(playback[1])

        # The processing algorithm is implemented here
        if frontend == 'tensorflow':
            if overlap == 0:
                if buffersize == 0:
                    while True:
                        # the input frame is extracted from the queue and saved in the 3d array noisy
                        noisy[0,:,0] = qin.get() #.astype(precision) 
                        # the input frame is processed by the DNN model and saved in the 3d array clean
                        clean = sess.run(prob_tensor, {input_node: noisy })
                        # the output frame 'clean' is converted to 1d and added in the output queue
                        qout.put(clean.ravel())
                else:
                    while True:
                        data = qin.get() #.astype(precision)
                        noisy[0,:-blocksize,0] = noisy[0,blocksize:,0] # move old data
                        noisy[0,-blocksize:,0] = data # add new data
                        clean = sess.run(prob_tensor, {input_node: noisy })
                        data = clean[0,buffer_blocksize:,0]
                        qout.put(data)
            elif overlap == 0.5:
                while True:
                    data = qin.get() #.astype(precision)
                    noisy[0,:-blocksize,0] = noisy[0,blocksize:,0]
                    noisy[0,-blocksize:,0] = data
                    clean = sess.run(prob_tensor, {input_node: noisy })
                    # 50% overlap
                    data = overlap*(cleanb+clean[0,buffer_blocksize-blocksize:buffer_blocksize,0])
                    cleanb=clean[0,buffer_blocksize:,0]
                    qout.put(data)
            else:
                raise RuntimeError('Overlap percentage must be 0 or 0.5')
        else:
            if overlap == 0:
                if buffersize == 0:
                    while True:
                        noisy[0,:,0] = qin.get() #.astype(precision)
                        clean = G_loaded.predict(noisy)
                        qout.put(clean.ravel())
                else:
                    while True:
                        data = qin.get() #.astype(precision)
                        noisy[0,:-blocksize,0] = noisy[0,blocksize:,0]
                        noisy[0,-blocksize:,0] = data
                        clean = G_loaded.predict(noisy)
                        data = clean[0,buffer_blocksize:,0]
                        qout.put(data)
            elif overlap == 0.5:
                while True:
                    data = qin.get() #.astype(precision)
                    noisy[0,:-blocksize,0] = noisy[0,blocksize:,0]
                    noisy[0,-blocksize:,0] = data
                    clean = G_loaded.predict(noisy)
                    data = overlap*(cleanb+clean[0,buffer_blocksize-blocksize:buffer_blocksize,0])
                    cleanb=clean[0,buffer_blocksize:,0]
                    qout.put(data)
            else:
                raise RuntimeError('Overlap percentage must be 0 or 0.5')
except (queue.Full):
    raise RuntimeError('Queue full')
except KeyboardInterrupt:
    print('\nInterrupted by User')
command = 'killall jackd'
check_call(command.split())
