#!/usr/bin/env python

"""
Real-time single-channel denoising using AECNN model and jackd audio server

Benchmark script for all settings and models in the specified directory.

Fotis Drakopoulos, UGent
"""

from __future__ import division, print_function
from time import time
import jack
import sys
from os import listdir
import numpy as np
from argparse import ArgumentParser
from threading import Event
try:
    import queue  # Python 3.x
except ImportError:
    import Queue as queue  # Python 2.x
from scipy.io.wavfile import read
from math import floor
from subprocess import check_call

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Wav file location for benchmarking - should have the same sampling rate as the model (16 kHz)", required=True, type=str)
    parser.add_argument("-d", "--directory", help="Directory containing the model folder(s) to benchmark", required=True, type=str)
    parser.add_argument("-f", "--frontend", help="keras (for .h5, .json files) or tensorflow (for .pb files)", default='tensorflow', type=str)
    parser.add_argument("-n", "--framesize", help="Size of the input/output frames of the model", default=0, type=int)
    parser.add_argument("-it", "--iterations", help="Iterations for averaging execution time", default=10, type=int)
    parser.add_argument("-st", "--savetxt", help="Save results in txt format", default=1, type=int)
    parser.add_argument("-q", "--queuesize", help="Size of the input/output queues in buffers", default=4, type=int)
    parser.add_argument("-p", "--precision", help="Float precision of the model", default='float32', type=str)
    parser.add_argument("-s", "--summary", help="Print summary of the model", default=0, type=int)
    parser.add_argument("-fs", "--sampling_rate", help="16 kHz sampling rate is used for AECNN models by default", default=16000, type=int)

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
        data = np.zeros((datain.shape))
        #qin.put(datain)
        #data = q.get_nowait()
        client.outports[0].get_array()[:] = data
    except queue.Empty:
        stop_callback('Buffer is empty: increase queuesize?')


args = build_argparser().parse_args()

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

directory = args.directory
save_flag = bool(args.savetxt)
for modeldir in listdir(directory):
    if True: #modeldir.startswith('AECNN'):

        if save_flag:
            f = open('processing_times_' + frontend + '.txt','a')
            f.write('\n' + modeldir + '\n')
            f.close()
            
        if args.framesize == 0:
            if int(modeldir[6])==1:
                if int(modeldir[7])==0:
                    args.framesize = 1024
                else:
                    args.framesize = 128
            if int(modeldir[6])==2:
                args.framesize = 256
            if int(modeldir[6])==5:
                args.framesize = 512
        if args.framesize == 1024:
                buf_j = 3
        elif args.framesize == 128:
                buf_j = 2
        elif args.framesize == 256:
            buf_j = 3
        elif args.framesize == 512:
            buf_j = 3
        else:
            print('Frame size should be 1024, 512, 256 or 128 samples')
            buf_j = 1
            
        buffersize=0.
        for buf_i in range(0,buf_j):
            if buf_i != 0:
                buffersize += 0.5/buf_i
            for num_i in range(0,2):
                overlap = 0.5*num_i

                # Start jackd server (to have almost the same level of computational load)
                fs = args.sampling_rate
                blocksize = int((1-overlap) * (1-buffersize) * args.framesize)
                command = './start_jackd.sh %d %d' % (blocksize,fs)
                check_call(command.split())

                if args.queuesize < 1:
                    print('Queuesize must be at least 1')
                    queuesize = 1
                else:
                    queuesize = args.queuesize

                q = queue.Queue(maxsize=queuesize)
                qin = queue.Queue(maxsize=queuesize)
                event = Event()

                # Load DNN model
                precision = args.precision
                fullmodeldir = directory + modeldir
                print ("Loading model from " + fullmodeldir + "/Gmodel")
                if frontend == 'tensorflow':
                    sess=Session()
                    graph_def = GraphDef()
                    with gfile.FastGFile(fullmodeldir + '/Gmodel.pb', 'rb') as f:
                        graph_def.ParseFromString(f.read())
                        import_graph_def(graph_def, name='')
                    output_layer = 'g_output/Reshape:0'
                    for n in graph_def.node:
                        if n.op == 'Placeholder':
                            input_node = n.name + ':0'
                    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
                    del n, graph_def, output_layer
                else:
                    g_opt = Adam(lr=0.0002) # Define optimizers
                    json_file = open(fullmodeldir + "/Gmodel.json", "r")
                    loaded_model_json = json_file.read()
                    json_file.close()
                    G_loaded = model_from_json(loaded_model_json)
                    G_loaded.compile(loss='mean_squared_error', optimizer=g_opt)
                    G_loaded.load_weights(fullmodeldir + "/Gmodel.h5")
                    if bool(args.summary):
                        G_loaded.summary()

                # Initialise the model
                model_blocksize = args.framesize
                if buffersize != 0 or overlap != 0:
                    buffer_blocksize = int(model_blocksize - blocksize)
                    if overlap != 0:
                        cleanb=np.zeros((blocksize,),dtype='float32')
                noisy=np.zeros((1,model_blocksize,1),dtype=precision)
                data=np.zeros((blocksize,),dtype=precision)
                if frontend == 'tensorflow':
                    clean = sess.run(prob_tensor, {input_node: noisy })
                else:
                    clean = G_loaded.predict(noisy)

                # Initialise jackd client
                client = jack.Client("thru_client")
                blocksize = client.blocksize
                samplerate = client.samplerate
                client.set_xrun_callback(xrun)
                client.set_shutdown_callback(shutdown)
                #client.set_process_callback(process)

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
                #q.put_nowait(data)

                with client:
                    i.connect(capture[0])
                    # Connect mono file to stereo output
                    o.connect(playback[0])
                    o.connect(playback[1])

                    _, signal = read(args.input)
                    scaled = np.array(signal/32767).astype('float32')
                    ie = int(floor(len(scaled) / blocksize))
                    del signal

                    t_total = 0
                    if frontend == 'tensorflow':
                        if overlap == 0:
                            if buffersize == 0:
                                for iter in range(0,args.iterations):
                                    t_total1 = 0
                                    for i in range(0, ie):
                                        noisy[0,:,0] = scaled[i*blocksize:(i+1)*blocksize] #.astype(precision)
                                        t = time()
                                        clean = sess.run(prob_tensor, {input_node: noisy })
                                        clean = clean.ravel()
                                        t_total1 += time() - t
                                    mean_time = t_total1 / ie
                                    t_total += mean_time
                                mean_time = 1000 * t_total / args.iterations
                            else:
                                for iter in range(1,args.iterations+1):
                                    t_total1 = 0
                                    for i in range(0, ie):
                                        data = scaled[i*blocksize:(i+1)*blocksize] #.astype(precision)
                                        t = time()
                                        noisy[0,:-blocksize,0] = noisy[0,blocksize:,0]
                                        noisy[0,-blocksize:,0] = data
                                        clean = sess.run(prob_tensor, {input_node: noisy })
                                        data = clean[0,buffer_blocksize:,0]
                                        t_total1 += time() - t
                                    mean_time = t_total1 / ie
                                    t_total += mean_time
                                mean_time = 1000 * t_total / args.iterations
                        elif overlap == 0.5:
                            for iter in range(1,args.iterations+1):
                                t_total1 = 0
                                for i in range(0, ie):
                                    data = scaled[i*blocksize:(i+1)*blocksize] #.astype(precision)
                                    t = time()
                                    noisy[0,:-blocksize,0] = noisy[0,blocksize:,0]
                                    noisy[0,-blocksize:,0] = data
                                    clean = sess.run(prob_tensor, {input_node: noisy })
                                    data = overlap*(cleanb+clean[0,buffer_blocksize-blocksize:buffer_blocksize,0])
                                    cleanb=clean[0,buffer_blocksize:,0]
                                    t_total1 += time() -t
                                mean_time = t_total1 / ie
                                t_total += mean_time
                            mean_time = 1000 * t_total / args.iterations
                        else:
                            print('Overlap percentage must be 0 or 0.5')
                            KeyboardInterrupt
                    else:
                        if overlap == 0:
                            if buffersize == 0:
                                for iter in range(0,args.iterations):
                                    t_total1 = 0
                                    for i in range(0, ie):
                                        noisy[0,:,0] = scaled[i*blocksize:(i+1)*blocksize] #.astype(precision)
                                        t = time()
                                        clean = G_loaded.predict(noisy)
                                        clean = clean.ravel()
                                        t_total1 += time() - t
                                    mean_time = t_total1 / ie
                                    t_total += mean_time
                                mean_time = 1000 * t_total / args.iterations
                            else:
                                for iter in range(1,args.iterations+1):
                                    t_total1 = 0
                                    for i in range(0, ie):
                                        data = scaled[i*blocksize:(i+1)*blocksize] #.astype(precision)
                                        t = time()
                                        noisy[0,:-blocksize,0] = noisy[0,blocksize:,0]
                                        noisy[0,-blocksize:,0] = data
                                        clean = G_loaded.predict(noisy)
                                        data = clean[0,buffer_blocksize:,0]
                                        t_total1 += time() - t
                                    mean_time = t_total1 / ie
                                    t_total += mean_time
                                mean_time = 1000 * t_total / args.iterations
                        elif overlap == 0.5:
                            for iter in range(1,args.iterations+1):
                                t_total1 = 0
                                for i in range(0, ie):
                                    data = scaled[i*blocksize:(i+1)*blocksize] #.astype(precision)
                                    t = time()
                                    noisy[0,:-blocksize,0] = noisy[0,blocksize:,0]
                                    noisy[0,-blocksize:,0] = data
                                    clean = G_loaded.predict(noisy)
                                    data = overlap*(cleanb+clean[0,buffer_blocksize-blocksize:buffer_blocksize,0])
                                    cleanb=clean[0,buffer_blocksize:,0]
                                    t_total1 += time() -t
                                mean_time = t_total1 / ie
                                t_total += mean_time
                            mean_time = 1000 * t_total / args.iterations
                print('Average processing time: %f ms' % mean_time)
                if save_flag:
                    f = open('processing_times_' + frontend + '.txt','a')
                    f.write('Buffersize = ' + str(buffersize) + '\n')
                    f.write('Overlap = ' + str(overlap) + '\n')
                    f.write(str(mean_time) + '\n')
                    f.close()

                if frontend == 'tensorflow':
                    sess.close()
                    tf.reset_default_graph()
                else:
                    clear_session()
                command = 'killall jackd'
                check_call(command.split())
