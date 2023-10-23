import math
import argparse

import matplotlib.pyplot as plt
import utils,model
import time
import numpy as np
import tensorflow as tf
import time, datetime
import os
import matplotlib
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 12,
                    help = 'prediction steps')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'batch size')
parser.add_argument('--traffic_file', default = 'data/march_all.h5',
                    help = 'traffic file')
parser.add_argument('--SE_file', default = 'data/SE(PeMS).txt',
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = 'data/GMAN(PeMS)_march_all_SW4L2lr4',
                    help = 'pre-trained model')
parser.add_argument('--log_file', default = 'data/GMAN(PeMS)_march_all_testcode',
                    help = 'log file')
parser.add_argument('--weather_rain_file', default='data/data_rain_all.h5',
                    help='weather_rain file')
parser.add_argument('--weather_clouds_file', default='data/data_visibility_all.h5',
                    help='weather_clouds file')
parser.add_argument('--weather_pressure_file', default='data/data_pressure_all.h5',
                    help='weather_pressure file')
parser.add_argument('--weather_temp_file', default='data/data_temp_all.h5',
                    help='weather_temp file')
args = parser.parse_args()

start = time.time()

log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')

(trainX, trainTE, trainY,trainW, valX, valTE, valY,valW, testX, testTE, testY,testW, SE,
  mean, std) = utils.loadData_weather(args)
num_train, num_val, num_test = trainX.shape[0], valX.shape[0], testX.shape[0]
utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
graph = tf.Graph()
with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(graph = graph, config = config) as sess:
    saver.restore(sess, args.model_file)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    pred = graph.get_collection(name = 'pred')[0]
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')

    testPred = []
    num_batch = math.ceil(num_test / args.batch_size)
    start_test = time.time()
    print(testX.shape)
    print(testTE.shape)
    print(testW.shape)
    for batch_idx in range(num_batch):
        print(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "third loop batch idx:" + str(batch_idx))
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            'Placeholder:0': testX[start_idx : end_idx],
            'Placeholder_1:0': testTE[start_idx : end_idx],
            'Placeholder_3:0': testW[start_idx: end_idx],
            'Placeholder_5:0': False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        testPred.append(pred_batch)
    print("-------------------------------------")
    print(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "third loop done")
    end_test = time.time()
    testPred = np.concatenate(testPred, axis = 0)

log.close()

