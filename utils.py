import numpy as np
import pandas as pd
import tensorflow as tf

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        #print(pred.shape)
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        #print(mask.shape)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y

# weatherProcess
def weatherProcess(Weather, wtrain_steps, wval_steps, wtest_steps):
    wtrain = Weather[: wtrain_steps]
    wval = Weather[wtrain_steps: wtrain_steps + wval_steps]
    wtest = Weather[-wtest_steps:]

    trainX, trainY = seq2instance(wtrain, args.P, args.Q)
    wtrain = np.concatenate((trainX, trainY), axis=1).astype(np.int32)
    valX, val = seq2instance(wval, args.P, args.Q)
    wval = np.concatenate((valX, valY), axis=1).astype(np.int32)
    testX, testY = seq2instance(wtest, args.P, args.Q)
    wtest = np.concatenate((testX, testY), axis=1).astype(np.int32)

    wtrain = np.expand_dims(wtrain, axis=3)
    wval = np.expand_dims(wval, axis=3)
    wtest = np.expand_dims(wtest, axis=3)

    return wtrain, wval, wtest

def loadData_weather(args):
    # Traffic
    df = pd.read_hdf(args.traffic_file)
    Traffic = df.values
    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding
    f = open(args.SE_file, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]

    # temporal embedding
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
    #            // Time.freq.delta.total_seconds()
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // 300
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps: train_steps + val_steps]
    test = Time[-test_steps:]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)

    #weather embedding
    dfrain = pd.read_hdf(args.weather_wind_speed_file)
    WeatherRain = dfrain.values

    dfclouds = pd.read_hdf(args.weather_clouds_file)
    Weatherclouds = dfclouds.values

    dfpressure = pd.read_hdf(args.weather_pressure_file)
    Weatherpressure = dfpressure.values

    dftemp = pd.read_hdf(args.weather_visibility_file)
    Weathertemp = dftemp.values

    Pressuremean, Pressurestd = np.mean(Weatherpressure), np.std(Weatherpressure)
    Weatherpressure = (Weatherpressure - Pressuremean) / Pressurestd

    # train/val/test
    wnum_step = dfrain.shape[0]
    wtrain_steps = round(args.train_ratio * wnum_step)
    wtest_steps = round(args.test_ratio * wnum_step)
    wval_steps = wnum_step - wtrain_steps - wtest_steps

    trainRain, valRain, testRain = weatherProcess(WeatherRain, wtrain_steps, wval_steps, wtest_steps)
    trainRain, valRain, testRain = weatherProcess(Weatherclouds, wtrain_steps, wval_steps, wtest_steps)
    trainRain, valRain, testRain = weatherProcess(Weatherpressure, wtrain_steps, wval_steps, wtest_steps)
    trainRain, valRain, testRain = weatherProcess(Weathertemp, wtrain_steps, wval_steps, wtest_steps)

    trainW = np.concatenate((trainRain, trainclouds), axis=3).astype(np.int32)
    valW = np.concatenate((valRain, valclouds), axis=3).astype(np.int32)
    testW = np.concatenate((testRain, testclouds), axis=3).astype(np.int32)
    trainW = np.concatenate((trainW, trainpressure), axis=3).astype(np.int32)
    valW = np.concatenate((valW, valpressure), axis=3).astype(np.int32)
    testW = np.concatenate((testW, testpressure), axis=3).astype(np.int32)
    trainW = np.concatenate((trainW, traintemp), axis=3).astype(np.int32)
    valW = np.concatenate((valW, valtemp), axis=3).astype(np.int32)
    testW = np.concatenate((testW, testtemp), axis=3).astype(np.int32)


    return (trainX, trainTE, trainY, trainW, valX, valTE, valY, valW, testX, testTE, testY,testW,
            SE, mean, std)

