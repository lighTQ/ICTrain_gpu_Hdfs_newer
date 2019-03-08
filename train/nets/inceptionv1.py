# -*- coding:utf-8 -*-


import argparse
import os
import subprocess
import sys

import cv2
import keras.backend.tensorflow_backend as KTF
from keras.regularizers import l2
from jsonrpc import Server
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from trainmonitor import TrainingMonitor


# from sklearn.utils import shuffle
# import  matplotlib.pyplot as plt
# from keras.callbacks import TensorBoard
#
# 参数定义
def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-gpus', type=str, help='gpu_list')
    ap.add_argument('-img_src', type=str, help='train image path')
    ap.add_argument('-width', type=int, help='img_resize_width')
    ap.add_argument('-height', type=int, help='img__resize_height')
    ap.add_argument('-test_percent', type=float, help='test_dataset_percent')
    ap.add_argument('-validata_percent', type=float, help='validata_dataset_percent')
    ap.add_argument('-channel_num', type=int, help='channels_num')
    # ap.add_argument('-num_classes', type=int , help='num_classes')
    ap.add_argument('-batch_size', type=int, help='batch_size')
    ap.add_argument('-lr', type=float, help='learning_rate')
    ap.add_argument('-epochs', type=int, help='epochs')
    # ap.add_argument('data_augmentation', type=bool, , help='data_augmentation')
    ap.add_argument('-save_dir', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model h5  file')
    ap.add_argument('-jsonrpcMlClientPoint', type=str, help='IP address of server endpoint...')
    ap.add_argument('-model_name', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model file')
    ap.add_argument('-model_id', type=str, help="define the unique model identifier id")
    ap.add_argument('-model_userid', type=str, help="define the model user who are")
    ap.add_argument('-model_version', type=str, help="define the model version is what")
    ap.add_argument('-user_Optimizer', type=str, help="user selective optimizers")
    ap.add_argument('-ams_id', type=str, help="callback the trained model id")

    return vars(ap.parse_args())

arguments = parse_arguments(sys.argv[1:])
gpus = arguments['gpus']

print(' vgg16 get gpus is ', gpus)
# 控制gpu使用,指定使用哪一块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print(" current gpu list is : ", gpus)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用 GPU 0

global G
# 读取显卡个数"0"-->1, "0,1"--->2,"0,1,2"--->3
if gpus.strip() != '':
    G = len(gpus.split(","))
    print('current G is ', G)

    # 设置GPU使用动态增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)
else:
    G = -1
print('current G is ', G)


# 数据预处理
def preprocessing_img(img_src, width, height, validate_percent, test_percent, HDFS_data=True):
    """

    :param img_src:   图盘路径
    :param width:     图片resize 宽度
    :param height:    图片resize 高度
    :param validate_percent: 验证集百分比
    :param test_percent:     测试集百分比
    :param HDFS_data:        是否使用HDFS数据源
    :return:
    """
    X, y = [], []

    if HDFS_data:
        # 将数据从HDFS上爬取下来
        print(" 使用HDFS 数据源")
        prefix = "/root/Data"
        model_userid = "u1934234525601"
        downloaded_path = os.path.join(prefix, model_userid)
        local_img_path = os.path.join(downloaded_path, img_src.split("/")[-1])
        if not os.path.exists(downloaded_path):
            os.makedirs(downloaded_path)

        try:
            ret = subprocess.run(['/root/hadoop-2.7.3/bin/hadoop', "fs", "-get", img_src, downloaded_path])
            if ret.returncode == 0:
                print('download successfully, data saved in local path : %s' % local_img_path)
        except Exception as e:
            print("un expected error").format(e)

    else:
        print("使用本地数据源", img_src)
        
        local_img_path = img_src

    categorical_num = len(set(os.listdir(local_img_path)))

    for i, folder in enumerate(os.listdir(local_img_path)):
        # print(i, folder)
        for index, filename in enumerate(os.listdir(os.path.join(local_img_path, folder))):
            # imgFile = os.path.join(img_src, (folder+"/"+filename))
            imgFile = os.path.join(local_img_path, folder, filename)
            #print(index, imgFile)
            image = cv2.imread(imgFile)
            if isinstance(image, np.ndarray):
                pass
            else:
                continue
            img = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            X.append(img)
            y.append(i)
    X = np.array(X).astype('float32') / 255.0
    y = np_utils.to_categorical(y, categorical_num)
    print(X.shape)

    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validate_percent, shuffle=True)
    print('preprocessing images is OK..')
    return X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_num,local_img_path

# keras.backend.clear_session()
# 模型构建

def inception_model(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                    filters_pool_proj):
    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding='same', activation='relu',
                      kernel_regularizer=l2(0.01))(input)

    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                             kernel_regularizer=l2(0.01))(input)

    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=l2(0.01))(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                             kernel_regularizer=l2(0.01))(input)

    conv_5x5 = Conv2D(filters=filters_5x5, kernel_size=(5, 5), padding='same', activation='relu',
                      kernel_regularizer=l2(0.01))(conv_5x5_reduce)

    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)

    maxpool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), strides=(1, 1), padding='same',
                          activation='relu', kernel_regularizer=l2(0.01))(maxpool)

    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)  # use tf as backend

    return inception_output

def build_model(width, height, channel_num, num_classes, UserOptimizer='adam'):
    # input = Input(shape=(224, 224, 3))
    input = Input(shape=(width, height, channel_num))

    conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu',
                          kernel_regularizer=l2(0.01))(input)

    maxpool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_7x7_s2)

    conv2_3x3_reduce = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=l2(0.01))(maxpool1_3x3_s2)

    conv2_3x3 = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(
        conv2_3x3_reduce)

    maxpool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_3x3)

    inception_3a = inception_model(input=maxpool2_3x3_s2, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
                                   filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)

    inception_3b = inception_model(input=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                                   filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)

    maxpool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_3b)

    inception_4a = inception_model(input=maxpool3_3x3_s2, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
                                   filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)

    inception_4b = inception_model(input=inception_4a, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
                                   filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)

    inception_4c = inception_model(input=inception_4b, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
                                   filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)

    inception_4d = inception_model(input=inception_4c, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
                                   filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64)

    inception_4e = inception_model(input=inception_4d, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                                   filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)

    maxpool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_4e)

    inception_5a = inception_model(input=maxpool4_3x3_s2, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                                   filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)

    inception_5b = inception_model(input=inception_5a, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384,
                                   filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128)

    averagepool1_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(inception_5b)

    x = Dropout(rate=0.4)(averagepool1_7x7_s1)
    x = Flatten()(x)

    if (int(num_classes >= 2)):

        x = Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)
        model = Model(inputs=input, outputs=x)

        if G <= 1 and G > 0:
            print("[INFO] training with 1 GPU...")
            parallel_model = model

        # otherwise, we are compiling using multiple GPUs
        elif G < 0:
            print("[INFO] training with CPU...")
            parallel_model = model
        else:
            print("[INFO] training with {} GPUs...".format(G))
            # we'll store a copy of the model on *every* GPU and then combine
            # the results from the gradient updates on the CPU
            # with tf.device("/cpu:0"):
            # initialize the model
            # make the model parallel(if you have more than 2 GPU)
            parallel_model = multi_gpu_model(model, gpus=G)

        parallel_model.compile(loss='categorical_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
        print(parallel_model.summary())

    else:
        x = Dense(units=num_classes, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

        model = Model(inputs=input, outputs=x)

        if G <= 1 and G > 0:
            print("[INFO] training with 1 GPU...")
            parallel_model = model

        # otherwise, we are compiling using multiple GPUs
        elif G < 0:
            print("[INFO] training with CPU...")
            parallel_model = model
        else:
            print("[INFO] training with {} GPUs...".format(G))
            # we'll store a copy of the model on *every* GPU and then combine
            # the results from the gradient updates on the CPU
            # with tf.device("/cpu:0"):
            # initialize the model
            # make the model parallel(if you have more than 2 GPU)
            parallel_model = multi_gpu_model(model, gpus=G)

        parallel_model.compile(loss='binary_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
        print(parallel_model.summary())

    return parallel_model


# 定义计算模型评估指标
def metrics_result(y_label, y_pred):
    _recall = recall_score(y_label, y_pred, average='weighted')
    _precision = precision_score(y_label, y_pred, average='weighted')
    _f1 = f1_score(y_label, y_pred, average='macro')
    _acc = accuracy_score(y_label, y_pred)
    _cm = confusion_matrix(y_label, y_pred)
    return _recall, _precision, _f1, _acc, _cm


# topK计算：

def topK_acc(y_test, X_test, num_classes):
    topk_dict = {}

    top1 = 0.0
    top2 = 0.0
    top3 = 0.0
    top4 = 0.0
    top5 = 0.0

    class_probs = model.predict(X_test)
    for i, l in enumerate(np.argmax(y_test, axis=1)):
        class_prob = class_probs[i]
        top_5_values = (-class_prob).argsort()[:5]
        top_4_values = (-class_prob).argsort()[:4]
        top_3_values = (-class_prob).argsort()[:3]
        top_2_values = (-class_prob).argsort()[:2]

        if top_5_values[0] == l:
            top1 += 1.0
        if np.isin(np.array([l]), top_2_values):
            top2 += 1.0
        if np.isin(np.array([l]), top_3_values):
            top3 += 1.0
        if np.isin(np.array([l]), top_4_values):
            top4 += 1.0
        if np.isin(np.array([l]), top_5_values):
            top5 += 1.0

    top_1 = top1 / len(y_test)
    top_2 = top2 / len(y_test)
    top_3 = top3 / len(y_test)
    top_4 = top4 / len(y_test)
    top_5 = top5 / len(y_test)

    print("top1 acc", top_1)
    print("top2 acc", top_2)
    print("top3 acc", top_3)
    print("top4 acc", top_4)
    print("to5 acc", top_5)

    topk_dict.setdefault('top_1', top_1)
    topk_dict.setdefault('top_2', top_2)
    topk_dict.setdefault('top_3', top_3)
    topk_dict.setdefault('top_4', top_4)
    topk_dict.setdefault('top_5', top_5)

    return topk_dict


# 具体计算评估指标函数
# 具体计算评估指标函数
def call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test, model,local_img_path):

    topk_acc_res = topK_acc(y_test, X_test, num_classes)
    print('top_k acc is : ', topk_acc_res)
    train_y_pred = np.argmax(np.asarray(model.predict(X_train)), axis=1)
    valid_y_pred = np.argmax(np.asarray(model.predict(X_valid)), axis=1)
    test_y_pred = np.argmax(np.asarray(model.predict(X_test)), axis=1)

    y_train = np.argmax(y_train, axis=1)
    y_valid = np.argmax(y_valid, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # 训练集模型评估结果
    metric_train = metrics_result(y_train, train_y_pred)
    metric_valid = metrics_result(y_valid, valid_y_pred)
    metric_test = metrics_result(y_test, test_y_pred)
    label_name = os.listdir(local_img_path)
    print('lable_name is : ', label_name)

    # 调用metric_result 方法返回的是tuple： 元素顺序为：  _recall, _precision, _f1, _acc, _cm
    call_res = {'model_id': modelId, 'model_userid': model_userid, 'model_version': model_version, 'ams_id': ams_id,
                'calculationType': 'result',
                'train_recall_score': metric_train[0], 'train_precision_score': metric_train[1],
                'train_f1_score': metric_train[2], 'train_acc': metric_train[3], 'train_cm': metric_train[4].tolist(),
                'valid_recall_score': metric_valid[0], 'valid_precision_score': metric_valid[1],
                'valid_f1_score': metric_valid[2], 'valid_acc': metric_valid[3], 'valid_cm': metric_valid[4].tolist(),
                'test_recall_score': metric_test[0], 'test_precision_score': metric_test[1],
                'test_f1_score': metric_test[2], 'test_acc': metric_test[3], 'test_cm': metric_test[4].tolist(),
                'label_name': label_name,
                'tok_acc_res': topk_acc_res
                }
    return  call_res


if __name__ == '__main__':

    # 参数动态传递
    arguments = parse_arguments(sys.argv[1:])
    print(type(arguments))
    print('dict content is :', arguments.items())
    img_src = arguments['img_src']
    width = arguments['width']
    height = arguments['height']
    test_percent = arguments['test_percent']
    validata_percent = arguments['validata_percent']
    channel_num = arguments['channel_num']
    # num_classes = arguments['num_classes']
    batch_size = arguments['batch_size']
    lr = arguments['lr']
    epochs = arguments['epochs']
    save_dir = arguments['save_dir']
    model_name = arguments['model_name']
    modelId = arguments["model_id"]
    user_optimizer = arguments['user_Optimizer']
    # pretrain_model_name = arguments['pretrain_model'][0]
    model_version = arguments['model_version']
    model_userid = arguments['model_userid']
    ams_id = arguments['ams_id']
    jsonrpcMlClientPoint = arguments['jsonrpcMlClientPoint']
    print('ip address is ', jsonrpcMlClientPoint)
    print('数据路径 :', img_src)

    # 数据集预处理
    X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes,local_img_path  = preprocessing_img(img_src, width, height,
                                                                                        validata_percent,
                                                                                        test_percent, HDFS_data=True)
    # 模型构建
    model = build_model(width, height, channel_num, num_classes, user_optimizer)

    http_client = Server(jsonrpcMlClientPoint)
    # 训练可视化，返回val_acc, val_loss, train_acc, train_loss
    callbacks = TrainingMonitor(http_client=http_client, model_id=modelId, model_userid=model_userid,
                                model_version=model_version, ams_id=ams_id)

    # checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    namepath = "trained_best_weights.h5"
    filepath = os.path.join(save_dir, namepath)
    print('current file path is : ', filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='max', period=1)

    # Fit the model
    if os.path.exists(filepath):
        print('now loadding the  weights file ', filepath)
        model.load_weights(filepath)
        # 若成功加载前面保存的参数，输出下列信息
        print("-----=-=-=-  checkpoint_loaded=-=-=--------------")

    try:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                            callbacks=[checkpoint,
                                       callbacks
                                       ],
                            validation_data=(X_valid, y_valid), verbose=1)
    except Exception as e:
        runtimeInvalidInfo = {'calculationType': 'exception','info':str(format(e))}
        response = http_client.modelTrain(str(runtimeInvalidInfo))
        print("un expected error").format(e)

    # 指标返回
    call_res = call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test, model,local_img_path)
    # 回调，向服务端发送评估指标


    try:
        response = http_client.modelTrain(str(call_res))
    except Exception as e:
        print("un expected error").format(e)

    # http_client.call("sayHelloWorld",call_res)

    model.save(os.path.join(save_dir, model_name) + '.h5', overwrite=True)
    print('\r\nmodel has been saved in  ', os.path.join(save_dir, model_name) + '.h5')