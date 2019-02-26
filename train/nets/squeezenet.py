# -*- coding:utf-8 -*-


import argparse
import os
import subprocess
import sys

import cv2
import keras.backend.tensorflow_backend as KTF
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
            ret = subprocess.run(["hadoop", "fs", "-get", img_src, downloaded_path])
            if ret.returncode == 0:
                print('download successfully, data saved in local path : %s' % local_img_path)
        except Exception as e:
            print("un expected error").format(e)

    else:
        print("使用本地数据源")

        local_img_path = img_src

    categorical_num = len(set(os.listdir(local_img_path)))

    for i, folder in enumerate(os.listdir(local_img_path)):
        # print(i, folder)
        for index, filename in enumerate(os.listdir(os.path.join(local_img_path, folder))):
            # imgFile = os.path.join(img_src, (folder+"/"+filename))
            imgFile = os.path.join(local_img_path, folder, filename)
           # print(index, imgFile)
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
def build_model(width, height, channel_num, num_classes, user_optimizer='adam'):
    inputs = (width, height, channel_num)
    input_img = Input(shape=inputs)

    conv1 = Conv2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_last")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_last")(conv1)
    fire2_squeeze = Conv2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_last")(maxpool1)
    fire2_expand1 = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_last")(fire2_squeeze)
    fire2_expand2 = Conv2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_last")(fire2_squeeze)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Conv2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_last")(merge2)
    fire3_expand1 = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_last")(fire3_squeeze)
    fire3_expand2 = Conv2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_last")(fire3_squeeze)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Conv2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_last")(merge3)
    fire4_expand1 = Conv2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_last")(fire4_squeeze)
    fire4_expand2 = Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_last")(fire4_squeeze)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_last")(merge4)

    fire5_squeeze = Conv2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_last")(maxpool4)
    fire5_expand1 = Conv2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_last")(fire5_squeeze)
    fire5_expand2 = Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_last")(fire5_squeeze)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Conv2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_last")(merge5)
    fire6_expand1 = Conv2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_last")(fire6_squeeze)
    fire6_expand2 = Conv2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_last")(fire6_squeeze)
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Conv2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_last")(merge6)
    fire7_expand1 = Conv2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_last")(fire7_squeeze)
    fire7_expand2 = Conv2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_last")(fire7_squeeze)
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_last")(merge7)
    fire8_expand1 = Conv2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_last")(fire8_squeeze)
    fire8_expand2 = Conv2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_last")(fire8_squeeze)
    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_last")(merge8)
    fire9_squeeze = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format="channels_last")(maxpool8)
    fire9_expand1 = Conv2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format="channels_last")(fire9_squeeze)
    fire9_expand2 = Conv2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format="channels_last")(fire9_squeeze)
    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        num_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_last")(fire9_dropout)
    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)

    all_optimizers = {'sgd': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      'rmsprop': RMSprop(lr=0.001, rho=0.9, epsilon=1e-6),
                      'adagrad': Adagrad(lr=0.01, decay=1e-6),
                      'adadelta': Adadelta(lr=1.0, rho=0.95, epsilon=1e-6),
                      'adam': Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
                      # 'adamax': Adamax,
                      # 'nadam': Nadam,
                      # 'tfoptimizer': TFOptimizer,
                      }

    UserOptimizer = all_optimizers[user_optimizer]
    print("use define optimier is ", UserOptimizer)

    if (int(num_classes > 2)):
        softmax = Activation("softmax", name='softmax')(global_avgpool10)
        model = Model(input_img, softmax)
        if G <= 1 and G > 0:
            print("[INFO] training with 1 GPU...")
            parallel_model = model

        # otherwise, we are compiling using multiple GPUs
        elif G < 0:
            print("[INFO] training with CPU...")
            parallel_model = model
        else:
            print("[INFO] training with {} GPUs...".format(G))
            parallel_model = multi_gpu_model(model, gpus=G)

        parallel_model.compile(loss='categorical_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
        print(parallel_model.summary())
        parallel_model.compile(optimizer=UserOptimizer,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
    else:
        softmax = Activation("sigmoid", name='softmax')(global_avgpool10)
        model = Model(input_img, softmax)
        if G <= 1 and G > 0:
            print("[INFO] training with 1 GPU...")
            parallel_model = model
            # otherwise, we are compiling using multiple GPUs
        elif G < 0:
            print("[INFO] training with CPU...")
            parallel_model = model
        else:
            print("[INFO] training with {} GPUs...".format(G))
            parallel_model = multi_gpu_model(model, gpus=G)

        parallel_model.compile(loss='binary_crossentropy', optimizer=UserOptimizer, metrics=['accuracy'])
        print(parallel_model.summary())

        parallel_model.compile(optimizer=UserOptimizer,
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
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
    print(call_res)
    return  call_res



if __name__ == '__main__':

    # 参数动态传递
    arguments = parse_arguments(sys.argv[1:])
    print(type(arguments))
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

    # 数据集预处理
    X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes ,local_img_path= preprocessing_img(img_src, width, height,
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

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=[checkpoint,
                     callbacks
                         ],
                        validation_data=(X_valid, y_valid), verbose=1)

    # 指标返回
    call_res = call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test, model,local_img_path)
    # 回调，向服务端发送评估指标

    response = http_client.modelTrain(str(call_res))
    # http_client.call("sayHelloWorld",call_res)

    model.save(os.path.join(save_dir, model_name) + '.h5', overwrite=True)
    print('\r\nmodel has been saved in  ', os.path.join(save_dir, model_name) + '.h5')
