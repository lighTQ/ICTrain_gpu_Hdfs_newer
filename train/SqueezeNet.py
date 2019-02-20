# -*- coding:utf-8 -*-


import sys
import os
import cv2
import argparse
import numpy as np
from keras.layers import *
from keras.layers import  merge
from  keras.models import *
from keras.optimizers import *
from keras.utils import np_utils
from trainmonitor import TrainingMonitor
from  sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from jsonrpc import  Server
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score



def preprocessing_img(img_src, width, height, validate_percent, test_percent):
    X, y = [], []
    categorical_num = len(set(os.listdir(img_src)))
    for i, folder in enumerate(os.listdir(img_src)):
        print(i, folder)
        for index, filename in enumerate(os.listdir(os.path.join(img_src, folder))):
            # imgFile = os.path.join(img_src, (folder+"/"+filename))
            imgFile = os.path.join(img_src, folder, filename)
            print(index, imgFile)
            image = cv2.imread(imgFile)
            if isinstance(image, np.ndarray):
                pass
            else:
                continue
            img = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            X.append(img)
            y.append(i)
    X = np.array(X).astype('float32') / 255.0
    y_label = np_utils.to_categorical(y, categorical_num)
    # y = np_utils.to_categorical(y).reshape(-1, categorical_num)
    print(X.shape)
    print(y_label.shape)
    print(len(y))

    x_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=test_percent, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validate_percent, shuffle=True)
    print('preprocessing images is OK..')
    return X_train, X_valid, X_test, y_train, y_valid, y_test,y,categorical_num

def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-img_src', type=str, help='train image path')
    ap.add_argument('-width', type=int, help='img_resize_width')
    ap.add_argument('-height', type=int, help='img__resize_height')
    ap.add_argument('-test_percent', type=float, help='test_dataset_percent')
    ap.add_argument('-validata_percent', type=float , help='validata_dataset_percent')
    ap.add_argument('-channel_num', type=int , help='channels_num')
    # ap.add_argument('-num_classes', type=int , help='num_classes')

    ap.add_argument('-batch_size', type=int , help='batch_size')
    ap.add_argument('-lr', type=float, help='learning_rate')
    ap.add_argument('-epochs', type=int, help='epochs')
    # ap.add_argument('data_augmentation', type=bool, , help='data_augmentation')
    ap.add_argument('-save_dir', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model h5  file')
    ap.add_argument('-jsonrpcMlClientPoint', type=str,help='IP address of server endpoint...')
    ap.add_argument('-model_name', type=str,
                    help='Could be either a directory containing the meta_file and ckpt_file or a model file')
    ap.add_argument('-model_id', type=str, help="define the unique model identifier id")
    ap.add_argument('-model_userid', type=str, help="define the model user who are")
    ap.add_argument('-model_version', type=str, help="define the model version is what")
    ap.add_argument('-user_Optimizer', type=str , help="user selective optimizers")
    ap.add_argument('-ams_id', type=str , help="callback the trained model id")

    return vars(ap.parse_args())

# keras.backend.clear_session()


def build_model(width, height, channel_num, num_classes, user_optimizer='adam'):

    inputs = (width, height,channel_num)
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

    softmax = Activation("softmax", name='softmax')(global_avgpool10)
    model = Model(input_img, softmax)
    model.compile(optimizer=UserOptimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 定义计算模型评估指标
def metrics_result(y_label, y_pred):
    _recall = recall_score(y_label, y_pred, average='weighted')
    _precision = precision_score(y_label, y_pred, average='weighted')
    _f1 = f1_score(y_label, y_pred, average='macro')
    _acc = accuracy_score(y_label, y_pred)
    _cm = confusion_matrix(y_label, y_pred)
    return _recall, _precision, _f1, _acc, _cm



# 具体计算评估指标函数
def call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test,model):


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


    # 调用metric_result 方法返回的是tuple： 元素顺序为：  _recall, _precision, _f1, _acc, _cm
    call_res = {'model_id': modelId, 'model_userid': model_userid, 'model_version': model_version, 'ams_id': ams_id,
                'calculationType': 'result',
                'train_recall_score': metric_train[0], 'train_precision_score': metric_train[1],
                'train_f1_score': metric_train[2], 'train_acc': metric_train[3], 'train_cm': metric_train[4].tolist(),
                'valid_recall_score': metric_valid[0], 'valid_precision_score': metric_valid[1],
                'valid_f1_score': metric_valid[2], 'valid_acc': metric_valid[3], 'valid_cm': metric_valid[4].tolist(),
                'test_recall_score': metric_test[0], 'test_precision_score': metric_test[1],
                'test_f1_score': metric_test[2], 'test_acc': metric_test[3], 'test_cm': metric_test[4].tolist()}
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
    batch_size = arguments['batch_size']
    lr = arguments['lr']
    epochs = arguments['epochs']
    save_dir = arguments['save_dir']
    model_name = arguments['model_name']
    modelId = arguments["model_id"]
    user_optimizer = arguments['user_Optimizer']
    model_version = arguments['model_version']
    model_userid = arguments['model_userid']
    ams_id = arguments['ams_id']
    jsonrpcMlClientPoint= arguments['jsonrpcMlClientPoint']
    print('ip address is ',jsonrpcMlClientPoint)


    # 数据集预处理
    X_train, X_valid, X_test, y_train, y_valid, y_test, y, categorical_num= preprocessing_img(img_src, width, height,
                                                                           validata_percent, test_percent)
    # 模型构建
    model = build_model(width, height, channel_num, categorical_num, user_optimizer)
    print(model.summary())

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
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

    # Fit the model
    if os.path.exists(filepath):
        print('now loadding the  weights file ', filepath)
        model.load_weights(filepath)
        # 若成功加载前面保存的参数，输出下列信息
        print("-----=-=-=-  checkpoint_loaded=-=-=--------------")

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=[checkpoint, callbacks],
                        validation_data=(X_valid, y_valid), verbose=1)

    # 指标返回
    call_res = call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test, model)
    # 回调，向服务端发送评估指标
    response = http_client.modelTrain(str(call_res))
    # http_client.call("sayHelloWorld",call_res)

    model.save(os.path.join(save_dir, model_name) + '.h5', overwrite=True)
    print('\r\nmodel has been saved in  ', os.path.join(save_dir, model_name) + '.h5')
