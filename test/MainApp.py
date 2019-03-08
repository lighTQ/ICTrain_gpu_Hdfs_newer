# coding:utf-8
# import  pyjsonrpc

import argparse
import os
import subprocess
import sys

import cv2
import keras
import numpy as np
import tensorflow as tf
from hdfs.client import Client
from keras.models import load_model


def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-network', type=str, nargs='+', help='network_name')
    ap.add_argument('-hdfslabel', type=str, nargs='+', help='hdfs label uri')
    ap.add_argument('-width', type=int, nargs='+', help='img_resize_width')
    ap.add_argument('-height', type=int, nargs='+', help='img__resize_height')
    ap.add_argument('-model_path', type=str, nargs='+',
                    help='in order to load the file ,which is the abs path')
    ap.add_argument('-testDataset', type=str, nargs='+', help='test dataset  path ')
    ap.add_argument('-label_path', type=str, nargs='+', help='label name path')
    return vars(ap.parse_args())


def preprocessImageFolder(img_src, width=128, height=128, HDFS_data=True):
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
        print("使用本地数据源")

        local_img_path = img_src

    print("processing image....")
    test_imgFiles = []
    X = []
    if os.path.isdir(local_img_path):
        print(os.path.isdir(local_img_path))
        print("test images file reading now")
        for imgfile in os.listdir(local_img_path):
            img = os.path.join(local_img_path, imgfile)
            if (img.lower().endswith('jpg') or img.lower().endswith('jpeg') or img.lower().endswith('png')):
                test_imgFiles.append(img)  # JAVA调用的path
                # print(img)
                try:
                    image = cv2.imread(img)
                    if isinstance(image, np.ndarray):
                        pass
                    else:
                        continue
                    img = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                    # test_imgFiles.append(img)  #  plt 打印使用的ndarray
                    X.append(img)
                except Exception as e:
                    print("un expected error").format(e)
            else:
                tf.logging.error("file type is not support")
                # sys.exit("program exited!")
        X = np.array(X).astype('float32') / 255.0
        print('X. shape is ', X.shape)
        return X, test_imgFiles

    elif (os.path.isfile(local_img_path)):
        print(os.path.isfile(local_img_path))
        img = local_img_path.lower()
        print(img)
        # if (img.endswith('jpg') or img.endswith('jpeg') or img.endswith('png')):
        print("测试单张图片")
        test_imgFiles.append(local_img_path)
        image = cv2.imread(local_img_path)
        # assert isinstance(imagePath,np.ndarray)
        if isinstance(image, np.ndarray):
            pass
        else:
            os._exit(-1)

        img = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        X = np.array(img).astype('float32') / 255.0
        # test_imgFiles.append(img)
        X = X.reshape((1,) + X.shape)
        print(X.shape)
        return X, test_imgFiles
    # else:
    #     tf.logging.error(" input image file  type is not support")
    #     sys.exit(" program exited!")

    else:
        tf.logging.error(" input must be a directory or a image format File  ")
        sys.exit("program exited !")


def resutlApp(dict_parameters):
    # step1 ： 获取动态参数
    print('hello wolrd')
    #    arguments = parse_arguments(sys.argv[1:])
    #    width = arguments['width'][0]
    #    height = arguments['height'][0]
    #    model_path = arguments['model_path'][0]
    #    label_path = arguments['label_path'][0]
    #    testDataset = arguments['testDataset'][0]

    print("接受参数中..................")
    print(type(dict_parameters), dict_parameters)
    print("\n" * 3)

    network = dict_parameters.get('network')
    hdfs_label_ip = dict_parameters.get('hdfslabel')
    img_src = dict_parameters.get('img_src')
    width = dict_parameters.get("width")
    height = dict_parameters.get("height")
    testDataset = dict_parameters.get("testDataset")
    model_path = dict_parameters.get("model_path")
    label_path = dict_parameters.get("label_path")
    print(network, img_src, width, height, model_path, label_path, testDataset)
    print("model path is ", model_path)

    #
    # step2测试数据预处理

    X, ImgFiles = preprocessImageFolder(testDataset, width, height)

    # step3 : 获取预测的类别标签名称：
    print('hdfs web interface address is :', hdfs_label_ip)
    client = Client(hdfs_label_ip)
    # client = hdfs.Client("http://172.10.236.21:50070")

    types = client.list(label_path)

    # step4 加载模型
    tf.logging.info("模型加载中.....")
    print(" model loading .....", model_path)
    keras.backend.clear_session()
    model = load_model(model_path)
    print(model.summary())
    np.set_printoptions(precision=2, suppress=True)

    # step 5使用模型进行预测
    print("model predicting ......")

    if 'fcn' in network:
        try:
            print(" test in FCN network structure")
            classes = np.argmax(np.squeeze(model.predict(X)), axis=1)
        except Exception as e:
            print("un expected error").format(e)
    else:
        try:
            print(" test in Dense network structure")
            classes = np.argmax(model.predict(X), axis=1)
        except Exception as e:
            print("un expected error").format(e)
    # print('classes: ', classes)

    #  step 6：测试结果返回
    back_testResult = {}
    print('types:  ', types)

    print("predicted result in your kerboard floder is： \n")

    for i, index in enumerate(classes):
        key = str(ImgFiles[i])
        value = str(types[index])
        back_testResult.setdefault(key, value)
    # plt.imshow(ImgFiles[i])
    # plt.title(str(types[index]))
    # plt.show()
    print('call bck type ', type(back_testResult))
    # return 'abc'
    return str(back_testResult)
