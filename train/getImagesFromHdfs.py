# coding:utf-8

"""
    该模块需要合并到数据预处理部分
"""
import os
import cv2
import subprocess
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def preprocessing_img(img_src, width, height, categorical_num, validate_percent, test_percent):
    """

    :param img_src: 图片文件夹位置
    :param width:   resize width
    :param height:  resize height
    :param categorical_num:   分类数目
    :param validate_percent:  验证集百分比
    :param test_percent:      测试集百分比
    :return:                  处理后的数据
    """
    X, y = [], []

    # 将数据从HDFS上爬取下来
    prefix = "/root/Data"
    model_userid="u0934234525601"
    downloaded_path = os.path.join(prefix,model_userid)
    local_img_path=os.path.join(downloaded_path,img_src.split("/")[-1])
    if not os.path.exists(downloaded_path):
        os.makedirs(downloaded_path)
    try:
        ret = subprocess.run(["hadoop", "fs", "-get", img_src, downloaded_path])
        if ret.returncode==0:
            print('download successfully, data saved in local path : %s' %local_img_path)
    except Exception as e:
        print("un expected error").format(e)


    for i, folder in enumerate(os.listdir(local_img_path)):
        print(i, folder)
        for index, filename in enumerate(os.listdir(os.path.join(local_img_path, folder))):
            # imgFile = os.path.join(img_src, (folder+"/"+filename))
            imgFile = os.path.join(local_img_path, folder, filename)
            print(index, imgFile)
            image = cv2.imread(imgFile)
            if isinstance(image, np.ndarray):
                pass
            else:
                continue
            img = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

            # img = cv2.resize(cv2.imread(imgFile), dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            X.append(img)
            y.append(i)
    X = np.array(X).astype('float32') / 255
    y = np_utils.to_categorical(y, categorical_num)
    # y = np_utils.to_categorical(y).reshape(-1, categorical_num)
    print(X.shape)
    print(y.shape)

if __name__ =='__main__':

    preprocessing_img("/root/Datasets/101Category", 128, 128, 3, 0.2, 0.1)
