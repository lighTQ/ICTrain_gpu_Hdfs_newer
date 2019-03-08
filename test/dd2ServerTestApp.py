# -*- coding:utf-8 -*-

import os
import cv2
import subprocess
from  jsonrpcserver import  method, serve
from time import sleep
import sys, os
import tensorflow as tf
from  sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from  keras.models import load_model
from  keras.preprocessing.image import load_img
import argparse
import MainApp


@method
def modelTest( dict_parameters):
    """Test method"""
    dict_parameters = dict(dict_parameters)
    print("------------ wtf!!!!!---------------")
    print(type(dict_parameters), dict_parameters)

    # network_name = dict_parameters.get('network').lower()

    global back_testResult

    try:
        network_name = dict_parameters.get('network').lower()

        img_src = dict_parameters.get('img_src')
        hdfs_label = dict_parameters.get('hdfslabel')
        width = dict_parameters.get("width")
        height = dict_parameters.get("height")
        testDataset = dict_parameters.get("testDataset")
        model_path = dict_parameters.get("model_path")
        label_path = dict_parameters.get("label_path")

        print(network_name,img_src, hdfs_label,width, height, model_path, label_path, testDataset)

        back_testResult = MainApp.resutlApp(dict_parameters)

    except Exception as e:
        print("Unexpected Error: {}".format(e))

    print(back_testResult)
    return back_testResult

# Threading HTTP-Server
# http_server = pyjsonrpc.ThreadingHttpServer(
#     server_address=('192.168.10.101', 10074),
#     RequestHandlerClass=RequestHandler
# )
# print("Starting HTTP server ...")
# print("URL: http://192.168.10.101:10074")
# http_server.serve_forever()

serve('192.168.144.2', 10073)
print("URL: http://192.168.144.2:10073")
