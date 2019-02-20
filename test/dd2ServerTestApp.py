# -*- coding:utf-8 -*-
import os
import subprocess
from  jsonrpcserver import  method, serve
from time import sleep
import sys, os
import tensorflow as tf
from  sklearn.model_selection import train_test_split
import numpy as np
import cv2
from keras.utils import np_utils
from  keras.models import load_model
from  keras.preprocessing.image import load_img
import argparse
import MainApp

@method
def add( dict_parameters):
    """Test method"""
    dict_parameters = dict(dict_parameters)
    print("------------ wtf!!!!!---------------")
    print(type(dict_parameters), dict_parameters)
#        img_src = dict_parameters.get('img_src')
#        width = dict_parameters.get("width")
#        height = dict_parameters.get("height")
#        testDataset = dict_parameters.get("testDataset")
#        model_path = dict_parameters.get("model_path")
#        label_path = dict_parameters.get("label_path")
#        print(img_src, width, height, model_path, label_path, testDataset)
#
    try:
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

serve('192.168.10.101', 10074)
print("URL: http://192.168.10.101:10074")
