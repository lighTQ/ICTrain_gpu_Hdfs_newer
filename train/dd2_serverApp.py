# -*- coding:utf-8 -*-

import os
import subprocess

from jsonrpcserver import serve


# from  jsonrpc import Server
@staticmethod
# @method
def modelTrain(dict_parameters):
    """Test method"""
    print(type(dict_parameters))
    # dict_parameters = dict(dict_parameters)

    network_name = dict_parameters.get('network_name').lower()
    gpus = dict_parameters.get('gpus')
    img_src = dict_parameters.get('img_src')
    width = dict_parameters.get("width")
    height = dict_parameters.get("height")
    test_percent = dict_parameters.get("test_percent")
    validata_percent = dict_parameters.get("validata_percent")
    channel_num = dict_parameters.get("channel_num")
    #    num_classes =dict_parameters.get("num_classes")
    batch_size = dict_parameters.get("batch_size")
    lr = dict_parameters.get("lr")
    epochs = dict_parameters.get("epochs")
    save_dir = dict_parameters.get("save_dir")
    model_name = dict_parameters.get("model_name")
    model_id = dict_parameters.get('model_id')
    model_userid = dict_parameters.get('model_userid')
    model_version = dict_parameters.get('model_version')
    user_optimizer = dict_parameters.get('user_Optimizer')
    ams_id = dict_parameters.get('ams_id')
    jsonrpcMlClientPoint = dict_parameters.get('jsonrpcMlClientPoint')

    print(gpus, img_src, width, height, test_percent, validata_percent, channel_num, batch_size, lr, epochs, save_dir,
          model_name, model_id, model_userid, model_version, user_optimizer, ams_id, jsonrpcMlClientPoint)
    print('wtf.,................')
    print('start execute a.py')

    # 待执行脚本路径

    script_path = os.path.join(os.getcwd(), 'nets', network_name + '.py')
    try:
        result = subprocess.call(['python ', script_path,
                                  '-gpus=%s' % (gpus),
                                  '-img_src=%s' % (img_src),
                                  '-width=%d' % (width),
                                  '-height=%d' % (height),
                                  '-test_percent=%f' % (test_percent),
                                  '-validata_percent=% f' % (validata_percent),
                                  '-channel_num=%d' % (channel_num),
                                  '-batch_size=%d' % (batch_size),
                                  '-lr=%f' % (lr),
                                  '-epochs=%d' % (epochs),
                                  '-save_dir=%s' % (save_dir),
                                  '-model_name=%s' % (model_name),
                                  '-model_id=%s' % (model_id),
                                  '-model_userid=%s' % (model_userid),
                                  '-model_version=%s' % (model_version),
                                  '-user_Optimizer=%s' % (user_optimizer),
                                  '-ams_id=%s' % (ams_id),
                                  '-jsonrpcMlClientPoint=%s' % (jsonrpcMlClientPoint)])

    except Exception as e:
        print("Unexpected Error: {}".format(e))

    print('current server ip address', jsonrpcMlClientPoint)
    print("starting executing the python scripts.......")
    # return a + b
    Codes_path = '/root/Codes/uptest/made_modelApp.py'


if __name__ == '__main__':
    print("serve is running!!!")
    serve("192.168.10.101", 10073)
    print("serve is stop!")
