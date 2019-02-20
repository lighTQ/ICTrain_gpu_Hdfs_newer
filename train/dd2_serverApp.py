# -*- coding:utf-8 -*-

from jsonrpcserver import  method, serve
#from  jsonrpc import Server

import os
import  subprocess
from time import sleep
@staticmethod

#@method
def modelTrain(dict_parameters):
    """Test method"""
    print(type(dict_parameters))
        #dict_parameters = dict(dict_parameters)

    network_name = dict_parameters.get('network_name').lower()
    img_src =dict_parameters.get('img_src')
    width = dict_parameters.get("width")
    height = dict_parameters.get("height")
    test_percent =  dict_parameters.get("test_percent")
    validata_percent = dict_parameters.get("validata_percent")
    channel_num =dict_parameters.get("channel_num")
#    num_classes =dict_parameters.get("num_classes")
    batch_size =dict_parameters.get("batch_size")
    lr =dict_parameters.get("lr")
    epochs = dict_parameters.get("epochs")
    save_dir = dict_parameters.get("save_dir")
    model_name = dict_parameters.get("model_name")
    model_id = dict_parameters.get('model_id')
    model_userid = dict_parameters.get('model_userid')
    model_version = dict_parameters.get('model_version')
    user_optimizer = dict_parameters.get('user_Optimizer')
    ams_id = dict_parameters.get('ams_id')
    jsonrpcMlClientPoint = dict_parameters.get('jsonrpcMlClientPoint')

    print(img_src, width, height, test_percent,validata_percent, channel_num, batch_size, lr, epochs,save_dir,model_name,model_id,model_userid,model_version,user_optimizer,ams_id,jsonrpcMlClientPoint)
    print('wtf.,................')
    print('start execute a.py')
    os.system('python a.py')
#    print(''' recevived paramters...........................
#                    python /root/Codes/newServer_kerasTrain/madel_model.py not /online_train.py  -img_src %s -width %d
#                    -height %d
#                    -test_percent %f
#            -validata_percent %f
#                    -channel_num %d
#                    -batch_size %d
#                    -lr %f
#                    -epochs %d
#                    -save_dir %s
#                    -model_name %s
#            -model_id  %s
#            -model_userid %s
#            -model_version %s
#                -user_Optimizer %s
#            -ams_id  %s
#	    -jsonrpcMlClientPoint  %s
#                    '''
#                  % (img_src,width,height,test_percent, validata_percent,channel_num,batch_size,lr,epochs,save_dir,model_name,model_id,model_userid, model_version,user_optimizer,ams_id,jsonrpcMlClientPoint))
   # 待执行脚本路径
    script_path =os.path.join('/root/Codes/newServer_kerasTrain/',network_name+'.py')
    # result = os.system('''python /root/Codes/newServer_kerasTrain/made_modelApp.py  -img_src %s -width %d -height %d   -test_percent %f  -validata_percent  %f  -channel_num %d     -batch_size %d -lr %f -epochs %d -save_dir %s  -model_name %s -model_id %s -model_userid %s -model_version %s -user_Optimizer %s  -ams_id  %s -jsonrpcMlClientPoint %s'''%(img_src,width,height,test_percent, validata_percent,channel_num,batch_size,lr,epochs,save_dir,model_name, model_id, model_userid, model_version,user_optimizer,ams_id,jsonrpcMlClientPoint))
    try:
        result = subprocess.call(['''python ''',script_path,'''-img_src %s -width %d -height %d   -test_percent %f  -validata_percent  %f  -channel_num %d     -batch_size %d -lr %f -epochs %d -save_dir %s  -model_name %s -model_id %s -model_userid %s -model_version %s -user_Optimizer %s  -ams_id  %s -jsonrpcMlClientPoint %s'''%(img_src,width,height,test_percent, validata_percent,channel_num,batch_size,lr,epochs,save_dir,model_name, model_id, model_userid, model_version,user_optimizer,ams_id,jsonrpcMlClientPoint)])
    except Exception as e:
        print("Unexpected Error: {}".format(e))




    print('current server ip address',jsonrpcMlClientPoint)
    print("starting executing the python scripts.......")
        #return a + b
    #Codes_path = '/root/Codes/online_train.py'
    Codes_path = '/root/Codes/uptest/made_modelApp.py'
    return  Codes_path


# # Threading HTTP-Server
# http_server = pyjsonrpc.ThreadingHttpServer(
#     server_address=('192.168.10.101', 10073),
#    RequestHandlerClass=RequestHandler
# )
# print("Starting HTTP server ...")
# print("URL: http://192.168.10.101:10073")
# http_server.serve_forever()
if __name__ =='__main__':
    print("serve is running!!!")
    serve("192.168.10.101", 10073)
    rint("serve is stop!")
