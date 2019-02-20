# -*- coding:utf-8 -*-
from keras.callbacks import BaseLogger
import json
import os
import  logging
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

class TrainingMonitor(BaseLogger):
    def __init__(self, http_client, model_id,model_userid, model_version,ams_id ,startAt=0):
        super(TrainingMonitor, self).__init__()
        # self.figPath = figPath
        # self.jsonPath = jsonPath
        # # 开始模型开始保存的开始epoch
        self.startAt = startAt
        self.http_client=http_client
        self.model_id = model_id
        self.model_userid = model_userid
        self.model_version = model_version
        self.ams_id = ams_id

    def on_train_begin(self, logs={}):
        # 初始化保存文件的目录dict
        self.H = {}
        # 判断是否存在文件和该目录
        # if self.jsonPath is not None:
            # if os.path.exists(self.jsonPath):
                # self.H = json.loads(open(self.jsonPath).read())
                # 开始保存的epoch是否提供
#        if self.startAt > 0:
#            for k in self.H.keys():
#                # 循环保存历史记录，从startAt开始
#                self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # 不断更新logs和loss accuracy等等
        #val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        #val_targ = self.validation_data[1]

        #val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])),axis=1)
        #val_targ = np.argmax(self.validation_data[1],axis=1)

        #val_f1 = f1_score(val_targ, val_predict, average='macro')
        #val_recall = recall_score(val_targ, val_predict, average='weighted')
        #val_precision = precision_score(val_targ, val_predict, average='weighted')

        if self.model_id is not None:
	    #call_res = {'model_id':self.model_id,'model_userid':self.model_userid,'model_version':self.model_version,'epoch':epoch, 'f1_score':val_f1,'recall':val_recall, 'val_precision':val_precision, 'ams_id':self.ams_id ,'calculationType':'process'}
            call_res = {'model_id':self.model_id,'model_userid':self.model_userid,'model_version':self.model_version,'epoch':epoch,  'ams_id':self.ams_id ,'calculationType':'process'}
            for (k, v) in logs.items():
                self.H[k] = v
                print(k, self.H[k])
                call_res.setdefault(k,self.H[k])
            status = self.http_client.modelTrain(str(call_res))

	    # status = self.http_client.call("sayHelloWorld", call_res)
            print(call_res)
            print("-=-=-=--==---=-= received staussis "+status)

