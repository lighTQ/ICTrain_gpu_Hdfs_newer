import argparse
import sys
import os
import subprocess

def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-gpus', type=str, help='gpu_list')
    ap.add_argument('-network', type=str, help='network name ')
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


if __name__ == '__main__':

    arguments = parse_arguments(sys.argv[1:])
    network_name = arguments['network'].lower()
    gpus = arguments['gpus']
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

    print(gpus, img_src, width, height, test_percent, validata_percent, channel_num, batch_size, lr, epochs, save_dir,
          model_name, "model_id", model_userid, model_version, user_optimizer, ams_id, jsonrpcMlClientPoint)
    print('wtf.,................')
    print('start execute a.py')
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

    script_path = os.path.join(os.getcwd(), 'nets', network_name + '.py')
    print('/root/anaconda3/envs/cv/bin/python', script_path,
          '-gpus %s -img_src %s -width %d -height %d   -test_percent %f  -validata_percent  %f  -channel_num %d     -batch_size %d -lr %f -epochs %d -save_dir %s  -model_name %s -model_id %s -model_userid %s -model_version %s -user_Optimizer %s  -ams_id  %s -jsonrpcMlClientPoint %s'
          % (gpus, img_src, width, height, test_percent, validata_percent, channel_num, batch_size,
             lr, epochs, save_dir, model_name, "model_id", model_userid, model_version,
             user_optimizer, ams_id, jsonrpcMlClientPoint))
    # result = os.system('''python /root/Codes/newServer_kerasTrain/made_modelApp.py  -img_src %s -width %d -height %d   -test_percent %f  -validata_percent  %f  -channel_num %d     -batch_size %d -lr %f -epochs %d -save_dir %s  -model_name %s -model_id %s -model_userid %s -model_version %s -user_Optimizer %s  -ams_id  %s -jsonrpcMlClientPoint %s'''%(img_src,width,height,test_percent, validata_percent,channel_num,batch_size,lr,epochs,save_dir,model_name, model_id, model_userid, model_version,user_optimizer,ams_id,jsonrpcMlClientPoint))
    #    try:
    #        result = subprocess.call(['/root/anaconda3/envs/cv/bin/python', script_path,
    #                                  '-gpus %s -img_src %s -width %d -height %d   -test_percent %f  -validata_percent  %f  -channel_num %d     -batch_size %d -lr %f -epochs %d -save_dir %s  -model_name %s -model_id %s -model_userid %s -model_version %s -user_Optimizer %s  -ams_id  %s -jsonrpcMlClientPoint %s' % (
    #                                  gpus, img_src, width, height, test_percent, validata_percent, channel_num, batch_size,
    #                                  lr, epochs, save_dir, model_name, "model_id", model_userid, model_version,
    #                                  user_optimizer, ams_id, jsonrpcMlClientPoint)])
    #    except Exception as e:
    #        print("Unexpected Error: {}".format(e))
    #

    try:
        result = subprocess.call(['/root/anaconda3/envs/cv/bin/python', script_path,
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
                                  '-model_id=%s' % (modelId),
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
    # Codes_path = '/root/Codes/online_train.py'
    Codes_path = '/root/Codes/uptest/made_modelApp.py'
