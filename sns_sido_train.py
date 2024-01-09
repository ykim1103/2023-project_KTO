# -*- coding: utf-8 -*-
import os
import sys
abs_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(abs_path)
sys.path.append(f"{abs_path[:abs_path.rfind('/')]}/common")

import time
import traceback
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from datetime import datetime,timedelta

from modules import data_loader
from modules import loggers
from modules import lstm
from modules import retrain_test
from modules import request_info
#from modules import dbconn
from dbconn import mysql_conn

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(3011)


def scaler(data,path,model_id,logger):
    try:
        minmaxscaler = MinMaxScaler(feature_range=(0,1))
        data['scaled_sum'] = minmaxscaler.fit_transform(np.array(data['place_cnt_sum']).reshape(-1,1))
        if os.path.isdir(path+"/output/sido/"+model_id):
            pass
        else:
            os.makedirs(path+"/output/sido/"+model_id)
        
        dump(minmaxscaler,open(path+"/output/sido/"+model_id+"/sns_data_scaler.pkl","wb"))
        logger.info(" scaling SUCCESS ")
        
    except:
        error_message = traceback.format_exc()
        logger.info(error_message)
    
    return data    
   
   
def print_best_callback(study,trial):
    global logger
    logger.info(f"Best trial_number : {study.best_trial.number}, Best value : {study.best_trial.value}, Best params : {study.best_trial.params}")
    
    
def objectives(trial):
    global logger 
    
    logger.info("************************************************************************************")
    
    ### 파라미터 설정 ###
    hidden = trial.suggest_categorical('hidden_size',[64,128,256,512])
    lr = trial.suggest_float('lr',low = 1e-5, high = 1e-3, step = 1e-5)
    num_layer = trial.suggest_int('num_layer',low = 3, high = 30, step = 1, log=False)
    batch_size = trial.suggest_int('batch_size',low = 6, high = 12, step = 1, lof=False)
    
    global param_history
    if trial.params in param_history:
        study.stop()
        logger.info(" ★★ duplicated trial param_info ★★ ")
        logger.info(f"trial_num : {trial.number}, trial_param : {trial.params}")
        
    model = lstm.LSTM(input_size = input_size, hidden_size = hidden, num_layers = num_layer, device = device).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    best_loss = 1e+6    ## last_loss와 비교할 베스트로스 세팅. 일부러 큰 값으로 시작
    no_update_cnt = 0   ## best_loss가 업데이트 되지 않은 횟수
    
    logger.info{f"number of trial : {trial.number}, trial_param : {trial.params}"}
    trial_start_time = time.time()
    
    global x_train
    global y_train
    global x_valid
    global y_valid
    
    train_set = data_loader.train_data_loader(x_train,y_train,2**batch_size)
    valid_set = data.loader.valid_data_loader(x_valid,y_valid,2**batch_size)
    
    for epoc in range(num_epoch):
        batch_loss = 0.0
        for xx,yy in train_set:
            xx = xx.to(device)
            yy = yy.to(device)
            y_pred = model(xx)
            
            loss = nn.MSELoss()(y_pred,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            
        valid_loss = 0.0
        for v_xx,v_yy in valid_set:
            v_xx = v_xx.to(device)
            v_yy = v_yy.to(device)
            v_y_pred = model(v_xx)
            val_loss = nn.MSELoss()(v_y_pred,v_yy)
            valid_loss += val_loss.item()
            
        if epoc % 100 == 0:
            logger.info(f"[epoch: {epoc}] train_loss : {batch_loss} valid_loss : {valid_loss}")
            
        last_loss = valid_loss
        
        if (best_loss >= last_loss) or (best_loss == 1e+6):
            no_update_cnt = 0
            best_loss = last_loss
            
        else:
            no_update_cnt += 1
            
        if no_update_cnt == 5:
            break
    
    trial_end_time = time.time()
    
    logger.info("총 학습 소요시간 : {}초".format(trial_end_time - trial_start_time)) 
    logger.info("총 epoch : {}".format(epoc))
    param_history.append(trial.params)
    
    return best_loss

        
        
if __name__ == "__main__":
    try:
        path = os.path.dirname(os.path.abspath(__file__))
        logger = loggers.make_logger(path,"train_sido")
        parser = argparse.ArgumentParser()
        parser.add_argument("-trial",type=int,default=1000)
        args = parser.parse_args()
        n_trial = args.trial
        
        ### 경로/디바이스세팅/ mysql 업데이트 ###
        stat = 999
        
        mysql_schema_nm = "kto_datalab"
        mysql_host = "10.1.113.83"
        mysql_port = 23306
        mysql_user = "kto"
        mysql_password = "Kto2020!"
        mysql_db = "kto_datalab"
        mysql = mysql_conn(logger, mysql_host, mysql_port, mysql_user, mysql_password, mysql_db)
        
        #os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        #os.environ["CUDA_VISIBLE_DEVICES"] = 1
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logger.info("=======================================")
        logger.info(f"device setting success : {device}")
        logger.info(f"path setting success : {path}")
    
        ### 학습 요청여부 및 날짜 확인 ###
        request_yn,model_id,use_data_start, use_data_end, test_data_start = request_info.request_check_sido(logger, mysql, mysql_schema_nm)
        if request_yn != 'train_start':
            raise("Nothing to train")
        else:
            logger.info(f"~ {model_id} TRAIN_SGG START ~")
            logger.info(f"use_data_start : {use_data_start}")
            logger.info(f"use_data_end : {use_data_end}")
            logger.info(f"test_data_start : {test_data_start}")
        
        ## x_test_start_day : 테스트 시작 날짜를 예측하기 위한 50일치의 x 시작 날짜
        x_test_start_day = datetime.strptime(test_data_start,"%Y%m%d") - timedelta(days=50)
        x_test_start_day = datetime.strftime(x_test_start_day,"%Y%m%d")
        
        ## x_train_last_set_first_day : x_train의 50일치 묶음 마지막 세트에서 첫번째 날짜를
        x_train_last_set_first_day = datetime.strptime(test_data_start,"%Y%m%d") - timedelta(days=51)
        x_train_last_set_first_day = datetime.strftime(x_train_last_set_first_day,"%Y%m%d")
        
        
        ### 데이터 불러오기 ###
        stat = 3
        ## 학습 요청 테이블 업데이트
        mysql.run_sql([f''' UPDATE {mysql_schema_nm}.TFM_TRAIN_DMND_TBL SET TRAIN_STAT=1, TRAIN_BGNG_DT = NOW(), MDFCN_DT = NOW(), MDFP_NM = 'sns_user' WHERE MDL_ID = "{model_id}" '''],True )
        data_sido = data_loader.read_query(use_data_start,use_data_end,['sido_nm','base_ymd'],logger)
        
        ### 스케일링 적용 ###
        stat = 4
        scaled_data = scaler(data_sido,path,model_id,logger)
        
        ### 데이터셋 가공 ###
        stat = 5
        total_test, x_train_origin,y_train_origin, x_train, y_train, x_valid, y_valid, x_test_origin, y_test_origin = data_loader.making_dataset_sido(scaled_data, use_data_start, use_data_end, test_data_start, x_test_start_day, x_train_last_set_first_day, logger)
        
        ### 텐서로 변환 ###
        stat = 6
        x_train_origin, y_train_origin, x_train, y_train, x_valid, y_valid, x_test_origin, y_test_origin = data_loader.data_to_tensor(x_train_origin, y_train_origin, x_train, y_train, x_valid, y_valid, x_test_origin, y_test_origin, logger)
        
        ### optuna 학습시작 ###
        stat = 7
        input_size  = x_train.size(2)
        num_epoch = 1000
        param_history = []
        
        logger.info("★☆ ALL OPTUNA START ★☆")
        
        optuna_start_time = time.time()
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        sampler = TPESampler(seed=3011)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objectives, n_trials = n_trial, callbacks = [print_best_callback])
        
        optuna_end_time = time.time()
        
        optuna_time = round((optuna_end_time-optuna_start_time)/3600,2)
        logger.info("★☆ ALL OPTUNA END ★☆")
        logger.info("#############################################################")
        logger.info("★☆ ALL OPTUNA RESULT ★☆")
        logger.info("총 optuna 소요시간 : {}시간".format(optuna_time))
        logger.info(f"Best trial_number : {study.best_trial.number}")
        logger.info(f"Best trial_parameters : {study.best_trial.params}")
        logger.info("#############################################################")
        
        
        ### 재학습 시작 ###
        stat = 8
        best_parameters = study.best_trial.params
        retrain_test.retrain(x_train_origin, y_train_origin, input_size, best_parameters, device, num_epoch, path, model_id,'sido', logger)

    
        ### 테스트 시작 ### 
        stat = 9
        best_parameters = study.best_trial.params
        result,pred_value = retrain_test.test(x_test_origin,y_test_origin,path,model_id,device,'sido',logger)
        
        
        ### 학습요청 테이블 업데이트 ###
        stat = 10
        mysql.run_sql([f''' UPDATE {mysql_schema_nm}.TFM_TRAIN_DMND_TBL SET TRAIN_STAT = 9, TRAIN_END_DT = NOW(), MDFCN_DT = NOW(), MDFP_NM = 'sns_user' WHERE MDL_ID = "{model_id}" '''],True)
        
    except Exception as e:
        if stat < 999:
            stat_final = stat * -1
            error_message = traceback.format_exc()
            logger.error(f"SNS_TRAIN_SGG FAIL_error_num : {stat_final}")
            logger.error(error_message)
            error_message = error_message.replace('"',"")
            
            myquery = f''' UPDATE {mysql_schema_nm}.TFM_TRAIN_DMND_TBL SET TRAIN_STAT = {stat_final}, TRAIN_END_DT = NOW(), MDFCN_DT=NOW(),MDFP_NM='sns_user', ERR_MSG = "{error_message}" WHERE MDL_ID="{model_id}" '''
            mysql.run_sql([myquery],True)
        else:
            logger.error(e)
        exit()
        
    ### 최종결과 log에 저장 ###
    logger.info("~~~ FINAL_RESULT ~~~")
    logger.info(f"MAPE : {result}")
    logger.info(f"BEST Model_path : {path}/output/sido"+f"/{model_id}")
    logger.info("ALL TRAIN PROCESS DONE!")
    logger.info(f"use_date_info : {use_data_start,use_data_end,test_data_start}")
    
    
    ### 학습결과 데이터프레임 생성 ###
    try:
    
        stat = 11
        best_params = ''
        for k,v in best_parameters.item():
            best_params += k
            best_params += ':'
            best_params += str(v)
            best_params += '\n'
            
        train_result_df =pd.DataFrame({'MDL_ID': [model_id],
                        'USE_YN':['N'],
                        'MDL_ROUT':[path+"/output/sido/"+model_id],
                        'MAPE_VAL': round(result * 100,4),
                        'BEST_PRM': best_params,
                        'CRT_DT':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'CRTP_NM':'sns_user',
                        'MDFCN_DT':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'MDFP_NM':'sns_user'})

        mysql.save_by_dataframe(mysql_schema_nm,'TFM_TRAIN_RST_TBL',train_result_df)
        
        logger.info("TFM_TRAIN_RST_TBL UPDATE")
        
        ### 학습결과 테스트 데이터프레임 생성 ###
        stat = 12
        total_test['y_red'] = pred_value
        
        logger.info("CODE LOADING START")
        code_data = data_loader.sido_code_query(logger)
        logger.info("CODE LOADING END")
        
        total_test2 = pd.merge(total_test, code_data, how='inner',on = ['sido_nm'])
        total_test2['MDL_ID'] = model_id
        
        del total_test2[sido_nm]
        
        total_test2.rename(columns={'new_sido_cd':'SIDO_CD','new_sido_nm':'SIDO_NM','ymd':'TEST_TRGT_YMD','y_val':'TEST_REAL_VAL','y_pred':'TEST_ESTI_VAL'}, inplace=True)    
        total_test2['LDNG_DT'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_test2['MDFCN_DT'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_test2['SGG_CD'] = total_test2['SIDO_CD']+"000"
        total_test2['SGG_NM'] = total_test2['SIDO_NM']
        total_test2['TAR_CD'] = ''
        total_test2['TAR_NM'] = ''
       
        
        total_test3 = total_test2[['MDL_ID','SIDO_CD','SIDO_NM','SGG_CD','SGG_NM','TAR_CD','TAR_NM','TEST_TRGT_YMD','TEST_REAL_VAL','TEST_ESTI_VAL','LDNG_DT','MDFCN_DT']]
        total_test3.reset_index(inplace=True,drop=True)
        
        mysql.save_by_dataframe(mysql_schema_nm,'TFM_TRAIN_TEST_RST_TBL', final_train_test3)
        
        logger.info("TFM_TRAIN_TEST_RST_TBL UPDATE")
       
    except:
        stat_final = stat * -1
        error_message = traceback.format_exc()
        logger.error(f"SNS_TRAIN_SIDO FAIL_error_num : {stat_final}")
        logger.error(error_message)
        error_message = error_message.replace('"',"")
        
        myquery = f''' UPDATE {mysql_schema_nm}.TFM_TRAIN_DMND_TBL SET TRAIN_STAT = {stat_final}, TRAIN_END_DT = NOW(), MDFCN_DT=NOW(),MDFP_NM='sns_user', ERR_MSG = "{error_message}" WHERE MDL_ID="{model_id}" '''
        mysql.run_sql([myquery],True)
        sys.exit()