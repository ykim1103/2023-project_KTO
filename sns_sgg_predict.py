# -*- coding: utf-8 -*-
import os
import sys
abs_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(abs_path)
sys.path.append(f"{abs_path[:abs_path.rfind('/')]}/common")

import argparse
import traceback
import torch
import numpy as np
import pandas as pd
from pickle import load
from torch.autograd import Variable
from datetime import datetime,timedelta

from modules import data_loader
from modules import loggers
from modules import lstm
from modules import request_info
#from modules import dbconn
from dbconn import mysql_conn
#
def predict(data_sgg_clust, model_path, model_id, logger):
    global last_month
    global use_data_end
    global today_date

    total_total_df = pd.DataFrame()
    for clust in data_sgg_clust['clust_no'].unique():
        if clust == 'C00':
            model_nm = '/sns_predict_all.pt'
            scaler_nm = '/sns_data_scaler_all.pkl'
            
        else:
            model_nm = f'/sns_predict_{clust}.pt'
            scaler_nm = f'/sns_data_scaler_{clust}.pkl'
        
        data_df = data_sgg_clust[data_sgg_clust['clust_no'] == clust]
        load_model = torch.load(model_path+model_nm)
        load_scaler = load(open(model_path+scaler_nm,'rb'))
        
        if use_data_end in data_df['base_ymd'].unique():
            predict_dt = pd.date_range(last_month + timedelta(days=1), last_month+timedelta(days=75)).normalize()
            total_df = pd.DataFrame()
            
            for sido in data_df['sido_cd'].unique():
                for sgg in data_df[data_df['sido_cd'] == sido]['sgg_cd'].unique():
                    test_df = data_df[(data_df['sido_cd'] == sido) & (data_df['sgg_cd']==sgg)]
                    x_test_list = test_df['place_cnt_sum'].tolist()
                    pred_list = []
                    
                    for day in range(75):
                        x_test_list = np.array(x_test_list)
                        x_test = load_scaler.transform(x_test_list.reshape(-1,1))
                        x_test = Variable(torch.Tensor(x_test))
                        x_test = torch.reshape(x_test, (x_test.shape[1],1,x_test.shape[0]))  ## (1,1,50)
                        x_test2 = x_test.to(device)
                        y_pred = load_model(x_test2)
                        predict_temp = y_pred.data.detach().cpu().numpy().reshape(len(y_pred),1)
                        predict_temp = load_scaler.inverse_transform(predict_temp)
                        x_test_list = x_test_list.tolist()
                        x_test_list.append(float(predict_temp))
                        pred_list.append(float(predict_temp))
                        x_test_list = x_test_list[1:]
                        
                    temp_df = pd.DataFrame(columns = ["MDL_ID","SIDO_CD","SIDO_NM","SGG_CD","SGG_NM","ESTI_EXEC_DT","ESTI_TRGT_YMD","ESTI_VAL","BASE_YM","LDNG_DT"])
                    temp_df["ESTI_TRGT_YMD"] = predict_dt
                    temp_df["MDL_ID"] = model_id
                    temp_df["ESTI_VAL"] = pred_list
                    temp_df["SIDO_CD"] = sido
                    temp_df["SIDO_NM"] = test_df['sido_nm'].unique()[0]
                    temp_df["SGG_CD"] = sgg
                    temp_df["SGG_NM"] = test_df['sgg_nm'].unique()[0]
                    #temp_df["ESTI_EXEC_DT"] = datetime.today()
                    temp_df["BASE_YM"] = today_date.strftime("%Y%m%d")[:6]
                    #temp_df["LDNG_DT"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    total_df = total_df.append(temp_df)
                    
        else:
            sys.exit()
            
    total_total_df = total_total_df.append(total_df)
    total_total_df["ESTI_EXEC_DT"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_total_df["LDNG_DT"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
    return total_total_df

                 
if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    
    default_date = datetime.today()
    default_date = int(default_date.strftime("%Y%m%d")[:6])
    parser = argparse.ArgumentParser()
    parser.add_argument("-yyyymm",type=int,default=default_date)
    args = parser.paese_args()
    today_date = args.yyyymm
    logger = loggers.make_logger(path,"predict_sgg")
    
    if today_date == default_date:
        os.system('python sns_sgg_valid.py')
    elif today_date > default_date:
        logger.info("NO VALID AND NO PREDICT")
    else:
        logger.info("ONLY PREDICT START (NO VALID) ")
        
    mysql_schema_nm = 'kto_datalab'
    mysql_host = '10.1.113.83'
    mysql_port = 23306
    mysql_user = 'kto'
    mysql_password = 'Kto2020!'
    mysql_db = 'kto_datalab'
    mysql = mysql_conn(logger,mysql_host,mysql_port,mysql_user,mysql_password,mysql_db)
    
    
    ### 사용여부 모델 확인 ###
    try:
        model_id, model_path = request_info.using_model_check(logger,'sgg',mysql_schema_nm,mysql)
        logger.info(f"model_id : {model_id}")
        logger.info(f"model_path : {model_path}")
    except:
        logger.error("★ AVAILABLE MODEL SEARCH FAIL ★")
        error_message = traceback.format_exc()
        logger.error(error_message)
        sys.exit()
        
    ### 날짜 세팅 및 확인 ###
    today_date = pd.to_datetime(str(today_date),format='%Y%m')
    last_month = today_date.replace(day=1) - timedelta(days=1)
    
        
    ## x 테으스의 마지막 날짜에서 49를 빼면 테스트의 첫번째 날짜가 나옴. 이 날짜가 사용 시작일
    use_data_start = last_month - timedelta(days=49)
    use_data_start = use_data_start.strftime("%Y%m%d")
    
    ## 사용 마지막 일
    use_data_end = last_month.strftime("%Y%m%d")
    
    try:
        data_sgg_clust = data_loader.cluster_sns_load(use_data_start,use_data_end,logger)
        data_sgg_clust = data_sgg_clust.sort_values(['base_ymd','sido_cd','sgg_cd']).reset_index(drop=True)
        data_sgg_clust.rename(columns = {'cnt':'place_cnt_sum'}, inplace=True)
        
        logger.info("data loading SUCCESS")
        if use_data_end not in data_sgg_clust['base_ymd'].unique():
            logger.info("last date is not in DataFrame")
            sys.exit()
        else:
            logger.info("CHECK last date is in DataFrame")
    except:
        logger.error("data loading FAIL")
        error_message = traceback.format_exc()
        logger.error(error_message)
        sys.exit()
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        logger.info("predict START")
        result_df = predict(data_sgg_clust,model_path,model_id,logger)
        logger.info("predict END")
        logger.info(f"USING_DATE : {use_data_start} ~ {use_data_end}")
        logger.info(f"Predict DATE : {last_month + timedelta(days=1)} ~ {last_month+timedelta(days=75)}")
        
    except:
        logger.error("predict FAIL")
        error_message = traceback.format_exc()
        logger.error(error_message)
        sys.exit()
        
    
    ### 예측값 ESTI_RST_TBL 저장 ### 
    try:
        mysql.save_by_dataframe(schema_nm = mysql_schema_nm, table_nm = "TFM_SNS_ESTI_RST_TBL", df = result_df)
        logger.info(" TFM_SNS_ESTI_RST_TBL insert SUCCESSS ")
        del mysql
    except:
        logger.error("DB save FAIL")
        error_message = traceback.format_exc()
        logger.error(error_message)
        sys.exit()
    
                 
      
            





    
