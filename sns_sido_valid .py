# -*- coding: utf-8 -*-
import os
import sys
abs_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(abs_path)
sys.path.append(f"{abs_path[:abs_path.rfind('/')]}/common")

import pandas as pd
from datetime import date
import traceback
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta

from modules import data_loader
from modules import loggers
from modules import request_info
#from modules import dbconn
from dbconn import mysql_conn

def code_sns_data_merge(sns_data,code_data,logger):
    try:
        code_real_df = pd.merge(sns_data,code_data,how='inner',on=['sido_nm'])
        del code_real_df['sido_nm']
          
        
        code_real_df.rename(columns={"new_sgg_cd":"SGG_CD","base_ymd":"ESTI_TRGT_YMD"},inplace=True)
        code_real_df['ESTI_TRGT_YMD'] = pd.to_datetime(code_real_df['ESTI_TRGT_YMD'],format="%Y-%m-%d")
        code_real_df['ESTI_TRGT_YMD'] = code_real_df['ESTI_TRGT_YMD'].apply(lambda x : x.date())
        
        logger.info("code and sns merge SUCCESS")
        
    except:
        logger.error("시도(시군구) 코드와 SNS데이터 결합 실패")
        error_message = traceback.format_exc()
        logger.error(error_message)
        sys.exit()
        
    return code_real_df


def valid_data(pred_data,code_real_df,logger):
    try:
        temp_df = pd.merge(pred_data,code_real_df,how='left',on=['ESTI_TRGT_YMD','SGG_CD'])
        
        ## 실제 데이터 값 부재로 계산되지 못해 NAN이 생긴 데이터들의 첫번째 날짜가
        nan_data_start_day = temp_df[temp_df.isna().any(axis=1)]['ESTI_TRGT_YMD'].tolist()[0]
        
        ## 실제 데이터가 없어 계산 못하는 컬럼 제외 (날짜기준으로)
        temp_df = temp_df[temp_df['ESTI_TRGT_YMD']<nan_data_start_day]
        temp_df.reset_index(drop=True, inplace=True)
        
        ## 검즘용 데이터셋 만들기
        valid_data =temp_df.copy()
        valid_data['ERR_VAL'] = valid_data['place_cnt_sum'] - valid_data['ESTI_VAL']
        valid_data.rename(columns= {'ESTI_TRGT_YMD':'CMPR_TRGT_DT',"place_cnt_sum":"REAL_VAL"}, inplace=True)
        valid_data = valid_data[['MDL_ID','CMPR_TRGT_DT','SIDO_CD','SIDO_NM','SGG_CD','SGG_NM','REAL_VAL','ESTI_VAL','ERR_VAL','BASE_YM']]
        valid_data['LDNG_DT'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        valid_data['CMPR_TRGT_DT'] = valid_data['CMPR_TRGT_DT'].apply(lambda x : pd.to_datetime(x))
        
        logger.info("(code/sns) and pred_data merge SUCCESS")
        logger.info(f"VALID DATE : {valid_data['CMPR_TRGT_DT'].sort_values().tolist()[0]} ~ {valid_data['CMPR_TRGT_DT'].sort_values(ascending=False).tolist()[0]}")
    
    except:
        logger.error("코드 및 SNS데이터와 예측 값 테이블 결합 실패")
        error_message = traceback.format_exc()
        logger.error(error_message)
        sys.exit()
        
    return valid_data
        
        

             
if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    logger = loggers.make_logger(path,"valid_sido")
    mysql_schema_nm = 'kto_datalab'
    mysql_host = '10.1.113.83'
    mysql_port = 23306
    mysql_user = 'kto'
    mysql_password = 'Kto2020!'
    mysql_db = 'kto_datalab'
    mysql = mysql_conn(logger,mysql_host,mysql_port,mysql_user,mysql_password,mysql_db)
    
    ### 사용여부 모델 확인 ###
    try:
        model_id, model_path = request_info.using_model_check(logger,'sido',mysql_schema_nm,mysql)
        logger.info(f"model_id : {model_id}")
        logger.info(f"model_path : {model_path}")
    except:
        logger.error("사용가능한 모델 없음")
        error_message - traceback.format_exc()
        logger.error(error_message)
        sys.exit()
        
    
    ## 검증 테이블에 적재된 검증 마지막 날짜 확인
    valid_last_date_df = mysql.read_by_sql(f"SELECT DISTINCT(CMPR_TRGT_DT) FROM {mysql_schema_nm}.TFM_SNS_VRFC_RST_TBL WHERE MDL_ID LIKE 'SNS_SGG%' ORDER BY CMPR_TRGT_DT DESC")
    if len(valid_last_date_df) == 0:
        pred_data = mysql.read_by_sql(f"SELECT * FROM {mysql_schema_nm}.TFM_SNS_ESTI_RST_TBL WHERE MDL_ID = '{model_id}' ")
    else:
        valid_last_date = valid_last_date_df.iloc[0][0].strftime("%Y-%m-%d")
        pred_data = mysql.read_by_sql(f"SELECT * FROM {mysql_schema_nm}.TFM_SNS_ESTI_RST_TBL WHERE ESTI_TRGT_YMD > '{valid_last_date}' AND MDL_ID = '{model_id}'")
    ## 검증 마지막날짜의 다음 날짜부터 예측 테이블에서 가져오기
    #pred_data = mysql.read_by_sql(f"SELECT * FROM {mysql_schema_nm}.TFM_SNS_ESTI_RST_TBL WHERE ESTI_TRGT_YMD > '{valid_last_date}'")
    #pred_data = pred_data[pred_data["MDL_ID"].str.contains('SGG')]

    ## 검증이 필요한 시작 날짜
    require_valid_start_date = pred_data['ESTI_TRGT_YMD'].sort_values().tolist()[0]
    require_valid_start_date = require_valid_start_date.strftime("%Y%m%d")
    
    ## 검증이 필요한 마지막 날짜
    require_valid_end_date = pred_data['ESTI_TRGT_YMD'].sort_values(ascending = False).tolist()[0]
    require_valid_end_date = require_valid_end_date.strftime("%Y%m%d")
    
    
    ### 검증이 필요한 날짜에 맞춰서 실제 데이터 불러오기 ###
    sns_data = data_loader.read_query(require_valid_start_date,require_valid_end_date,['sido_nm','base_ymd'],logger)
    if len(sns_data) == 0:
        logger.info(f"NO DATA : {require_valid_start_date} ~ {require_valid_end_date} is not in data")
        sys.exit()
    else:
        ### 시군구 코드와 실제 sns데이터 결합 ###
        code_data = data_loader.sgg_code_query(logger)
        code_real_df = code_sns_data_merge(sns_data,code_data,logger)
        
    ### 시군구코드와 sns실제 값 데이터를 예측 값 테이블과 결합 ###
    valid_data = valid_data(pred_data,code_real_df,logger)
    
    ### DB INSERT ###
    try:
        mysql.save_by_dataframe(schema_nm=mysql_schema_nm,table_nm='TFM_SNS_VRFC_RST_TBL',df=valid_data)
    except:
        logger.info("예측결과검증 DB INSERT 실패")
        error_message = traceback.format_exc()
        logger.error(error_message)
        sys.exit()
    del mysql
    
    



