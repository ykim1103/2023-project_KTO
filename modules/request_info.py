import torch
import os
import sys
from pickle import load
from modules import dbconn
from modules import data_loader


def request_check_sido(logger,mysql,schema_nm):
    logger.info("                                              ")
    logger.info("★☆★☆★☆★☆★☆ REQUEST SEARCH STARTCH ★☆★☆★☆★☆★☆")
                

    request_data = mysql.read_by_sql(f"SELECT * FROM {schema_nm}.TFM_TRAIN_DMND_TBL")
    request_data = request_data[(request_data['TRAIN_STAT']==0) & (request_data['MDL_ID'].str.contains('SIDO')) & (request_data['MDL_ID'].str.contains('SNS'))]
    
    logger.info(f"TRAIN AVAILABLE REQUEST NUM : {len(request_data)}")
    
    ## 요청 데이터가 1개일 때만 작업 정상수행
    if len(request_data) == 1:
        result = 'train start'
        model_id = request_data['MDL_ID'].to_list()[0]
        use_data_start = request_data['USE_DATA_BGNG_DT'].to_list()[0]
        use_data_end = request_data['USE_DATA_END_DT'].to_list()[0]
        test_data_start = request_data['TEST_DATA_BGNG_DT'].to_list()[0]
        
        logger.info("★☆★ This model will train!! ★☆★ ")
        
    elif len(request_data) > 1:
        result = 'too much train'
        model = 'no model_id'
        use_data_start = 'no_use_data_start'
        use_data_end = 'no_use_data_end'
        test_data_start = 'no_test_data_start'
        
        logger.info("★☆★ TOO MUCH train model ★☆★ ")
    
    else:
        result = 'nothing train'
        model = 'no model_id'
        use_data_start = 'no_use_data_start'
        use_data_end = 'no_use_data_end'
        test_data_start = 'no_test_data_start'
        
        logger.info("★☆★ NOTHING train model ★☆★ ")
        
       
    logger.info("★☆★★☆★★☆★~ Request search END ~★☆★★☆★★☆★★☆★")
    
    return result,model_id,use_data_start,use_data_end,test_data_start
    
    
def request_check_sgg(logger,mysql,schema_nm):
    logger.info("                                              ")
    logger.info("★☆★☆★☆★☆★☆ REQUEST SEARCH STARTCH ★☆★☆★☆★☆★☆")
                
    request_data = mysql.read_by_sql(f"SELECT * FROM {schema_nm}.TFM_TRAIN_DMND_TBL")
    request_data = request_data[(request_data['TRAIN_STAT']==0) & (request_data['MDL_ID'].str.contains('SGG')) & (request_data['MDL_ID'].str.contains('SNS'))]
    
    logger.info(f"TRAIN AVAILABLE REQUEST NUM : {len(request_data)}")
    
    ## 요청 데이터가 1개일 때만 작업 정상수행
    if len(request_data) == 1:
        result = 'train start'
        model_id = request_data['MDL_ID'].to_list()[0]
        use_data_start = request_data['USE_DATA_BGNG_DT'].to_list()[0]
        use_data_end = request_data['USE_DATA_END_DT'].to_list()[0]
        test_data_start = request_data['TEST_DATA_BGNG_DT'].to_list()[0]
        
        logger.info("★☆★ This model will train!! ★☆★ ")
        
    elif len(request_data) > 1:
        result = 'too much train'
        model = 'no model_id'
        use_data_start = 'no_use_data_start'
        use_data_end = 'no_use_data_end'
        test_data_start = 'no_test_data_start'
        
        logger.info("★☆★ TOO MUCH train model ★☆★ ")

    
    else:
        result = 'nothing train'
        model = 'no model_id'
        use_data_start = 'no_use_data_start'
        use_data_end = 'no_use_data_end'
        test_data_start = 'no_test_data_start'
        
        logger.info("★☆★ NOTHING train model ★☆★ ")

       
    logger.info("★☆★★☆★★☆★~ Request search END ~★☆★★☆★★☆★")
    
    return result,model_id,use_data_start,use_data_end,test_data_start
        

def using_model_check(logger, sido_or_sgg,schema_nm,mysql):
    
    logger.info("★☆★★☆★★☆★~ AVAILABLE model search START ~★☆★★☆★★☆★")
    

    train_rst_tbl = mysql.read_by_sql(f"SELECT * FROM {schema_nm}.TFM_TRAIN_RST_TBL")
    
    if sido_or_sgg == 'sgg':
        train_rst_tbl = train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SGG')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))]
    else:
        train_rst_tbl = train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SIDO')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))]
    
    
    logger.info(f"AVAILABLE model num : {len(train_rst_tbl)}")
    
    if len(train_rst_tbl)==0:
        sys.exit()
    else:
        if sido_or_sgg == 'sgg':
            df_len = len(train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SGG')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))])
        else:
            df_len = len(train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SIDO')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))])
    
        if df_len != 1:
            sys.exit()
        else:
            logger.info("★☆★★☆★★☆★~ AVAILABLE model search END ~★☆★★☆★★☆★")
    
    if sido_or_sgg == 'sgg':
        model_path = train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SGG')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))]['MDL_ROUT'].values[0]  
        model_id = train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SGG')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))]['MDL_ID'].values[0]
    else:
        model_path = train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SIDO')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))]['MDL_ROUT'].values[0]  
        model_id = train_rst_tbl[(train_rst_tbl['USE_YN'] == 'Y') & (train_rst_tbl['MDL_ID'].str.contains('SIDO')) & (train_rst_tbl['MDL_ID'].str.contains('SNS'))]['MDL_ID'].values[0]
        
    #load_model = torch.load(model_path.values[0])
    #load_scaler = load(open(model_path.values[0].replace("sns_predict.pt","sns_data_scaler.pkl"),'rb'))
    
    logger.info(f"model_path : {model_path}")
    #logger.info(f"using_model : {load_model}")
    
    del mysql
    
    return model_id,model_path
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
