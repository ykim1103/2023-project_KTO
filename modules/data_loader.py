from modules import dbconn
import pandas as pd
import numpy as np
import traceback
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def read_query(use_data_start, use_data_end,cols,logger):
    hive = dbconn.hive_conn(logger)
    col_str = ','.join(cols)
    data = hive.read_by_sql(f"""    
        SELECT {col_str}, SUM(cnt) as place_cnt_sum 
        FROM DW.TRD_ST_SNS_KWRD_DD_SUM
        WHERE sns_cate_mcls_nm = '장소' AND base_ymd >= {use_data_start} and base_ymd <= {use_data_end}
        GROUP BY {col_str}  """)
        
    data = data.sort_values(cols)

    return data
                

def read_query_predict(use_data_start, use_data_end,cols,logger):
    hive = dbconn.hive_conn(logger) 
    data = hive.read_by_sql(f"""
    SELECT 
        T1.base_ymd, 
        NVL(T3.new_sido_cd, T2.sido_cd) as sido_cd,
        NVL(T3.new_sido_nm, T2.sido_nm) as sido_nm,
        NVL(T3.new_sgg_cd, T2.sgg_cd) as sgg_cd,
        NVL(T3.new_sgg_nm, T2.sgg_nm) as sgg_nm,
        T1.cnt_sum
    FROM (
        SELECT 
            base_ymd,sido_nm,sgg_nm, SUM(cnt) as cnt_sum
        FROM
            DW.TRD_ST_SNS_KWRD_DD_SUM
        WHERE 1=1
            AND sns_cate_mcls_nm='장소'
            AND base_ymd BETWEEN {use_data_start} AND {use_data_end}
        GROUP BY sido_nm,sgg_nm,base_ymd
        ) T1    
    LEFT JOIN (
            SELECT sido_cd,sido_nm,sgg_cd,sgg_nm
            FROM DW.TRD_MS_ADONG_BAS T201
            WHERE base_ym = '202306'
            GROUP BY sido_cd,sido_nm,sgg_cd,sgg_nm

        ) T2
    ON T1.sido_nm = T2.sido_nm AND T1.sgg_nm =T2.sgg_nm
    LEFT JOIN (
        SELECT old_sido_cd, new_sido_cd, old_sido_nm,new_sido_nm,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
        FROM DW.TRD_MS_ADONG_BAS
        GROUP BY old_sgg_cd,new_sido_cd,old_sido_nm,new_sido_nm,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
        ) T3
    ON T2.sido_cd = T3.old_sido_cd AND T2.sgg_cd = T3.old_sgg_cd
    ORDER BY sgg_cd  """ )
    
    data = data.sort_values(['sido_cd','sgg_cd','base_ymd'])
    
    return data
    
    
def making_dataset_sgg(data,use_data_start,use_data_end,test_data_start,x_test_start_day,x_train_last_set_first_day,logger):
    total_train = pd.DataFrame()
    total_test = pd.DataFrame(0
    
    try:
        for sido in data['sido_nm'].unique():
            for sgg in data[data['sido_nm']==sido]['sgg_nm'].unique():
                temp = data.groupby(['sido_nm','sgg_nm']).get_group((sido,sgg))
                temp_train_df = temp[temp['base_ymd'] < f'{test_data_start}']
                temp_test_df = temp[temp['base_ymd']>= f'{x_test_start_day}']
                
                temp_train = temp_train_df['scaled_sum'].values
                temp_test = temp_test_df['scaled_sum'].values
                temp_y_test = temp_test_df['place_cnt_sum'].values
                
                temp_train = temp_train.reshape(-1,1)
                temp_test = temp_test.reshape(-1,1)
                temp_y_test = temp_y_test.reshape(-1,1)
                
                x_train = []
                y_train = []
                x_test = []
                y_test = []
                training_datasize = len(temp_train_df)
                testing_datasize = len(temp_test_df)
                
                for i in range(50,training_datasize):
                    x_train.append(temp_train[i-50:i,0])
                    y_train.append(temp_train[i,0])
                    
                for i in range(50,testing_datasize)    :
                    x_test.append(temp_test[i-50:i,0])
                    y_test.append(temp_y_test[i,0])
                    
                df_train = pd.DataFrame()
                df_train['ymd'] = pd.date_range(f'{use_data_start}',f'{x_train_last_set_first_day}')
                df_train['x_val'] = x_train
                df_train['y_val'] = y_train
                df_train['sido_nm'] = sido
                df_train['sgg_nm'] = sgg
                df_train['sido_cd'] = temp['sido_cd'].values[0]
                df_train['sgg_cd'] = temp['sgg_cd'].values[0]
                  
                df_test = pd.DataFrame()
                df_test['ymd'] = pd.date_range(f'{test_data_start}',f'{use_data_end}')
                df_test['x_val'] = x_test
                df_test['y_val'] = y_test
                df_test['sido_nm'] = sido
                df_test['sgg_nm'] = sgg
                df_test['sido_cd'] = temp['sido_cd'].values[0]
                df_test['sgg_cd'] = temp['sgg_cd'].values[0]
                
                if len(total_train) == 0:
                    total_train = df_train
                    total_test = df_test
                else:
                    total_train = pd.concat([total_train,df_train])
                    total_test = pd.concat([total_test,df_test])
        
        total_train = total_train.sort_values(['ymd','sido_nm','sgg_nm'])
        total_test = total_test.sort_values(['ymd','sido_nm','sgg_nm'])
        
        x_train_origin = total_train['x_val'].tolist()
        y_train_origin = total_train['y_val'].tolist()
        x_test_origin = total_test['x_val'].tolist()       
        y_test_origin = total_test['y_val'].tolist()
        
        ## train/valied split
        x_train,x_valid,y_train,y_valid = train_test_split(x_train_origin,y_train_origin,test_size=0.1,random_state=3011)
        
        logger.info("                                      ")
        logger.info("★ making dataset SUCESS ★")
        
    except:
        error_message = traceback.format_exc()
        logger.info(error_message)
        
    return total_test,x_train_origin,y_train_origin,x_train,y_train,x_valid,y_valid,x_test_origin,y_test_origin
    
    
def making_dataset_sido(data,use_data_start,use_data_end,test_data_start,x_test_start_day,x_train_last_set_first_day,logger):
    total_train = pd.DataFrame()
    total_test = pd.DataFrame(0
    
    try:
        for sido in data['sido_nm'].unique():
            temp = data.groupby(['sido_nm']).get_group((sido))
            temp_train_df = temp[temp['base_ymd'] < f'{test_data_start}']
            temp_test_df = temp[temp['base_ymd']>= f'{x_test_start_day}']
            
            temp_train = temp_train_df['scaled_sum'].values
            temp_test = temp_test_df['scaled_sum'].values
            temp_y_test = temp_test_df['place_cnt_sum'].values
            
            temp_train = temp_train.reshape(-1,1)
            temp_test = temp_test.reshape(-1,1)
            temp_y_test = temp_y_test.reshape(-1,1)
            
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            training_datasize = len(temp_train_df)
            testing_datasize = len(temp_test_df)
            
            for i in range(50,training_datasize):
                x_train.append(temp_train[i-50:i,0])
                y_train.append(temp_train[i,0])
                
            for i in range(50,testing_datasize)    :
                x_test.append(temp_test[i-50:i,0])
                y_test.append(temp_y_test[i,0])
                
            df_train = pd.DataFrame()
            df_train['ymd'] = pd.date_range(f'{use_data_start}',f'{x_train_last_set_first_day}')
            df_train['x_val'] = x_train
            df_train['y_val'] = y_train
            df_train['sido_nm'] = sido
    
            df_test = pd.DataFrame()
            df_test['ymd'] = pd.date_range(f'{test_data_start}',f'{use_data_end}')
            df_test['x_val'] = x_test
            df_test['y_val'] = y_test
            df_test['sido_nm'] = sido
            
            if len(total_train) == 0:
                total_train = df_train
                total_test = df_test
            else:
                total_train = pd.concat([total_train,df_train])
                total_test = pd.concat([total_test,df_test])
        
        total_train = total_train.sort_values(['ymd','sido_nm'])
        total_test = total_test.sort_values(['ymd','sido_nm'])
        
        x_train_origin = total_train['x_val'].tolist()
        y_train_origin = total_train['y_val'].tolist()
        x_test_origin = total_test['x_val'].tolist()       
        y_test_origin = total_test['y_val'].tolist()
        
        ## train/valied split
        x_train,x_valid,y_train,y_valid = train_test_split(x_train_origin,y_train_origin,test_size=0.1,random_state=3011)
        
        logger.info("                                      ")
        logger.info("★ making dataset SUCESS ★")
        
    except:
        error_message = traceback.format_exc()
        logger.info(error_message)
        
    return total_test,x_train_origin,y_train_origin,x_train,y_train,x_valid,y_valid,x_test_origin,y_test_origin


def data_to_tensor(x_train_origin,y_train_origin,x_train,y_train,x_valid,y_valid,x_test_origin,y_test_origin,logger):
    try:
        ## type : list to nparray
        x_train_origin = np.array(x_train_origin)
        y_train_origin = np.array(y_train_origin)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)
        x_test_origin = np.array(x_test_origin)
        y_test_origin = np.array(y_test_origin)
        
        
        ## type : nparray to tensor
        x_train_origin = Variable(torch.Tensor(x_train_origin))
        y_train_origin = Variable(torch.Tensor(y_train_origin))
        x_train = Variable(torch.Tensor(x_train))
        y_train = Variable(torch.Tensor(y_train))
        x_valid = Variable(torch.Tensor(x_valid))
        y_valid = Variable(torch.Tensor(y_valid))
        x_test_origin = Variable(torch.Tensor(x_test_origin))
        
        ## tensor reshape
        x_train_origin = torch.reshape(x_train_origin, (x_train_origin.shape[0],1,x_train_origin.shape[1]))
        y_train_origin = torch.reshape(y_train_origin, (y_train_origin.shape[0],1))
        x_train = torch.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
        y_train = torch.reshape(y_train, (y_train.shape[0],1))
        x_valid = torch.reshape(x_valid, (x_valid.shape[0],1,x_valid.shape[1]))
        y_valid = torch.reshape(y_valid, (y_valid.shape[0],1))
        x_test_origin = torch.reshape(x_test_origin, (x_test_origin.shape[0],1,x_test_origin.shape[1]))
        
        logger.info("                                      ")
        logger.info("★ dataset to tensor SUCESS ★")
    except:
        error_message = traceback.format_exc()
        logger.info(error_message)
        
    return x_train_origin,y_train_origin,x_train,y_train,x_valid,y_valid,x_test_origin,y_test_origin
    
    
def train_data_loader(x_train,y_train,batch_size):
    train_set = TensorDataset(x_train,y_train)
    torch.manual_seed(3011)
    train_set2 = DataLoader(train_set, batch_size=batch_size,shuffle=True)
    
    return train_set2

        
def valid_data_loader(x_valid,y_valid,batch_size):
    valid_set = TensorDataset(x_valid,y_valid)
    torch.manual_seed(3011)
    valid_set2 = DataLoader(valid_set, batch_size=batch_size,shuffle=True)
    
    return valid_set2       
        
        
def sgg_code_query(logger):
    hive = dbconn.hive_conn(logger)
    code_data = hive.read_by_sql(f"""
    SELECT 
        NVL(T2.new_sido_cd, T1.sido_cd) as new_sido_cd,
        NVL(T2.new_sido_nm, T1.sido_nm) as new_sido_nm,
        NVL(T2.old_sido_nm, T1.sido_nm) as sido_nm,
        NVL(T2.new_sgg_cd, T1.sgg_cd) as new_sgg_cd,
        NVL(T2.new_sgg_nm, T1.sgg_nm) as new_sgg_nm,
        NVL(T2.old_sgg_nm, T1.sgg_nm) as sgg_nm
    FROM (
        SELECT sido_cd,sido_nm,sgg_cd,sgg_nm
        FROM DW.TRD_MS_ADONG_BAS
        GROUP BY sido_cd,sido_nm,sgg_cd,sgg_nm
        )T1
    LEFT JOIN (
        SELECT old_sido_cd,new_sido_cd,old_sido_nm,new_sido_nm,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
        FROM DW.TRD_MS_ADONG_BAS
        GROUP BY old_sido_cd,new_sido_cd,old_sido_nm,new_sido_nm,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
            ) T2
    ON T1.sido_cd = T2.old_sido_cd AND T1.sgg_cd = T2.old_sgg_cd """)
    
    return code_data
    
    
def sido_code_query(logger):
    hive = dbconn.hive_conn(logger)
    code_data = hive.read_by_sql(f"""
    SELECT 
        new_sido_cd,new_sido_nm,sido_nm
    FROM i=(
        SELECT 
            NVL(T2.new_sido_cd, T1.sido_cd) as new_sido_cd,
            NVL(T2.new_sido_nm, T1.sido_nm) as new_sido_nm,
            NVL(T2.old_sido_nm, T1.sido_nm) as sido_nm,
            NVL(T2.new_sgg_cd, T1.sgg_cd) as new_sgg_cd,
            NVL(T2.new_sgg_nm, T1.sgg_nm) as new_sgg_nm,
            NVL(T2.old_sgg_nm, T1.sgg_nm) as sgg_nm
        FROM (
            SELECT sido_cd,sido_nm,sgg_cd,sgg_nm
            FROM DW.TRD_MS_ADONG_BAS
            GROUP BY sido_cd,sido_nm,sgg_cd,sgg_nm
            )T1
        LEFT JOIN (
            SELECT old_sido_cd,new_sido_cd,old_sido_nm,new_sido_nm,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
            FROM DW.TRD_MS_ADONG_BAS
            GROUP BY old_sido_cd,new_sido_cd,old_sido_nm,new_sido_nm,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
            ) T2
        ON T1.sido_cd = T2.old_sido_cd AND T1.sgg_cd = T2.old_sgg_cd 
        
        ) T0 WHERE sgg_nm <> '군위군' 
            GROUP BY new_sido_cd, new_sido_nm,sido_nm  """)
    
    return code_data    
    
    
def cluster_sns_load(use_data_start,use_data_end,logger):
    hive = dbconn.hive_conn(logger)
    data = hive.read_by_sql(f"""
        SELECT base_ymd, sido_cd, sido_nm, sgg_cd, sgg_nm, sum(cnt) as cnt, clust_no 
        FROM (
            SELECT 
                T1.base_ymd, 
                NVL(T4.new_sido_cd, T3.sido_cd) as sido_cd,
                NVL(T4.new_sido_nm, T3.sido_nm) as sido_nm,
                NVL(T2.old_sido_nm, T1.sido_nm) as sido_nm,
                NVL(T4.new_sgg_cd, T3.sgg_cd) as sgg_cd,
                NVL(T4.new_sgg_nm, T3.sgg_nm) as sgg_nm,
                T1.cnt,
                T2.clust_no
            FROM    
                DW.TRD_ST_SNS_KWRD_DD_SUM T1
            LEFT JOIN COM.TRC_MS_LOCGO_CTRG_BAS T2
                ON T1.sido_nm = T2.sido_nm and T1.sgg_nm = T2.sgg_nm
            LFFT JOIN (
                    SELECT sido_cd,sido_nm,sgg_cd,sgg_nm
                    FROM DW.TRD_MS_ADONG_BAS
                    GROUP BY sido_cd,sido_nm,sgg_cd,sgg_nm           
            ) T3 
                ON T1.sido_nm = T3.sido_nm AND T1.sgg_nm = T3.sgg_nm
            LEFT JOIN (
                    SELECT old_sido_cd,new_sido_cd,old_sido_nm,new_sido_cd,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
                    FROM DW.TRD_MS_CHG_ADONG_BAS
                    GROUP BY old_sido_cd,new_sido_cd,old_sido_nm,new_sido_nm,old_sgg_cd,new_sgg_cd,old_sgg_nm,new_sgg_nm
            ) T4
                ON T1.sido_nm = T4.old_sido_nm AND T1.sgg_nm = T4.old_sgg_nm
            WHERE 1=1
                AND T1.base_ymd BETWEEN {use_data_start} AND {use_data_end}
                AND T1.sns_cate_mcls_nm = '장소'
        ) T101 
        GROUP BY base_ymd, sido_cd,sido_nm,sgg_cd,sgg_nm,clust_no  """)
        
    return data    
        