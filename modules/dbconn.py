"""
필수사항 : python>=3.8
        dbapi
        pymysql

사용법
    from dbconn import hive_conn
    import logging
    logger = ...<create logger>...
    hive = hive_conn(logger)
    data = hive.read_by_sql("SELECT * FROM com.trc_ms_crsu_cvt_cd")
    
    del hive
"""

from impala import dbapi
import pymysql
import pandas as pd
import numpy as np
from datetime import datetime as dt
import logging
import traceback

qry_prn_num = 300

class hive_conn:
    def __init__(self,logger,host:str='10.1.113.30', port: int=10000, user:str='admin', password: str='KTOit699!', db:str='temp_dw'):
        self.__HOST = host
        self.__PORT = port
        self.__USER = user
        self.__PASSWORD = password
        self.__DATABASE = db
        self.__logger = logger
        self.__conn = self.__connection()
        
        
    def __del__(self):
        if self.__conn:
            self.__conn.close()
            
            
    def __connection(self):
        conn = None
        try:
            conn = dbapi.connect(
                        host = self.__HOST,
                        port = self.__PORT,
                        user = self.__USER,
                        password = self.__PASSWORD,
                        database = self.__DATABASE,
                        auth_mechanism = 'PLAIN'
            )
            self.__logger.info("Hive connection succeed")
        except Exception as e:
            self.__logger.error(f"""Hive connection is failed with {self.__HOST}:{self.__PORT} {self.__USER}/{self.__PASSWORD}""")
            self.__logger.error(traceback.print_exc())
            
        return conn    
        
        
    def read_by_sql(self, query:str, fetch_num = -1, ret_type = 1, engine: str = 'tez'):
        cursor = self.__conn.cursor()
        df = None
        try:
            if engine not in ['tez','mr']:
                raise Exception(f"""hive execution egine setting error {engine} """)
            self.__logger.debug(f"""use '{engine}' engine in hive """)
            cursor.execute(f""" set hive.execution.engine={engine}""")
            self.__logger.debug(f""" hive query ==> \n{query}""")
            cursor.execute(query)
            
            if fetch_num == -1:
                data = cursor.fetchall()
            elif fetch_num == 0:
                log_max_len = qry_prn_num if len(query) > qry_prn_num else len(query)
                self.__logger.info(f"""Hive query executiondone \n {query[:log_max_len]} """)
                cursor.close()
                return None
            else:
                data = cursor.fetchmany(fetch_num)
                
            data_len = len(data)
            if ret_type ==1:
                cols = [col[0] for col in cursor.description]
                df = pd.DataFrame(data, columns = cols)
                self.__logger.info(f"""Return {data_len} hive data in DataFrame""")
            else:
                df = data
                self.__logger.info(f"""Return {data_len} hive data in List""")
        except Exception as e:
            log_max_len = qry_prn_num if len(query) > qry_prn_num else len(query)
            self.__logger.error(f"""Hive read_by_sql is failed with \n {query[:log_max_len]}""")
            self.__logger.error(traceback.print_exc())
        finally:
            cursor.close()
            
        return df

        
    def __make_value_str(self,df):
        final_str = """"""
        for value in df.values:
            tmp_list = []
            for item in value:
                if type(item) in [str,pd._libs.tslibs.timestamps.Timestamp]:
                    tmp_list.append(f""" '{item}' """)
                elif pd.isna(item):
                    tmp_list.append(""" '<NULL>' """)
                else:
                    tmp_list.append(f""" {item} """)
                    
            value_str = ','.join(tmp_list)
            final_str += f"""({value_str}),"""
            ret_str = final_str.replace("'<NULL>'","NULL")

        return ret_str[:-1]
        
        
    def save_by_dataframe(self, schema_nm:str, table_nm:str, df:pd.DataFrame, insert_type:str='into', partition_col_nm:str=None):
        cursor = self.__conn.cursor()
        data_len = len(df)
        try:
            insert_query = f""" insert {insert_type} {schema_nm}.{table_nm} """
            if partition_col_nm is not None:
                cols = df.columns.tolist()
                cols.remove(partition_col_nm)
                partitions = df[partition_col_nm].unique()
                
                for part in partitions:
                    df_temp = df[df[partition_col_nm] ==part ]
                    df_temp = df_temp[cols]
                    
                    query = f""" alter table {schema_nm}.{table_nm} add if not exists partition ({partition_col_nm}='{part}') """
                    self.__logger.debug(f""" hive query ==> \n{query}""")
                    cursor.execute(query)
                    
                    insert_query += f"""partition ({partition_col_nm}='{part}') values """ + self.__make_value_str(df_temp)
                
                log_max_len = qry_prn_num if len(insert_query) > qry_prn_num else len(insert_query)
                self.__logger.debug(f""" hive query ==>\n {insert_query[:log_max_len]}""")
                
                cursor.execute(insert_query)
                self.__logger.info(f""" Done inserting {data_len} data with partition ({partition_col_nm])""")
            
            else:
                insert_query += 'values' + self.__make_value_str(df)
                
                log_max_len = qry_prn_num if len(insert_query) > qry_prn_num else len(insert_query)
                self.__logger.debug(f""" hive query==>\n {insert_query[log_max_len]}""")

                cursor.execute(insert_query)
                self.__logger.info(f"""Done inserting {data_len} data""")
                
        except Exception as e:
            log_max_len = qry_prn_num if len(insert_query) > qry_prn_num else len(insert_query)
            self.__logger.error(f""" Hive save_by_dataframe is failed with\n {insert_query[:log_max_len]}""")
            self.__logger.error(traceback.print_exc())
            
        finally:
            cursor.close()
            
            
            
            
class mysql_conn:
    def __init__(self, logger, host='10.1.113.83', port=23306, user='kto', password='Kto2020!', db='kto_datalab'):
        self.__HOST = host
        self.__PORT = port
        self.__USER = user
        self.__PASSWORD = password
        self.__DATABASE = db
        self.__logger = logger
        self.__conn = self.__connection()
        
    
    def __del__(self):
        if self.__conn:
            self.__conn.close()
            
    
    def __connection(self):
        conn = None
        try:
            conn == pymysql.connect(
                        host = self.__HOST,
                        port = self.__PORT,
                        user = self.__USER,
                        password = self.__PASSWORD,
                        database = self.__DATABASE,
                        charset = 'UTF8' )
                        
            self.__logger.info("MySQL connection succeed")
        except Exception as e:
            self.__logger.error(f"""MySQL connection is failed with {self.__HOST}:{self.__PORT} {self.__USER}/{self.__PASSWORD} """)
            self.__logger.error(traceback.print_exc())
         
        return conn
    

    def read_by_sql(self, query:str, fetch_num = -1, ret_type = 1):
        cursor = self.__conn.cursor()
        df = None
        try:
            cursor.execute(query)
            
            if fetch_num == -1:
                data = cursor.fetchall()
            elif fetch_num == 0:
                log_max_len = qry_prn_num if len(query) > qry_prn_num else len(query)
                self.__logger.info(f"""MySQL query execution done\n {query[:log_max_len]}""")
                cursor.close()
                return None
            else:
                data = cursor.fetchmany(fetch_num)
                
            data_len = len(data)
            if ret_type == 1:
                cols = [col[0] for col in cursor.description]
                df = pd.DataFrame(data, columns=cols)
                self.__logger.info(f"""Return {data_len} MySQL data in DataFrame """)
            else:
                df = data
                self.__logger.info(f"""Return {data_len} MySQL data in List """)
        except Exception as e:
            log_max_len = qry_prn_num if len(query) > qry_prn_num else len(query)
            self.__logger.error(f"""MySQLread_by_sql is failed with \n{query[:log_max_len]}""")
            self.__logger.error(traceback.print_exc())
            
        finally:
            cursor.close()
            
        return df
        
        
    def __make_value_str(self,df):
        final_str = """"""
        for value in df.values:
            tmp_list = []
            for item in value:
                if type(item) in [str,pd._libs.tslibs.timestamps.Timestamp]:
                    tmp_list.append(f""" '{item}' """)
                elif pd.isna(item):
                    tmp_list.append(""" '<NULL>' """)
                else:
                    tmp_list.append(f""" {item} """)
                    
            value_str = ','.join(tmp_list)
            final_str += f"""({value_str}),"""
            ret_str = final_str.replace("'<NULL>'","")

        return ret_str[:-1]
                
                
    def save_by_dataframe(self, schema_nm:str, table_nm:str, df:pd.DataFrame, bcommit=True, fetch_num = 20000):
        cursor = self.__conn.cursor()
        data_len = len(df)
        try:
            share,remained = divmod(data_len, fetch_num)
            loop_num = share
            loop_num += 1 if remained >0 else 0 
            
            for i in range(loop_num):
                begin_row = fetch_num * i
                end_row - data_len if fetch_num * i + fetch_num > data_len else fetch_num * i + fetch_num
                df_trgt - df.loc[begin_row:end_row]
                
                insert_query = f""" insert into {schema_nm}.{table_nm} """
                insert_query += "values " + self.__make_value_str(df_trgt)
            
                log_max_len = qry_prn_num if len(insert_query) > qry_prn_num else len(insert_query)
                self.__logger.debug(f"""mysql {i+1}/{loop_num} insert query from {begin_row} to {end_row} ==>\n{insert_query[:log_max_len]}""")
                cursor.execute(insert_query)
            
            if bcommit:
                cursor.execute("commit")
            self.__logger.info(f"""Done inserting {data_len} data """)
        except Exception as e:
            log_max_len = qry_prn_num if len(insert_query) > qry_prn_num else len(insert_query)
            self.__logger.error(f""" MySQL save_by_dataframe is failed with\n{insert_query[:log_max_len]}""")
            self.__logger.error(traceback.print_exc())
            
        finally:
            cursor.close()
            
            
    def run_sql(self, queries:list = ['select * from db.tables'], bcommit=False):
        cursor = self.__conn.cursor()
        try:
            if bcommit:
                queries.append('commit')
            for query in queries:
                cursor.execute(query)
                self.__logger.info(f"""Done running MySQL query\n{query}""")
        except Exception as e:
            self.__logger.error(f"""MySQL run sql is failed with\n{query}""")
            self.__logger.error(traceback.print_exc())
        finally:
            cursor.close()

        
                
                
                
                
                
                
                
                
                
                
                
                

