import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys

def make_logger(path, prgm_nm):
    ## 로그 생성 ##
    logger = logging.getLogger(prgm_nm)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s%(message)s]")
    log_fname = f"""logs/{prgm_nm}.log"""
    
    if os.path.isdir(path+"/logs"):
        pass
    else:
        os.mkdir(path+'/logs')
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fhandler = TimedRotatingFileHandler(os.path.join(path,log_fname), when = 'midnight', interval=1, backupCount=100)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    
    return logger
                
                
                
                
                
                
                
                
                

