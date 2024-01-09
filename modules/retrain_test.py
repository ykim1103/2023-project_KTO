import os
import time
import numpy as np
from pickle import load
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error
from modules import lstm
from modules import data_loader


def retrain(x_train_origin,y_train_origin,input_size,best_parameters,device,num_epoch,path,model_id,div,logger):
    logger.info("~RETRAIN START~")
    
    retrain_data_set = data_loader.train_data_loader(x_train_origin,y_train_origin,2**best_parameters['batch_size'])
    num_layers = best_parameters['num_layer']
    hidden_size = best_parameters['hidden_size']
    learning_rate = best_parameters['lr']
    
    model = lstm.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, device = device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    best_loss = 1e+6
    no_update_cnt = 0
    
    total_start_time = time.time()
    
    for epoc in range(num_epoch):
        batch_loss = 0.0
        
        for xx,yy in retrain_data_set:
            xx = xx.to(device)
            yy = yy.to(device)
            y_pred = model(xx)
            
            loss = criterion(y_pred,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            
        if epoc % 100 == 0:
            logger.info("[epoch: %d] loss: %.4f"%(epoc,batch_loss))
        last_loss = batch_loss

        if (best_loss >= last_loss) or (best_loss == 1e+6) :
            no_update_cnt = 0
            best_loss = last_loss
        else:
            no_update_cnt += 1
            
        if no_update_cnt == 10:
            break
    
    total_end_time = time.time()
    
    if div == 'sido':
        sub_path = '/output/sido/'

    elif div == 'sgg':
        sub_path = '/output/sgg/'
    else:
        sub_path = '/output/total/'

    if os.path.isdir(path+sub_path+model_id):
        pass
    else:
        os.makedirs(path+sub_path+model_id)
        
    logger.info(f"retrain_final_epoc: {epoc}")
    logger.info("best_parameters 학습 소요시간 : {}초".format(total_end_time - total_start_time))
    logger.info("~ RETRAIN END ~")
    torch.save(model,path+sub_path+model_id+'/sns_predict.pt')
    logger.info("★ BEST model SAVE ★")
    
    del xx 
    del yy
    del y_pred
    del model
    torch.cuda.empty_cache()
    


def test(x_test_origin,y_test_origin,path,model_id,device,div,logger):
    logger.info(" TEST START ")
    
    if div == 'sido':
        sub_path = "/output/sido/"
    elif div == 'sgg':
        sub_path = "/output/sgg/"
    else:
        sub_path = "/output/total/"
    
    total_y_pred = []
    load_scaler = load(open(path+sub_path+model_id+"/sns_data_scaler.pkl","rb"))
    model = torch.load(path+sub_path+model_id+"/sns_predict.pt")
    
    for i in range(0,len(x_test_origin),250):
        x_test2 = x_test_origin[i:i+250].to(device)
        y_pred = model(x_test2)
        predict_temp = y_pred.data.detach().cpu().numpy().reshape(len(y_pred),1)
        predict_temp - load_scaler.inverse_transform(predict_temp)
        total_y_pred.append(predict_temp)
    
    result = mean_absolute_percentage_error(y_test_origin, np.concatenate(total_y_pred))
    
    logger.info(" TEST END ")
    
    del x_test2
    del y_pred
    del model
    torch.cuda.empty_cache()
    
    return result, np.concatenate(total_y_pred)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
