import os
import torch
import argparse
import numpy as np
import torch.optim as optim

from model import MODEL
from trainer import train,test
from autoencoder import AUTOENCODER,di_reduction
from data import DATA,DATASET
import utils as utils


def autoencoding(f1,f2,f3,f4,f5,autoencoder):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    encoder=autoencoder.double().to(device)
    
    f3=f3.reshape(encoder.x,-1)
    f5=f5.reshape(encoder.x,-1)
    
    f1=torch.from_numpy(f1)
    f2=torch.from_numpy(f2)
    f3=torch.from_numpy(f3)
    f4=torch.from_numpy(f4)
    f5=torch.from_numpy(f5)

    xxx=torch.cat([f1,f2,f3,f4,f5],dim=1)
    feature_data=di_reduction(encoder,xxx)
    return feature_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=50, help='number of iterations')
    parser.add_argument('--decay_epoch', type=int, default=20, help='number of iterations')
    
    #MODEL ADDON
    parser.add_argument('--auto_encoder', type=bool, default=False, help='Autoencoder')
    parser.add_argument('--fuzzy_logic', type=bool, default=True, help='attention')
    parser.add_argument('--feedforward', type=bool, default=False, help='LSTM')
    
    parser.add_argument('--trans', type=bool, default=False, help='print progress')
    

    parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='learning rate decay')
    parser.add_argument('--final_lr', type=float, default=1E-5,
                        help='learning rate will not decrease after hitting this threshold')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

    dataset = 'mydata'

    if dataset == 'mydata':
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--qa_embed_dim', type=int, default=20, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--n_question', type=int, default=7486, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=1090, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
        parser.add_argument('--data_name', type=str, default=dataset, help='data set name')
        parser.add_argument('--test_name', type=str, default='mytest', help='save model name')
        parser.add_argument('--load', type=str, default='mydata', help='model file to load')
        parser.add_argument('--save', type=str, default='mydata', help='')
        
    params = parser.parse_args()
    params.lr = params.init_lr
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim
    print(params)
    
    if params.trans:
        tr_d=DATASET(params.data_dir,mode=0)
        training_set=[tr_d[ele] for ele in tr_d.student_id]
        tr_d.save(training_set,params.data_dir,params.data_name)
    else:
        pass
    data=DATA(params.data_dir,params.data_name)
    train_q_data,train_qa_data=data.loader()
    
    if params.trans:
        te_d=DATASET(params.data_dir,mode=1)
        test_set=[te_d[ele] for ele in te_d.student_id]
        te_d.save(test_set,params.data_dir,params.test_name)
    else:
        pass
    tst=DATA(params.data_dir,params.test_name)
    test_q_data,test_qa_data=tst.loader()
    
    if params.auto_encoder==True:
        f1,f2,f3,f4,f5=data.feat_loader()
        train_encoder=AUTOENCODER(mode=0)
        feature_data=autoencoding(f1,f2,f3,f4,f5,train_encoder)
        
        test_encoder=AUTOENCODER(mode=1)
        t1,t2,t3,t4,t5=tst.feat_loader()
        test_feat_data=autoencoding(t1,t2,t3,t4,t5,test_encoder)
    else:
        feature_data=np.random.rand(7486,1090)
        test_feat_data=np.random.rand(8093,9)
    
    model = MODEL(params=params)

    model.init_embeddings()
    model.init_params()
    
    optimizer = optim.Adam(params=model.parameters(),
                           lr=params.lr,
                           betas=(0.9, 0.9))

    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()
        
    all_train_auc = []
    all_train_loss = []
    all_train_accuracy = []
    max_auc=0
    
    for idx in range(params.max_iter):
        train_loss,train_accuracy,train_auc=train(idx,
                                                  model,
                                                  params,
                                                  optimizer,
                                                  train_q_data,
                                                  train_qa_data,
                                                  feature_data)
        if idx % 5 == 0:
            print('Epoch %d/%d'%(idx,params.max_iter))
            print('loss %3.5f'%(train_loss))
            print('auc %3.5f'%(train_auc))
            print('accuracy %3.5f'%(train_accuracy))
            
        all_train_auc.append(train_auc)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)
        
        if train_auc>max_auc:
            max_auc = train_auc
            torch.save(model,"model.pth")
            
                       
    train_result=[all_train_auc,all_train_loss,all_train_accuracy,max_auc]
    utils.save_data(train_result,'train_result')    
    pred=test(model,
              params,
              optimizer,
              test_q_data,
              test_qa_data,
              test_feat_data)
    utils.save_data(pred,'pred')
    utils.save_pred(pred,'pred')

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    