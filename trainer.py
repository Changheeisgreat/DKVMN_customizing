import math
import torch
import random
import numpy as np
from torch import nn
from sklearn import metrics

import utils as utils

def train(epoch_num, model, params, optimizer, q_data, qa_data,feat_data):
    N = int(math.floor(len(q_data)/params.batch_size))

    feature = feat_data
    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    for idx in range(N):
        q_one_seq=q_data[idx * params.batch_size:(idx+1)*params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = (target - 1) / params.n_question
        target = np.floor(target)
        
        input_q = utils.varible(torch.LongTensor(q_one_seq), params.gpu)
        input_qa = utils.varible(torch.LongTensor(qa_batch_seq), params.gpu)
        
        target = utils.varible(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        model.zero_grad()

        f_one_seq = feature[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        input_f = utils.varible(torch.LongTensor(f_one_seq), params.gpu)
        
        loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d,input_f)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), params.maxgradnorm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())

        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc

def test(model, params, optimizer, q_data, qa_data,test_feat_data):
    N = int(math.floor(len(q_data) / params.batch_size))
    feature=test_feat_data
    model.mem.iswrite=False
    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):
        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.varible(torch.LongTensor(q_one_seq), params.gpu)
        input_qa = utils.varible(torch.LongTensor(qa_batch_seq), params.gpu)
        target = utils.varible(torch.FloatTensor(target), params.gpu)

        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)
        
        f_one_seq = feature[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        input_f = utils.varible(torch.LongTensor(f_one_seq), params.gpu)
        
        loss, filtered_pred, filtered_target = model.forward(input_q,
                                                             input_qa,
                                                             target_1d,input_f)
        
        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    
    return all_pred
