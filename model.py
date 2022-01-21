import torch
import numpy as np
import torch.nn as nn

import utils as utils
from lstm import LSTM

class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write

        if self.is_write:
            self.erase = torch.nn.Linear(
                self.memory_state_dim,
                self.memory_state_dim,
                bias=True)

            self.add = torch.nn.Linear(
                self.memory_state_dim,
                self.memory_state_dim,
                bias=True)

            nn.init.kaiming_normal(self.erase.weight)
            nn.init.kaiming_normal(self.add.weight)

            nn.init.constant(self.erase.bias, 0)
            nn.init.constant(self.add.bias, 0)


    def addressing(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score=torch.matmul(control_input, torch.t(memory))
        correlation_weight=torch.nn.functional.softmax(similarity_score,dim=1)
        # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content
        '''
        with open('read_content.pickle', 'wb') as f:
            pickle.dump(read_content, f)
        '''


    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory

class DKVMN(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        super(DKVMN, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False)
        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True)
        self.memory_key = init_memory_key

        # self.memory_value = self.init_memory_value
        self.memory_value = None

    def init_value_memory(self, memory_value):
        self.memory_value = memory_value

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(control_input=control_input,
                                                      memory=self.memory_key)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(memory=self.memory_value,
                                            read_weight=read_weight)
        return read_content

    def write(self, write_weight, control_input, if_write_memory):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=self.memory_value,
                                             write_weight=write_weight)
        self.memory_value = nn.Parameter(memory_value.data)

        return self.memory_value

class MODEL(nn.Module):
    def __init__(self,params,student_num=None):

        super(MODEL, self).__init__()
        self.auto_encoder = params.auto_encoder
        self.fuzzy_logic=params.fuzzy_logic
        self.feedforward=params.feedforward
        
        self.n_question = params.n_question
        self.batch_size = params.batch_size
        self.memory_size = params.memory_size
        
        self.q_embed_dim = params.q_embed_dim
        self.qa_embed_dim = params.qa_embed_dim
        self.final_fc_dim = params.final_fc_dim
        
        self.memory_key_state_dim = params.memory_key_state_dim
        self.memory_value_state_dim = params.memory_value_state_dim
        self.student_num = student_num
        self.input_embed_linear = nn.Linear(
            self.q_embed_dim,
            self.final_fc_dim,
            bias=True)

        #r_t
        self.read_embed_linear = nn.Linear(
            self.memory_value_state_dim+self.final_fc_dim,
            self.final_fc_dim,
            bias=True)

        #feedforward
        if self.auto_encoder:
            x=10
        else:
            x=0
        if self.fuzzy_logic:
            y=2
        else:
            y=0
        if self.feedforward==True:
            self.predict_linear=LSTM(self.batch_size,self.final_fc_dim+x+y,1)
        else:
            self.predict_linear=nn.Linear(self.final_fc_dim+x+y,1,bias=True)

        self.init_memory_key = nn.Parameter(
            torch.randn(self.memory_size,
                        self.memory_key_state_dim))
        nn.init.kaiming_normal(self.init_memory_key)

        self.init_memory_value = nn.Parameter(
            torch.randn(self.memory_size,
                        self.memory_value_state_dim))
        nn.init.kaiming_normal(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                         init_memory_key=self.init_memory_key,
                         memory_key_state_dim=self.memory_key_state_dim,
                         memory_value_state_dim=self.memory_value_state_dim)

        memory_value = nn.Parameter(
            torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(self.batch_size)], 0).data)

        self.mem.init_value_memory(memory_value)
        self.q_embed = nn.Embedding(self.n_question + 1,
                                    self.q_embed_dim,
                                    padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1,#고유 데이터만 임베딩이 가능함.
                                     self.qa_embed_dim,
                                     padding_idx=0)
        
    def init_params(self):
        if self.feedforward==False:
            nn.init.kaiming_normal(self.predict_linear.weight)
        nn.init.kaiming_normal(self.read_embed_linear.weight)
                                   
        if self.feedforward==False:
            nn.init.constant(self.predict_linear.bias, 0)
        nn.init.constant(self.read_embed_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal(self.q_embed.weight)
        nn.init.kaiming_normal(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target, input_f, student_id=None):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(
            torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)

        self.mem.init_value_memory(memory_value)

        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        correlation_l=[]
        value_read_content_l = []
        input_embed_l = []
        predict_logs = []
        
        if self.auto_encoder:
            input_f=input_f.reshape(-1,10)
        else:
            input_f=input_f.reshape(-1,1)

        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)
            correlation_l.append(correlation_weight)
            if_memory_write = slice_q_data[i].squeeze(1).ge(1)
            if_memory_write = utils.varible(torch.FloatTensor(if_memory_write.data.tolist()), 1)

            ## Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            ## Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, qa, if_memory_write)

        all_correlation = torch.cat([correlation_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        attention_weight=utils.pca_kmeans_tri(all_correlation)
        attention_weight=attention_weight.reshape(-1,2)

        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        predict_input = torch.cat([all_read_value_content,input_embed_content], 2) #여기다 뭐 추가하면 콘캇가능
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size*seqlen, -1)))
        
        if self.fuzzy_logic==True:
            read_content_embed = torch.cat([read_content_embed,attention_weight],1)
        else:
            pass
        #add features
        if self.auto_encoder==True:
            read_content_embed = torch.cat([read_content_embed,input_f],1)
        else:
            pass
        
        pred = self.predict_linear(read_content_embed)

        target_1d = target                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target
