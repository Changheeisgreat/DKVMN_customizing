import os
import csv
import numpy as np
import pandas as pd

class DATASET():
    def __init__(self,data_path,mode=bool):
        self.data_path=data_path
        
        self.train_data=data_path+'train.csv'
        self.test_data=data_path+'test.csv'
        
        if mode==0:
            self.mode='train'
            self.data=pd.read_csv(self.train_data)
        elif mode==1:
            self.mode='test'
            self.data=pd.read_csv(self.test_data)
        self.features=list(self.data.columns.values)[5:-1]
        self.feat_unique=[len(self.data[c].unique()) for c in self.features]
        self.student_id=self.data['student_id'].unique().tolist()
        
    def __len__(self):
        return len(self.data['student_id'].unique())
    
    def show_unique(self):
        self.features=list(self.data.columns.values)[5:-1]
        self.feat_unique=[len(self.data[c].unique()) for c in self.features]
        return self.feat_unique
        
    def __getitem__(self,items):
        '''
        output
        number of questions:   5
        exercise sets:         1,4,4,6,10
        answer:                0,1,0,1,1
        '''     
        self.data.sort_values(by=['row_id'])
        

        #use student_id as a key to map dataset.
        st=list()
        cond=self.data['student_id']==items

        exercise_li=list()
        for _,x in enumerate(self.data[cond]['bundle_id']):
            exercise_li.append(x)
        st.append(exercise_li)

        correct_li=list()
        for _,y in enumerate(self.data[cond]['correct']):
            correct_li.append(y)
        st.append(correct_li)

        if len(st[0])!=len(st[-1]):
            print('I am your father')

        #features 
        features=self.data.iloc[:,5:-1].columns.tolist()
        for feat in features:
            feature_li=list()
            for _,f_n in enumerate(self.data[cond][feat]):
                feature_li.append(f_n)
            st.append(feature_li)
        
        return st
            
    def save(self,total_list,data_dir,data_name):
        name=data_dir+data_name+'.csv' 
        with open(name,'w') as f: #path 추가할것. 
            writer=csv.writer(f)
            for ele in total_list:
                writer.writerow([len(ele[0])])
                for data in ele:
                    writer.writerow(data)
                    
import os
import csv
import numpy as np
import pandas as pd

class DATA():
    def __init__(self,data_dir,data_name):
        self.f_data=open(data_dir+data_name+'.csv','r')
        self.data=self.f_data.read().split('\n')[:-1]
        self.f_data.close()
        
        assert len(self.data)%8==0,"You underestimate my power!" 
        self.leng=int(len(self.data))
        self.n_question=self.leng/8
        
        self.answer_idx=self.get_every_nth(2)
        self.answer=self.get_feat(self.answer_idx)
        self.maximum_q=max([len(x) for x in self.answer])
  
        self.question_idx=self.get_every_nth(1)
        self.qdata=self.get_feat(self.question_idx)
        
        self.qadata=self.get_qa()
        
        self.feat_1_idx=self.get_every_nth(3)
        self.feat_2_idx=self.get_every_nth(4)
        self.feat_3_idx=self.get_every_nth(5)
        self.feat_4_idx=self.get_every_nth(6)
        self.feat_5_idx=self.get_every_nth(7)
        
        self.feat_1=self.get_feat(self.feat_1_idx)
        self.feat_2=self.get_feat(self.feat_2_idx)
        self.feat_3=self.get_feat(self.feat_3_idx)
        self.feat_4=self.get_feat(self.feat_4_idx)
        self.feat_5=self.get_feat(self.feat_5_idx)
        
        self.f_1=self.array_convert(self.feat_1)
        self.f_2=self.array_convert(self.feat_2)
        self.f_3=self.array_convert(self.feat_3)
        self.f_4=self.array_convert(self.feat_4)
        self.f_5=self.array_convert(self.feat_5)
        self.onehotencoder()
        
    def get_every_nth(self,nth=int):
        return [x for x in range(self.leng) if x%8==nth]
    
    def get_feat(self,idx_li):
        return [self.data[ele].split(',') for ele in idx_li]
    
    def array_convert(self,data_li):
        n=self.n_question
        m=self.maximum_q
        array=np.zeros((len(data_li),m))
        for i in range(len(data_li)):
            dat=data_li[i]
            array[i,:len(dat)]=list(map(int,dat))
        assert array.shape==(n,m),"I hate you"
        return array 
    
    def get_qa(self):
        qa=[]
        for i in range(int(self.n_question)):
            tmp=[]
            for j in range(len(self.qdata[i])):
                x=int(self.qdata[i][j])+int(self.answer[i][j])*self.n_question
                tmp.append(x)
            qa.append(tmp)
        return qa
    
    def loader(self):
        return self.array_convert(self.qdata),self.array_convert(self.qadata)
    
    def onehotencoder(self):
        """
                train    test   total
    feature1:   1760     494    1781
    feature2:   1898     580    1924
    feature3:   9        11     11
    feature4:   18       18     18
    feature5:   3        3      3
     
        """        
        f3_metrix=np.eye(int(11)).astype(int)
        #f4_metrix=np.eye(int(18)).astype(int)
        f5_metrix=np.eye(int(3)).astype(int)
        
        self.f_3_=f3_metrix[self.f_3.astype(int)]
        #self.f_4_=f4_metrix[self.f_4.astype(int)]
        self.f_5_=f5_metrix[self.f_5.astype(int)]
        
    def feat_loader(self):
        return self.f_1,self.f_2,self.f_3_,self.f_4,self.f_5_