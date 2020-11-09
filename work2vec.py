#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
import warnings
import random


# In[2]:


df=pd.read_csv("topo.txt",sep='\t',names=['head_node','tail_node'],header=None) #导入数据
df.head()
df['tail_node']=df['tail_node'].apply(lambda x: x.split(","))
df.head()


# In[3]:


data=df.iloc[:,:]
data_dict=dict()
for i in tqdm(range(0,data.shape[0])):
    data_dict[str(data.iloc[i,0])]=data.iloc[i,1]  #拆分成字典



# In[4]:


##随机游走
step=3
work_vec_row=[]

for i in tqdm(range(0,700000)):
    
    str_i=str(i)
    if str_i in data_dict:
        work_vec_loc=[]
        work_vec_loc.insert(1,str_i )       
        rd=random.choice(data_dict[str_i])
        str_j=rd
        for j in range(0,step):
            
            if str_j in data_dict:
                work_vec_loc.insert(1,str_j)
                str_j=random.choice(data_dict[str_j])
                  
                
            else :
                work_vec_loc.insert(1,str_j)
        work_vec_row.insert(1,work_vec_loc)
            
work_vec_row
        


# In[5]:


##写入文件
df_result = pd.DataFrame(columns=['head', 'step_1', 'step_2', 'step_3'])
work_vec_row=np.array(work_vec_row)
df_result['head'] = work_vec_row[:,0]
df_result['step_1'] =  work_vec_row[:,1]
df_result['step_2'] =  work_vec_row[:,2]
df_result['step_3'] =  work_vec_row[:,3]
df_result.to_csv("word2vec_input.csv", index=False)


# In[11]:


sentences=work_vec_row[:,:]
sentences=sentences.tolist()
type(sentences)


# In[19]:


##Word2vec生成词向量
from gensim.models import Word2Vec



model = Word2Vec(sentences, min_count=1,workers=12,iter=128)

print(len(model.wv.vocab))


# In[20]:


model.save('word2vec_test_1.model')


# In[21]:


print(model['458966'])


# In[8]:


#导出文件
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
import warnings
import random
from gensim.models import Word2Vec
model = Word2Vec()


# In[10]:


model = Word2Vec.load('word2vec_test_1.model')


# In[12]:


model.wv.save_word2vec_format('word2vec_test_1.csv',binary = False)

