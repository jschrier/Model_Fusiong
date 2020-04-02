#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import pandas as pd
import modelFusion as MF
import numpy as np


# In[2]:


# Support vector radial basis classifier
A = pd.read_csv('Crank 44/w9zj18ewvMD2D.csv')
# Random forest classification
B = pd.read_csv('Crank 44/w3qaqvdepL2JZ.csv')
# Weighted logistic classifier
C = pd.read_csv('Crank 44/wZWGwwoZOBZwZ.csv')
# Gradient boosted tree
D = pd.read_csv('Crank 44/w3M6mXamm38Or.csv')


# In[3]:


SVM = MF.preclean(A)
RF = MF.preclean(B)
WLC = MF.preclean(C)
GBT = MF.preclean(D)

raw = MF.raw_def([SVM,RF,WLC,GBT])


# In[4]:


RSCG = MF.RSC_graph(raw)


# In[5]:


RA, A_prec = MF.rank_raw(SVM)
RB, B_prec = MF.rank_raw(RF)
RC, C_prec = MF.rank_raw(WLC)
RD, D_prec = MF.rank_raw(GBT)


# In[6]:


DS_A, DS_B,DS_C, DS_D = MF.diversityS(raw)


# In[7]:


df = MF.cognitiveD(raw)
df


# In[8]:


SCAB = MF.two_fusionW(SVM,RF,[DS_A,DS_B])
SCAC = MF.two_fusionW(SVM,WLC,[DS_A,DS_C])
SCAD = MF.two_fusionW(SVM,GBT,[DS_A,DS_D])
SCBC = MF.two_fusionW(RF,WLC,[DS_B,DS_C])
SCBD = MF.two_fusionW(RF,GBT,[DS_B,DS_D])
SCCD = MF.two_fusionW(RF,GBT,[DS_C,DS_D])
SCABC = MF.three_fusionW(SVM,RF,WLC,[DS_A,DS_B,DS_C])
SCABD = MF.three_fusionW(SVM,RF,GBT,[DS_A,DS_B,DS_D])
SCACD = MF.three_fusionW(SVM,WLC,GBT,[DS_A,DS_C,DS_D])
SCBCD = MF.three_fusionW(SVM,WLC,GBT,[DS_B,DS_C,DS_D])
SCABCD = MF.four_fusionW(SVM,RF,WLC,GBT,[DS_A,DS_B,DS_C,DS_D])


# In[9]:


RCAB = MF.two_fusionW(RA,RB,[DS_A,DS_B],False)
RCAC = MF.two_fusionW(RA,RC,[DS_A,DS_C],False)
RCAD = MF.two_fusionW(RA,RD,[DS_A,DS_D],False)
RCBC = MF.two_fusionW(RB,RC,[DS_B,DS_C],False)
RCBD = MF.two_fusionW(RB,RD,[DS_B,DS_D],False)
RCCD = MF.two_fusionW(RC,RD,[DS_C,DS_D],False)
RCABC = MF.three_fusionW(RA,RB,RC,[DS_A,DS_B,DS_C],False)
RCABD = MF.three_fusionW(RA,RB,RD,[DS_A,DS_B,DS_D],False)
RCACD = MF.three_fusionW(RA,RC,RD,[DS_A,DS_C,DS_D],False)
RCBCD = MF.three_fusionW(RB,RC,RD,[DS_B,DS_C,DS_D],False)
RCABCD = MF.four_fusionW(RA,RB,RC,RD,[DS_A,DS_B,DS_C,DS_D],False)


# In[10]:


#find the rank of each data point in each single ML model
rawRank = pd.DataFrame([RA['ID'],RA['truth'],RA['rank'],RB['rank'],RC['rank'],RD['rank']])
rawRank = rawRank.T


# In[11]:


# combined all the fusion model into one for later
List = [SCAB,SCAC,SCAD,SCBC,SCBD,SCCD,SCABC,SCABD,SCACD,SCBCD,SCABCD]
ListName = ['AB','AC','AD','BC','BD','CD','ABC','ABD','ACD','BCD','ABCD']


# In[12]:


for i in range(11):
    print(ListName[i])
    print(MF.find_df(List[i],rawRank))


# In[13]:


for i in range(11):
    print(ListName[i])
    print(MF.find_dfop(List[i],rawRank))


# In[14]:


TPlist = []
for i in range(len(List)):
    x = MF.find_TP(List[i], sum(List[i]['truth']))
    MF.addtolist(x,TPlist)
    
precision =[]

for i in List:
    x = MF.rank_function(i)
    MF.addtolist(x,precision)


# In[15]:


dict = { 'True positive': TPlist,'SC': precision}
df = pd.DataFrame(dict, index=['AB','AC','AD','BC','BD','CD',
                               'ABC','ABD','ACD','BCD','ABCD'])
df


# In[16]:


ListRank = [RCAB,RCAC,RCAD,RCBC,RCBD,RCCD,RCABC,RCABD,RCACD,RCBCD,RCABCD]
ListRankName = ['AB','AC','AD','BC','BD','CD','ABC','RCABD','RCACD','RCBCD','RCABCD']


# In[17]:


sum(ListRank[5]['truth'])


# In[18]:


TPlistr = []
for i in range(len(ListRank)):
    x = MF.find_TP(ListRank[i], sum(ListRank[i]['truth']))
    MF.addtolist(x,TPlistr)
    
precisionr =[]

for i in ListRank:
    x = MF.rank_function(i)
    MF.addtolist(x,precisionr)


# In[19]:


dict = { 'True positive': TPlistr,'RC': precisionr}
rc = pd.DataFrame(dict, index=ListName)
rc


# In[20]:


result=pd.DataFrame()
for i in range(11):
    print(ListRankName[i])
    print(MF.find_df(ListRank[i],rawRank))


# In[21]:


resultss=pd.DataFrame()
for i in range(11):
    print(ListRankName[i])
    print(MF.find_dfop(ListRank[i],rawRank))


# In[22]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# In[23]:


newDF = pd.concat([df.SC,rc.RC], axis = 1)
newDF


# In[24]:


newDF.plot(kind='bar', color=tuple(["b","orange"]))
x = np.arange(len(ListName))

l1 = plt.hlines(A_prec,-1,22,'c')
l2 = plt.hlines(B_prec,-1,22,'m')
l3 = plt.hlines(C_prec,-1,22,'g')
l4 = plt.hlines(D_prec,-1,22,'r')

SC_patch = mpatches.Patch(color='blue', label='SC')
RC_patch = mpatches.Patch(color='orange', label='RC')
l1_patch = mpatches.Patch(color='c', label='A')
l2_patch = mpatches.Patch(color='Y', label='B')
l3_patch = mpatches.Patch(color='g', label='C')
l4_patch = mpatches.Patch(color='r', label='D')

plt.legend(handles=[SC_patch, RC_patch, l1_patch, l2_patch,l3_patch,l4_patch],loc = 0, ncol =6)


plt.ylim(0.3,0.8)
plt.figure(figsize=(8, 5))


# In[ ]:




