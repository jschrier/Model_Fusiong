#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd


# In[1]:


# pre-progress first model
def preclean(frame,columnDrop = ['dataset','_rxn_M_inorganic','_rxn_M_organic', '_rxn_M_acid'], nameList = ['name','truth','pred','score']):
    """
    this function preclean the result of each model. 
    droping the unused columns from the result.
    leave the name, truth result, pred result and possibility 
    change name of the column to [name, thruth, pred, score]
    then insert an ID column at beginning
    
    """
    frame = frame.drop(columnDrop, axis=1)
    frame.columns = nameList
    frame.iloc[:,-1] = (frame.iloc[:,-1] -frame.iloc[:,-1].min())/(frame.iloc[:,-1].max()-frame.iloc[:,-1].min())
    frame.insert(0,'ID',range(1,len(frame)+1))
    return frame


# In[2]:


def raw_def(frame_list,dropList = ['name','truth','pred'], nameList = ['ID','SVM','RF','WLC','GBT']):
    """
    The fucntion merge all single model from the preclean function into one dataframe by ID column.
    and rename the column name
    """
    raw = pd.DataFrame(frame_list[0].iloc[:,0])
    for i in range(len(frame_list)):
        frame = frame_list[i]
        frame = frame.drop(dropList,axis = 1)
        raw =pd.merge(raw, frame,how='outer', on='ID')
    raw.columns = nameList
    # not normailize the ID
    raw['ID'] = frame_list[0]['ID']
    raw.insert(1,'truth',frame_list[0].truth)
    return raw


# In[3]:

def rank_function(rank_function):
    """
    this fucntion calculate the precision of the rank fucntion
    """
    prec = rank_function[:rank_function.truth.sum()].truth.sum()/rank_function.truth.sum()
    
    return prec

#find true positive
def find_TP(df, rank):
    """
    This function calculate the number of true positive in model base on rank
    rank = the total positive result in model
    """
    TP = 0
    for i in range(len(df)):
        if df.iloc[i][-1] <= rank and df.iloc[i][1] ==1:
            TP += 1
    return TP

#add number to the list
def addtolist(x,list):
    list.append(x)
    return list




def rank_raw(frame, sortby = ['score']):
    """
    This function create the rank fucntion of the input data frame from the score fucntion
    
    return the rank fucntion and the precision at number of the truth positive in total
    """
    rankFun = frame.sort_values(by=sortby, ascending = False)
    rankFun.insert(len(rankFun.columns),'rank',range(1,len(rankFun)+1))
    frame_prec = rank_function(rankFun)
    
    rankFun = rankFun.sort_values(by=['ID'])
    return rankFun, frame_prec


# In[4]:


# draw the RSC graph
def RSC_graph(raw):
    """
    This fucntion draw the RSC graph from the noramlized raw dataframe 
    """
    RSCG = raw.drop(['ID','truth'], axis = 1)
    for col in RSCG.columns:
        RSCG[col] = RSCG[col].sort_values(ascending=False).values
    RSCG.insert(0,'Rank',range(1,len(raw)+1))
    RSCG = RSCG.set_index('Rank')
    #RSCG.plot(title='Rank-Score Characteristic (RSC) function graph', grid = True)
    RSCG.plot(grid = True)
    plt.ylabel('score')
    plt.xlabel('rank')
    return RSCG


# In[5]:
#calculate the cognitive diversities

def cognitiveD(raw):
    """
    This function calculate the cognitive diversities table between models from raw
    """
    df = pd.DataFrame({"SVM" :[abs(raw.SVM.sum()-raw.SVM.sum()), abs(raw.RF.sum()-raw.SVM.sum()), abs(raw.WLC.sum()-raw.SVM.sum()), abs(raw.GBT.sum()-raw.SVM.sum())],
                       "RF" :[abs(raw.SVM.sum()-raw.RF.sum()), abs(raw.RF.sum()-raw.RF.sum()) , abs(raw.WLC.sum()-raw.RF.sum()), abs(raw.GBT.sum()-raw.RF.sum())],
                       "WLC" :[abs(raw.SVM.sum()-raw.WLC.sum()), abs(raw.RF.sum()-raw.WLC.sum()), abs(raw.WLC.sum()-raw.WLC.sum()), abs(raw.GBT.sum()-raw.WLC.sum())],
                       "GBT": [abs(raw.SVM.sum()-raw.GBT.sum()), abs(raw.RF.sum()-raw.GBT.sum()), abs(raw.WLC.sum()-raw.GBT.sum()), abs(raw.GBT.sum()-raw.GBT.sum())]},
                       index = ["SVM", "RF","WLC","GBT"])
    df = df[["SVM", "RF","WLC","GBT"]]
    return df
                                                                                                                                   
#calculate the diveristy strength
def diversityS(raw):
    """
    This function calculate the diversity strength of four single model
    input raw should contain four models' normalized sorce
    """
    strength_A = (abs(raw.GBT.sum()-raw.SVM.sum()) + abs(raw.RF.sum()-raw.SVM.sum()) + abs(raw.WLC.sum()-raw.SVM.sum()))/3
    strength_B = (abs(raw.GBT.sum()-raw.RF.sum()) + abs(raw.WLC.sum()-raw.RF.sum()) + abs(raw.SVM.sum()-raw.RF.sum()))/3
    strength_C = (abs(raw.GBT.sum()-raw.WLC.sum()) + abs(raw.RF.sum()-raw.WLC.sum()) + abs(raw.SVM.sum()-raw.WLC.sum()))/3
    strength_D = (abs(raw.WLC.sum()-raw.GBT.sum()) + abs(raw.RF.sum()-raw.GBT.sum()) + abs(raw.SVM.sum()-raw.GBT.sum()))/3
    return strength_A, strength_B, strength_C, strength_D


#two models combination
def two_fusionW(first, second, weight = [1,1], score = True):
    """
    This function combination two model into one, then change the new model to the rank fucntion
    socre = True is score combiantion, False is the rank combination
    weight defult = [1,1] which is average combiantion
    """
    CMB = first.iloc[:,[0,2]] 
    CMB['score'] = (first.iloc[:,[-1]]*weight[0] + second.iloc[:,[-1]]*weight[1] )/(weight[0]+weight[1])
    if score:
        CMB = CMB.sort_values(by=['score'], ascending = False)
    else:
        CMB = CMB.sort_values(by=['score'], ascending = True)
        
    CMB.insert(len(CMB.columns),'rank',range(1,len(CMB)+1))
    return CMB


# In[6]:

#three models combination
def three_fusionW(first, second, third, weight =[1,1,1], score = True):
    """
    This function combination three model into one, then change the new model to the rank fucntion
    socre = True is score combiantion, False is the rank combination
    weight defult = [1,1] which is average combiantion
    """
    CMB = first.iloc[:,[0,2]]
    CMB['score'] = (first.iloc[:,[-1]]*weight[0] + second.iloc[:,[-1]]*weight[1]+third.iloc[:,[-1]]*weight[2] )/(weight[0]+weight[1]+weight[2])
    
    if score:
        CMB = CMB.sort_values(by=['score'], ascending = False)
    else:
        CMB = CMB.sort_values(by=['score'], ascending = True)
    CMB.insert(len(CMB.columns),'rank',range(1,len(CMB)+1))

    return CMB


# In[7]:

#four models combination
def four_fusionW(first, second, third, fourth, weight = [1,1,1,1], score = True):
    """
    This function combination four model into one, then change the new model to the rank fucntion
    socre = True is score combiantion, False is the rank combination
    weight defult = [1,1] which is average combiantion
    """
    CMB = first.iloc[:,[0,2]]
    CMB['score'] = (first.iloc[:,[-1]]*weight[0] + second.iloc[:,[-1]]*weight[1] +third.iloc[:,[-1]]*weight[2]                    + fourth.iloc[:,[-1]]*weight[3])/(weight[0]+weight[1]+weight[2]+weight[3])
    if score:
        CMB = CMB.sort_values(by=['score'], ascending = False)
    else:
        CMB = CMB.sort_values(by=['score'], ascending = True)
    CMB.insert(len(CMB.columns),'rank',range(1,len(CMB)+1))
    return CMB


# In[8]:



# In[9]:


def find_df(combRes, raw):
    """
    this fucntion compare the fusion model withe all singe model to find new data points 
    which are the truth and only find by the fusion model"""
    result = pd.DataFrame()
    for i in range(len(combRes)):
        if combRes['rank'][i] <= 357 and raw.loc[i][2] > 357 and raw.loc[i][3] > 357 and raw.loc[i][4] > 357 and raw.loc[i][5] > 357 and raw.loc[i][1]== 1:
            result = result.append(combRes.loc[i])
    print(result)


# In[ ]:

def find_dfop(combRes, raw):
    """
    this fucntion compare the fusion model withe all singe model to find new data points 
    which are the truth and only find by the fusion model"""
    result = pd.DataFrame()
    for i in range(len(combRes)):
        if combRes['rank'][i] > 357 and raw.loc[i][1]== 1 and raw.loc[i][2] <= 357 and raw.loc[i][3] <= 357 and raw.loc[i][4] <= 357 and raw.loc[i][5] <= 357:
            result = result.append(combRes.loc[i])
    print(result)



# In[ ]:





# In[ ]:




