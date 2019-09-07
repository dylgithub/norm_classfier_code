#encoding=utf-8
import pandas as pd
df=pd.read_csv('data/cut_pseg_train_data.csv')
content_list=list(df['content'])[:6000]
label_list=list(df['label'])[:6000]
content_list=[''.join(i.split('_sp_')) for i in content_list]
label2id_dict={'时尚':1, '家居':2, '娱乐':3, '财经':4, '体育':5, '房产':6, '游戏':7, '时政':8, '科技':9,'教育':10,}
label_list=[label2id_dict[i] for i in label_list]
train_content=content_list[:5001]
train_label=label_list[:5001]
dev_content=content_list[5001:5501]
dev_label=label_list[5001:5501]
test_content=content_list[5501:]
test_label=label_list[5501:]
print(set(train_label))
print(set(dev_label))
print(set(test_label))
df_tr=pd.DataFrame({'content':train_content,'label':train_label})
df_de=pd.DataFrame({'content':dev_content,'label':dev_label})
df_te=pd.DataFrame({'content':test_content,'label':test_label})
df_tr.to_csv('data/train.csv',encoding='utf-8',index=False)
df_de.to_csv('data/dev.csv',encoding='utf-8',index=False)
df_te.to_csv('data/test.csv',encoding='utf-8',index=False)