#encoding=utf-8
from sklearn.metrics import classification_report
#多分类时计算宏平均的precision,recall以及f1-score
def cal_mac_prf(label_list,predict_list):
    label_set=set(label_list)
    _len=len(label_list)
    precision_list=[]
    recall_list=[]
    f1_score_list=[]
    for label in label_set:
        #所有预测为正的数量，计算precision的分母
        predict_true_num=0
        #真实为正的数量，计算recall的分母
        label_true_num=0
        #真实label和预测值同为pos,计算precision以及recall的分子
        pre_pos_and_label_pos=0
        for i in range(_len):
            if label_list[i]==label:
                label_true_num+=1
            if predict_list[i]==label:
                predict_true_num+=1
            if label_list[i]==label and predict_list[i]==label:
                pre_pos_and_label_pos+=1
        precision=pre_pos_and_label_pos/predict_true_num
        recall=pre_pos_and_label_pos/label_true_num
        f1_score=2*precision*recall/(precision+recall)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
    final_precision=round(sum(precision_list)/len(label_set),4)
    final_recall=round(sum(recall_list)/len(label_set),4)
    final_f1_score=round(sum(f1_score_list)/len(label_set),4)
    return final_precision,final_recall,final_f1_score
if __name__ == '__main__':
    label_list=[6,6,9,9,3,3,4,4,5,5]
    predict_list=[6,6,9,6,4,3,4,4,5,6]
    final_precision, final_recall, final_f1_score=cal_mac_prf(label_list,predict_list)
    print(final_precision, final_recall, final_f1_score)
    print(classification_report(label_list,predict_list))