#encoding=utf-8
import tensorflow as tf
import time
import os
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import tcnn_data_helper
from tcnn_model2 import TcnnModel
from tcnn_config import Config
from new_norm_text_cnn.calculate_prf import cal_mac_prf
#网络模型的保存文件夹
model_save_location="checkpoints/textcnn"
#网络模型保存的相对路径
save_path = os.path.join(model_save_location, 'best_model')
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
def train_val_test_model(train_data,val_data,test_data,model,tcnn_config):
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    #删除原来已存在的tensorboard文件
    else:
        file_list=os.listdir(tensorboard_dir)
        if len(file_list)>0:
            for file in file_list:
                os.remove(os.path.join(tensorboard_dir,file))
    tf.summary.scalar("loss", model.losses)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver，用以保存模型
    saver = tf.train.Saver()
    if not os.path.exists(model_save_location):
        os.makedirs(model_save_location)
    print('Training and Testing...')
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer.add_graph(sess.graph)
        #批量获得数据
        for epoch in range(tcnn_config.num_epochs):
            batch_train = tcnn_data_helper.batch_iter(train_data[0],train_data[1],train_data[2],tcnn_config.batch_size)
            total_batch = 0
            for x_batch,x_pseg_batch,y_batch in batch_train:
                total_batch+=1
                feed_dict={model.input_x:x_batch,model.input_x_pseg:x_pseg_batch,model.input_y:y_batch,model.keep_prob:tcnn_config.keep_prob}
                if total_batch%tcnn_config.save_per_batch==0:
                    summary_str = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(summary_str,total_batch)  # 将summary 写入文件
                if total_batch%tcnn_config.print_per_batch == 0:
                    train_accuracy = model.accuracy.eval(feed_dict=feed_dict)
                    print("Epoch %d:Step %d accuracy is %f" % (epoch+1,total_batch,train_accuracy))
                sess.run(model.optim, feed_dict=feed_dict)
            here2(sess,val_data)
        saver.save(sess,save_path)
        time_dif = get_time_dif(start_time)
        print("train usage:", time_dif)
        do_tes(sess,test_data)
def do_evaluate(sess,val_data):
    # 没训练完一个批次都用验证集验证模型的性能
    start_time = time.time()
    batch_train = tcnn_data_helper.batch_iter(val_data[0], val_data[1], val_data[2], tcnn_config.batch_size)
    all_val_pred = []
    for x_batch, x_pseg_batch, y_batch in batch_train:
        test_pred = sess.run(model.pred_label, feed_dict={model.input_x: x_batch, model.input_x_pseg: x_pseg_batch,
                                                          model.input_y: y_batch, model.keep_prob: 1.0})
        all_val_pred.extend(test_pred)
    val_label = np.argmax(val_data[2], 1)
    time_dif = get_time_dif(start_time)
    print('do eval usage:',time_dif)
    final_precision, final_recall, final_f1_score=cal_mac_prf(val_label,all_val_pred)
    print("mac precision is:",final_precision,"mac recall is:",final_recall,"mac f1-score is:",final_f1_score)
    # print(classification_report(val_label,all_val_pred))
def here2(sess,val_data):
    # 没训练完一个批次都用验证集验证模型的性能
    start_time = time.time()
    test_pred,loss = sess.run([model.pred_label,model.losses], feed_dict={model.input_x: val_data[0], model.input_x_pseg: val_data[1],
                                                          model.input_y: val_data[2], model.keep_prob: 1.0})
    val_label = np.argmax(val_data[2], 1)
    time_dif = get_time_dif(start_time)
    print('do eval usage:',time_dif)
    final_precision, final_recall, final_f1_score=cal_mac_prf(val_label,test_pred)
    print("mac precision is:",final_precision,"mac recall is:",final_recall,"mac f1-score is:",final_f1_score,'loss is:',loss)
    # print(classification_report(val_label,all_val_pred))
def do_tes(sess,test_data):
    # 训练完之后通过测试集测试模型
    start_time = time.time()
    batch_train = tcnn_data_helper.batch_iter(test_data[0], test_data[1],test_data[2],tcnn_config.batch_size)
    all_test_pred=[]
    for x_batch, x_pseg_batch, y_batch in batch_train:
        test_pred=sess.run(model.pred_label,feed_dict={model.input_x:x_batch,model.input_x_pseg:x_pseg_batch,model.input_y:y_batch,model.keep_prob:1.0})
        all_test_pred.extend(test_pred)
    test_label = np.argmax(test_data[2],1)
    time_dif = get_time_dif(start_time)
    print("test usage:", time_dif)
    #要和id所代表的类别标签顺序相同
    categories=['教育','时尚', '家居', '娱乐', '财经', '体育','房产','游戏','时政','科技']
    # 评估
    print("Precision, Recall and F1-Score...")
    print(classification_report(test_label,all_test_pred,target_names=categories))
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = confusion_matrix(test_label,all_test_pred)
    print(cm)
if __name__ == '__main__':
    tcnn_config=Config()
    model=TcnnModel(tcnn_config)
    # 获得训练数据和测试数据
    start_time = time.time()
    _, sen_index, _, sen_pseg_index, one_hot_label = tcnn_data_helper.process_file(
        tcnn_config.file_location, tcnn_config.w2v_model_location, tcnn_config.words_location,
        tcnn_config.psegs_location,
        False, tcnn_config.sentence_length, tcnn_config.vector_size, tcnn_config.pseg_size)
    X_train, X_test, X_pseg_train, X_pseg_test, y_train, y_test = train_test_split(sen_index, sen_pseg_index,
                                                                                   one_hot_label, test_size=0.1)
    time_dif = get_time_dif(start_time)
    print("load data usage:", time_dif)
    train_val_test_model((X_train, X_pseg_train, y_train),(X_test, X_pseg_test, y_test),(X_test, X_pseg_test, y_test),model,tcnn_config)
