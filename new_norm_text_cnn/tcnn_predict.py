#encoding=utf-8
from tcnn_model import TcnnConfig,TcnnModel
from tcnn_data_helper import get_word_and_pseg2id_dict
import tensorflow.contrib.keras as kr
import tensorflow as tf
import jieba.posseg as pseg
import os
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_model')  # 最佳验证结果保存路径
class TCNNModel:
    def __init__(self):
        #重建图
        tf.reset_default_graph()
        self.config = TcnnConfig()
        self.model = TcnnModel(self.config)
        _,self.word2id_dict,_,self.pseg2id_dict=get_word_and_pseg2id_dict(self.config.words_location,self.config.psegs_location)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
    def process_data(self,message):
        content_id_list=[]
        pseg_id_list=[]
        for w in pseg.cut(message):
            if w.word in self.word2id_dict:
                content_id_list.append(self.word2id_dict[w.word])
            else:
                content_id_list.append(self.word2id_dict['UNKNOW'])
            if w.flag[0] in ['v','n','l','y','r']:
                _key=w.flag[0]
            else:
                _key='x'
            pseg_id_list.append(self.pseg2id_dict[_key])
        input_x=kr.preprocessing.sequence.pad_sequences([content_id_list], self.config.sentence_length)
        input_x_pseg=kr.preprocessing.sequence.pad_sequences([pseg_id_list], self.config.sentence_length)
        return input_x,input_x_pseg
    def do_predict(self,message):
        self._input_x,self._input_x_pseg=self.process_data(message)
        feed_dict = {
            self.model.input_x:self._input_x,
            self.model.input_x_pseg:self._input_x_pseg,
            self.model.keep_prob:1.0
        }
        y_pred_cls = self.session.run(self.model.pred_label, feed_dict=feed_dict)
        print(y_pred_cls)
if __name__ == '__main__':
    tcnn_model = TCNNModel()
    for i in range(2):
        str=input("Enter your input:")
        tcnn_model.do_predict(str)
    # test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
    #              '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    # for i in test_demo:
    #     tcnn_model.do_predict(i)