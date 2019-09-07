#encoding=utf-8
from datahelper import Batcher,read_data,batch
from elmo.model import ELMOClassifierModel
import os
import tensorflow as tf
import time
import pandas as pd
from sklearn.metrics import classification_report
os.environ['CUDA_VISIBLE_DEVICES']='1'

def train(train_data,valid_data,model,epochs=10):
    lowest_loss=float('inf')
    best_step=0
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    graph=model.graph
    with graph.as_default():
        with tf.Session(config=config,graph=graph) as sess:
            ckpt=tf.train.get_checkpoint_state(model.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                tf.global_variables_initializer().run()
            global_step=model.global_step.eval()
            summary_writer=tf.summary.FileWriter(model.log_dir,graph=graph)
            for i,(data_x,data_y) in enumerate(batch(train_data[0],train_data[1],epochs=epochs)):
                if len(data_x)!=len(data_y):
                    continue
                t0=time.clock()
                feed_dict=model.get_feed_dict(data_x,data_y,True)
                step,summaries,loss,accuracy,neg_f1,pos_f1,_=sess.run([
                    model.global_step,
                    model.summary_op,
                    model.loss,
                    model.accuracy,
                    model.neg_f1,
                    model.pos_f1,
                    model.train_op
                ],feed_dict=feed_dict)
                summary_writer.add_summary(summaries,global_step=step+global_step)
                if loss<lowest_loss:
                    lowest_loss=loss
                    best_step=step+global_step
                time_cost=time.clock()-t0
                if step%100==0:
                    print(
                        'step {},loss={:.4f},time={:.4f},acc={:.4f},neg_f1={:.4f},pos_f1={:.4f}'.format(
                            step,loss,
                            time_cost,
                            accuracy,neg_f1,pos_f1
                        )
                    )
                if step%1000==0:
                    print('checkpoint......')
                    model.saver.save(sess,model.checkpoint_path,global_step=step+global_step)
                    print('checkpoint done!')
                    print('evaluate---------------------')

                    print('evaluate done!++++++++++++++++')
    print('model is trained!!, lowest_loss={} at step={}'.format(lowest_loss,best_step))
def evaluate(sess,model,dataset):
    predicts=[]
    labels=[]
    pos_probs=[]
    neg_probs=[]
    for x,y in batch(dataset[0],dataset[1],epochs=1):
        labels.extend(y)
        feed_dict=model.get_feed_dict(x,y,is_training=False)
        predict,prob=sess.run([model.predicts,model.probs],feed_dict=feed_dict)
        predicts.extend(predict)
        pos_probs.extend(prob[:,0])
        pos_probs.extend(prob[:,1])
    df=pd.DataFrame(
        {'predictions':predicts,'labels':labels,'pos_probs':pos_probs,'neg_probs':neg_probs}
    )
    print(classification_report(labels, predicts))
    return df
if __name__ == '__main__':
    train_data_file='data/aug_clean_seg_train.txt'
    test_data_file='data/clean_seg_test.txt'
    char_vocab_file='config/elmo_char_vocab.txt'
    seg_vocab_file='config/elmo_seg_vocab.txt'
    options_file='config/options.json'
    lm_weights_file='config/lm_weights.hdf5'
    epochs=5
    max_word_char_len=10
    batch_size=128
    n_dims=256
    train_x,train_y=read_data(train_data_file)
    test_x,test_y=read_data(test_data_file)
    bacher=Batcher(seg_vocab_file,char_vocab_file,max_word_char_len)
    train_x_ids=bacher.batch_sentences(train_x)
    test_x_ids=bacher.batch_sentences(test_x)
    graph=tf.get_default_graph()
    elmo_model=ELMOClassifierModel(n_dims,max_word_char_len,graph,options_file,lm_weights_file,batch_size)
    train((train_x_ids,train_y),(test_x_ids,test_y),elmo_model)