#encoding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tcnn_data_helper
#用于产生权重向量
def W_generate(shape):
    inital=tf.truncated_normal(shape)
    return tf.Variable(inital)
#用于产生偏置值
def bias_generate(shape):
    inital=tf.constant(0.1,shape=[shape])
    return tf.Variable(inital)
'''
该类用于完成cnn模型的正向传播，返回一个正向传播的结果
sentence_length：每个句子的长度
vector_size：每个单词向量的维度
num_classes：分类的类别数
num_filters：卷积核的个数
filter_hs：所使用的各个高度的卷积核
'''
class TcnnModel():
    def __init__(self,config):
        self.config = config
        # 四个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sentence_length], name='input_x')
        self.input_x_pseg = tf.placeholder(tf.int32, [None, self.config.sentence_length], name='input_x_pseg')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.tcnn()
    #建立模型，执行正向传播，返回正向传播得到的值
    def tcnn(self):
        # 嵌入层、词向量映射
        word_vecs, sen_index, psegs_vec, sen_pseg_index, one_hot_label = tcnn_data_helper.process_file(
            self.config.file_location, self.config.w2v_model_location, self.config.words_location, self.config.psegs_location,
            False, self.config.sentence_length, self.config.vector_size, self.config.pseg_size)
        with tf.device('/cpu:0'):
            input_x_content = tf.nn.embedding_lookup(word_vecs,self.input_x)
            input_x_pseg = tf.nn.embedding_lookup(psegs_vec,self.input_x_pseg)
            _input_x = tf.concat((input_x_content, input_x_pseg), axis=-1)
            x = tf.expand_dims(_input_x, -1)

        #随后进行卷积池化操作
        print('begin conve_pool')
        pool_outputs = []
        for filter_h in self.config.filter_hs:
            with tf.variable_scope('conv_pool_{}'.format(filter_h)):
                #注意这里不加偏置值，默认的是加
                conv=tf.layers.conv2d(x,filters=self.config.num_filters,kernel_size=(filter_h,self.config.vector_size+self.config.pseg_size)
                                      ,kernel_initializer=tf.glorot_normal_initializer(),use_bias=False,activation=tf.nn.relu,name='conve')
                pooled=tf.nn.max_pool(conv,ksize=[1,self.config.sentence_length-filter_h+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pool_outputs.append(pooled)

        #全连接层操作
        print('begin full_connection')
        with tf.name_scope("full_connection"):
            h_pool = tf.concat(pool_outputs, 3)  # 把3种大小卷积核卷积池化之后的值进行连接
            num_filters_total = self.config.num_filters * len(self.config.filter_hs)
            # 因为随后要经过一个全连接层得到与类别种类相同的输出，而全连接接收的参数是二维的，所以进行维度转换
            h_pool_flaten = tf.reshape(h_pool, [-1, num_filters_total])
            h_drop = tf.nn.dropout(h_pool_flaten, self.keep_prob)
            #分类器
            W = tf.Variable(tf.truncated_normal([num_filters_total, self.config.num_classes]))
            self.l2_loss = tf.nn.l2_loss(W)
            b = tf.Variable(tf.constant(0., shape=[self.config.num_classes]), name="b")
            self.y_pred = tf.nn.xw_plus_b(h_drop, W, b, name="scores")  # wx+b
            #预测类别
            self.pred_label = tf.argmax(tf.nn.softmax(self.y_pred), 1)
        with tf.name_scope('optimize'):
            # 计算损失值,会自动计算softmax故必须传入没计算之前的，否则相当于计算了两次
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.y_pred)
            #self.losses = tf.reduce_mean(cross_entropy)
            # 加L2正则化
            self.losses = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda *self.l2_loss
            #优化器
            self.optim = tf.train.AdamOptimizer(self.config.init_learning_rate).minimize(self.losses)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.pred_label, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
