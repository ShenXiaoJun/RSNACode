# encoding:utf-8
import tensorflow as tf

# 创建一个reader来读取TFRecord文件中的样例
tfreader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表，在7.3.2小节中将更加详细的介绍
# tf.train.string_input_producer函数
tf_record_file  = "/home/shenxj/RSNA/code/test/dicom100_nolabel_test.tfrecords"
dicom_queue = tf.train.string_input_producer([tf_record_file])

# 从文件中读出一个样例。也可是使用read_up_to函数一次性读取多个样例
_, serialized_example = tfreader.read(dicom_queue)
# 解析读入的一个样例。如果需要解析多个样例，可以用parser-example函数
features = tf.parse_single_example(
    serialized_example,
    features={
        # tensorflow提供两种不同的属性解析方法。一种方法是tf.FixedLenFeature,
        # 这种方法解析的结果为一个Tensor。另一种方法是tf.VarLenFeature,这种方法
        # 得到的解析结果为SpareTensor，用于处理稀疏数据。这里解析数据的格式需要和
        # 写入程序的格式一致。
        #'image_raw': tf.FixedLenFeature([], tf.string),
        #'pixel': tf.FixedLenFeature([], tf.int64),
        'target':   tf.FixedLenFeature([], tf.int64),
        'image_raw':    tf.FixedLenFeature([], tf.string),
    }
)

targets = tf.cast(features['target'], tf.int32)

# tf.records_raw可以将字符串解析成图像对应的像素数组
image_raws = tf.decode_raw(features['image_raw'], tf.uint8)

sess = tf.Session()
#启动多线程处理输入数据，7.3节将更加详细的介绍tensorflow多线程处理
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行可以读取TFRecord文件中的一个样例。当所有样例都读完之后，在此样例中程序
# 会在重头读取
for i in range(100):
    target,image_raw = sess.run([targets, image_raws])
    print("-debug-target:%d"%(target))