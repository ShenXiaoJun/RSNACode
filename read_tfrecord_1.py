# encoding:utf-8
import tensorflow as tf

# 创建一个reader来读取TFRecord文件中的样例
tfreader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表，在7.3.2小节中将更加详细的介绍
# tf.train.string_input_producer函数
tf_record_file  = "/home/shenxj/RSNA/code/test/dicom100.tfrecords"
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
        'pic_pixel':   tf.FixedLenFeature([], tf.int64),

        'label_topx_0': tf.FixedLenFeature([], tf.int64),
        'label_topy_0': tf.FixedLenFeature([], tf.int64),
        'label_lowx_0': tf.FixedLenFeature([], tf.int64),
        'label_lowy_0': tf.FixedLenFeature([], tf.int64),
        'label_class_0':tf.FixedLenFeature([], tf.int64),

        'label_topx_1': tf.FixedLenFeature([], tf.int64),
        'label_topy_1': tf.FixedLenFeature([], tf.int64),
        'label_lowx_1': tf.FixedLenFeature([], tf.int64),
        'label_lowy_1': tf.FixedLenFeature([], tf.int64),
        'label_class_1':tf.FixedLenFeature([], tf.int64),

        'label_topx_2': tf.FixedLenFeature([], tf.int64),
        'label_topy_2': tf.FixedLenFeature([], tf.int64),
        'label_lowx_2': tf.FixedLenFeature([], tf.int64),
        'label_lowy_2': tf.FixedLenFeature([], tf.int64),
        'label_class_2':tf.FixedLenFeature([], tf.int64),

        'label_topx_3': tf.FixedLenFeature([], tf.int64),
        'label_topy_3': tf.FixedLenFeature([], tf.int64),
        'label_lowx_3': tf.FixedLenFeature([], tf.int64),
        'label_lowy_3': tf.FixedLenFeature([], tf.int64),
        'label_class_3':tf.FixedLenFeature([], tf.int64),

        'image_raw':    tf.FixedLenFeature([], tf.string),
    }
)

pic_pixels = tf.cast(features['pic_pixel'], tf.int32)
label_topxs_0 = tf.cast(features['label_topx_0'], tf.int32)
label_topys_0 = tf.cast(features['label_topy_0'], tf.int32)
label_lowxs_0 = tf.cast(features['label_lowx_0'], tf.int32)
label_lowys_0 = tf.cast(features['label_lowy_0'], tf.int32)
label_classes_0 = tf.cast(features['label_class_0'], tf.int32)

label_topxs_1 = tf.cast(features['label_topx_1'], tf.int32)
label_topys_1 = tf.cast(features['label_topy_1'], tf.int32)
label_lowxs_1 = tf.cast(features['label_lowx_1'], tf.int32)
label_lowys_1 = tf.cast(features['label_lowy_1'], tf.int32)
label_classes_1 = tf.cast(features['label_class_1'], tf.int32)

label_topxs_2 = tf.cast(features['label_topx_2'], tf.int32)
label_topys_2 = tf.cast(features['label_topy_2'], tf.int32)
label_lowxs_2 = tf.cast(features['label_lowx_2'], tf.int32)
label_lowys_2 = tf.cast(features['label_lowy_2'], tf.int32)
label_classes_2 = tf.cast(features['label_class_2'], tf.int32)

label_topxs_3 = tf.cast(features['label_topx_3'], tf.int32)
label_topys_3 = tf.cast(features['label_topy_3'], tf.int32)
label_lowxs_3 = tf.cast(features['label_lowx_3'], tf.int32)
label_lowys_3 = tf.cast(features['label_lowy_3'], tf.int32)
label_classes_3 = tf.cast(features['label_class_3'], tf.int32)

# tf.records_raw可以将字符串解析成图像对应的像素数组
image_raws = tf.decode_raw(features['image_raw'], tf.uint8)

sess = tf.Session()
#启动多线程处理输入数据，7.3节将更加详细的介绍tensorflow多线程处理
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行可以读取TFRecord文件中的一个样例。当所有样例都读完之后，在此样例中程序
# 会在重头读取
for i in range(100):
    pic_pixel,\
    label_topx_0, label_topy_0, label_lowx_0, label_lowy_0, label_class_0,\
    label_topx_1, label_topy_1, label_lowx_1, label_lowy_1, label_class_1,\
    label_topx_2, label_topy_2, label_lowx_2, label_lowy_2, label_class_2,\
    label_topx_3, label_topy_3, label_lowx_3, label_lowy_3, label_class_3,\
    image_raw = sess.run([pic_pixels,
                                    label_topxs_0, label_topys_0, label_lowxs_0, label_lowys_0, label_classes_0,
                                    label_topxs_1, label_topys_1, label_lowxs_1, label_lowys_1, label_classes_1,
                                    label_topxs_2, label_topys_2, label_lowxs_2, label_lowys_2, label_classes_2,
                                    label_topxs_3, label_topys_3, label_lowxs_3, label_lowys_3, label_classes_3,
                                    image_raws])
    print("-debug-label_topx_0:%d, label_topy_0:%d, label_lowx_0:%d, label_lowy_0:%d, label_class_0:%d"
          %(label_topx_0, label_topy_0, label_lowx_0, label_lowy_0, label_class_0))
    print("-debug-label_topx_1:%d, label_topy_1:%d, label_lowx_1:%d, label_lowy_1:%d, label_class_1:%d"
          % (label_topx_1, label_topy_1, label_lowx_1, label_lowy_1, label_class_1))
    print("-debug-label_topx_2:%d, label_topy_2:%d, label_lowx_2:%d, label_lowy_2:%d, label_class_2:%d"
          % (label_topx_2, label_topy_2, label_lowx_2, label_lowy_2, label_class_2))
    print("-debug-label_topx_3:%d, label_topy_3:%d, label_lowx_3:%d, label_lowy_3:%d, label_class_3:%d"
          % (label_topx_3, label_topy_3, label_lowx_3, label_lowy_3, label_class_3))