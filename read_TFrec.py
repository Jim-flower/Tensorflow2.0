import tensorflow as tf
do = tf.data.TFRecordDataset('output.tfrec')
di = tf.data.TFRecordDataset('output.tfrec')
###反序列化
def parse(x):
  result = tf.io.parse_tensor(x, out_type=tf.float32)
  result = tf.reshape(result, [32, 32, 3])
  return result
def parse_o(x):
  result = tf.io.parse_tensor(x, out_type=tf.float32)
  result = tf.reshape(result, [32, 32, 1])
  return result

if __name__ == '__main__':

    do = do.map(parse_o,)
    di = di.map(parse,)
    da = tf.data.Dataset.zip((di, do))
    print(do,"\n",di,"\n",da)