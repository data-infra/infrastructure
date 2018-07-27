
# 只是读取和生成文件,并不能识别网络是否正确
from tensorflow.python.platform import gfile
import tensorflow as tf
model = 'quantized_graph.pb'
model = 'frozen_graph.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
print(graph_def)

#with tf.gfile.FastGFile('expert-graph.pb', mode='wb') as f:
 #   f.write(graph_def.SerializeToString())

