import pandas as pd
import numpoy as np
import cv2
import tensorflow as tf, sys

# provide the path of the test image
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, the retrained_labels file contains the labels to tf_files/output_labels.txt
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# output_graph.pb stores a representation of our Inception v3 network with the final layer trained to our categories
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # the image is now fed to the graph to find accuracy scores for all the categories used to classify the image.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    pred_list = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    #print the result
    for node_id in pred_list:
        category = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (category, score))