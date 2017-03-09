import os
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features
from tfwrapper.utils.images import copy_image_folder
from tfwrapper.nets.pretrained import InceptionV3

def transform_data(src, dest):
	if not os.path.isdir(dest):
		os.mkdir(dest)

	for foldername in os.listdir(src):
		src_folder = os.path.join(src, foldername)

		if os.path.isdir(src_folder):
			dest_folder = os.path.join(dest, foldername)

			if not os.path.isdir(dest_folder):
				os.mkdir(dest_folder)

			copy_image_folder(src_folder, dest_folder, bw=True, h_flip=True, v_flip=True)

def main():
	transform_data('checkboxes', 'refined')

	inception = InceptionV3(graph_file='/Users/esten/ml/imagenet/classify_image_graph_def.pb')
	features = inception.extract_features_from_datastructure('refined', feature_file='checkbox_features.csv')

	dataset = Dataset(features=features)
	X, y, test_X, test_y, labels = dataset.getdata(balance=True, translate_labels=True, shuffle=True, onehot=True, split=True)
	print('Labels: ' + str(labels))
	graph = tf.Graph()
	with graph.as_default():
		with tf.Session(graph=graph) as sess:
			nn = SingleLayerNeuralNet(X.shape[1], 2, 1024, name='Checkboxes', sess=sess, graph=graph)
			nn.train(X, y, epochs=10, sess=sess)
			nn.save('models/checkboxes', sess=sess)

	nn = SingleLayerNeuralNet(X.shape[1], 2, 1024, name='Checkboxes')
	graph = tf.Graph()
	with graph.as_default():
		with tf.Session(graph=graph) as sess:
			nn.load('models/checkboxes', sess=sess)
			_, acc = nn.validate(X, y, sess=sess)
			print('Test accuracy: %d %%' % (acc*100))

if __name__ == '__main__':
	main()
