''' Author: Aaditya Prakash '''

import argparse

def get_arguments():

	parser = argparse.ArgumentParser()
	# model
	parser.add_argument('-model'                  , type=str   , default='simple_mlp')
	parser.add_argument('-num_hidden_units_mlp'   , type=int   , default=1024)
	parser.add_argument('-num_hidden_units_lstm'  , type=int   , default=512)
	parser.add_argument('-num_hidden_layers_mlp'  , type=int   , default=3)
	parser.add_argument('-num_hidden_layers_lstm' , type=int   , default=1)
	parser.add_argument('-dropout'                , type=float , default=0.5)
	parser.add_argument('-activation_1'           , type=str   , default='tanh')
	parser.add_argument('-activation_2'           , type=str   , default='relu')

	# training
	parser.add_argument('-seed'                   , type=int   , default=1337)
	parser.add_argument('-optimizer'              , type=str   , default='rmsprop')
	parser.add_argument('-nb_epoch'               , type=int   , default=300)
	parser.add_argument('-nb_iter'                , type=int   , default=200000)
	parser.add_argument('-model_save_interval'    , type=int   , default=19)
	parser.add_argument('-batch_size'             , type=int   , default=128)

	# language features
	parser.add_argument('-word_vector'            , type=str   , default='glove')
	parser.add_argument('-word_emb_dim'           , type=int   , default=300)
	parser.add_argument('-vocabulary_size'        , type=int   , default=12603)
	parser.add_argument('-max_ques_length'        , type=int   , default=26)
	parser.add_argument('-data_type'              , type=str   , default='TRAIN')

	# image features
	parser.add_argument('-img_vec_dim'            , type=int   , default=2048)
	parser.add_argument('-img_features'           , type=str   , default='resnet')
	parser.add_argument('-img_normalize'          , type=int   , default=0)

	# evaluations
	parser.add_argument('-nb_classes'             , type=int   , default=1000)
	parser.add_argument('-class_activation'       , type=str   , default='softmax')
	parser.add_argument('-loss'                   , type=str   , default='categorical_crossentropy')
	parser.add_argument('-save_folder'            , type=str   , default='')

	# data
	parser.add_argument('-ans_file'               , type=str   , default='data/val_all_answers_dict.json')
	parser.add_argument('-input_json'             , type=str   , default='data/data_prepro.json')
	parser.add_argument('-input_img_h5'           , type=str   , default='data/data_img.h5')
	parser.add_argument('-input_ques_h5'          , type=str   , default='data/data_prepro.h5')


	return parser.parse_args()
