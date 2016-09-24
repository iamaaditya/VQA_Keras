'''
Original: https://github.com/JamesChuanggg/vqa-tf/blob/master/model_VQA.py
Modified: Added Val answers to the return of get_data_test
Aaditya Prakash

'''

from __future__ import print_function
import numpy as np
import h5py  as hf
import json


def get_train_data(args):

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(args.input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(args.input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(args.input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        #-----1~82460-----
        tem = hf.get('img_pos_train')
    # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,args.img_vec_dim)))

    return dataset, img_feature, train_data

def get_data_test(args):
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(args.input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(args.input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(args.input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        # -----1~82460-----
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
    # MC_answer_test
    tem = hf.get('MC_ans_test')
    test_data['MC_ans_test'] = np.array(tem)

    print('Normalizing image feature')
    if img_norm:
        tem =  np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,args.img_vec_dim)))


    # Added by Adi, make sure the ans_file is provided
    nb_data_test = len(test_data[u'question'])
    val_all_answers_dict = json.load(open(args.ans_file))
    val_answers = np.zeros(nb_data_test, dtype=np.int32)

    ans_to_ix = {v: k for k, v in dataset[u'ix_to_ans'].items()}
    count_of_not_found = 0
    for i in xrange(nb_data_test):
        qid = test_data[u'ques_id'][i]
        try : 
            val_ans_ix =int(ans_to_ix[utils.most_common(val_all_answers_dict[str(qid)])]) -1
        except KeyError:
            count_of_not_found += 1
            val_ans_ix = 480
        val_answers[i] = val_ans_ix
    print("Beware: " + str(count_of_not_found) + " number of val answers are not really correct")

    return dataset, img_feature, test_data
