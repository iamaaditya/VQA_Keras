''' Author: Aaditya Prakash '''

import json
import numpy as np
from keras.utils import np_utils

# custom
from utils.get_data import get_test_data, get_train_data
from utils.arguments import get_arguments


def main():
    
    args = get_arguments()
    print(args)
    np.random.seed(args.seed)


    dataset, train_img_feature, train_data = get_train_data(args)
    dataset, test_img_feature,  test_data, val_answers = get_test_data(args)

    train_X = [train_data[u'question'], train_img_feature]
    train_Y = np_utils.to_categorical(train_data[u'answers'], args.nb_classes)

    test_X = [test_data[u'question'], test_img_feature]
    test_Y = np_utils.to_categorical(val_answers, args.nb_classes)


    model_name = importlib.import_module("models."+args.model)
    model = model_name.model(args)
    model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])
    model.summary() # prints model layers with weights

    history = model.fit(train_X, train_Y, batch_size = args.batch_size, nb_epoch=args.nb_epoch, validation_data=(test_X, test_Y))

    return history.history

if __name__ == "__main__":
    main()


