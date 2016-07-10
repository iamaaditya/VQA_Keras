from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout, Highway
from keras.layers import LSTM, Merge, Dense, Embedding


def model(args):

    # Image model
    model_image = Sequential()
    model_image.add(Reshape((args.img_vec_dim,), input_shape=(args.img_vec_dim,)))

    # Language Model
    model_language = Sequential()
    model_language.add(Embedding(args.vocabulary_size, args.word_emb_dim, input_length=args.max_ques_length))
    model_language.add(LSTM(args.num_hidden_units_lstm, return_sequences=True, input_shape=(args.max_ques_length, args.word_emb_dim)))
    model_language.add(LSTM(args.num_hidden_units_lstm, return_sequences=True))
    model_language.add(LSTM(args.num_hidden_units_lstm, return_sequences=False))

    # combined model
    model = Sequential()
    model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))


    for i in xrange(args.num_hidden_layers_mlp):
        model.add(Dense(args.num_hidden_units_mlp))
        model.add(Dropout(args.dropout))

    model.add(Dense(args.nb_classes))
    model.add(Activation(args.class_activation))

    return model
