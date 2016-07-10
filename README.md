# Modular & Simple approach to VQA in Keras

## Dataset

Use the [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN) link to obtain the data. You will need to place 
the following three files in the data directory --
 
	* 'data/data_prepro.json'
	* 'data/data_img.h5'
	* 'data/data_prepro.h5'


## Models

Write the keras models and place them inside 'models' directory. See the two models provided. 
simple_mlp is a very basic model for VQA whereas  'DeeperLSTM' is based on [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN)

To call your own model pass the argument -model and name of the file. For e.g

`python train -model DeeperLSTM`

