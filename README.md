# cnn-project

Convolutional neural network project done as a project assignment for university subject Projekt under mentorship of prof. dr. sc. Marko Subašić at Faculty of Electrical Engineering and Computing, Zagreb (FER). 
Network is trained to enhance colors of an image. Dataset is not included in this repository. 

Network architecture: 
  Convolutional layer 9x9 followed by three residual blocks, each of the blocks consists of two 3x3 layers alternated with batch normalization layers. After the three residual blocks, two additional convolutional layers are used of size 3x3 and one final layer with 9x9 filters. All layers have 64 channels and are followed by ReLU activation function except the last one where a tanh function is applied to the outputs. 

To train the model, run train_model.py. 

Parts of this project are taken from: https://arxiv.org/pdf/1704.02470.pdf.
