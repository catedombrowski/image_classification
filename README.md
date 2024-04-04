# Convolutional Neural Network

To achieve 99% accuracy with our CNN, we adjusted several parameters of our model. Below is a brief summary of the model architecture and the parameters used: 

**Input Layer**

The architecture begins with an input convolutional layer that accommodates images with dimensions of 28x28 pixels.

**Convolution Layers**

Following this, our model includes two additional convolutional layers, that has 32 filters/kernels of size (3, 3), each utilizing the ReLU activation function to introduce non-linearity.
Both convolutional layers are followed by max-pooling layers (MaxPooling2D) with pool size (2, 2) to down-sample the spatial dimensions. There is another convolutional layer with 128 filters of size (3, 3) and ReLU activation.

**Flatten Layer** 

The output of the last convolutional layer is flattened into a 1D array to be fed into fully connected layers.

**Fully Connected Layers** 

There is a fully connected dense layer (Dense) with 128 units and ReLU activation. A dropout layer (Dropout) is added with a dropout rate of 0.5 to reduce overfitting. The output layer consists of 10 units (equal to the number of classes) with softmax activation for multi-class classification.


**Training Parameters**
The model is optimized using the Adam optimizer with a learning rate of 0.001, a careful choice that balances the need for efficient learning without risking significant overfitting.
Our training regimen involves a batch size of 64 and a validation split of 0.2 to monitor the model's performance on unseen data during the learning process.

## CNN Accuracy
Training Accuracy - **98.95%**

Validation Accuracy - **99.15%**

Training Accuracy After Retraining With Entire Data - **99.15%**

Test Accuracy - **99.09%**

# Transformer Model

To achieve optimal performance with our transformer model, we engineered its architecture and hyperparameters. The model is structured to process input images divided into blocks, each with RGB values. Below is a summary of the model architecture and parameters used: 

**Class Token Layer:**

A crucial component of our model is the inclusion of a class token to aggregate global information for classification. This token is connected to the sequence of block vectors before being processed by the transformer layers.

**Number of Blocks:**

We used a 4X4 structure of blocks each having 7X7 pixels in them.
The initial layer transforms these blocks into vectors with a hidden dimension, followed by the addition of learned positional embeddings to retain spatial information.

**Attention Heads & Layers**

The core of our model consists of transformer layers, each featuring multi-head self-attention with a specific number of heads. 
In our case 6 heads of attention and 6 layers of attention blocks provided the best results. We tried different combinantions of both values ranging from 2 to 8.

**Dimension of the Embeddings (Hidden_Dim)**

Dimension of Embeddings was set to 64. We tried values ranging from 32 to 512 but got the best results with 64.

**Dense Layers**

Following the attention mechanism, we employ a feed-forward network with a hidden dimension activated by the GeLU function. We tried ReLU as well but the results were similar.
Dimension of the feedforward network's hidden layer (MLP_Dim) was set equal to that of the embeddings, i.e. 64.
Dropout layers are strategically placed to prevent overfitting. We tried different values of dropout ranging from 0 to 30% but ended up using 10%.

**Dimension of the Key Vector**

This was set to hidden_dim/num_heads.

**Final Output**

The output of the transformer layers is normalized, and the class token is extracted to make the final classification. This token is passed through a dense layer with Softmax activation to produce probabilities for each class.

**Training Parameters**

The model is compiled with the Adam optimizer, using crossentropy as the loss function.
We train for 50 epochs with a batch size of 40 and 20% validation split.

## Transformer Accuracy

Training Accuracy - **99.52%**

Validation Accuracy - **98.43%**

Training Accuracy After Retraining With Entire Data - **99.74%**

Test Accuracy - **98.43%**
