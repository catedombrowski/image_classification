#!/usr/bin/env python
# coding: utf-8

# In[1]:


import anvil.server 
import anvil.media
anvil.server.connect("server_WFII5EQUQ2P3ATWZXYBO3Z5P-3QJNR3G5GLA4DNFA")

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as plt
from io import BytesIO
from tensorflow import keras
from PIL import Image


# In[2]:


# fn will read in csv; load our trained models; predict from each model the classification; then output image and its classifications
@anvil.server.callable
def is_normalized(data):
    # Check if the data is already normalized (values between 0 and 1)
    return np.all(data <= 1.0) and np.all(data >= 0)

@anvil.server.callable
def process_and_classify_image_20(file):
    with anvil.media.TempFile(file) as filename:
        # Read the CSV file containing pixel intensities
        pixel_data = pd.read_csv(filename, header=None).values #send media object to python to read_csv
        pixel_data = pixel_data.astype(np.float32).reshape(28, 28)

        if not is_normalized(pixel_data):
            pixel_data = pixel_data / 255.0  # Normalize to [0, 1]

        # Create an image from the pixel data
        image = Image.fromarray(pixel_data*255)

        # Convert the image to 'L' (grayscale) mode
        image = image.convert('L')

        # Save the image to a buffer
        buf = BytesIO()
        image.save(buf, format='png')
        buf.seek(0)

        # Create a bytes-like object from the buffer's contents
        image_bytes = buf.getvalue()

        # Create a media object from the bytes-like object
        image_media = anvil.BlobMedia(content_type='image/png', content=image_bytes, name='digit.png')
        
        
        
        pixel_data_ravel = np.zeros((1,16,49))
        ind = 0
        #pixel_data = np.array(pixel_data[0].reshape(28, 28))
        # Iterate over 28x28 pixel blocks
        for row in range(4):
            for col in range(4):
                # Extract the current 4x4 pixel block
                pixel_data_ravel[0, ind, :] = pixel_data[(row*7):((row+1)*7), (col*7):((col+1)*7)].ravel()
                ind += 1

        pos_feed_classify = np.array([list(range(16))])
        
        @tf.keras.saving.register_keras_serializable(package="MyLayers")
        class ClassToken(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(ClassToken, self).__init__(**kwargs)  # Pass the keyword arguments to the superclass

            def build(self, input_shape):
                w_init = tf.random_normal_initializer()
                self.w = tf.Variable(
                    initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype='float32'),
                    trainable=True,
                )

            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]
                hidden_dim = self.w.shape[-1]
                cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
                cls = tf.cast(cls, dtype=inputs.dtype)
                return cls
        custom_objects = {'ClassToken': ClassToken}
        
        # Load the CNN model and pass the image for classification
        cnn_model = tf.keras.models.load_model('CNN_OptiProj_Team5_Model_V5.h5')
        cnn_prediction = cnn_model.predict(pixel_data.reshape(-1, 28, 28, 1).astype('float32'))
        cnn_classification = str(np.argmax(cnn_prediction, axis=1)[0])

        # Load the ViT model and pass the image for classification
        vit_model = tf.keras.models.load_model('ViT_OptiProj_Team5_Model_V7.h5', custom_objects=custom_objects)
#         with tf.keras.utils.custom_object_scope(custom_objects):
#             vit_model = tf.keras.models.load_model('ViT_OptiProj_Team5_Model_V5.h5')
        vit_prediction = vit_model.predict([pixel_data_ravel,pos_feed_classify])
        vit_classification = str(np.argmax(vit_prediction, axis=1)[0])

        return image_media, cnn_classification, vit_classification


# In[ ]:


anvil.server.wait_forever()


# In[ ]:




