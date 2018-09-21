
# coding: utf-8

# In[1]:


from tensorflow.python.keras.layers import Conv2D, ReLU, BatchNormalization, Dense, GlobalAveragePooling2D, Input, MaxPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import Nadam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from image_generators import train_generator, validation_generator
import tensorflow as tf

img_height, img_width = 200, 200
epochs = 50
train_steps = 3030
val_steps = 918


# In[2]:


with tf.device('cpu:0'):    
    input_image = Input(shape=(img_height, img_width, 3), name='image_imput')
    #Block1 
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block1_conv1')(input_image)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    #Block2 
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    #Block3 
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block3_conv4')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    #Block4
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block4_conv4')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    #Block5
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name ='block5_conv4')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=53, activation='softmax', name='output_predictions')(x)
    model = Model(inputs = input_image, outputs = x, name = 'Classifier')


# In[3]:


parallel_model = multi_gpu_model(model=model, gpus=4)
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(filepath='models/top_weights.h5', monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
rlr = ReduceLROnPlateau()
callbacks = [tb, mc, es, rlr]
nadam = Nadam(lr=1e-3)
parallel_model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


parallel_model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs = epochs, validation_data=validation_generator, validation_steps=val_steps, workers=8, callbacks=callbacks)

