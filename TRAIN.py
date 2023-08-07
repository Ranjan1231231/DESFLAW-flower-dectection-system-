import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator#allows your model to receive new variations of the images at each epoch
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation,BatchNormalization
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
import cv2
import os
from random import shuffle
from PIL import  Image
from tqdm import tqdm
import random
import tensorflow as tf
from keras.callbacks import  ReduceLROnPlateau
x=[]
z=[]
img_size=150
flower_daisy_dir='flowers/daisy'
flower_sunflower_dir='flowers/sunflower'
flower_tulip_dir='flowers/tulip'
flower_rose_dir='flowers/rose'
flower_dandelion_dir='flowers/dandelion'

def assign_label(img,flower_type):
    return flower_type

def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path=os.path.join(DIR,img)
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        img=cv2.resize(img,(img_size,img_size))
        x.append(np.array(img))
        z.append(str(label))
make_train_data('Daisy',flower_daisy_dir)
print(len(x),'images')
make_train_data('sunflower',flower_sunflower_dir)
print(len(x),'images')
make_train_data('tulip',flower_tulip_dir)
print(len(x),'images')
make_train_data('rose',flower_rose_dir)
print(len(x),'images')
make_train_data('dandelion',flower_dandelion_dir)
print(len(x),'images')

#visualizing some random images
# fig, ax = plt.subplots(5, 2)
# fig.set_size_inches(15, 15)
# for i in range(5):
#     for j in range(2):
#         l = random.randint(0, len(z)-1)
#         ax[i, j].imshow(x[l])
#         ax[i, j].set_title('Flower: ' + z[l])
# plt.tight_layout()
# plt.show()

#LABEL ENCODING THE Y ARRAY(Dasy-0,Rose-1..etc)
le=LabelEncoder()
y=le.fit_transform(z)
y=to_categorical(y,5)
x=np.array(x)
x=x/255

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.25,random_state = 42)

np.random.seed(42)
random.seed(42)
# tf.set_random_seed(42)


# modelling
model=Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))
batch_size=128
epoch=50
red_lr=ReduceLROnPlateau(monitor = 'val_acc',patience = 3,verbose = 1,factor = 0.1)


#data augmentation to prevent overfitting
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
#
model.compile(optimizer = Adam(learning_rate = 0.001),loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()
model=tf.keras.models.load_model('flower3.keras')

#fittiing on the training set and making predictions on the validation set
History=model.fit(datagen.flow(x_train,y_train,batch_size = batch_size),
                            epochs=epoch,validation_data=(x_test,y_test),
                            verbose=1,steps_per_epoch=x_train.shape[0]//batch_size)


#evaluating the model performance
# plt.plot(History.history['loss'])
# plt.plot(History.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend(['train','test'])
# plt.show()

# plt.plot(History.history['acc'])
# plt.plot(History.history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend(['train', 'test'])
# plt.show()



#
pred=model.predict(x_test)
# Get the predicted class indices
predicted_classes = np.argmax ( pred , axis = 1 )
# Inverse transform the predicted class indices to get flower names
predicted_flower_names = le.inverse_transform ( predicted_classes )
# Get the actual class indices for the test dataset
actual_classes = np.argmax ( y_test , axis = 1 )
# Inverse transform the actual class indices to get flower names
actual_flower_names = le.inverse_transform ( actual_classes )
# Print the predicted and actual flower names along with the corresponding images
for i in range ( len ( predicted_flower_names ) ) :
    print ( "Predicted: " , predicted_flower_names [ i ] , " | Actual: " , actual_flower_names [ i ] )
    # Optionally, you can display the corresponding image for each prediction
    plt.imshow ( x_test [ i ] )
    plt.show ( )




#VISUALIZING PREDICTIONS ON THE VALIDATION SET
#getting prediction on validation set
# pred=model.predict(x_test)
# pred_digits=np.argmax(pred,axis=1)
#
# #non storing some properly as well as misclassified indexes
# i=0
# prop_class=[]
# mis_class=[]
# for i in range(len(y_test)):
#     if(np.argmax(y_test[i])==pred_digits[i]):
#         prop_class.append(i)
#     if(len(prop_class)==8):
#         break
# i=0
# for i in range(len(y_test)):
#     if(not np.argmax(y_test[i])==pred_digits[i]):
#         mis_class.append(i)
#     if(len(mis_class)==8):
#         break




# count = 0
# fig, ax = plt.subplots(4, 2)
# fig.set_size_inches(15, 15)
# for i in range(4):
#     for j in range(2):
#         ax[i, j].imshow(x_test[prop_class[count]])
#         predicted_flower = le.inverse_transform([pred_digits[prop_class[count]]])[0]
#         actual_flower = le.inverse_transform([np.argmax(y_test[prop_class[count]])])[0]
#         ax[i, j].set_title("Predicted Flower: " + str(predicted_flower) + "\n" + "Actual Flower: " + str(actual_flower))
#         count += 1
# plt.tight_layout()
# plt.show()


model.save('flower.keras')





