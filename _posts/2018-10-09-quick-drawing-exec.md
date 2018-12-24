---
title: "Quick drawing - dogs and cats"
categories: 
  - Keras
comments: true
last_modified_at: 2018-10-09

---

### 개와 고양이는 어떻게 구분되는가

* quick drawing은 구글에서 공개하는 오픈소스 데이터셋입니다.
* 345개 종류의 5백만장의 그림으로 이루어져있습니다.

* 이 포스팅에서는 그 중 개와 고양이 그림을 이용해 개와 고양이 그림을 구분하는 모델을 학습하고, 모델이 그림을 어떻게 인식하는지 시각화해보았습니다. 


```python
import numpy as np
import matplotlib.pyplot as plt

## Quick! drawing dataset
## https://quickdraw.withgoogle.com/data
## https://github.com/googlecreativelab/quickdraw-dataset
## download : https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

dogs = np.load('full%2Fnumpy_bitmap%2Fdog.npy')
cats = np.load('full%2Fnumpy_bitmap%2Fcat.npy')
plt.subplot(121)
plt.imshow(dogs[0].reshape(28,28), plt.get_cmap('Greys'), vmin=0, vmax=255)
plt.title('dog')
plt.subplot(122)
plt.imshow(cats[0].reshape(28,28), plt.get_cmap('Greys'), vmin=0, vmax=255)
plt.title('cat')
plt.suptitle('Wow! so cute!')
plt.show()
```

<img src= "/assets/img/2018-10-09/output_1_0.png">



```python
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

print(dogs.shape, cats.shape)

x_train = dogs.reshape(-1, 28, 28, 1)
y_train = np.zeros((dogs.shape[0], 1))
x_train = np.concatenate((x_train, cats.reshape(-1, 28, 28, 1)))
y_train = np.concatenate((y_train, np.ones((cats.shape[0], 1))))

x_train = x_train / 255.
y_train = to_categorical(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
```

    (152159, 784) (123202, 784)
    (192752, 28, 28, 1) (192752, 2) (82609, 28, 28, 1) (82609, 2)



```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


batch_size = 128
epochs = 12


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Train on 192752 samples, validate on 82609 samples
    Epoch 1/12
    192752/192752 [==============================] - 310s 2ms/step - loss: 0.2997 - acc: 0.8717 - val_loss: 0.2500 - val_acc: 0.8949
    Epoch 2/12
    192752/192752 [==============================] - 317s 2ms/step - loss: 0.2490 - acc: 0.8966 - val_loss: 0.2328 - val_acc: 0.9034
    Epoch 3/12
    192752/192752 [==============================] - 308s 2ms/step - loss: 0.2336 - acc: 0.9036 - val_loss: 0.2295 - val_acc: 0.9059
    Epoch 4/12
    192752/192752 [==============================] - 302s 2ms/step - loss: 0.2242 - acc: 0.9087 - val_loss: 0.2333 - val_acc: 0.9058
    Epoch 5/12
    192752/192752 [==============================] - 306s 2ms/step - loss: 0.2158 - acc: 0.9122 - val_loss: 0.2195 - val_acc: 0.9099
    Epoch 6/12
    192752/192752 [==============================] - 306s 2ms/step - loss: 0.2102 - acc: 0.9152 - val_loss: 0.2195 - val_acc: 0.9085
    Epoch 7/12
    192752/192752 [==============================] - 308s 2ms/step - loss: 0.2058 - acc: 0.9174 - val_loss: 0.2246 - val_acc: 0.9118
    Epoch 8/12
    192752/192752 [==============================] - 321s 2ms/step - loss: 0.2015 - acc: 0.9190 - val_loss: 0.2151 - val_acc: 0.9126
    Epoch 9/12
    192752/192752 [==============================] - 316s 2ms/step - loss: 0.1972 - acc: 0.9211 - val_loss: 0.2160 - val_acc: 0.9132
    Epoch 10/12
    192752/192752 [==============================] - 320s 2ms/step - loss: 0.1945 - acc: 0.9226 - val_loss: 0.2274 - val_acc: 0.9126
    Epoch 11/12
    192752/192752 [==============================] - 320s 2ms/step - loss: 0.1908 - acc: 0.9244 - val_loss: 0.2327 - val_acc: 0.9122
    Epoch 12/12
    192752/192752 [==============================] - 304s 2ms/step - loss: 0.1881 - acc: 0.9253 - val_loss: 0.2281 - val_acc: 0.9135
    Test loss: 0.22813269033429004
    Test accuracy: 0.9135203186107955



```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_8 (Conv2D)            (None, 26, 26, 32)        320       
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 12, 12, 64)        0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 128)               1179776   
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 2)                 258       
    =================================================================
    Total params: 1,198,850
    Trainable params: 1,198,850
    Non-trainable params: 0
    _________________________________________________________________



```python
model.save('quick_drawing_model.h5')
```


```python
import matplotlib.image as mpimg
a_dog=mpimg.imread('articles/quick-drawing/yjucho-dog.png')
a_cat=mpimg.imread('articles/quick-drawing/yjucho-cat.png')

plt.subplot(121)
plt.imshow(a_dog, plt.get_cmap('Greys'), vmin=0, vmax=255)
plt.title('dog')
plt.subplot(122)
plt.imshow(a_cat, plt.get_cmap('Greys'), vmin=0, vmax=255)
plt.title('cat')
plt.suptitle('yjucho\'s drawing')
plt.show()
```

<img src= "/assets/img/2018-10-09/output_6_0.png">



```python
tmp = np.mean(a_dog, axis=-1)
plt.imshow(tmp, plt.get_cmap('gray'))
plt.show()
y_pred = model.predict(tmp.reshape(1, 28,28,1))
y_pred
```


<img src= "/assets/img/2018-10-09/output_7_0.png">





    array([[0.5746763 , 0.42532378]], dtype=float32)




```python
tmp = np.mean(a_cat, axis=-1)
plt.imshow(tmp, plt.get_cmap('gray'))
plt.show()
y_pred = model.predict(tmp.reshape(1, 28,28,1))
y_pred
```


<img src= "/assets/img/2018-10-09/output_8_0.png">





    array([[0.5847676 , 0.41523245]], dtype=float32)




```python
import keras.backend as K
from keras.models import load_model
model = load_model('quick_drawing_model.h5')

layer_dict = dict([(layer.name, layer) for layer in model.layers])
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def vis_img_in_filter(img = np.array(x_train[13]).reshape((1, 28, 28, 1)).astype(np.float64), 
                      layer_name = 'conv2d_8'):
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        img_asc = np.array(img)
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((28, 28)))
        
    if layer_output.shape[3] >= 35:
        plot_x, plot_y = 6, 6
    elif layer_output.shape[3] >= 23:
        plot_x, plot_y = 4, 6
    elif layer_output.shape[3] >= 11:
        plot_x, plot_y = 2, 6
    else:
        plot_x, plot_y = 1, 2
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    ax[0, 0].imshow(img.reshape((28, 28)), cmap = 'gray')
    ax[0, 0].set_title('Input image')
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        if x == 0 and y == 0:
            continue
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))

vis_img_in_filter()
```
<img src= "/assets/img/2018-10-09/output_9_0.png">

reference : https://www.kaggle.com/ernie55ernie/mnist-with-keras-visualization-and-saliency-map