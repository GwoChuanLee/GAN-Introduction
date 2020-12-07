
#
# 
# Ref1: 默默地學 Deep Learning (8)-淺玩GAN
# https://medium.com/%E7%94%A8%E5%8A%9B%E5%8E%BB%E6%84%9B%E4%B8%80%E5%80%8B%E4%BA%BA%E7%9A%84%E8%A9%B1-%E5%BF%83%E4%B9%9F%E6%9C%83%E7%97%9B%E7%9A%84/%E9%BB%98%E9%BB%98%E5%9C%B0%E5%AD%B8-deep-learning-8-%E6%B7%BA%E7%8E%A9gan-c2493ad79929
#
# Ref2:[實戰系列] 使用 Keras 搭建一個 GAN 魔法陣（模型）
# https://ithelp.ithome.com.tw/articles/10208478
#


import numpy as np
import os

from keras.datasets import mnist
from keras.layers import Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY

#
# 我們定義一個叫做Generator的function，是用來生成假的手寫數字用的。
# 本範例是從10個數字，擴展到256、512、1024，最後再整理成28*28*1的灰階array。
#

def generator():
    """ Declare generator """

    model = Sequential()
    model.add(Dense(256, input_shape=(10,)))
    model.add(LeakyReLU(alpha=0.2))  # 使用 LeakyReLU 激活函數
    model.add(BatchNormalization(momentum=0.8))  # 使用 BatchNormalization 優化
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(width  * height * channels, activation='tanh'))
    model.add(Reshape((width, height, channels)))
    model.summary()

    return model
	
#
#再來是Discriminator，這是用來判斷這張圖是否為真的手寫數字用的。
#如果說Generator是製造偽鈔的人，那Discriminator就是驗鈔師了。
#我們的目的就是訓練Generator成為一個優秀的偽鈔製造者。
#
def discriminator():
    """ Declare discriminator """

    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense((width * height * channels), input_shape=shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(int((width * height * channels)/2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model

#
# 再來是把Generator和Discriminator疊在一起，就好像是G做完一張假鈔之後，
# 交給D去檢驗，看看自己做的像不像真的。這邊要注意的是，我們把Discriminator給Freeze住了，
# 在訓練的過程中，D是不會更動的。
#
def stacked_generator_discriminator(G,D):

    D.trainable = False

    model = Sequential()
    model.add(G)
    model.add(D)

    return model
    
def plot_images(G, height, width, noise_plot,step=0, samples=16, save2file=False):
    ''' Plot and generated images '''
    if not os.path.exists("./images"):
        os.makedirs("./images")
    filename = "./images/mnist_%d.png" % step
    noise = noise_plot
    images = G.predict(noise)

    plt.figure(figsize=(10, 10))

    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [height, width])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()

    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()
		
#
# 而下一個function，就是將這次的結果可視化，
# 我們用隨機的十個數字，讓Generator來生成一張手寫數字，總共生成4*4=16張，然後拼成一張大圖。
#

# 一開始先設定圖片的長、寬等，還有生成一個等等會用到的noise_plot，該變數是一個16*10的array，
# 也就是如同上面提到的，要生成16張手寫數字，每張手寫數字是由10個隨機數生成的。
#
#  這邊使用Adam(lr=0.0002, beta_1=0.5, decay=8e-8)作為訓練時的optimizer
#
# 之後就是我們常見的訓練流程了，可以調整訓練的次數、batch size等等來觀察看看結果有何變化。
#
# 最後的部分就是每訓練100次，就把Generator的成果輸出成一張含16個手寫數字的圖片檔。

width=28
height=28
channels=1
noise_plot = np.random.normal(0, 1, (16, 10))

shape = (width, height, channels)

optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

G = generator()
G.compile(loss='binary_crossentropy', optimizer=optimizer)

D = discriminator()
D.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

stacked_generator_discriminator = stacked_generator_discriminator(G,D)
    
stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

#epochs=6000
epochs = 10000

#batch = 320
batch = 32

save_interval = 100

(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

for cnt in range(epochs):

    ## train discriminator
    random_index = np.random.randint(0, len(X_train) - batch/2)
    legit_images = X_train[random_index : random_index + int(batch/2)].reshape(int(batch/2), width, height, channels)

    gen_noise = np.random.normal(0, 1, (int(batch/2), 10)) 
    syntetic_images = G.predict(gen_noise)

    x_combined_batch = np.concatenate((legit_images, syntetic_images))
    y_combined_batch = np.concatenate((np.ones((int(batch/2), 1)), np.zeros((int(batch/2), 1))))

    d_loss = D.train_on_batch(x_combined_batch, y_combined_batch)


    # train generator

    noise = np.random.normal(0, 1, (batch, 10))  # 添加高斯噪聲
    y_mislabled = np.ones((batch, 1))

    g_loss = stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

    print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

    if cnt % save_interval == 0:
        plot_images(G, height, width,noise_plot,save2file=True, step=cnt)

