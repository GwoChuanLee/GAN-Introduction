#
# 2 DCGAN for MNIST
#
# Ref: https://blog.csdn.net/bryant_meng/article/details/81210656
#


# 2.1 导入必要的库
# 
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys

import numpy as np

#
# 2.2 搭建 generator
#
# 输入 100 维的 noisy，输出 （28，28，1）的图片

#100 → 128*7*7 reshape 成（7，7，128）→上采样（14，14，128）→ 
# conv（14，14，128）→上采样（28，28，128）→ conv（28，28，64）→ conv（28，28，1）
#
# 

# build_generator(self)
model = Sequential()

model.add(Dense(128 * 7 * 7, activation="relu", input_dim=100)) # 7,7,128
model.add(Reshape((7, 7, 128)))

model.add(UpSampling2D()) # 14,14,128

model.add(Conv2D(128, kernel_size=3, padding="same"))
model.add(BatchNormalization(momentum=0.8))
model.add(Activation("relu"))

model.add(UpSampling2D()) # 28,28,128

model.add(Conv2D(64, kernel_size=3, padding="same")) 
model.add(BatchNormalization(momentum=0.8))
model.add(Activation("relu"))

model.add(Conv2D(1, kernel_size=3, padding="same"))
model.add(Activation("tanh"))

model.summary()

noise = Input(shape=(100,))
img = model(noise)        
        
generator = Model(noise, img)

# 2.3 搭建 discriminator
#
#输入（28，28，1），输出概率值
#（28，28，1）→ conv（14，14，32）→ conv（7，7，64）→ padding （8，8，64）→ 
#conv（4，4，128）→ conv（4，4，256）→ flatten 4*4*256 → 1

# build_discriminator
model = Sequential()

model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28,28,1), padding="same")) # 14,14,32
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))  # 7,7,64
model.add(ZeroPadding2D(padding=((0,1),(0,1)))) # 8,8,64
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) # 4,4,128
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=3, strides=1, padding="same")) # 4,4,256
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Flatten()) # 4*4*256
model.add(Dense(1, activation='sigmoid'))

model.summary()

img = Input(shape=(28,28,1)) # 输入 （28，28，1）
validity = model(img) # 输出二分类结果

discriminator = Model(img, validity)

#
# 2.4 compile 模型，对学习过程进行配置
#
# 
optimizer = Adam(0.0002, 0.5)

# discriminator
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])


# The combined model  (stacked generator and discriminator)
z = Input(shape=(100,))
img = generator(z)
validity = discriminator(img)
# For the combined model we will only train the generator
discriminator.trainable = False

# Trains the generator to fool the discriminator
combined = Model(z, validity)
combined.summary()
combined.compile(loss='binary_crossentropy', 
                 optimizer=optimizer)
				 
#
#
# 2.5 保存生成的图片
#

def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()

# 2.6 训练
#

batch_size = 32
sample_interval = 50
# Load the dataset
(X_train, _), (_, _) = mnist.load_data() # (60000,28,28)
# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)  # (60000,28,28,1)
# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(4001):
    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size) # 0-60000 中随机抽  
    imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))# 生成标准的高斯分布噪声

    # Generate a batch of new images
    gen_imgs = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(imgs, valid) #真实数据对应标签１
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake) #生成的数据对应标签０
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

	 # ---------------------
    #  Train Generator
    # ---------------------
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Train the generator (to have the discriminator label samples as valid)
    g_loss = combined.train_on_batch(noise, valid)

    # Plot the progress
    if epoch % 50==0:
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        sample_images(epoch)
        
# 每隔 50 iteration 打印一次结果，保存一次图片（代码中的 epoch 理解为 iteration）
# 2.7 结果展示
