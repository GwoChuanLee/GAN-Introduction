
# Keras-MLP-GAN for Mnist
#
# https://blog.csdn.net/bryant_meng/article/details/81024890
#
# GAN 生成 MNIST
#
# 2.1 导入必要的库

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
# generator 输入噪声（100），可以产生图片（28，28，1），这里用 MLP 的形式，
# 也即都是 fully connection！
#
# 100→256→512→1024→784→reshape 成 （28,28,1）
#
# build_generator

# -----------------------------------------------------------------------------------
# ReLu=max(0,x), LeakyReLu=max(alpa*x,x), LeakyReLu 避免 Dead ReLu 問題
# -----------------------------------------------------------------------------------
# 當一個 Deep Network 非常深的時候，狀況也是一樣的，當我們後面的 Layer 根據前面
# 的結果做出調整，但是前面的 Layer 也根據更前面的結果更動了，大家都動的情況下，
# 造成結果還是壞掉了

# 為了解決這個問題，過去傳統的方法就是 Learning  Rate 設小一點,  Learning Rate 設小一點的缺點就是慢

# Normalization 的好處：把每一個  Layer 的 feature 都做 Normalization 後，
# 對於每一個 Layer 而言，確保 output 的 Statistic 是固定的

# 但是麻煩的地方在於，我們不太容易讓他的 Statistic 固定，
# 因為在整個 training 的過程中，Network 的參數是不斷在變化的，
# 所以每一個 Hidden Layer 的 mean 跟 variance 是不斷在變化的，
# 所以我們需要一個新的技術，這個技術就叫做 “Batch Normalization” 
# ------------------------------------------------------------------------------------

model = Sequential()

model.add(Dense(256, input_dim=100))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))

model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
          
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
          
model.add(Dense(np.prod((28,28,1)), activation='tanh')) # 即 Dense(784)
model.add(Reshape((28,28,1))) # 圖形大小 28x28 , 1 表單色

model.summary()

noise = Input(shape=(100,)) # input 100,这里写成100不加逗号不行哟
img = model(noise) # output (28,28,1)
        
generator = Model(noise, img) # input:noise , output: img

#
# 2.3 搭建 discriminator
# 
# 分类网络，输入（28，28，1），输出概率值（sigmoid），也都是用的 MLP

#（28，28，1）flatten 为 784→512→256→1

# build_discriminator
model = Sequential()

model.add(Flatten(input_shape=(28,28,1))) # 將輸入 28x28x1 維度資料變成 1維 784 的資料輸入 

model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))

model.add(Dense(256))
model.add(LeakyReLU(alpha=0.2))

model.add(Dense(1, activation='sigmoid'))
model.summary()

img = Input(shape=(28,28,1)) # 输入 （28，28，1）
validity = model(img) # 输出二分类结果

discriminator = Model(img, validity) # input: img, output: 0 或 1 

#
# 2.4 compile 模型，对学习过程进行配置
#
# 这里训练 GAN 分为两个过程

#训练 discriminator，图片由固定 generator 产生
#训练 generator，联合 discriminator 和 generator，
#但是 discriminator 的梯度不更新，所以 discriminator 固定住了

optimizer = Adam(0.0002, 0.5) # learning rate=0.0002 

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
	fig.savefig("images/%d.png"%epoch)
	plt.close()

#
# 2.6 训练
# 
# batch_size 设置为 32，每隔 500 次 iteration（代码中叫 epoch 不太合理）
#，打印一下结果，保存生成的图片！
#

batch_size = 32
sample_interval = 500

# Load the dataset
(X_train, _), (_, _) = mnist.load_data() # (60000,28,28)
# Rescale -1 to 1
X_train = X_train / 127.5 - 1. # tanh 的结果是 -1~1，所以这里 0-1 归一化后减1
X_train = np.expand_dims(X_train, axis=3)  # (60000,28,28,1)
# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(30001):
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
	if epoch % 500==0:
		print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss)) 
	
	# If at save interval => save generated image samples
	if epoch % sample_interval == 0:
		sample_images(epoch)
		
