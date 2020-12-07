# GAN-Introduction


# keras 實現GAN（生成對抗網路）
# Ref: https://www.itread01.com/content/1543548422.html

# keras_gan_cifar10.py


'''
生成器（generator）
首先，建立一個“生成器（generator）”模型，它將一個向量（從潛在空間 - 在訓練期間隨機取樣）轉換為候選影象。
GAN通常出現的許多問題之一是generator卡在生成的影象上，看起來像噪聲。一種可能的解決方案是在鑑別器（discriminator）
和生成器（generator）上使用dropout。

'''
'''python

import keras
from keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# 首先，將輸入轉換為16x16 128通道的feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

# 然後，添加捲積層
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# 上取樣至 32 x 32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# 新增更多的卷積層
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# 生成一個 32x32 1-channel 的feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()


'''
