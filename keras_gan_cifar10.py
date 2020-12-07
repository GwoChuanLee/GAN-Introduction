# -*- coding: utf-8 -*-
'''
  keras 實現GAN（生成對抗網路）

  https://www.itread01.com/content/1543548422.html
    
生成器（generator）

首先，建立一個“生成器（generator）”模型，它將一個向量（從潛在空間 - 在訓練期間隨機取樣）轉換為候選影象。
GAN通常出現的許多問題之一是generator卡在生成的影象上，看起來像噪聲。一種可能的解決方案是在鑑別器（discriminator）
和生成器（generator）上使用dropout。

'''
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
discriminator(鑑別器)

建立鑑別器模型，它將候選影象（真實的或合成的）作為輸入，並將其分為兩類：“生成的影象”或“來自訓練集的真實影象”。

'''
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# 重要的技巧（新增一個dropout層）
x = layers.Dropout(0,4)(x)

# 分類層
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# 為了訓練穩定，在優化器中使用學習率衰減和梯度限幅（按值）。
discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

'''
The adversarial network:對抗網路
最後，設定GAN，它連結生成器（generator）和鑑別器（discrimitor）。 這是一種模型，經過訓練，
將使生成器（generator）朝著提高其愚弄鑑別器（discrimitor）能力的方向移動。 該模型將潛在的空間點轉換為分類決策，
“假的”或“真實的”，並且意味著使用始終是“這些是真實影象”的標籤來訓練。 所以訓練`gan`將以一種方式更新
“發生器”的權重，使得“鑑別器”在檢視假影象時更可能預測“真實”。 非常重要的是，將鑑別器設定為在訓練
期間被凍結（不可訓練）：訓練“gan”時其權重不會更新。 如果在此過程中可以更新鑑別器權重，那麼將訓練鑑別
器始終預測“真實”。
'''
# 將鑑別器（discrimitor）權重設定為不可訓練（僅適用於`gan`模型）
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

'''
  開始訓練了。
  每個epoch：
   *在潛在空間中繪製隨機點（隨機噪聲）。
   *使用此隨機噪聲生成帶有“generator”的影象。
   *將生成的影象與實際影象混合。
   *使用這些混合影象訓練“鑑別器”，使用相應的目標，“真實”（對於真實影象）或“假”（對於生成的影象）。
   *在潛在空間中繪製新的隨機點。
   *使用這些隨機向量訓練“gan”，目標都是“這些是真實的影象”。 這將更新發生器的權重（僅因為鑑別器在“gan”內被凍結）
   以使它們朝向獲得鑑別器以預測所生成影象的“這些是真實影象”，即這訓練發生器欺騙鑑別器。
'''
import os
from keras.preprocessing import image

# 匯入CIFAR10資料集
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

# 從CIFAR10資料集中選擇frog類（class 6）
x_train = x_train[y_train.flatten() == 6]

# 標準化資料
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
#save_dir = '.\\gan_image'
save_dir = 'keras_gan_cifar10'

start = 0 
# 開始訓練迭代
for step in range(iterations):
    # 在潛在空間中抽樣隨機點
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    # 將隨機抽樣點解碼為假影象
    generated_images = generator.predict(random_latent_vectors)
    
    # 將假影象與真實影象進行比較
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    
    # 組裝區別真假影象的標籤
    labels = np.concatenate([np.ones((batch_size, 1)),
                            np.zeros((batch_size, 1))])
    # 重要的技巧，在標籤上新增隨機噪聲
    labels += 0.05 * np.random.random(labels.shape)
    
    # 訓練鑑別器（discrimitor）
    d_loss = discriminator.train_on_batch(combined_images, labels)
    
    # 在潛在空間中取樣隨機點
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    # 彙集標有“所有真實影象”的標籤
    misleading_targets = np.zeros((batch_size, 1))
    
    # 訓練生成器（generator）（通過gan模型，鑑別器（discrimitor）權值被凍結）
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:
        # 儲存網路權值
        gan.save_weights('gan.h5')

        # 輸出metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        # 儲存生成的影象
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))

        # 儲存真實影象，以便進行比較
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))

# 繪圖
import matplotlib.pyplot as plt

# 在潛在空間中抽樣隨機點
random_latent_vectors = np.random.normal(size=(10, latent_dim))

# 將隨機抽樣點解碼為假影象
generated_images = generator.predict(random_latent_vectors)

for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)
    
plt.show()
