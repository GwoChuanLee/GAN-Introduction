# 
#  让我们跑一个最简单的GAN网络吧！（附Jupyter Notebook 代码）
#
#  程式參考:
#  https://zhuanlan.zhihu.com/p/85908702
#
# Vanilla GAN
#

# 先导入包
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#from google.colab import drive

# 读取Keras自带的mnist数据集。在这里我们给出一个读取数据的函数load_data()。
#
# Load the dataset
def load_data():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = (x_train.astype(np.float32) - 127.5)/127.5
  
  # Convert shape from (60000, 28, 28) to (60000, 784)
  x_train = x_train.reshape(60000, 784)
  return (x_train, y_train)

X_train, y_train = load_data()
print(X_train.shape, y_train.shape)

# 实现最原始的GAN网络，因此用最简单MLP全连接层来构建生成器
# 

def build_generator():
    model = Sequential()
    
    model.add(Dense(units=256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=784, activation='tanh'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

generator = build_generator()
generator.summary()

# 然后建一个判别器，也是一个MLP全连接神经网络：
#

def build_discriminator():
    model = Sequential()
    
    model.add(Dense(units=1024 ,input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
       
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
       
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
      
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model
  
discriminator = build_discriminator()
discriminator.summary()

# 然后，我们建立一个GAN网络，由discriminator和generator组成。
#

def build_GAN(discriminator, generator):
    discriminator.trainable=False
    GAN_input = Input(shape=(100,))
    x = generator(GAN_input)
    GAN_output= discriminator(x)
    GAN = Model(inputs=GAN_input, outputs=GAN_output)
    GAN.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return GAN

GAN = build_GAN(discriminator, generator)
GAN.summary()

# 然后我们给出绘制图像的函数，用于把generator生成的假图片画出来:
#

def draw_images(generator, epoch, examples=25, dim=(5,5), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(25,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='Greys')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Generated_images %d.png' %epoch)
	
# 最后一步，写一个train函数，来训练GAN网络。在这里我们设置最大迭代次数400，每次迭代生成128张假图片：
#

def train_GAN(epochs=1, batch_size=128):
    
  #Loading the data
  X_train, y_train = load_data()

  # Creating GAN
  # 建立一个GAN网络，GAN由两个神经网络（generator, discriminator）连接而成。
  generator= build_generator()
  discriminator= build_discriminator()
  GAN = build_GAN(discriminator, generator)

  # 建立一个循环（400次迭代）。tqdm用来动态显示每次迭代的进度。
  for i in range(1, epochs+1):
    print("Epoch %d" %i)
    
    for _ in tqdm(range(batch_size)):
      # Generate fake images from random noiset
      # 接着，我们生成呈高斯分布的噪声，利用generator，来生成batch_size（128张）图片。每张图片的输入就是一个1*100的噪声矩阵。
      noise= np.random.normal(0,1, (batch_size, 100))
      fake_images = generator.predict(noise)

      # Select a random batch of real images from MNIST
      # 我们从Mnist数据集中随机挑选128张真实图片。我们给真实图片标注1，给假图片标注0，然后将256张真假图片混合在一起。
      real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

      # Labels for fake and real images           
      label_fake = np.zeros(batch_size)  # 假圖標 0
      label_real = np.ones(batch_size)   # 真圖標 1
	  
      # Concatenate fake and real images 
      X = np.concatenate([fake_images, real_images])
      y = np.concatenate([label_fake, label_real])

      # Train the discriminator
      # 我们利用上文提到的256张带标签的真假图片，训练discriminator。训练完毕后，discriminator的weights得到了更新。
      # （打个比方，警察通过研究市面上流通的假币，在一起开会讨论，努力研发出了新一代鉴定假钞的方法）。
      discriminator.trainable=True
      discriminator.train_on_batch(X, y)  # 開始訓練 Discriminator , 資料集在 X 訓練目標為 fake_images --> label_fake, real_images ---> lable_real

      # Train the generator/chained GAN model (with frozen weights in discriminator) 
      # 然后，我们冻结住discriminator的weights，让discriminator不再变化。然后就开始训练generator (chained GAN)。
      # 在GAN的训练中，我们输入一堆噪声，期待的输出是将假图片预测为真。在这个过程中，generator继续生成假图片，送到discriminator检验，得到检验结果，
      # 如果被鉴定为假，就不断更新自己的权重（假钞贩子不断改良造假技术），直到discriminator将加图片鉴定为真图片（直到当前鉴定假钞的技术无法识别出假钞）。
      discriminator.trainable=False
      GAN.train_on_batch(noise, label_real)  # 開始訓練 Generator , 資料集在 noise 目標為訓練成 real image

    # Draw generated images every 15 epoches     
    if i == 1 or i % 10 == 0:
      draw_images(generator, i)
train_GAN(epochs=400, batch_size=128)


# 现在，我们总结一下每次迭代发生了什么：

# 1. Generator利用自己最新的权重，生成了一堆假图片。
# 2. Discrminator根据真假图片的真实label，不断训练更新自己的权重，直到可以顺利鉴别真假图片。
# 3. 此时discriminator权重被固定，不再发生变化。generator利用最新的discrimintor，苦苦思索，不断训练自己的权重，最终使discriminator将假图片鉴定为真图片。




