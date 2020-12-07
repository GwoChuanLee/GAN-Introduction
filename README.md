# (深耕計畫) 短期實務集訓課程 : 資工系, 教學發展中心
# 講題: 生成對抗網路之專題實作開發 [2020/12/11]


# Deep Learning 背景知識 : 
### Keras深度學習(Deep Learning)卷積神經網路(CNN)辨識Cifar-10影像 [林大貴, 博碩書局]
### http://tensorflowkeras.blogspot.com/2017/10/kerasdeep-learningcnncifar-10.html

<p align="center">
    <img src="https://github.com/GwoChuanLee/GAN-Introduction/blob/main/CNN1.png" alt="Sample"  width="1000" height="500">
    <p align="center">
        <b> CNN (卷積類神經網路) 示意圖 [參考:中正游寶達教授講義] </b>
    </p>
</p>

<p align="center">
    <img src="https://github.com/GwoChuanLee/GAN-Introduction/blob/main/CNN2.jpg" alt="Sample"  width="600" height="800">
    <p align="center">
        <b> CNN (卷積類神經網路) 架構圖 [參考:林大貴/博碩書局]</b>
    </p>
</p>


# GAN-Introduction

#### 資料參考:  https://www.leiphone.com/news/201703/Y5vnDSV9uIJIQzQm.html

### 生成對抗網路（Generative Adversarial Networks，GAN）最早由 Ian Goodfellow 在 2014 年提出，是目前深度學習領域最具潛力的研究成果之一。它的核心思想是：同時訓練兩個相互協助、同時又相互競爭的深度神經網路（一個稱為生成器 Generator，另一個稱為判别器 Discriminator）來處理非監督式學習的相關問題。在訓練過程中，兩個網路最終都要學習如何處理任務。

### 舉例來說明 GAN 的原理：
### 將警察視為判别器，製造假鈔的犯罪分子視為生成器。一開始，犯罪分子會首先向警察展示一张假鈔。警察識別出該假鈔，並向犯罪分子反饋哪些地方是假的。接着，根據警察的反饋，犯罪分子改進工藝，製作一張更逼真的假鈔给警方檢查。接著警方再反饋，犯罪分子再改進工藝。不斷重複這一過程，直到警察識别不出真假，那麼假鈔生成模型就訓練成功。

## GAN的運作模式:
### 有兩個需要被訓練的model，
### 一個是鑑別器網路 (Discriminator network): 偵探則是要分辨現在給他的data是真的還是假的
### 一個是生成器網路 (Generator network):工匠要做的事就是偽造出假的data，現在手上有真的data，並且會給出一個回饋。工匠根據這個回饋來「訓練」他現在的工藝，也就是調整model的parameter
### 一旦工匠的工藝成熟到偵探分辨不出來誰真誰假，就可以說我們訓練出了一個能夠模擬真正data分布的model。

<p align="center">
    <img src="https://github.com/GwoChuanLee/GAN-Introduction/blob/main/GAN1.png" alt="Sample"  width="400" height="300">
    <p align="center">
        <b>GAN 結構</b>
    </p>
</p>

## 深度卷積生成對抗網路 DCGAN (Deep Convolution GAN):

### DCGANs的基本架構就是使用幾層“反摺積”（Deconvolution）網路。“反摺積”類似於一種反向卷積，這跟用反向傳播演算法訓練監督的卷積神經網路（CNN）是類似的操作。

#### 資料參考:  https://www.leiphone.com/news/201703/Y5vnDSV9uIJIQzQm.html

### DCGAN 判别器的作用是判断一個模型生成的圖像和真實圖像比，有多逼真。它的基本结構就是如下圖所示的卷積神經網路（Convolutional Neural Network，CNN）。對於 MNIST 數據集來說，模型輸入是一個 28x28 像素的單通道圖像。Sigmoid 函數的輸出值在 0-1 之間，表示图像真實度的機率，其中 0 表示肯定是假的，1 表示肯定是真的。與典型的 CNN 結構相比，這里去掉了層之間的 max-pooling，而是採用了步進卷積來進行下採樣。這裡每個 CNN 層都以 LeakyReLU 為激活函數。而且為了防止過擬合和記憶效應，層之間的 dropout 值均被設置在 0.4-0.7 之間。

<p align="center">
    <img src="https://github.com/GwoChuanLee/GAN-Introduction/blob/main/DCGAN-G-Net.jpg" alt="Sample"  width="800" height="300">
    <p align="center">
        <b>DCGAN Generator Network</b>
    </p>
</p>

### DCGAN 生成器的作用是合成假的图像，其基本机构如下图所示。图中，我们使用了卷积的倒数，即转置卷积（transposed convolution），从 100 维的噪声（满足 -1 至 1 之间的均匀分布）中生成了假图像。如在 DCGAN 模型中提到的那样，去掉微步进卷积，这里我们采用了模型前三层之间的上采样来合成更逼真的手写图像。在层与层之间，我们采用了批量归一化的方法来平稳化训练过程。以 ReLU 函数为每一层结构之后的激活函数。最后一层 Sigmoid 函数输出最后的假图像。第一层设置了 0.3-0.5 之间的 dropout 值来防止过拟合。
<p align="center">
    <img src="https://github.com/GwoChuanLee/GAN-Introduction/blob/main/DCGAN-D-Net.jpg" alt="Sample"  width="800" height="300">
    <p align="center">
        <b>DCGAN Discriminator Network</b>
    </p>
</p>

# 範例(一) : 【Keras-MLP-GAN】MNIST: 以MLP方式建立GAN
### 參考文章 https://blog.csdn.net/bryant_meng/article/details/81024890
### 本範例僅採用多層類神經網路(MultiLayer Neural Networks)來建構GAN
<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/Keras-MLP-GAN-MNIST.py">Keras-MLP-GAN-MNIST.py </a>

# 範例(二) : keras 實現 GAN（生成對抗網路）
### 參考文章 https://www.itread01.com/content/1543548422.html

## 如何在Keras中以最小的形式實現 GAN。
## 具體實現一個深度卷積GAN，或DCGAN (Deep Convolution GAN)
#### 一個GAN，其中generator和discriminator是深度卷積網路，它利用`Conv2DTranspose`層對generator中的影象上取樣。
#### 然後將在CIFAR10的影象上訓練GAN，CIFAR10資料集由屬於10個類別（每個類別5,000個影象）的50,000個32x32 RGB影象構成。為了節約時間，本文將只使用“frog”類的影象。


#### 原理上，GAN的組成如下所示：
#### *`generator`網路將shape`（latent_dim，）`的向量對映到shape`（32,32,3）`的影象。
#### *“discriminator”網路將形狀（32,32,3）的影象對映到估計影象是真實的概率的二進位制分數。
####  *`gan`網路將generator和discriminator連結在一起：`gan（x）=discriminator（generator（x））`。因此，這個“gan”網路將潛在空間向量對映到discriminator對由generator解碼的這些潛在向量的#### 真實性的評估。
#### *使用真實和虛假影象以及“真實”/“假”標籤來訓練鑑別器，因此需要訓練任何常規影象分類模型。
#### *為了訓練generator，我們使用generator權重的梯度來減少“gan”模型的損失。這意味著，在每個step中，將generator的權重移動到使得discriminator更可能被分類為由generator解碼的影象“真實”的方向#### 上。即訓練generator來欺騙discriminator。

<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/keras_gan_cifar10.py">keras_gan_cifar10.py </a>

# 2. 其他範例:

### (1)【Keras-DCGAN】MNIST / CIFAR-10:  https://blog.csdn.net/bryant_meng/article/details/81210656
<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/DCGAN_Cifar10.py">DCGAN_Cifar10.py </a> <br>
<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/DCGAN_Mnist.py">DCGAN_Mnist.py </a>

### (2)【Keras-MLP-GAN】MNIST: https://blog.csdn.net/bryant_meng/article/details/81024890
#### 以MLP方式建立GAN
<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/Keras-MLP-GAN-MNIST.py">Keras-MLP-GAN-MNIST.py </a>

### (3) 使用 Keras 搭建一個 GAN 魔法陣（模型）: https://ithelp.ithome.com.tw/articles/10208478
<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/playGAN_mnist.py">playGAN_mnist.py</a>

# 3. 其他文章

### (1) 一文看懂生成式对抗网络GANs： https://36kr.com/p/1721743589377

### (2) 不到 200 行代码，教你如何用 Keras 搭建生成对抗网络（GAN）:  https://www.leiphone.com/news/201703/Y5vnDSV9uIJIQzQm.html


# 4. 參考網站
### (1) 李宏毅-HackMD: https://hackmd.io/@overkill8927/SyyCBk3Mr?type=view
### (2) 李宏毅(NTU-courses): http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html
### (3) G-Lab (seeprettyface) : https://seeprettyface.com/
### (4) Github (seeprettyface.com) https://github.com/a312863063
### (5) 人臉downloads : https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL






   
