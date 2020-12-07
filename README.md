# (深耕計畫) 短期實務集訓課程 : 資工系, 教學發展中心
## 講題: 生成對抗網路之專題實作開發 [2020/12/11]


# GAN-Introduction

### 生成對抗網路（Generative Adversarial Networks，GAN）最早由 Ian Goodfellow 在 2014 年提出，是目前深度學習領域最具潛力的研究成果之一。它的核心思想是：同時訓練兩個相互協助、同時又相互競爭的深度神經網路（一個稱為生成器 Generator，另一個稱為判别器 Discriminator）來處理非監督式學習的相關問題。在訓練過程中，兩個網路最終都要學習如何處理任務。

### 舉例來說明 GAN 的原理：
### 將警察視為判别器，製造假鈔的犯罪分子視為生成器。一開始，犯罪分子會首先向警察展示一张假鈔。警察識別出該假鈔，並向犯罪分子反饋哪些地方是假的。接着，根據警察的反饋，犯罪分子改進工藝，製作一張更逼真的假鈔给警方檢查。接著警方再反饋，犯罪分子再改進工藝。不斷重複這一過程，直到警察識别不出真假，那麼假鈔生成模型就訓練成功。

## GAN的運作模式:
### 有兩個需要被訓練的model，
### 一個是Discriminator network: 偵探則是要分辨現在給他的data是真的還是假的
### 另一個是Generator network:工匠要做的事就是偽造出假的data，現在手上有真的data，並且會給出一個回饋。工匠根據這個回饋來「訓練」他現在的工藝，也就是調整model的parameter
### 一旦工匠的工藝成熟到偵探分辨不出來誰真誰假，就可以說我們訓練出了一個能夠模擬真正data分布的model。


<p align="center">
    <img src="https://github.com/GwoChuanLee/GAN-Introduction/blob/main/GAN1.png" alt="Sample"  width="400" height="300">
    <p align="center">
        <b>GAN 結構</b>
    </p>
</p>


# 範例(一) : keras 實現 GAN（生成對抗網路）
### 參考文章 https://www.itread01.com/content/1543548422.html


## 如何在Keras中以最小的形式實現GAN。具體實現是一個深度卷積GAN，或DCGAN (Deep Convolution GAN)
#### 一個GAN，其中generator和discriminator是深度卷積網路，它利用`Conv2DTranspose`層對generator中的影象上取樣。
#### 然後將在CIFAR10的影象上訓練GAN，CIFAR10資料集由屬於10個類別（每個類別5,000個影象）的50,000個32x32 RGB影象構成。為了節約時間，本文將只使用“frog”類的影象。


#### 原理上，GAN的組成如下所示：
#### *`generator`網路將shape`（latent_dim，）`的向量對映到shape`（32,32,3）`的影象。
#### *“discriminator”網路將形狀（32,32,3）的影象對映到估計影象是真實的概率的二進位制分數。
####  *`gan`網路將generator和discriminator連結在一起：`gan（x）=discriminator（generator（x））`。因此，這個“gan”網路將潛在空間向量對映到discriminator對由generator解碼的這些潛在向量的#### 真實性的評估。
#### *使用真實和虛假影象以及“真實”/“假”標籤來訓練鑑別器，因此需要訓練任何常規影象分類模型。
#### *為了訓練generator，我們使用generator權重的梯度來減少“gan”模型的損失。這意味著，在每個step中，將generator的權重移動到使得discriminator更可能被分類為由generator解碼的影象“真實”的方向#### 上。即訓練generator來欺騙discriminator。

<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/keras_gan_cifar10.py">keras_gan_cifar10.py </a>
   
