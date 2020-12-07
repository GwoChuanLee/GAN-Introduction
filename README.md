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
    <img src="https://github.com/GwoChuanLee/GAN-Introduction/blob/main/GAN1.png" alt="Sample"  width="500" height="350">
    <p align="center">
        <b>GAN 結構</b>
    </p>
</p>


# 範例(一) : keras 實現 GAN（生成對抗網路）
### Ref: https://www.itread01.com/content/1543548422.html

<a href="https://github.com/GwoChuanLee/GAN-Introduction/blob/master/keras_gan_cifar10.py">keras_gan_cifar10.py </a>
   
