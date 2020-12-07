# GAN-Introduction


# keras 實現GAN（生成對抗網路）
# Ref: https://www.itread01.com/content/1543548422.html

### keras_gan_cifar10.py

### 生成器（generator）
### 首先，建立一個“生成器（generator）”模型，它將一個向量（從潛在空間 - 在訓練期間隨機取樣）轉換為候選影象。
### GAN通常出現的許多問題之一是generator卡在生成的影象上，看起來像噪聲。一種可能的解決方案是在鑑別器（discriminator）
### 和生成器（generator）上使用dropout。

