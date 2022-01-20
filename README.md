# NLP ChatBot
NLP入门练习 

## 分词准备
+ jieba 分词
+ 单字

## 语料准备
+ 小黄鸡50w
+ 百科问答125w
+ 微博400w

## 意图识别
+ 分类识别“问答”和“闲聊”
+ `fasttext`模型
+ 准备数据集，打标签

## 问答模型
+ `word2vector`
    + `tfidf`
    + `bm25`
    + `fasttext`
+ 召回
    + 对问题进行粗分类
    + `pysparnn` 计算相似度
+ 排序
    + 对问题进行细分类
    + `Siamese Netword` 孪生神经网络
        + `self Attention`

## 闲聊模型
+ `seq2seq`
+ `attention`
+ `teaching force`
+ `beam search`