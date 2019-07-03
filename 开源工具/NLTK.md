https://www.jianshu.com/p/0e1d51a7549d

https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/

#### 简介

NLTK 模块是一个巨大的工具包，目的是在整个自然语言处理（NLP）方法上帮助您。 NLTK 将为您提供一切，从将段落拆分为句子，拆分词语，识别这些词语的词性，高亮主题，甚至帮助您的机器了解文本关于什么。在这个系列中，我们将要解决意见挖掘或情感分析的领域。

在我们学习如何使用 NLTK 进行情感分析的过程中，我们将学习以下内容：

- 分词 - 将文本正文分割为句子和单词。
- 词性标注
- 机器学习与朴素贝叶斯分类器
- 如何一起使用 Scikit Learn（sklearn）与 NLTK
- 用数据集训练分类器
- 用 Twitter 进行实时的流式情感分析。
- ...以及更多。

#### 安装

```python
pip install nltk
```

接下来，我们需要为 NLTK 安装一些组件。通过你的任何常用方式打开 python，然后键入：

```python
    import nltk
    nltk.download()
```

会跳出一个GUI 里面包含了 分词器，分块器，其他算法和所有的语料库。 如果空间是个问题，您可以选择手动选择性下载所有内容。 NLTK 模块将占用大约 7MB，整个`nltk_data`目录将占用大约 1.8GB，其中包括您的分块器，解析器和语料库。

