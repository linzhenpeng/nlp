https://www.cnblogs.com/iloveai/p/gensim_tutorial.html

https://www.jianshu.com/p/9ac0075cc4c0

[Gensim 简介](https://blog.csdn.net/hohaizx/article/details/84068198)

[gensim  Introduction](https://radimrehurek.com/gensim/intro.html)

Gensim（generate similarity）是一个简单高效的自然语言处理Python库，用于抽取文档的语义主题（semantic topics）。Gensim的输入是原始的、无结构的数字文本（纯文本），内置的算法包括Word2Vec，FastText，潜在语义分析（Latent Semantic Analysis，LSA），潜在狄利克雷分布（Latent Dirichlet Allocation，LDA）等，通过计算训练语料中的统计共现模式自动发现文档的语义结构。这些算法都是非监督的，这意味着不需要人工输入——仅仅需要一组纯文本语料。一旦发现这些统计模式后，任何纯文本（句子、短语、单词）就能采用语义表示简洁地表达。

## 特点

Memory independence： 不需要一次性将整个训练语料读入内存，Gensim充分利用了Python内置的生成器（generator）和迭代器（iterator）用于流式数据处理，内存效率是Gensim设计目标之一。
Memory sharing： 训练好的模型可以持久化到硬盘，和重载到内存。多个进程之间可以共享相同的数据，减少了内存消耗。
多种向量空间算法的高效实现： 包括Word2Vec，Doc2Vec，FastText，TF-IDF，LSA，LDA，随机映射等。
支持多种数据结构。

基于语义表示的文档相似度查询

## 核心概念

### corpus

一组纯文本的集合，在Gensim中，语料有两个角色：

模型训练的输入。此时语料用于自动训练机器学习模型，如LsiModel，LdaModel，模型使用训练语料发现公共主题，初始化模型参数。因为Gensim聚焦于非监督模型，因此无需人工干预。
Documents to organize。模型训练好后，可以用于从新文档（训练语料中没有出现过的）抽取主题。

### 向量空间模型（vector space model，VSM）

在向量空间模型中，每一篇文档被表示成一组特征。特征可以认为是问答对（question-answer pair），例如：

How many times does the word splonge \textit{splonge}splonge appear in the document? Zero.
How many paragraphs does the document consist of? Two.
How many fonts does the document use? Five.
通常对于每一个question分配一个id，因此这篇文档可以表示成一系列的二元对(1,0.0),(2,2.0),(3,5.0) (1,0.0),(2,2.0),(3,5.0)(1,0.0),(2,2.0),(3,5.0)。如果我们事先知道所有的question，我们可以隐式的省略这些id只保留answer序列，简写成(0.0,2.0,5.0) (0.0,2.0,5.0)(0.0,2.0,5.0)，这组answer序列就可以被认为是一个向量，用于代表这篇文档。每一篇文档的questions都是相同的，因此在观察两个向量后，我们希望能够得到如下结论：两个向量很相似，因此原始文档一定也很相似。 当然，这个结论是否正确取决于questions选择的好坏。

我们最常用的词袋模型就是一种向量空间模型，question是词汇表中的词wi w_iw 
i
	
 是否出现在文档中，因此用词袋模型表示文档，向量的维度等于词汇表中单词的数量V VV。

### sparse vector

为了节约空间，在Gensim中省略了所有值为0.0的元素，例如，对于上面的向量(0.0,2.0,5.0) (0.0,2.0,5.0)(0.0,2.0,5.0)，我们只需要写[(2,2.0),(3,5.0)] [(2, 2.0),(3, 5.0)][(2,2.0),(3,5.0)]，向量中每一个元素是一个二元元组
$$
(feature_{id}, feature_{value})
$$
。如果某个特征在稀疏表示中缺省，可以很自然的认为其值为0.0。

### streamed corpus

Gensim没有规定任何指定的数据格式，语料是一组稀疏向量序列。例如：[[(2,2.0),(3,5.0)],[(3,1.0)]] [[(2, 2.0),(3, 5.0)],[(3, 1.0)]][[(2,2.0),(3,5.0)],[(3,1.0)]]是一个包含两篇文档的简单语料，两篇文档被表示成两个稀疏向量，第一个有两个非零元素，第二个有一个非零元素。这个例子中，我们将语料用Python中list表示，但是Gensim并没有规定语料必须表示成list，Numpy中array，Pandas中dataframe或者其他任何对象，迭代时，将依次产生这些稀疏向量。这个灵活性允许我们创建自己的语料类，直接从硬盘、网络、数据库……中流式产生稀疏向量。

### model, transformation

Gensim中用model指代将一篇文档转换（transform）为另一种形式的模型代码以及相关参数，因为文档被表示成向量，因此model可以认为是从一个向量空间到另一个向量空间的变换，这个变换的参数是从训练数据中学习得到的。训练好的models可以被持久化到硬盘，之后在重载回来，无论是在新的训练文档中继续训练还是用于转换一篇文档。Gensim实现了很多模型，比如：Word2Vec，LsiModel，LdaModel，FastText等，具体的可以参考 [Gensim API](https://radimrehurek.com/gensim/apiref.html)。



## 2 步骤一：训练语料的预处理

训练语料的预处理指的是将文档中原始的字符文本转换成Gensim模型所能理解的稀疏向量的过程。

通常，我们要处理的原生语料是一堆文档的集合，每一篇文档又是一些原生字符的集合。在交给Gensim的模型训练之前，我们需要将这些原生字符解析成Gensim能处理的稀疏向量的格式。由于语言和应用的多样性，我们需要先对原始的文本进行分词、去除停用词等操作，得到每一篇文档的特征列表。例如，在词袋模型中，文档的特征就是其包含的word：

texts = [['human', 'interface', 'computer'],

['survey', 'user', 'computer', 'system', 'response', 'time'],

['eps', 'user', 'interface', 'system'],

['system', 'human', 'system', 'eps'],

['user', 'response', 'time'],

['trees'],

['graph', 'trees'],

['graph', 'minors', 'trees'],

['graph', 'minors', 'survey']]

其中，corpus的每一个元素对应一篇文档。

接下来，我们可以调用Gensim提供的API建立语料特征（此处即是word）的索引字典，并将文本特征的原始表达转化成词袋模型对应的稀疏向量的表达。依然以词袋模型为例：

from gensim import corpora
 dictionary = corpora.Dictionary(texts)
 corpus = [dictionary.doc2bow(text) for text in texts]
 print corpus[0] # [(0, 1), (1, 1), (2, 1)]
 到这里，训练语料的预处理工作就完成了。我们得到了语料中每一篇文档对应的稀疏向量（这里是bow向量）；向量的每一个元素代表了一个word在这篇文档中出现的次数。值得注意的是，虽然词袋模型是很多主题模型的基本假设，这里介绍的doc2bow函数并不是将文本转化成稀疏向量的唯一途径。在下一小节里我们将介绍更多的向量变换函数。

最后，出于内存优化的考虑，Gensim支持文档的流式处理。我们需要做的，只是将上面的列表封装成一个Python迭代器；每一次迭代都返回一个稀疏向量即可。

class MyCorpus(object):
 def **iter**(self):
 for line in open('mycorpus.txt'):
 \# assume there's one document per line, tokens                   separated by whitespace
 yield dictionary.doc2bow(line.lower().split())

## 3 步骤二：主题向量的变换

对文本向量的变换是Gensim的核心。通过挖掘语料中隐藏的语义结构特征，我们最终可以变换出一个简洁高效的文本向量。

在Gensim中，每一个向量变换的操作都对应着一个主题模型，例如上一小节提到的对应着词袋模型的doc2bow变换。每一个模型又都是一个标准的Python对象。下面以TF-IDF模型为例，介绍Gensim模型的一般使用方法。

首先是模型对象的初始化。通常，Gensim模型都接受一段训练语料（注意在Gensim中，语料对应着一个稀疏向量的迭代器）作为初始化的参数。显然，越复杂的模型需要配置的参数越多。

from gensim import models
 tfidf = models.TfidfModel(corpus)
 其中，corpus是一个返回bow向量的迭代器。这两行代码将完成对corpus中出现的每一个特征的IDF值的统计工作。

接下来，我们可以调用这个模型将任意一段语料（依然是bow向量的迭代器）转化成TFIDF向量（的迭代器）。需要注意的是，这里的bow向量必须与训练语料的bow向量共享同一个特征字典（即共享同一个向量空间）。

doc_bow = [(0, 1), (1, 1)]
 print tfidf[doc_bow] # [(0, 0.70710678), (1, 0.70710678)]

注意，同样是出于内存的考虑，model[corpus]方法返回的是一个迭代器。如果要多次访问model[corpus]的返回结果，可以先将结果向量序列化到磁盘上。

我们也可以将训练好的模型持久化到磁盘上，以便下一次使用：

tfidf.save("./model.tfidf")
 tfidf = models.TfidfModel.load("./model.tfidf")
 Gensim内置了多种主题模型的向量变换，包括LDA，LSI，RP，HDP等。这些模型通常以bow向量或tfidf向量的语料为输入，生成相应的主题向量。所有的模型都支持流式计算。关于Gensim模型更多的介绍，可以参考这里：API Reference（[https://radimrehurek.com/gensim/apiref.html](https://link.jianshu.com?t=https%3A%2F%2Fradimrehurek.com%2Fgensim%2Fapiref.html)）



### 4 步骤三：文档相似度的计算

在得到每一篇文档对应的主题向量后，我们就可以计算文档之间的相似度，进而完成如文本聚类、信息检索之类的任务。在Gensim中，也提供了这一类任务的API接口。

以信息检索为例。对于一篇待检索的query，我们的目标是从文本集合中检索出主题相似度最高的文档。

首先，我们需要将待检索的query和文本放在同一个向量空间里进行表达（以LSI向量空间为例）：

```python
#构造LSI模型并将待检索的query和文本转化为LSI主题向量

#转换之前的corpus和query均是BOW向量

lsi_model = models.LsiModel(corpus, id2word=dictionary,num_topics=2)
 documents = lsi_model[corpus]
 query_vec = lsi_model[query]
```

接下来，我们用待检索的文档向量初始化一个相似度计算的对象：

```python
index = similarities.MatrixSimilarity(documents)
```


 我们也可以通过save()和load()方法持久化这个相似度矩阵：

```python
index.save('/tmp/test.index')
index = similarities.MatrixSimilarity.load('/tmp/test.index')
```


 注意，如果待检索的目标文档过多，使用similarities.MatrixSimilarity类往往会带来内存不够用的问题。此时，可以改用similarities.Similarity类。二者的接口基本保持一致。

最后，我们借助index对象计算任意一段query和所有文档的（余弦）相似度：

sims = index[query_vec]

**返回一个元组类型的迭代器：(idx, sim)**

## 5 补充

TF-IDF
 TF-IDF（注意：这里不是减号）是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
 字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。TF-IDF加权的各种形式常被搜索引擎应用，作为文件与用户查询之间相关程度的度量或评级。

1. 一个词预测主题能力越强，权重就越大，反之，权重就越小。我们在网页中看到“原子能”这个词，或多或少地能了解网页的主题。我们看到“应用”一次，对主题基本上还是一无所知。因此，“原子能“的权重就应该比应用大。
2. 应删除词的权重应该是零。

LDA文档主题生成模型
 LDA是一种文档主题生成模型，包含词、主题和文档三层结构。

所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布。

LDA是一种非监督机器学习技术，可以用来识别大规模文档集或语料库中潜藏的主题信息。它采用了词袋的方法，这种方法将每一篇文档视为一个词频向量，从而将文本信息转化为了易于建模的数字信息。

但是词袋方法没有考虑词与词之间的顺序，这简化了问题的复杂性，同时也为模型的改进提供了契机。每一篇文档代表了一些主题所构成的一个概率分布，而每一个主题又代表了很多单词所构成的一个概率分布。





#### word2vec 

　　　　在gensim中，word2vec 相关的参数都在包gensim.models.word2vec中。完整函数如下：

```python
gensim.models.word2vec.Word2Vec(sentences=None,size=100,alpha=0.025,window=5, min_count=5, max_vocab_size=None, sample=0.001,seed=1, workers=3,min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>,iter=5,null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
```


　　　　1) sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。对于大语料集，建议使用**BrownCorpus**,**Text8Corpus**或**lineSentence**构建。

　　　　2) size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，视语料库的大小而定。

​               3) alpha： 是初始的学习速率，在训练过程中会线性地递减到min_alpha。

　　　　4) window：即词向量上下文最大距离，skip-gram和cbow算法是基于滑动窗口来做预测。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。对于一般的语料这个值推荐在[5,10]之间。

　           5) min_count:：可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。

​               6) max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。

​               7) sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。

​               8) seed：用于随机数发生器。与初始化词向量有关。

​               9) workers：用于控制训练的并行数。此参数只有在安装了Cpython后才有效，否则只能使用单核。

​             10) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每    轮的迭代步长可以由iter，alpha， min_alpha一起得出。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。

　　　　 11) sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。

​                12)hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。

　　　　  13) negative:如果大于零，则会采用negativesampling，用于设置多少个noise words（一般是5-20）。

　　　　  14) cbow_mean: 仅用于CBOW在做投影的时候，为0，则采用上下文的词向量之和，为1则为上下文的词向量的平均值。默认值也是1,不推荐修改默认值。

​                 15) hashfxn： hash函数来初始化权重，默认使用python的hash函数。

　　　　 16) iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。

​                 17) trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。

​                 18) sorted_vocab： 如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。

​                 19) batch_words：每一批的传递给线程的单词的数量，默认为10000。






