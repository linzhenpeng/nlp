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





















