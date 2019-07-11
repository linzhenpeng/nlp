https://www.jianshu.com/p/4bc597e3bc72

https://blog.csdn.net/qq_41664845/article/details/82869596

https://blog.csdn.net/qq_14959801/article/details/51273360

http://blog.itpub.net/31562039/viewspace-2286669/

http://www.hankcs.com/nlp/textrank-algorithm-to-extract-the-keywords-java-implementation.html

http://www.hankcs.com/nlp/textrank-algorithm-java-implementation-of-automatic-abstract.html

[TextRank](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)是受到Google的[PageRank](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)的启发，通过把文本分割成若干组成单元(单词、句子)并建立图模型, 利用投票机制对文本中的重要成分进行排序, 仅利用单篇文档本身的信息即可实现关键词提取、文摘。和 LDA、HMM 等模型不同, TextRank不需要事先对多篇文档进行学习训练, 因其简洁有效而得到广泛应用。
 TextRank 一般模型可以表示为一个有向有权图 G =(V, E), 由点集合 V和边集合 E 组成, E 是V ×V的子集。图中任两点 Vi , Vj 之间边的权重为 wji , 对于一个给定的点 Vi, In(Vi) 为 指 向 该 点 的 点 集 合 , Out(Vi) 为点 Vi 指向的点集合。点 Vi 的得分定义如下:



![](F:\git\nlp\算法\images\TextRank\gif.gif)

其中, d 为阻尼系数, 取值范围为 0 到 1, 代表从图中某一特定点指向其他任意点的概率, 一般取值为 0.85。使用TextRank 算法计算图中各点的得分时, 需要给图中的点指定任意的初值, 并递归计算直到收敛, 即图中任意一点的误差率小于给定的极限值时就可以达到收敛, 一般该极限值取 0.0001。

TextRank中一个单词$i$的权重取决于与在$i$前面的各个点$j$组成的$(i,j)$这条边的权重，以及这个点到其他其他边的权重之和。


其中，$S(V_i)$表示句子i的权重(weight sum)，右侧的求和表示每个相邻句子对本句子的贡献程度。在单文档中，我们可以粗略认为所有句子都是相邻的，不需要像多文档一样进行多个窗口的生成和抽取，仅需单一文档窗口即可。求和的分母$w_{ji}$表示两个句子的相似度，而分母仍然表示权重，$S(V_j)​$代表上次迭代出的句子j的权重，因此，TextRank算法也是类似PageRank算法的、不断迭代的过程。



需要注意的是   做关键词提取时  需要指定  统计窗口  文字前后出现可以认为是  pagerank 中的指向的页面  然后进行迭代 最后取得收敛后的分数 

做摘要提取时  认为所有的 句子是相连的,需要计算手游句子的相互的相识度作为 初始的权重值, 相识度算法 有

**BM25算法**，通常用来作搜索相关性平分。一句话概况其主要思想：对Query进行语素解析，生成语素qi；然后，对于每个搜索结果D，计算每个语素qi与D的相关性得分，最后，将qi相对于D的相关性得分进行加权求和，从而得到Query与D的相关性得分。

**余弦相识度**

**编辑距离** 

等等 





我们可以通过TextRank算法，对文章做关键词的提取以及自动文摘提取。
 更详细的内容参见：[和textrank4ZH代码一模一样的算法详细解读](https://www.cnblogs.com/www-caiyin-com/p/9719897.html)

相关组件为textrank4zh，其用于抽取中文文章的关键字以及关键句（作为文摘）
安装方法很简单：

```python
pip install textrank4zh
```

我们用一篇新闻作为测试的数据 :

```
美军一架无人机在夏威夷坠毁，引发山火目前仍未扑灭
综合美国《星条旗报》和《陆军时报》的消息报道，美国军方称，当地时间7月10日，陆军第25步兵师的一架无人机在怀厄奈山脉进行的一次飞行训练中坠毁，引发灌木丛火灾。
根据美国陆军的声明，坠毁的无人机为一架RQ-7“暗影”战术无人机，这架无人机于10日下午3:30左右在瓦胡岛中部的斯科菲尔德兵营附近地区坠毁。第25步兵师发言人、高级军士长Andrew Porch说，无人机在檀香山以北约一小时车程的斯科菲尔德军营附近的怀厄奈山区坠毁。
火灾发生后，陆军荒地消防部门与联邦消防局和檀香山消防局一起做出了响应。截至10日晚上，灭火工作以及对该地区受火灾影响的评估工作仍在进行中。檀香山消防局消防队长苏格兰·西格兰特10日表示，檀香山消防局的一架直升机正向瓦胡岛中部发生火灾的地方吊水灭火。
美国陆军方面称，第25步兵师也派出了飞机对该地区进行评估并为灭火工作提供帮助。此次坠机事件发生时没有人员受伤，也没有其他飞机在该地区活动。坠机事故原因尚未确定。一旦官方调查完成，将会发布更多信息。
```

我们通过非常简单的代码实现TextRank的使用：

```python
# coding:utf-8

from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':
    f = open('./news.txt', mode='r', encoding='utf-8')
    text = f.read()
    f.close()

    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=5)
    print('关键词：')
    for item in tr4w.get_keywords(10, word_min_len=1):
        print(item['word'], item['weight'])

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source = 'no_stop_words')
    data = pd.DataFrame(data=tr4s.key_sentences)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(data['weight'], 'ro-', lw=2, ms=5, alpha=0.7, mec='#404040')
    plt.grid(b=True, ls=':', color='#606060')
    plt.xlabel('句子', fontsize=12)
    plt.ylabel('重要度', fontsize=12)
    plt.title('句子的重要度曲线', fontsize=15)
    plt.show()

    key_sentences = tr4s.get_key_sentences(num=10, sentence_min_len=2)
    for sentence in key_sentences:
        print(sentence['weight'], sentence['sentence'])
```



输出结果:

关键词：
无人机 0.036267977551917414
坠毁 0.028512397260500873
檀香山 0.027576012597350895
地区 0.02322843411465643
发生 0.021868517361964143
陆军 0.021018422062404306
火灾 0.020900593315012195
消防局 0.02067719243929721
灭火 0.02000556049849132
步兵师 0.019578160139906563

![](F:\git\nlp\算法\images\TextRank\下载.png)

0.13087981109160346 综合美国《星条旗报》和《陆军时报》的消息报道，美国军方称，当地时间7月10日，陆军第25步兵师的一架无人机在怀厄奈山脉进行的一次飞行训练中坠毁，引发灌木丛火灾
0.10625502440658581 根据美国陆军的声明，坠毁的无人机为一架RQ-7“暗影”战术无人机，这架无人机于10日下午3:30左右在瓦胡岛中部的斯科菲尔德兵营附近地区坠毁
0.10478944987284321 檀香山消防局消防队长苏格兰·西格兰特10日表示，檀香山消防局的一架直升机正向瓦胡岛中部发生火灾的地方吊水灭火
0.09606517766502452 截至10日晚上，灭火工作以及对该地区受火灾影响的评估工作仍在进行中
0.09090909090909091 一旦官方调查完成，将会发布更多信息
0.08995824909404901 第25步兵师发言人、高级军士长Andrew Porch说，无人机在檀香山以北约一小时车程的斯科菲尔德军营附近的怀厄奈山区坠毁
0.08961913013573382 美国陆军方面称，第25步兵师也派出了飞机对该地区进行评估并为灭火工作提供帮助
0.07804878094418242 此次坠机事件发生时没有人员受伤，也没有其他飞机在该地区活动
0.07326119907785457 火灾发生后，陆军荒地消防部门与联邦消防局和檀香山消防局一起做出了响应
0.07185551780768326 美军一架无人机在夏威夷坠毁，引发山火目前仍未扑灭





从结果看 还是很准确的找出 全文中最重要的关键词  无人机和坠毁   



```python
def solve(self): #针对抽关键句
        for cnt, doc in enumerate(self.docs):
            scores = self.bm25.simall(doc) #在本实现中，使用的不是前面提到的公式，而是使用的BM25算法，之前会有一个预处理（self.bm25 = BM25(docs)），然后求doc跟其他所有预料的相似程度。
            self.weight.append(scores)
            self.weight_sum.append(sum(scores)-scores[cnt])#需要减掉本身的权重。
            self.vertex.append(1.0)
        for _ in range(self.max_iter):
            m = []
            max_diff = 0
            for i in range(self.D):#每个文本都要计算与其他所有文档的链接，然后计算出重要程度。
                m.append(1-self.d)
                for j in range(self.D):
                    if j == i or self.weight_sum[j] == 0:
                        continue
                    m[-1] += (self.d*self.weight[j][i]
                              / self.weight_sum[j]*self.vertex[j])
                              #利用前面的公式求解
                if abs(m[-1] - self.vertex[i]) > max_diff:
                #找到该次迭代中，变化最大的一次情况。
                    max_diff = abs(m[-1] - self.vertex[i])
            self.vertex = m
            if max_diff <= self.min_diff:#当变化最大的一次，仍然小于某个阈值时认为可以满足跳出条件，不用再循环指定的次数。
                break
        self.top = list(enumerate(self.vertex))
        self.top = sorted(self.top, key=lambda x: x[1], reverse=True)
 
 
def solve(self):#针对抽关键词
        for doc in self.docs:
            que = []
            for word in doc:
                if word not in self.words:
                    self.words[word] = set()
                    self.vertex[word] = 1.0
                que.append(word)
                if len(que) > 5:
                    que.pop(0)
                for w1 in que:
                    for w2 in que:
                        if w1 == w2:
                            continue
                        self.words[w1].add(w2)
                        self.words[w2].add(w1)
        for _ in range(self.max_iter):
            m = {}
            max_diff = 0
            tmp = filter(lambda x: len(self.words[x[0]]) > 0,
                         self.vertex.items())
            tmp = sorted(tmp, key=lambda x: x[1] / len(self.words[x[0]]))
            for k, v in tmp:
                for j in self.words[k]:
                    if k == j:
                        continue
                    if j not in m:
                        m[j] = 1 - self.d
                    m[j] += (self.d / len(self.words[k]) * self.vertex[k]) #利用之前提到的公式，简化的结果。
            for k in self.vertex:
                if k in m and k in self.vertex:
                    if abs(m[k] - self.vertex[k]) > max_diff:
                        max_diff = abs(m[k] - self.vertex[k])
            self.vertex = m
            if max_diff <= self.min_diff:
                break
        self.top = list(self.vertex.items())
        self.top = sorted(self.top, key=lambda x: x[1], reverse=True)

```






