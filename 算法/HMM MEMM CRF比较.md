https://www.zhihu.com/question/35866596/answer/236886066

https://www.cnblogs.com/hellochennan/p/6624509.html




## **HMM**

最早接触的是HMM。较早做过一个项目，关于声波手势识别，跟声音识别的机制一样，使用的正是HMM的一套方法。后来又用到了*kalman filter*,之后做序列标注任务接触到了CRF，所以整个概率图模型还是接触的方面还蛮多。

## **理解HMM**

在2.2、2.3中提序列的建模问题时，我们只是讨论了常规的序列数据，e.g., ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88X_%7B1%7D%2C%5Ccdots%2CX_%7Bn%7D%EF%BC%89) ,像2.3的图片那样。像这种序列一般用马尔科夫模型就可以胜任。实际上我们碰到的更多的使用HMM的场景是每个节点 ![[公式]](https://www.zhihu.com/equation?tex=+X_%7Bi%7D) 下还附带着另一个节点 ![[公式]](https://www.zhihu.com/equation?tex=+Y_%7Bi%7D) ，正所谓**隐含**马尔科夫模型，那么除了正常的节点，还要将**隐含状态节点**也得建模进去。正儿八经地，将 ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bi%7D+%E3%80%81+Y_%7Bi%7D+) 换成 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bi%7D+%E3%80%81o_%7Bi%7D) ,并且他们的名称变为状态节点、观测节点。状态节点正是我的隐状态。

HMM属于典型的生成式模型。对照2.1的讲解，应该是要从训练数据中学到数据的各种分布，那么有哪些分布呢以及是什么呢？直接正面回答的话，正是**HMM的5要素**，其中有3个就是整个数据的不同角度的概率分布：

- ![[公式]](https://www.zhihu.com/equation?tex=N) ，隐藏状态集 ![[公式]](https://www.zhihu.com/equation?tex=N+%3D+%5Clbrace+q_%7B1%7D%2C+%5Ccdots%2C+q_%7BN%7D+%5Crbrace) , 我的隐藏节点不能随意取，只能限定取包含在隐藏状态集中的符号。
- ![[公式]](https://www.zhihu.com/equation?tex=M) ，观测集 ![[公式]](https://www.zhihu.com/equation?tex=M+%3D+%5Clbrace+v_%7B1%7D%2C+%5Ccdots%2C+v_%7BM%7D+%5Crbrace) , 同样我的观测节点不能随意取，只能限定取包含在观测状态集中的符号。
- ![[公式]](https://www.zhihu.com/equation?tex=A) ，状态转移概率矩阵，这个就是其中一个概率分布。他是个矩阵， ![[公式]](https://www.zhihu.com/equation?tex=+A%3D+%5Ba_%7Bij%7D%5D_%7BN+%5Ctimes+N%7D+) （N为隐藏状态集元素个数），其中 ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D+%3D+P%28i_%7Bt%2B1%7D%7Ci_%7Bt%7D%29%EF%BC%8C+i_%7Bt%7D) 即第i个隐状态节点,即所谓的状态转移嘛。
- ![[公式]](https://www.zhihu.com/equation?tex=B) ，观测概率矩阵，这个就是另一个概率分布。他是个矩阵， ![[公式]](https://www.zhihu.com/equation?tex=B+%3D+%5Bb_%7Bij%7D%5D_%7BN+%5Ctimes+M%7D) （N为隐藏状态集元素个数，M为观测集元素个数），其中 ![[公式]](https://www.zhihu.com/equation?tex=b_%7Bij%7D+%3D+P%28o_%7Bt%7D%7Ci_%7Bt%7D%29%EF%BC%8C+o_%7Bt%7D) 即第i个观测节点, ![[公式]](https://www.zhihu.com/equation?tex=+i_%7Bt%7D) 即第i个隐状态节点,即所谓的观测概率（发射概率）嘛。
- ![[公式]](https://www.zhihu.com/equation?tex=%CF%80) ，在第一个隐状态节点 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bt%7D) ,我得人工单独赋予，我第一个隐状态节点的隐状态是 ![[公式]](https://www.zhihu.com/equation?tex=N) 中的每一个的概率分别是多少，然后 ![[公式]](https://www.zhihu.com/equation?tex=%CF%80) 就是其概率分布。

所以图看起来是这样的：

![img](https://pic4.zhimg.com/50/v2-d4077c2dbd9899d8896751a28490c9c7_hd.jpg)

看的很清楚，我的模型先去学习要确定以上5要素，之后在inference阶段的工作流程是：首先，隐状态节点 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bt%7D) 是不能直接观测到的数据节点， ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bt%7D) 才是能观测到的节点，并且注意箭头的指向表示了依赖生成条件关系， ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bt%7D) 在A的指导下生成下一个隐状态节点 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bt%2B1%7D) ，并且 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bt%7D) 在 ![[公式]](https://www.zhihu.com/equation?tex=B) 的指导下生成依赖于该 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bt%7D) 的观测节点 ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bt%7D) , 并且我只能观测到序列 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D%29) 。

好，举例子说明（序列标注问题，POS，标注集BES）：

> input: "学习出一个模型，然后再预测出一条指定"
>
> expected output: 学/B 习/E 出/S 一/B 个/E 模/B 型/E ，/S 然/B 后/E 再/E 预/B 测/E ……
>
> 其中，input里面所有的char构成的字表，形成观测集 ![[公式]](https://www.zhihu.com/equation?tex=M) ，因为字序列在inference阶段是我所能看见的；标注集BES构成隐藏状态集 ![[公式]](https://www.zhihu.com/equation?tex=N) ，这是我无法直接获取的，也是我的预测任务；至于 ![[公式]](https://www.zhihu.com/equation?tex=A%E3%80%81B%E3%80%81%CF%80) ，这些概率分布信息（上帝信息）都是我在学习过程中所确定的参数。

然后一般初次接触的话会疑问：为什么要这样？……好吧，就应该是这样啊，根据具有同时带着隐藏状态节点和观测节点的类型的序列，在HMM下就是这样子建模的。

下面来点高层次的理解：

1. 根据概率图分类，可以看到HMM属于有向图，并且是生成式模型，直接对联合概率分布建模 ![[公式]](https://www.zhihu.com/equation?tex=P%28O%2CI%29+%3D+%5Csum_%7Bt%3D1%7D%5E%7BT%7DP%28O_%7Bt%7D+%7C+O_%7Bt-1%7D%29P%28I_%7Bt%7D+%7C+O_%7Bt%7D%29) (注意，这个公式不在模型运行的任何阶段能体现出来，只是我们都去这么来表示HMM是个生成式模型，他的联合概率 ![[公式]](https://www.zhihu.com/equation?tex=P%28O%2CI%29) 就是这么计算的)。
2. 并且B中 ![[公式]](https://www.zhihu.com/equation?tex=b_%7Bij%7D+%3D+P%28o_%7Bt%7D%7Ci_%7Bt%7D%29) ，这意味着o对i有依赖性。
3. 在A中， ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D+%3D+P%28i_%7Bt%2B1%7D%7Ci_%7Bt%7D%29) ，也就是说只遵循了一阶马尔科夫假设，1-gram。试想，如果数据的依赖超过1-gram，那肯定HMM肯定是考虑不进去的。这一点限制了HMM的性能。

## **模型运行过程**

模型的运行过程（工作流程）对应了HMM的3个问题。

## **学习训练过程**

对照2.1的讲解，HMM学习训练的过程，就是找出数据的分布情况，也就是模型参数的确定。

主要学习算法按照训练数据除了观测状态序列 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D%29) 是否还有隐状态序列 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88i_%7B1%7D%2C+%5Ccdots%2C+i_%7Bi%7D%29) 分为：

- 极大似然估计, with 隐状态序列
- Baum-Welch(前向后向), without 隐状态序列

感觉不用做很多的介绍，都是很实实在在的算法，看懂了就能理解。简要提一下。

**1. 极大似然估计**

一般做NLP的序列标注等任务，在训练阶段肯定是有隐状态序列的。所以极大似然估计法是非常常用的学习算法，我见过的很多代码里面也是这么计算的。比较简单。

- step1. 算A

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Ba_%7Bij%7D%7D+%3D+%5Cfrac%7BA_%7Bij%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BN%7DA_%7Bij%7D%7D+)

- step2. 算B

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bb_%7Bj%7D%7D%28k%29+%3D+%5Cfrac%7BB_%7Bjk%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BM%7DB_%7Bjk%7D%7D)

- step3. 直接估计 ![[公式]](https://www.zhihu.com/equation?tex=%CF%80)

比如说，在代码里计算完了就是这样的： 

![img](https://pic4.zhimg.com/50/v2-5343666b942f952f6c79c4fa5e2dfdd9_hd.jpg)![img](https://pic4.zhimg.com/80/v2-5343666b942f952f6c79c4fa5e2dfdd9_hd.jpg)

![img](https://pic4.zhimg.com/50/v2-0d695539c785fe16cfcfff3b9bd0c164_hd.jpg)![img](https://pic4.zhimg.com/80/v2-0d695539c785fe16cfcfff3b9bd0c164_hd.jpg)

![img](https://pic2.zhimg.com/50/v2-443053bd4342c71bf602ac15aa7c8731_hd.jpg)![img](https://pic2.zhimg.com/80/v2-443053bd4342c71bf602ac15aa7c8731_hd.jpg)



**2. Baum-Welch(前向后向)**

就是一个EM的过程，如果你对EM的工作流程有经验的话，对这个Baum-Welch一看就懂。EM的过程就是初始化一套值，然后迭代计算，根据结果再调整值，再迭代，最后收敛……好吧，这个理解是没有捷径的，去隔壁钻研EM吧。

这里只提一下核心。因为我们手里没有隐状态序列 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88i_%7B1%7D%2C+%5Ccdots%2C+i_%7Bi%7D%29) 信息，所以我先必须给初值 ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D%5E%7B0%7D%2C+b_%7Bj%7D%28k%29%5E%7B0%7D%2C+%5Cpi%5E%7B0%7D) ，初步确定模型，然后再迭代计算出 ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D%5E%7Bn%7D%2C+b_%7Bj%7D%28k%29%5E%7Bn%7D%2C+%5Cpi%5E%7Bn%7D) ,中间计算过程会用到给出的观测状态序列 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D%29) 。另外，收敛性由EM的XXX定理保证。

## **序列标注（解码）过程**

好了，学习完了HMM的分布参数，也就确定了一个HMM模型。需要注意的是，这个HMM是对我这一批全部的数据进行训练所得到的参数。

序列标注问题也就是“预测过程”，通常称为解码过程。对应了序列建模问题3.。对于序列标注问题，我们只需要学习出一个HMM模型即可，后面所有的新的sample我都用这一个HMM去apply。

我们的目的是，在学习后已知了 ![[公式]](https://www.zhihu.com/equation?tex=P%28Q%2CO%29) ,现在要求出 ![[公式]](https://www.zhihu.com/equation?tex=P%28Q%7CO%29) ，进一步

![[公式]](https://www.zhihu.com/equation?tex=Q_%7Bmax%7D+%3D+argmax_%7BallQ%7D%5Cfrac%7BP%28Q%2CO%29%7D%7BP%28O%29%7D)

再直白点就是，我现在要在给定的观测序列下找出一条隐状态序列，条件是这个隐状态序列的概率是最大的那个。



具体地，都是用Viterbi算法解码，是用DP思想减少重复的计算。Viterbi也是满大街的，不过要说的是，Viterbi不是HMM的专属，也不是任何模型的专属，他只是恰好被满足了被HMM用来使用的条件。谁知，现在大家都把Viterbi跟HMM捆绑在一起了, shame。

Viterbi计算有向无环图的一条最大路径，应该还好理解。如图：

![img](https://pic3.zhimg.com/50/v2-71f1ea9abbab357f7d9bad1138ee7344_hd.jpg)

关键是注意，每次工作热点区只涉及到t 与 t-1,这对应了DP的无后效性的条件。如果对某些同学还是很难理解，请参考[这个答案](https://www.zhihu.com/question/20136144)下@Kiwee的回答吧。

## **序列概率过程**

我通过HMM计算出序列的概率又有什么用？针对这个点我把这个问题详细说一下。 

实际上，序列概率过程对应了序列建模问题2.，即序列分类。
在3.2.2第一句话我说，在序列标注问题中，我用一批完整的数据训练出了一支HMM模型即可。好，那在序列分类问题就不是训练一个HMM模型了。我应该这么做（结合语音分类识别例子）： 

> 目标：识别声音是A发出的还是B发出的。 
> HMM建模过程： 
>       \1.   训练：我将所有A说的语音数据作为dataset_A,将所有B说的语音数据作为dataset_B（当然，先要分别对dataset A ,B做预处理encode为元数据节点，形成sequences）,然后分别用dataset_A、dataset_B去训练出HMM_A/HMM_B
>       \2.   inference：来了一条新的sample（sequence），我不知道是A的还是B的，没问题，分别用HMM_A/HMM_B计算一遍序列的概率得到 ![[公式]](https://www.zhihu.com/equation?tex=P_%7BA%7D%EF%BC%88S%EF%BC%89%E3%80%81P_%7BB%7D%EF%BC%88S%EF%BC%89) ，比较两者大小，哪个概率大说明哪个更合理，更大概率作为目标类别。



所以，本小节的理解重点在于，**如何对一条序列计算其整体的概率**。即目标是计算出 ![[公式]](https://www.zhihu.com/equation?tex=P%28O%7C%CE%BB%29) 。这个问题前辈们在他们的经典中说的非常好了，比如参考李航老师整理的：

- 直接计算法（穷举搜索）
- 前向算法
- 后向算法

后面两个算法采用了DP思想，减少计算量，即每一次直接引用前一个时刻的计算结果以避免重复计算，跟Viterbi一样的技巧。

还是那句，因为这篇文档不是专门讲算法细节的，所以不详细展开这些。毕竟，所有的科普HMM、CRF的博客貌似都是在扯这些算法，妥妥的街货，就不搬运了。



## **MEMM**

MEMM，即最大熵马尔科夫模型，这个是在接触了HMM、CRF之后才知道的一个模型。说到MEMM这一节时，得转换思维了，因为现在这MEMM属于判别式模型。

不过有一点很尴尬，MEMM貌似被使用或者讲解引用的不及HMM、CRF。

## **理解MEMM**

这里还是啰嗦强调一下，MEMM正因为是判别模型，所以不废话，我上来就直接为了确定边界而去建模，比如说序列求概率（分类）问题，我直接考虑找出函数分类边界。这一点跟HMM的思维方式发生了很大的变化，如果不对这一点有意识，那么很难理解为什么MEMM、CRF要这么做。

HMM中，观测节点 ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bi%7D) 依赖隐藏状态节点 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bi%7D) ,也就意味着我的观测节点只依赖当前时刻的隐藏状态。但在更多的实际场景下，观测序列是需要很多的特征来刻画的，比如说，我在做NER时，我的标注 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bi%7D) 不仅跟当前状态 ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bi%7D) 相关，而且还跟前后标注 ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bj%7D%28j+%5Cneq+i%29) 相关，比如字母大小写、词性等等。

为此，提出来的MEMM模型就是能够直接允许**“定义特征”**，直接学习条件概率，即 ![[公式]](https://www.zhihu.com/equation?tex=P%28i_%7Bi%7D%7Ci_%7Bi-1%7D%2Co_%7Bi%7D%29+%28i+%3D+1%2C%5Ccdots%2Cn%29) , 总体为：

![[公式]](https://www.zhihu.com/equation?tex=P%28I%7CO%29+%3D+%5Cprod_%7Bt%3D1%7D%5E%7Bn%7DP%28i_%7Bi%7D%7Ci_%7Bi-1%7D%2Co_%7Bi%7D%29%2C+i+%3D+1%2C%5Ccdots%2Cn)

并且， ![[公式]](https://www.zhihu.com/equation?tex=P%28i%7Ci%5E%7B%27%7D%2Co%29) 这个概率通过最大熵分类器建模（取名MEMM的原因）:

![](./images/概率图模型比较/equation.svg)

重点来了，这是ME的内容，也是理解MEMM的关键： ![[公式]](https://www.zhihu.com/equation?tex=Z%28o%2Ci%5E%7B%27%7D%29) 这部分是归一化； ![[公式]](https://www.zhihu.com/equation?tex=f_%7Ba%7D%28o%2Ci%29) 是**特征函数**，具体点，这个函数是需要去定义的; ![[公式]](https://www.zhihu.com/equation?tex=%CE%BB) 是特征函数的权重，这是个未知参数，需要从训练阶段学习而得。

比如我可以这么定义特征函数：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+f_%7Ba%7D%28o%2Ci%29+%3D+%5Cbegin%7Bcases%7D+1%26+%5Ctext%7B%E6%BB%A1%E8%B6%B3%E7%89%B9%E5%AE%9A%E6%9D%A1%E4%BB%B6%7D%EF%BC%8C%5C%5C+0%26+%5Ctext%7Bother%7D+%5Cend%7Bcases%7D+%5Cend%7Bequation%7D)

其中，特征函数 ![[公式]](https://www.zhihu.com/equation?tex=+f_%7Ba%7D%28o%2Ci%29) 个数可任意制定， ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88a+%3D+1%2C+%5Ccdots%2C+n%EF%BC%89)



所以总体上，MEMM的建模公式这样：

![[公式]](https://www.zhihu.com/equation?tex=P%28I%7CO%29+%3D+%5Cprod_%7Bt%3D1%7D%5E%7Bn%7D%5Cfrac%7B+exp%28%5Csum_%7Ba%7D%29%5Clambda_%7Ba%7Df_%7Ba%7D%28o%2Ci%29+%7D%7BZ%28o%2Ci_%7Bi-1%7D%29%7D+%2C+i+%3D+1%2C%5Ccdots%2Cn)



是的，公式这部分之所以长成这样，是由ME模型决定的。

请务必注意，理解**判别模型**和**定义特征**两部分含义，这已经涉及到CRF的雏形了。

所以说，他是判别式模型，直接对条件概率建模。 上图： 

![img](https://pic4.zhimg.com/80/v2-cb2cc25593fcaf06e682191d551ba03b_hd.jpg)

MEMM需要两点注意：

1. 与HMM的 ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bi%7D) 依赖 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bi%7D) 不一样，MEMM当前隐藏状态 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bi%7D) 应该是依赖当前时刻的观测节点 ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bi%7D) 和上一时刻的隐藏节点 ![[公式]](https://www.zhihu.com/equation?tex=i_%7Bi-1%7D)
2. 需要注意，之所以图的箭头这么画，是由MEMM的公式决定的，而公式是creator定义出来的。

好了，走一遍完整流程。

> step1. 先预定义特征函数 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Ba%7D%28o%2Ci%29) ，
> step2. 在给定的数据上，训练模型，确定参数，即确定了MEMM模型
> step3. 用确定的模型做序列标注问题或者序列求概率问题。

## **模型运行过程**

MEMM模型的工作流程也包括了学习训练问题、序列标注问题、序列求概率问题。

## **学习训练过程**

一套MEMM由一套参数唯一确定，同样地，我需要通过训练数据学习这些参数。MEMM模型很自然需要学习里面的特征权重λ。

不过跟HMM不同的是，因为HMM是生成式模型，参数即为各种概率分布元参数，数据量足够可以用最大似然估计。而判别式模型是用函数直接判别，学习边界，MEMM即通过特征函数来界定。但同样，MEMM也有极大似然估计方法、梯度下降、牛顿迭代法、拟牛顿下降、BFGS、L-BFGS等等。各位应该对各种优化方法有所了解的。

嗯，具体详细求解过程貌似问题不大。

## **序列标注过程**

还是跟HMM一样的，用学习好的MEMM模型，在新的sample（观测序列 ![[公式]](https://www.zhihu.com/equation?tex=+o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D) ）上找出一条概率最大最可能的隐状态序列 ![[公式]](https://www.zhihu.com/equation?tex=i_%7B1%7D%2C+%5Ccdots%2C+i_%7Bi%7D) 。

只是现在的图中的每个隐状态节点的概率求法有一些差异而已,正确将每个节点的概率表示清楚，路径求解过程还是一样，采用viterbi算法。

## **序列求概率过程**

跟HMM举的例子一样的，也是分别去为每一批数据训练构建特定的MEMM，然后根据序列在每个MEMM模型的不同得分概率，选择最高分数的模型为wanted类别。

应该可以不用展开，吧……

## **标注偏置？**

MEMM讨论的最多的是他的labeling bias 问题。

**1. 现象**

是从街货上烤过来的…… 

![img](https://pic4.zhimg.com/50/v2-40f9945cdffb12cfec84bebc7b7e3be5_hd.jpg)![img](https://pic4.zhimg.com/80/v2-40f9945cdffb12cfec84bebc7b7e3be5_hd.jpg)

用Viterbi算法解码MEMM，状态1倾向于转换到状态2，同时状态2倾向于保留在状态2。 解码过程细节（需要会viterbi算法这个前提）：

> P(1-> 1-> 1-> 1)= 0.4 x 0.45 x 0.5 = 0.09 ，
> P(2->2->2->2)= 0.2 X 0.3 X 0.3 = 0.018，
> P(1->2->1->2)= 0.6 X 0.2 X 0.5 = 0.06，
> P(1->1->2->2)= 0.4 X 0.55 X 0.3 = 0.066 

但是得到的最优的状态转换路径是1->1->1->1，为什么呢？因为状态2可以转换的状态比状态1要多，从而使转移概率降低,即MEMM倾向于选择拥有更少转移的状态。

**2. 解释原因**

直接看MEMM公式：

![[公式]](https://www.zhihu.com/equation?tex=P%28I%7CO%29+%3D+%5Cprod_%7Bt%3D1%7D%5E%7Bn%7D%5Cfrac%7B+exp%28%5Csum_%7Ba%7D%29%5Clambda_%7Ba%7Df_%7Ba%7D%28o%2Ci%29+%7D%7BZ%28o%2Ci_%7Bi-1%7D%29%7D+%2C+i+%3D+1%2C%5Ccdots%2Cn)

![[公式]](https://www.zhihu.com/equation?tex=%E2%88%91) 求和的作用在概率中是归一化，但是这里归一化放在了指数内部，管这叫local归一化。 来了，viterbi求解过程，是用dp的状态转移公式（MEMM的没展开，请参考CRF下面的公式），因为是局部归一化，所以MEMM的viterbi的转移公式的第二部分出现了问题，导致dp无法正确的递归到全局的最优。

![[公式]](https://www.zhihu.com/equation?tex=+%5Cdelta_%7Bi%2B1%7D+%3D+max_%7B1+%5Cle+j+%5Cle+m%7D%5Clbrace+%5Cdelta_%7Bi%7D%28I%29+%2B+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29+%5Crbrace+)



## **CRF**

我觉得一旦有了一个清晰的工作流程，那么按部就班地，没有什么很难理解的地方，因为整体框架已经胸有成竹了，剩下了也只有添砖加瓦小修小补了。有了上面的过程基础，CRF也是类似的，只是有方法论上的细微区别。

## **理解CRF**

请看第一张概率图模型构架图，CRF上面是马尔科夫随机场（马尔科夫网络），而条件随机场是在给定的随机变量 ![[公式]](https://www.zhihu.com/equation?tex=X) （具体，对应观测序列 ![[公式]](https://www.zhihu.com/equation?tex=o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D) ）条件下，随机变量 ![[公式]](https://www.zhihu.com/equation?tex=Y) （具体，对应隐状态序列 ![[公式]](https://www.zhihu.com/equation?tex=i_%7B1%7D%2C+%5Ccdots%2C+i_%7Bi%7D) 的马尔科夫随机场。
广义的CRF的定义是： 满足 ![[公式]](https://www.zhihu.com/equation?tex=P%28Y_%7Bv%7D%7CX%2CY_%7Bw%7D%2Cw+%5Cneq+v%29+%3D+P%28Y_%7Bv%7D%7CX%2CY_%7Bw%7D%2Cw+%5Csim+v%29+) 的马尔科夫随机场叫做条件随机场（CRF）。

不过一般说CRF为序列建模，就专指CRF线性链（linear chain CRF）：

![img](https://pic3.zhimg.com/50/v2-c5e2e782e35f6412ed65e58cdda0964e_hd.jpg)

在2.1.2中有提到过，概率无向图的联合概率分布可以在因子分解下表示为：

![[公式]](https://www.zhihu.com/equation?tex=P%28Y+%7C+X%29%3D%5Cfrac%7B1%7D%7BZ%28x%29%7D+%5Cprod_%7Bc%7D%5Cpsi_%7Bc%7D%28Y_%7Bc%7D%7CX+%29+%3D+%5Cfrac%7B1%7D%7BZ%28x%29%7D+%5Cprod_%7Bc%7D+e%5E%7B%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28c%2Cy%7Cc%2Cx%29%7D+%3D+%5Cfrac%7B1%7D%7BZ%28x%29%7D+e%5E%7B%5Csum_%7Bc%7D%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28y_%7Bi%7D%2Cy_%7Bi-1%7D%2Cx%2Ci%29%7D)

而在线性链CRF示意图中，每一个（ ![[公式]](https://www.zhihu.com/equation?tex=I_%7Bi%7D+%5Csim+O_%7Bi%7D) ）对为一个最大团,即在上式中 ![[公式]](https://www.zhihu.com/equation?tex=c+%3D+i) 。并且线性链CRF满足 ![[公式]](https://www.zhihu.com/equation?tex=P%28I_%7Bi%7D%7CO%2CI_%7B1%7D%2C%5Ccdots%2C+I_%7Bn%7D%29+%3D+P%28I_%7Bi%7D%7CO%2CI_%7Bi-1%7D%2CI_%7Bi%2B1%7D%29+) 。

**所以CRF的建模公式如下：**

![[公式]](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+%5Cprod_%7Bi%7D%5Cpsi_%7Bi%7D%28I_%7Bi%7D%7CO+%29+%3D+%5Cfrac%7B1%7D%7BZ%28O%29%7D+%5Cprod_%7Bi%7D+e%5E%7B%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D+%3D+%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D)

我要敲黑板了，这个公式是非常非常关键的，注意递推过程啊，我是怎么从 ![[公式]](https://www.zhihu.com/equation?tex=%E2%88%8F) 跳到 ![[公式]](https://www.zhihu.com/equation?tex=e%5E%7B%5Csum%7D) 的。

不过还是要多啰嗦一句，想要理解CRF，必须判别式模型的概念要深入你心。正因为是判别模型，所以不废话，我上来就直接为了确定边界而去建模，因为我创造出来就是为了这个分边界的目的的。比如说序列求概率（分类）问题，我直接考虑找出函数分类边界。所以才为什么会有这个公式。所以再看到这个公式也别懵逼了，he was born for discriminating the given data from different classes. 就这样。不过待会还会具体介绍特征函数部分的东西。



除了建模总公式，关键的CRF重点概念在MEMM中已强调过：**判别式模型**、**特征函数**。

**1. 特征函数**

上面给出了CRF的建模公式：

![[公式]](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D)

- 下标*i*表示我当前所在的节点（token）位置。
- 下标*k*表示我这是第几个特征函数，并且每个特征函数都附属一个权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_%7Bk%7D) ，也就是这么回事，每个团里面，我将为 ![[公式]](https://www.zhihu.com/equation?tex=token_%7Bi%7D) 构造M个特征，每个特征执行一定的限定作用，然后建模时我再为每个特征函数加权求和。
- ![[公式]](https://www.zhihu.com/equation?tex=Z%28O%29) 是用来归一化的，为什么？想想LR以及softmax为何有归一化呢，一样的嘛，形成概率值。
- 再来个重要的理解。 ![[公式]](https://www.zhihu.com/equation?tex=P%28I%7CO%29) 这个表示什么？具体地，表示了在给定的一条观测序列 ![[公式]](https://www.zhihu.com/equation?tex=O%3D%28o_%7B1%7D%2C%5Ccdots%2C+o_%7Bi%7D%29) 条件下，我用CRF所求出来的隐状态序列 ![[公式]](https://www.zhihu.com/equation?tex=I%3D%28i_%7B1%7D%2C%5Ccdots%2C+i_%7Bi%7D%29) 的概率，注意，这里的*I*是一条序列，有多个元素（一组随机变量），而至于观测序列 ![[公式]](https://www.zhihu.com/equation?tex=O%3D%28o_%7B1%7D%2C%5Ccdots%2C+o_%7Bi%7D%29) ，它可以是一整个训练语料的所有的观测序列；也可以是在inference阶段的一句sample，比如说对于序列标注问题，我对一条sample进行预测，可能能得到 ![[公式]](https://www.zhihu.com/equation?tex=P_%7Bj%7D%28I+%7C+O%29%EF%BC%88j%3D1%2C%E2%80%A6%2CJ%EF%BC%89)*J*条隐状态*I*，但我肯定最终选的是最优概率的那条（by viterbi）。这一点希望你能理解。

对于CRF，可以为他定义两款特征函数：转移特征&状态特征。 我们将建模总公式展开：

![[公式]](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B+%5B+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bj%7D%5E%7BJ%7D%5Clambda_%7Bj%7Dt_%7Bj%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29+%2B+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bl%7D%5E%7BL%7D%5Cmu_%7Bl%7Ds_%7Bl%7D%28O%2CI_%7Bi%7D%2Ci%29+%5D+%7D)



其中：

- ![[公式]](https://www.zhihu.com/equation?tex=t_%7Bj%7D) 为i处的转移特征，对应权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_%7Bj%7D) ,每个 ![[公式]](https://www.zhihu.com/equation?tex=token_%7Bi%7D) 都有J个特征,转移特征针对的是前后token之间的限定。

- - 举个例子：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+t_%7Bk%3D1%7D%28o%2Ci%29+%3D+%5Cbegin%7Bcases%7D+1%26+%5Ctext%7B%E6%BB%A1%E8%B6%B3%E7%89%B9%E5%AE%9A%E8%BD%AC%E7%A7%BB%E6%9D%A1%E4%BB%B6%EF%BC%8C%E6%AF%94%E5%A6%82%E5%89%8D%E4%B8%80%E4%B8%AAtoken%E6%98%AF%E2%80%98I%E2%80%99%7D%EF%BC%8C%5C%5C+0%26+%5Ctext%7Bother%7D+%5Cend%7Bcases%7D+%5Cend%7Bequation%7D)

- sl为i处的状态特征，对应权重μl,每个tokeni都有L个特征

- - 举个例子：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+s_%7Bl%3D1%7D%28o%2Ci%29+%3D+%5Cbegin%7Bcases%7D+1%26+%5Ctext%7B%E6%BB%A1%E8%B6%B3%E7%89%B9%E5%AE%9A%E7%8A%B6%E6%80%81%E6%9D%A1%E4%BB%B6%EF%BC%8C%E6%AF%94%E5%A6%82%E5%BD%93%E5%89%8Dtoken%E7%9A%84POS%E6%98%AF%E2%80%98V%E2%80%99%7D%EF%BC%8C%5C%5C+0%26+%5Ctext%7Bother%7D+%5Cend%7Bcases%7D+%5Cend%7Bequation%7D)



不过一般情况下，我们不把两种特征区别的那么开，合在一起：

![[公式]](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D)



满足特征条件就取值为1，否则没贡献，甚至你还可以让他打负分，充分惩罚。

再进一步理解的话，我们需要把特征函数部分抠出来：

![[公式]](https://www.zhihu.com/equation?tex=Score+%3D+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29+)



是的，我们为 ![[公式]](https://www.zhihu.com/equation?tex=token_%7Bi%7D) 打分，满足条件的就有所贡献。最后将所得的分数进行log线性表示，求和后归一化，即可得到概率值……完了又扯到了log线性模型。现在稍作解释：

> log-linear models take the following form:
> ![[公式]](https://www.zhihu.com/equation?tex=P%28y%7Cx%3B%5Comega%29+%3D+%5Cfrac%7B+exp%28%5Comega%C2%B7%5Cphi%28x%2Cy%29%29+%7D%7B+%5Csum_%7By%5E%7B%27%7D%5Cin+Y+%7Dexp%28%5Comega%C2%B7%5Cphi%28x%2Cy%5E%7B%E2%80%98%7D%29%29+%7D)

我觉得对LR或者sotfmax熟悉的对这个应该秒懂。然后CRF完美地满足这个形式，所以又可以归入到了log-linear models之中。

## **模型运行过程**

模型的工作流程，跟MEMM是一样的：

- step1. 先预定义特征函数 ![[公式]](https://www.zhihu.com/equation?tex=+f_%7Ba%7D%28o%2Ci%29) ，
- step2. 在给定的数据上，训练模型，确定参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_%7Bk%7D)
- step3. 用确定的模型做`序列标注问题`或者`序列求概率问题`。

可能还是没做到100%懂，结合例子说明：

> ……

## **学习训练过程**

一套CRF由一套参数λ唯一确定（先定义好各种特征函数）。

同样，CRF用极大似然估计方法、梯度下降、牛顿迭代、拟牛顿下降、IIS、BFGS、L-BFGS等等。各位应该对各种优化方法有所了解的。其实能用在log-linear models上的求参方法都可以用过来。

嗯，具体详细求解过程貌似问题不大。

## **序列标注过程**

还是跟HMM一样的，用学习好的CRF模型，在新的sample（观测序列 ![[公式]](https://www.zhihu.com/equation?tex=o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D) ）上找出一条概率最大最可能的隐状态序列 ![[公式]](https://www.zhihu.com/equation?tex=i_%7B1%7D%2C+%5Ccdots%2C+i_%7Bi%7D) 。

只是现在的图中的每个隐状态节点的概率求法有一些差异而已,正确将每个节点的概率表示清楚，路径求解过程还是一样，采用viterbi算法。

啰嗦一下，我们就定义i处的局部状态为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta_%7Bi%7D%28I%29) ,表示在位置i处的隐状态的各种取值可能为*I*，然后递推位置i+1处的隐状态，写出来的DP转移公式为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta_%7Bi%2B1%7D+%3D+max_%7B1+%5Cle+j+%5Cle+m%7D%5Clbrace+%5Cdelta_%7Bi%7D%28I%29+%2B+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29+%5Crbrace)

这里没写规范因子 ![[公式]](https://www.zhihu.com/equation?tex=Z%28O%29) 是因为不规范化不会影响取最大值后的比较。

具体还是不展开为好。

## **序列求概率过程**

跟HMM举的例子一样的，也是分别去为每一批数据训练构建特定的CRF，然后根据序列在每个MEMM模型的不同得分概率，选择最高分数的模型为wanted类别。只是貌似很少看到拿CRF或者MEMM来做分类的，直接用网络模型不就完了不……

应该可以不用展开，吧……

## **CRF++分析**

本来做task用CRF++跑过baseline,后来在对CRF做调研时，非常想透析CRF++的工作原理，以identify以及verify做的各种假设猜想。当然，也看过其他的CRF实现源码。

所以干脆写到这里来，结合CRF++实例讲解过程。

有一批语料数据，并且已经tokenized好了：

> Nuclear
> theory
> devoted
> major
> efforts
> …… 

并且我先确定了13个标注元素：

> B_MAT
> B_PRO
> B_TAS
> E_MAT
> E_PRO
> E_TAS
> I_MAT
> I_PRO
> I_TAS
> O
> S_MAT
> S_PRO
> S_TAS 

**1. 定义模板**

按道理应该是定义特征函数才对吧？好的，在CRF++下，应该是先定义特征模板，然后用模板自动批量产生大量的特征函数。我之前也蛮confused的，用完CRF++还以为模板就是特征，后面就搞清楚了：每一条模板将在每一个token处生产若干个特征函数。

CRF++的模板（template）有U系列（unigram）、B系列(bigram)，不过我至今搞不清楚B系列的作用，因为U模板都可以完成2-gram的作用。

> U00:%x[-2,0]
> U01:%x[-1,0]
> U02:%x[0,0]
> U03:%x[1,0]
> U04:%x[2,0] 
>
> U05:%x[-2,0]/%x[-1,0]/%x[0,0]
> U06:%x[-1,0]/%x[0,0]/%x[1,0]
> U07:%x[0,0]/%x[1,0]/%x[2,0]
> U08:%x[-1,0]/%x[0,0]
> U09:%x[0,0]/%x[1,0] 
>
> B 

所以，U00 - U09 我定义了10个模板。

**2. 产生特征函数**

是的，会产生大量的特征。 U00 - U04的模板产生的是状态特征函数；U05 - U09的模板产生的是转移特征函数。

在CRF++中，每个特征都会try每个标注label（这里有13个），总共将生成 ![[公式]](https://www.zhihu.com/equation?tex=N+%2A+L+%3D+i+%2A+k%5E%7B%27%7D+%2A+L) 个特征函数以及对应的权重出来。N表示每一套特征函数 ![[公式]](https://www.zhihu.com/equation?tex=N%3D+i+%2A+k%5E%7B%27%7D) ，L表示标注集元素个数。

比如训练好的CRF模型的部分特征函数是这样存储的：

> 22607 B
> 790309 U00:%
> 3453892 U00:%)
> 2717325 U00:&
> 2128269 U00:'t
> 2826239 U00:(0.3534
> 2525055 U00:(0.593–1.118
> 197093 U00:(1)
> 2079519 U00:(1)L=14w2−12w−FμνaFaμν
> 2458547 U00:(1)δn=∫−∞En+1ρ˜(E)dE−n
> 1766024 U00:(1.0g
> 2679261 U00:(1.1wt%)
> 1622517 U00:(100)
> 727701 U00:(1000–5000A)
> 2626520 U00:(10a)
> 2626689 U00:(10b)
> ……
> 2842814 U07:layer/thicknesses/Using
> 2847533 U07:layer/thicknesses/are
> 2848651 U07:layer/thicknesses/in
> 331539 U07:layer/to/the
> 1885871 U07:layer/was/deposited
> ……（数量非常庞大） 

其实也就是对应了这样些个特征函数：

> func1 = if (output = B and feature="U02:一") return 1 else return 0
> func2 = if (output = M and feature="U02:一") return 1 else return 0
> func3 = if (output = E and feature="U02:一") return 1 else return 0
> func4 = if (output = S and feature="U02:一") return 1 else return 0 

比如模板U06会从语料中one by one逐句抽出这些各个特征：

> 一/个/人/……
> 个/人/走/……

**3. 求参**

对上述的各个特征以及初始权重进行迭代参数学习。

在CRF++ 训练好的模型里，权重是这样的：

> 0.3972716048310705
> 0.5078838237171732
> 0.6715316559507898
> -0.4198827647512405
> -0.4233310655891150
> -0.4176580083832543
> -0.4860489836004728
> -0.6156475863742051
> -0.6997919485753300
> 0.8309956709647820
> 0.3749695682658566
> 0.2627347894057647
> 0.0169732441379157
> 0.3972716048310705
> 0.5078838237171732
> 0.6715316559507898
> ……（数量非常庞大，与每个label的特征函数对应，我这有300W个）

**4. 预测解码**

结果是这样的：

> Nuclear B*TAS*
> *theory E*TAS
> devoted O
> major O
> efforts O
> …… 

## **LSTM+CRF**

LSTM+CRF这个组合其实我在知乎上答过问题，然后顺便可以整合到这里来。

**1、perspectively**

大家都知道，LSTM已经可以胜任序列标注问题了，为每个token预测一个label（LSTM后面接:分类器）；而CRF也是一样的，为每个token预测一个label。

但是，他们的预测机理是不同的。CRF是全局范围内统计归一化的条件状态转移概率矩阵，再预测出一条指定的sample的每个token的label；LSTM（RNNs，不区分here）是依靠神经网络的超强非线性拟合能力，在训练时将samples通过复杂到让你窒息的高阶高纬度异度空间的非线性变换，学习出一个模型，然后再预测出一条指定的sample的每个token的label。

**2、LSTM+CRF**

既然LSTM都OK了，为啥researchers搞一个LSTM+CRF的hybrid model? 

哈哈，因为a single LSTM预测出来的标注有问题啊！举个segmentation例子(BES; char level)，plain LSTM 会搞出这样的结果：

> **input**: "学习出一个模型，然后再预测出一条指定"
> **expected output**: 学/B 习/E 出/S 一/B 个/E 模/B 型/E ，/S 然/B 后/E 再/E 预/B 测/E ……
> **real output**: 学/B 习/E 出/S 一/B 个/B 模/B 型/E ，/S 然/B 后/B 再/E 预/B 测/E ……

看到不，用LSTM，整体的预测accuracy是不错indeed, 但是会出现上述的错误：在B之后再来一个B。这个错误在CRF中是不存在的，因为CRF的特征函数的存在就是为了对given序列观察学习各种特征（n-gram，窗口），这些特征就是在限定窗口size下的各种词之间的关系。然后一般都会学到这样的一条规律（特征）：B后面接E，不会出现E。这个限定特征会使得CRF的预测结果不出现上述例子的错误。当然了，CRF还能学到更多的限定特征，那越多越好啊！

好了，那就把CRF接到LSTM上面，把LSTM在time*step上把每一个hidden*state的tensor输入给CRF，让LSTM负责在CRF的特征限定下，依照新的loss function，学习出一套新的非线性变换空间。

最后，不用说，结果还真是好多了呢。

[BiLSTM+CRF codes](https://link.zhihu.com/?target=https%3A//github.com/scofield7419/sequence-labeling-BiLSTM-CRF), here. Go just take it.

这个代码比较早，CRF层中的transition matrix以及score的计算都是python from scratch. 目前tf 1.4早已将crf加入contrib中，4行代码即可实现LSTM拼接CRF的效果。



**3. CRF in TensorFlow V.S. CRF in discrete toolkit**

发现有的同学还是对general 实现的CRF工具包代码，与CRF拼接在LSTM网络之后的代码具体实现（如在TensorFlow），理解的稀里糊涂的，所以还得要再次稍作澄清。

在CRF相关的工具包里，CRF的具体实现是采用上述理论提到的为特征打分的方式统计出来的。统计的特征分数作为每个token对应的tag的类别的分数，输入给CRF解码即可。

而在TensorFlow中，LSTM每个节点的隐含表征vector：Hi的值作为CRF层对应的每个节点的统计分数，再计算每个序列（句子）的整体得分score，作为损失目标，最后inference阶段让viterbi对每个序列的transition matrix去解码，搜出一条最优路径。

**关键区别在于，在LSTM+CRF中，CRF的特征分数直接来源于LSTM传上来的Hi的值；而在general CRF中，分数是统计来的。所有导致有的同学认为LSTM+CRF中其实并没有实际意义的CRF。其实按刚才说的，Hi本身当做特征分数形成transition matrix再让viterbi进行路径搜索，这整个其实就是CRF的意义了。所以LSTM+CRF中的CRF没毛病。**





## **总结**

**1. 总体对比**

应该看到了熟悉的图了，现在看这个图的话，应该可以很清楚地get到他所表达的含义了。这张图的内容正是按照生成式&判别式来区分的，NB在sequence建模下拓展到了HMM；LR在sequence建模下拓展到了CRF。

![img](https://pic3.zhimg.com/50/v2-376fd85a490e161978130ddd759244d4_hd.jpg)

**2. HMM vs. MEMM vs. CRF**

将三者放在一块做一个总结：

1. HMM -> MEMM： HMM模型中存在两个假设：一是输出观察值之间严格独立，二是状态的转移过程中当前状态只与前一状态有关。但实际上序列标注问题不仅和单个词相关，而且和观察序列的长度，单词的上下文，等等相关。MEMM解决了HMM输出独立性假设的问题。因为HMM只限定在了观测与状态之间的依赖，而MEMM引入自定义特征函数，不仅可以表达观测之间的依赖，还可表示当前观测与前后多个状态之间的复杂依赖。
2. MEMM -> CRF:

- CRF不仅解决了HMM输出独立性假设的问题，还解决了MEMM的标注偏置问题，MEMM容易陷入局部最优是因为只在局部做归一化，而CRF统计了全局概率，在做归一化时考虑了数据在全局的分布，而不是仅仅在局部归一化，这样就解决了MEMM中的标记偏置的问题。使得序列标注的解码变得最优解。
- HMM、MEMM属于有向图，所以考虑了x与y的影响，但没讲x当做整体考虑进去（这点问题应该只有HMM）。CRF属于无向图，没有这种依赖性，克服此问题。



















