# 第三章：状态、事件、轨迹与观察等价

最终文本 $u_\star$ 看不出它是直接生成的，还是经历过查询、写入、重试和乱序传输。要保存这些差异，运行必须从“输入到值”的箭头展开为一串状态与事件。贯穿案例的一段压缩轨迹可以写成

$$
s_0\xrightarrow{\operatorname{query}(\mathtt{SP404})}s_1
\xrightarrow{\operatorname{return}(\mathtt{Cancelled})}s_2
\xrightarrow{\operatorname{write}(\mathtt{trip.md})}s_3
\xrightarrow{\operatorname{commit}}s_4.
$$

这还不是完整模型：每个 $s_i$ 的类型、错误状态、无穷等待和可见事件都尚未说明。本章用第二章的关系语言把这段缩写展开为带标签转移系统，并说明“相同输出”究竟观察了轨迹的哪一部分。

## 3.1 带标签转移系统

**定义 3.1（LTS）.** 带标签转移系统是四元组

$$
\mathcal L=(S,A,\longrightarrow,F),
$$

其中 $S$ 是状态集，$A$ 是事件标签集，
$\longrightarrow\subseteq S\times A\times S$ 是转移关系，
$F\subseteq S$ 是正常终止状态集。写
$s\overset{a}{\longrightarrow}s'$ 表示 $(s,a,s')\in\longrightarrow$。

本书要求正常终止状态无后继：

$$
s\in F\Longrightarrow
\neg\exists a\in A,\exists s'\in S,\
s\overset{a}{\longrightarrow}s'.
$$

内部事件记为 $\tau\in A$；它与轨迹符号不混用。

**定义 3.2（强确定性）.** 若

$$
\forall s\in S\;\forall a,b\in A\;\forall s',t'\in S,\quad
s\overset{a}{\longrightarrow}s'
\land s\overset{b}{\longrightarrow}t'
\Longrightarrow (a,s')=(b,t'),
$$

称 $\mathcal L$ 强确定。本定义同时要求下一标签和下一状态唯一。只要求下一状态唯一不足以保证事件日志唯一。

## 3.2 有限、无限与最大轨迹

长度 $n$ 的有限轨迹是交错序列

$$
t=(s_0,a_1,s_1,\ldots,a_n,s_n)
$$

满足 $s_{i-1}\overset{a_i}{\longrightarrow}s_i$。其初态、末态和标签词分别记
$\operatorname{src}(t)=s_0$、
$\operatorname{last}(t)=s_n$、
$\operatorname{lab}(t)=a_1\cdots a_n\in A^*$。

无限轨迹是 $(s_0,a_1,s_1,\ldots)$，每个有限前缀都满足转移关系。有限轨迹称为**最大**，若末态无后继；无限轨迹按定义为最大。记
$\operatorname{Tr}_{\max}(s)$ 为从 $s$ 出发的全部最大轨迹。

- 最大有限轨迹以 $F$ 中状态结束时称正常终止；
- 以 $S\setminus F$ 中无后继状态结束时称卡死；
- 无限轨迹称发散轨迹。

“没有最终文本”可以来自正常空结果、错误卡死、尚未完成、观察者取消或发散；这些状态不能合并。

## 3.3 确定性与轨迹唯一

**定理 3.3（固定长度轨迹至多唯一）.** 若 $\mathcal L$ 强确定，则对每个 $s_0\in S$ 与 $n\in\mathbb N$，从 $s_0$ 出发的长度 $n$ 轨迹至多一条。

**证明.** 对 $n$ 归纳。$n=0$ 时只有仅含 $s_0$ 的轨迹。设结论对 $n$ 成立。若有两条长度 $n+1$ 轨迹，其首步分别为
$s_0\overset{a}{\longrightarrow}s_1$ 和
$s_0\overset{b}{\longrightarrow}t_1$。强确定性给出
$(a,s_1)=(b,t_1)$。从共同状态 $s_1$ 出发的两个剩余长度 $n$ 轨迹由归纳假设相同，故原轨迹相同。证毕。

**推论 3.4（强确定系统的最大行为存在且唯一）.** 在强确定系统中，对每个初态 $s_0\in S$，集合
$\operatorname{Tr}_{\max}(s_0)$ 恰含一条轨迹；若该轨迹正常终止，则终止状态唯一。

**证明.** 强确定性使一步转移唯一确定一个部分函数

$$
d:S\rightharpoonup A\times S,
$$

其中 $d(s)=(a,s')$ 当且仅当
$s\overset{a}{\longrightarrow}s'$。从 $s_0$ 开始递归应用 $d$。若存在 $n\in\mathbb N$ 使第 $n$ 个状态 $s_n$ 不在
$\operatorname{dom}(d)$，取最小的此类 $n$；此前各步均由 $d$ 唯一确定，所得长度 $n$ 轨迹末态无后继，因而有限最大。若不存在这样的 $n$，递归对每个自然数都定义唯一的 $(a_{n+1},s_{n+1})=d(s_n)$，所得交错序列是无限轨迹，因而最大。两种情形互斥，所以最大轨迹存在。

唯一性也可直接由定理 3.3 得到：任意两条最大轨迹的共同有限前缀相同；若一条有限结束，末态无后继使另一条不能继续；若二者均无限，则逐项相同。正常终止状态是唯一最大轨迹的末态，故也唯一。证毕。

这里证明的是抽象 LTS 中轨迹的集合论存在性，不是每个实现步骤都有墙钟时间界。若程序求值本身可能挂起，必须把求值中间状态和对应无限轨迹纳入 LTS，而不能把一个可能不返回的过程冒充一步部分函数调用。

## 3.4 多步关系与输出投影

写

$$
s\overset{w}{\Longrightarrow}s'
$$

表示存在一条从 $s$ 到 $s'$、标签词为 $w\in A^*$ 的有限轨迹。忽略标签时写 $s\to^*s'$。

设可见输出片段集合为 $O$。给定单步投影
$e:A\to O^*$，其中内部、工具或审计事件可以映到 $\epsilon$。它唯一扩张为幺半群同态

$$
e^*:A^*\to O^*,
\qquad
e^*(a_1\cdots a_n)=e(a_1)\cdots e(a_n).
$$

若还有末态装配函数
$\operatorname{assemble}:O^*\times F\rightharpoonup Y$，则最终值只对正常终止且装配有定义的轨迹存在。撤回或 patch 协议不能用单纯连接同态表达，必须把可见文档纳入状态并解释编辑事件。

例如，令 $t_\star$ 是贯穿案例的完整最大轨迹。若 $e$ 只保留最终已提交文本片段而把工具、重试和确认事件映到 $\epsilon$，则
$e^*(\operatorname{lab}(t_\star))$ 可以装配为 $u_\star$。若另一次运行没有丢失第一次写入确认，它仍可能装配为同一 $u_\star$，但两条标签词不同。最终文本因此是轨迹的投影，不是轨迹本身。

## 3.5 单条轨迹的核等价

固定某类轨迹集合 $\mathcal T$、观察空间 $Z$ 和函数
$\pi:\mathcal T\to Z$。

**定义 3.5（轨迹核等价）.** 对 $t_1,t_2\in\mathcal T$，定义

$$
t_1\sim_\pi t_2
\Longleftrightarrow
\pi(t_1)=\pi(t_2).
$$

**命题 3.6.** $\sim_\pi$ 是 $\mathcal T$ 上的等价关系。

**证明.** 自反性、对称性与传递性分别由 $Z$ 上等号的对应性质直接得到。证毕。

这一定义只比较已经给定的两条轨迹。它没有量化系统的其他可能运行，因而不能代替程序或状态的 observational equivalence。

## 3.6 状态级观察等价

固定观察函数

$$
\pi:\bigcup_{s\in S}\operatorname{Tr}_{\max}(s)\to Z.
$$

定义状态 $s$ 的可能观察集

$$
\operatorname{Obs}_\pi(s)
=\{\pi(t):t\in\operatorname{Tr}_{\max}(s)\}\subseteq Z.
$$

**定义 3.7（may-observation 等价）.**

$$
s\approx_\pi^{\mathrm{may}}s'
\Longleftrightarrow
\forall z\in Z,\quad
\bigl(\exists t\in\operatorname{Tr}_{\max}(s),\pi(t)=z\bigr)
\Longleftrightarrow
\bigl(\exists t'\in\operatorname{Tr}_{\max}(s'),\pi(t')=z\bigr).
$$

等价地，$\operatorname{Obs}_\pi(s)=\operatorname{Obs}_\pi(s')$。这里的两个存在量词不能省略。若要比较所有调度都满足的性质，应另用 must 语义：

$$
s\models_\pi^{\mathrm{must}}Q
\Longleftrightarrow
\forall t\in\operatorname{Tr}_{\max}(s),\ \pi(t)\in Q,
\qquad Q\subseteq Z.
$$

may 集相同会使所有只依赖观察值集合的 must 性质相同；若观察还保留重数、概率或公平性，需采用相应更细语义。

## 3.7 上下文观察等价

令 $\mathcal K$ 为允许的上下文集合，$\operatorname{plug}(K,s)$ 为把状态或组件 $s$ 放入上下文 $K$ 后得到的初态。定义

$$
s\approx_{\mathcal K,\pi}s'
\Longleftrightarrow
\forall K\in\mathcal K,\quad
\operatorname{Obs}_\pi(\operatorname{plug}(K,s))
=
\operatorname{Obs}_\pi(\operatorname{plug}(K,s')).
$$

上下文类越大，等价关系通常越细。若不声明 $\mathcal K$、最大轨迹的范围和观察函数 $\pi$，“两个系统行为相同”没有封闭真值条件。

概率系统的对应定义不是比较支持集，而是比较每个上下文诱导的观察推前概率测度；第五章给出该接口。

## 3.8 弱观察与流式历史

若 $\pi$ 删除内部 $\tau$ 事件，则不同内部步数的轨迹可核等价。若 $\pi$ 只取最终文本，则延迟、工具调用、撤回和用户曾经看到的片段都被遗忘。安全审计应采用更细观察，例如保留：

- 发出、确认、撤回和提交事件；
- 工具请求、授权、响应与副作用确认；
- 错误、重试、时间区间和状态哈希；
- 被脱敏但仍可关联的运行标识。

现在可以把贯穿案例的两个运行作三种不同比较：只看最终文本时，它们可能由
$\sim_\pi$ 判为相同；比较某状态的全部可能观察时，需要
$\approx_\pi^{\mathrm{may}}$；要求任何允许的客户端、审计器或调度上下文都不能区分时，则要量化整个 $\mathcal K$。下一章选择其中一段尚未展开的轨迹，即 token 生成过程，并给出它的小步规则。工具和网络事件暂时不混入生成器，而会在第六、七章组成更大的乘积系统。

## 练习

**练习 3.1.** 为文件生成器定义正常空文件、普通成功、卡死、显式错误和发散状态。

**练习 3.2.** 给出两条完整轨迹不同但最终文本核等价的例子，并给出能区分它们的更细观察函数。

**练习 3.3.** 证明强确定转移系统从同一初态若正常终止，则终止状态唯一；指出“终止状态无后继”假设在哪里使用。

**练习 3.4.** 构造一个有无限轨迹但每个有限前缀唯一的系统，并求其最大轨迹集合。

**练习 3.5.** 为安全审计设计观察函数，并分别写出单轨迹核等价与上下文观察等价的完整量词。
