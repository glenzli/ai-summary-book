# 第五十二章：多项式函子、Species、解析函子与 W-types

和式、积式及变量代入可以统一编码为图形 $I\leftarrow E\to B\to J$，相应多项式函子由依赖拉回、依赖积和依赖和复合而成。Species 把有限标号集上的组合结构组织成群胚值函子，解析函子则对标号对称性取同伦商。多项式函子的初代数给出 W-types，因而同时描述良基树、容器和递归数据类型。本章比较这些构造的普通集合版本与保留 automorphisms 的群胚版本。

所需背景是 slice、局部 Cartesian closed 结构、Kan 延拓、初代数与 operads。解析函子的商必须区分 orbit set 与 homotopy quotient；W-type 的存在也会声明范畴具有何种初始代数或可表现性条件。

## 52.1 多项式函子

**定义 52.1.** 在有 pullback 和依赖和/积的范畴中，映射串

$$
I\xleftarrow{s}E\xrightarrow{p}B\xrightarrow{t}J
$$

定义多项式函子

$$
P:\mathcal C_{/I}\to\mathcal C_{/J}
$$

为复合

$$
P=\Sigma_t\Pi_p s^*.
$$

**命题 52.2.** 在 $\mathbf{Set}$ 中，若 $I=J=1$，则

$$
P(X)=\sum_{b\in B}X^{E_b}
$$

其中 $E_b=p^{-1}(b)$。

**证明.** $s^*$ 在终 slice 上不改变集合 $X$。$\Pi_p$ 沿纤维 $E_b$ 取依赖积，给出 $X^{E_b}$。$\Sigma_t$ 沿 $B\to1$ 取依赖和，即对所有 $b\in B$ 求不交并。因此得到公式。$\square$

## 52.2 容器

**定义 52.3.** Set 中的 container 由 shapes 集合 $B$ 和 positions 函数 $E\to B$ 组成，其扩张为

$$
X\mapsto\sum_{b\in B}X^{E_b}.
$$

**命题 52.4.** Set 中一元 container 与 $1\leftarrow E\to B\to1$ 型多项式函子等价。

**证明.** 给定 container $(B,E\to B)$，取 $1\leftarrow E\to B\to1$ 得到命题 52.2 的函子。反过来，任意该形状多项式只含中间映射 $E\to B$ 的数据，即 shapes 和每个 shape 的 positions。两构造互逆。$\square$

## 52.3 Species

**定义 52.5.** Species 是函子

$$
F:\mathbf{FinBij}\to\mathbf{Set},
$$

其中 $\mathbf{FinBij}$ 为有限集合与双射的 groupoid。

**定义 52.6.** Species $F$ 的解析函子为

$$
\widehat F(X)=\sum_{n\ge0}F[n]\times_{\Sigma_n}X^n.
$$

若 $F$ 取值于 groupoids 或 spaces，则应把轨道商替换为 homotopy quotient：

$$
\widehat F(X)\simeq
\coprod_{n\ge0}\bigl(F[n]\times X^n\bigr)_{h\Sigma_n}.
$$

普通轨道集会遗失结构的 automorphism groups。

**命题 52.7.** 若 $F[n]=1$ 对所有 $n$，则 $\widehat F(X)$ 是有限多重集函子。

**证明.** 公式化为

$$
\widehat F(X)=\sum_{n\ge0}X^n/\Sigma_n.
$$

$X^n/\Sigma_n$ 是大小为 $n$ 的无序带重复元素列表，即 $n$ 元多重集。对所有 $n$ 求并得到有限多重集。$\square$

## 52.4 解析函子与对称群作用

**命题 52.8.** 解析函子中的商 $\times_{\Sigma_n}$ 表示标号遗忘。

**证明.** $F[n]$ 是在标准 $n$ 元集合上的结构，$X^n$ 是给 $n$ 个位置贴 $X$-标签。对称群 $\Sigma_n$ 同时重排结构标号和标签位置。取轨道商后，依赖具体编号的差异被识别，只保留未编号有限集合上的 $F$-结构及其 $X$-标签。$\square$

## 52.5 W-types

**定义 52.9.** 多项式函子 $P(X)=\sum_{b\in B}X^{E_b}$ 的 W-type 是 $P$ 的初代数

$$
\alpha:P(W)\to W.
$$

**外部输入定理 52.10.** 若 $\mathcal C$ locally presentable 且 locally Cartesian closed，并且多项式端函子 $P:\mathcal C\to\mathcal C$ accessible，则 $P$ 有初代数，因而相应 W-type 存在。仅有 elementary topos 或 locally Cartesian closed 结构不自动保证所有多项式函子都有初代数；“有 W-types”在一般类型论模型中是一条额外公理。

**命题 52.11.** 若范畴有自然数对象 $N$，则 $N$ 是多项式函子 $P(X)=1+X$ 的 W-type；反过来，该初代数给出自然数对象。

**证明.** $P$-代数为对象 $A$ 配映射 $1+A\to A$，等价于点 $z:1\to A$ 和后继 $s:A\to A$。初 $P$-代数正是带点和自映射的初对象，这就是自然数对象的泛性质。$\square$

## 52.6 多项式单子与 Operads

**定义 52.12.** 多项式单子是多项式函子 $P$ 配单子结构

$$
\eta:\operatorname{id}\to P,\qquad \mu:P^2\to P
$$

且单位和乘法由多项式自然变换给出。

**外部输入定理 52.13.** 许多 colored operads、非对称 operads 和 trees 的组合结构可表示为多项式单子。

**命题 52.14.** List functor $X\mapsto\sum_{n\ge0}X^n$ 带有由空列表和拼接给出的单子结构。

**证明.** 单位 $\eta_X:X\to\sum_nX^n$ 把 $x$ 送到长度 1 的列表 $[x]$。乘法 $\mu_X$ 把列表的列表拼接为单个列表。空列表是幺元，拼接结合，因此满足单子单位律和结合律。$\square$

**命题 52.15.** Set 中一元多项式函子

$$
P(X)=\sum_{b\in B}X^{E_b}
$$

保持 pullbacks。

**证明.** 对 pullback $X\times_ZY$，有自然双射

$$
(X\times_ZY)^{E_b}\cong X^{E_b}\times_{Z^{E_b}}Y^{E_b}
$$

对每个 $b\in B$ 成立。于是

$$
P(X\times_ZY)
\cong
\sum_{b\in B}\bigl(X^{E_b}\times_{Z^{E_b}}Y^{E_b}\bigr).
$$

另一方面，$P(X)\times_{P(Z)}P(Y)$ 中元素必须有相同的 shape $b$，因为 $P(Z)$ 是按 $b\in B$ 的不交并；在固定 $b$ 上正是 $X^{E_b}\times_{Z^{E_b}}Y^{E_b}$。故两者自然同构。$\square$

## 52.7 组合结构与递归类型的统一表达

多项式函子以 $\Sigma\Pi\Delta$ 形式编码依赖和、依赖积与替换；containers 是 Set 中的同一思想；species 与解析函子把有限对称性纳入组合结构；W-types 给多项式函子的初代数；多项式单子则把 operad 和树形代数结构统一为范畴论对象。

## 练习

**练习 52.1.** 定义多项式函子 $P=\Sigma_t\Pi_ps^*$。

**练习 52.2.** 在 Set 中推导 $P(X)=\sum_{b\in B}X^{E_b}$。

**练习 52.3.** 定义 container。

**练习 52.4.** 证明 container 与一元多项式函子等价。

**练习 52.5.** 定义 species。

**练习 52.6.** 写出解析函子公式。

**练习 52.7.** 证明常值 species 给出有限多重集函子。

**练习 52.8.** 解释 $\Sigma_n$-商的意义。

**练习 52.9.** 定义 W-type。

**练习 52.10.** 证明自然数对象是 $1+X$ 的 W-type。

**练习 52.11.** 定义多项式单子。

**练习 52.12.** 证明 list functor 带单子结构。

**练习 52.13.** 证明 Set 中一元多项式函子保持 pullbacks。
