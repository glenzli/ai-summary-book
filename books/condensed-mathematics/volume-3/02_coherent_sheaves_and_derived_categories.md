# 第二章：相干层、perfect 复形与点支撑例子

复解析空间的结构层拥有无限维截面，但几何对象常由有限个局部生成元和关系控制。
这种“有限”发生在 stalk 和局部表示中，不是全局截面维数有限。进一步进入导出范畴
时，还要区分上同调层相干的复形与局部由有限秩向量丛复形表示的 perfect 对象。一个
最小而非平凡的例子是圆盘原点处的 skyscraper sheaf：它不是向量丛，却由乘以坐标函数
的两项自由 resolution 表示，并能实际计算 sheaf Ext。

第一章的 analytic 建模输入允许把这些经典对象解释为 liquid/analytic 模。这里先在
经典环化空间中完成有限表示、resolution 与派生 Hom 计算，再说明哪些结论仅靠 exact
与 monoidal 形式便可传入 analytic 派生范畴。

## 2.1 相干性的局部含义

**定义 2.1.** 复解析空间 $X$ 上的 $\mathcal O_X$-模 $\mathcal F$ 称为相干，如果
每个点都有邻域 $U$ 和有限整数 $m,n$，使存在正合列

$$
\mathcal O_U^{\oplus m}
\longrightarrow
\mathcal O_U^{\oplus n}
\longrightarrow
\mathcal F|_U
\longrightarrow0,
$$

并且有限生成子模的关系层仍局部有限生成。若 $\mathcal O_X$ 是 coherent ring sheaf，
后一个条件等价于局部有限表示。

**外部输入定理 2.2（Oka coherence）.** 复流形及复解析空间局部模型的结构层
$\mathcal O_X$ 是 coherent ring sheaf。因而相干层对 kernel、cokernel 和 extension
封闭，组成阿贝尔范畴 $\operatorname{Coh}(X)$。

复流形的局部环
$\mathcal O_{X,x}\cong\mathbb C\{z_1,\ldots,z_n\}$ 是正则 Noether 局部环。正则
局部环的有限整体维数是另一项经典输入，它推出相干层在每个点附近都有长度至多 $n$
的有限局部自由 resolution；完整 stalk 到 sheaf 的提升见[附录 W](W_regular_local_rings_and_coherent_resolutions.md)。
局部 resolution 不自动拼成全局 resolution，这一区别在第四、五章会保留。

## 2.2 两个导出子范畴

**定义 2.3.** $D^b_{\operatorname{coh}}(X)$ 是 $D(\mathcal O_X)$ 中上同调层相干且
仅有限多个非零的对象所成的全子范畴。

**定义 2.4.** 对象 $P\in D(\mathcal O_X)$ 称为 perfect，如果每个点附近都有一个
有限秩局部自由 $\mathcal O_X$-模组成的有界复形 $E^\bullet$，使
$P|_U\simeq E^\bullet$。

perfect 蕴含 bounded coherent，但反向需要正则性以及局部有限 Tor 维数。在复流形上，
正则局部环输入使每个 bounded coherent 对象局部 perfect；在奇异复空间上不能这样
断言。

**命题 2.5.** 有限秩局部自由层 $E$ 在闭幺半导出范畴中可对偶，其对偶为
$E^\vee=\mathcal Hom(E,\mathcal O_X)$。任意 perfect complex 局部可对偶。

**证明.** 局部平凡化后 $E\cong\mathcal O_X^{\oplus r}$，evaluation 与 coevaluation
分别由矩阵配对和单位矩阵给出，三角恒等式逐基向量成立。有限直和、shift 和 cone
保持可对偶性；perfect complex 由有限个局部自由项经这些有限操作构成，故局部可对偶。
证毕。

## 2.3 Worked example：原点的 skyscraper sheaf

令 $X=\Delta$，$i:\{0\}\hookrightarrow\Delta$，并记

$$
\mathbb C_0=i_*\mathbb C
$$

为原点处的 skyscraper $\mathcal O_\Delta$-模，其中 $f\in\mathcal O_{\Delta,0}$
通过 $f(0)$ 作用。

**命题 2.6.** 有正合列

$$
0\longrightarrow\mathcal O_\Delta
\xrightarrow{\,z\,}
\mathcal O_\Delta
\longrightarrow\mathbb C_0
\longrightarrow0.
$$

**证明.** 在 $x\ne0$ 的 stalk 上，$z$ 可逆，所以乘法是同构且 cokernel 为零。在
$x=0$ 的 stalk 上，$\mathcal O_{\Delta,0}=\mathbb C\{z\}$ 是整环，故乘以 $z$
单射；其 cokernel 为
$\mathbb C\{z\}/(z)\cong\mathbb C$，作用由取值 $f\mapsto f(0)$ 给出。stalk
正合性等价于 sheaf 正合性，结论成立。证毕。

这个 resolution 立刻给出派生 Hom，而不需要抽象 injective resolution。

**命题 2.7.** 有

$$
\mathcal Ext^q_{\mathcal O_\Delta}
(\mathbb C_0,\mathcal O_\Delta)
\cong
\begin{cases}
\mathbb C_0,&q=1,\\
0,&q\ne1.
\end{cases}
$$

**证明.** 对命题 2.6 的两项局部自由 resolution 施加
$\mathcal Hom(-,\mathcal O_\Delta)$，得到位于次数 $0,1$ 的复形

$$
\mathcal O_\Delta
\xrightarrow{\,z\,}
\mathcal O_\Delta.
$$

其次数零 kernel 为零，次数一 cokernel 由命题 2.6 等于 $\mathbb C_0$，更高次数没有
项。证毕。

这个计算说明“相干但非局部自由”的对象仍可由 perfect complex 控制，也预示第五章中
$R\mathcal Hom(\mathcal F,\omega_X)$ 为何要使用有限局部自由 resolution。

## 2.4 有限 resolution 的形式后果

**命题 2.8.** 若 $\mathcal F$ 有全局有限局部自由 resolution
$E^\bullet\to\mathcal F$，则

$$
R\mathcal Hom_X(\mathcal F,\mathcal G)
\simeq
\mathcal Hom_X(E^\bullet,\mathcal G)
$$

对任意 $\mathcal G\in D^+(\mathcal O_X)$ 成立。

**证明.** 对有限秩局部自由 $E$，
$\mathcal Hom(E,-)\cong E^\vee\otimes-$ 是精确函子，因此每一项都对第二变量的
导出 Hom 无需再解析。有限 resolution 的总 Hom 复形遂计算
$R\mathcal Hom$。证毕。

若对 $E^\bullet$ 施加 $R\Gamma$，得到超上同调谱序列

$$
E_1^{p,q}=H^q(X,E^p)
\Longrightarrow
\mathbb H^{p+q}(X,E^\bullet).
$$

当 $E^\bullet\simeq\mathcal F$ 时，abutment 就是 $H^{p+q}(X,\mathcal F)$。第四章
会完整证明有限维性如何沿这个有限谱序列传播。

## 2.5 进入 analytic 派生范畴

**外部输入定理 2.9（相干对象的 analytic 实现）.** 在定理 1.5 的适用范围内，有忠实
精确的实现

$$
D^b_{\operatorname{coh}}(X)
\longrightarrow
D_{\mathrm{an}}(X),
$$

它把有限局部自由层送到可对偶 analytic $\mathcal O_X$-模，并与有限 cone、派生
$\mathcal Hom$ 及 $R\Gamma$ 相容。

接受该输入后，命题 2.6 的正合列变成 analytic 派生范畴中的三角

$$
\mathcal O_\Delta
\xrightarrow{z}
\mathcal O_\Delta
\longrightarrow
\mathbb C_0
\longrightarrow
\mathcal O_\Delta[1],
$$

命题 2.7 的计算也由同一个两项复形给出。这些是 exact/monoidal 实现的形式后果；实现
本身及其拓扑正合性仍属于 Clausen--Scholze 建模输入。

## 2.6 从局部有限表示到全局上同调

相干性控制 stalk 上的生成元和关系，perfect 性控制派生对象能否由有限向量丛复形
表示；二者都不直接断言 $H^q(X,\mathcal F)$ 有限维。要计算全局上同调，还需一个
$\Gamma$-acyclic resolution。第三章构造的 Dolbeault fine resolution提供这种模型，
第四章再引入紧性与椭圆 Fredholm 输入，把无限维复形的同调压缩为有限维对象。

## 练习

**练习 2.1.** 对 $a\in\Delta$ 写出 $\mathbb C_a$ 的两项局部自由 resolution，并
计算 $\mathcal Ext^q(\mathbb C_a,\mathcal O_\Delta)$。

**练习 2.2.** 证明有限秩局部自由层 $E$ 满足
$R\mathcal Hom(E,\mathcal G)\simeq E^\vee\otimes^L\mathcal G$。

**练习 2.3.** 给出理由说明命题 2.6 只依赖 stalk 正合性；指出若把 $z$ 换成零函数，
证明在哪一步失败。
