# 第七章：两个站点上的投射局部对象

pro-etale topology 与凝聚数学都会用到 profinite 极限、sheaf 下降和能够分裂覆盖的局部
对象，但相同的证明形状并不等于相同的对象。若没有一个保持覆盖与纤维积的明确函子，
$X_{\mathrm{proet}}$ 上的值 $G(U)$ 与紧 Hausdorff 站点上的值 $F(S)$ 甚至不在同一个
比较问题中。真正可靠的接口必须先写出两个站点，再说明哪一步只是形式类比，哪一步
来自几何输入。

本章在 $X=\operatorname{Spec}k$、$k$ 代数闭的有限层上完成一个 worked example：有限
集合 $I$ 分别给出离散紧空间 $I$ 与有限 étale 对象
$\coprod_I X$，两边的 sheaf 等化子都化为按分支取乘积。随后证明好测试对象检测零
sheaf、分裂覆盖使 Čech 复形收缩这两个共同机制。无限 profinite 层、w-contractible
基的存在以及由此计算全部导出同调，均在需要的准确位置标为外部输入。最后还会登记
Wolf 的已知比较定理；它通过 Galois category 与 pyknotic spaces 建立接口，而不是把
两个原始站点直接等同。

## 7.1 先固定两个不同的站点

凝聚侧取

$$
\mathbf{CHaus}_\kappa
$$

及其有限 jointly surjective 覆盖。凝聚集合或凝聚阿贝尔群是这个站点上的 sheaf；
测试对象只携带紧 Hausdorff 拓扑，没有到某个 scheme 的结构映射。

几何侧固定 scheme $X$，取 Bhatt--Scholze 的小 pro-etale 站点

$$
X_{\operatorname{proet}}.
$$

按 Bhatt--Scholze 的原始口径，对象是 weakly étale 态射 $U\to X$：该态射平坦，且
对角态射 $U\to U\times_XU$ 也平坦；覆盖取这些对象之间的 fpqc covering。带 affine
转移的 étale 态射逆极限给 pro-etale 描述，而 weakly étale 与 pro-etale 口径定义
同一 Grothendieck topology 的比较、大小约定及站点基定理均采用 Bhatt--Scholze 的
外部输入。正文只使用有限 étale 态射平坦且对角仍 étale，因而属于该站点，以及
sheaf 对覆盖满足下降。

两个函子类别分别是

$$
\operatorname{Shv}(\mathbf{CHaus}_\kappa)
\quad\text{与}\quad
\operatorname{Shv}(X_{\operatorname{proet}}).
$$

同样写成“取 sheaf”不会在它们之间自动产生函子。特别地，一个 profinite 集合 $S$
本身不是 $X$-scheme，而一个 pro-etale 对象 $U\to X$ 也没有自动成为紧 Hausdorff
空间。

## 7.2 Worked example：有限分支上的同一个等化子形状

令 $k$ 为代数闭域，$X=\operatorname{Spec}k$，并取有限集合 $I$。在 pro-etale 侧令

$$
U_I=\coprod_{i\in I}X\longrightarrow X.
$$

这是有限 étale 对象。各分支嵌入 $X_i\to U_I$ 构成覆盖，而且

$$
X_i\times_{U_I}X_j
\cong
\begin{cases}
X,&i=j,\\
\varnothing,&i\ne j.
\end{cases}
$$

若 $G$ 是集合值 pro-etale sheaf，sheaf 等化子为

$$
G(U_I)\longrightarrow
\prod_{i\in I}G(X_i)
\rightrightarrows
\prod_{(i,j)\in I^2}G(X_i\times_{U_I}X_j).
$$

非对角交为空，对角上的两条限制映射相同；再用 $G(\varnothing)=*$，得到

$$
G(U_I)\cong\prod_{i\in I}G(X).
$$

对阿贝尔群值 sheaf，空对象上的值换成 $0$，结论仍然相同。

凝聚侧把同一个有限集合 $I$ 赋予离散拓扑，以各单点
$\{i\}\to I$ 覆盖。交空间也满足

$$
\{i\}\times_I\{j\}
\cong
\begin{cases}
*,&i=j,\\
\varnothing,&i\ne j.
\end{cases}
$$

所以任意凝聚 sheaf $F$ 满足

$$
F(I)\cong\prod_{i\in I}F(*).
$$

这里的输入分别是 $G$ 与 $F$，步骤都是把覆盖代入 sheaf 等化子，输出都是有限乘积。
但没有额外指定 $G(X)\cong F(*)$，两个输出之间仍没有同构。这个例子证明的是下降计算
的共同形式，而不是两个 topos 的等价。

## 7.3 好测试对象怎样检测 sheaf

下面的命题与几何来源无关，因而可以同时用于两个站点。

**命题 7.3.1（覆盖检测）。** 设 $\mathcal C$ 是站点，
$\mathcal P\subset\mathcal C$ 是一族对象，并假设每个 $U\in\mathcal C$ 都有覆盖
$\{P_a\to U\}$，其中 $P_a\in\mathcal P$。若阿贝尔群值 sheaf $F$ 对所有
$P\in\mathcal P$ 都满足 $F(P)=0$，则 $F=0$。

**证明。** 对任意 $U$ 选上述覆盖。sheaf 的分离性给单射

$$
F(U)\hookrightarrow\prod_aF(P_a).
$$

右侧为零，故 $F(U)=0$。由于 $U$ 任意，$F$ 为零 sheaf。证毕。

在凝聚站点中，极不连通紧 Hausdorff 空间对连续满射具有投射提升性质，并形成足够多
的测试对象；这是第一卷使用的外部拓扑输入及其站点比较后果。在 pro-etale 理论中，
相应工作由 w-contractible 局部对象承担；“存在足够多这类对象”以及其几何构造采用
Bhatt--Scholze 的外部输入。命题 7.3.1 说明，一旦各自的存在定理给出，零对象检测
这部分后果已经在书内完整证明。

## 7.4 分裂覆盖为什么杀死 Čech 障碍

投射局部对象最直接的用途，是把覆盖变成带截面的覆盖。设 $V\to P$ 是覆盖并有截面
$s:P\to V$。它的增广 Čech nerve 为

$$
\cdots
V\times_PV\times_PV
\rightrightarrows
V\times_PV
\rightrightarrows
V
\longrightarrow P.
$$

**命题 7.4.1（分裂 Čech 收缩）。** 对任意阿贝尔群值 sheaf $F$，上述覆盖的增广
Čech 上链复形在正次数正合。

**证明。** 截面 $s$ 给增广 simplicial 对象一个额外退化：在一个纤维积元组前插入
$s$ 所选的分支。对 $F$ 反变取值后，得到 Čech 上链复形的同伦算子 $h$。simplicial
恒等式给

$$
dh+hd=\operatorname{id}
$$

于正次数；在次数零，增广映射识别其核。因此正次数 cohomology 为零。证毕。

极不连通对象和 w-contractible 对象的共同技术角色正在这里：它们让适当覆盖分裂，
于是相同的形式收缩可用。由“这些对象上的 Čech 复形正合”进一步推出全部 sheaf
导出同调消失，还需要 Čech-to-derived 比较、足够基与 acyclicity 条件；这些条件不能
从上面的三行同伦公式自动得到。

## 7.5 无限 profinite 层需要真正的比较函子

在 $X=\operatorname{Spec}k$ 的特殊情形，profinite 集合
$S=\varprojlim_iS_i$ 可提示一个几何对象

$$
U_S=\varprojlim_i\coprod_{S_i}X.
$$

若 $k$ 取离散拓扑，其坐标环可写为 locally constant 函数环

$$
C(S,k)=\varinjlim_i k^{S_i}.
$$

把 $U_S\to X$ 识别为所选 pro-etale 站点中的对象，是 pro-etale 几何输入。即便有了
$S\mapsto U_S$，若想令 $F_G(S)=G(U_S)$ 成为凝聚 sheaf，还必须逐项验证：连续映射的
函子性、有限 jointly surjective 覆盖被送到有效覆盖、纤维积相容性，以及无限极限与
$G$ 的下降是否相容。

有限层的 7.2 节只使用有限 coproduct，因此没有这些问题。无限层若缺少上述验证，
$G(U_S)$ 与任意凝聚值 $F(S)$ 只是两个长得相似的表达式；把它们直接等同，就是失败
条件。对一般 base scheme $X$，还要处理几何连通分支、剩余域和 base change，不能把
$C(S,k)$ 的点情形公式原样搬过去。

## 7.6 已知的 pro-étale--pyknotic 比较

**外部输入定理 7.6.1（Wolf）.** 设 $X$ 是 coherent scheme，并令
$\operatorname{Gal}(X)$ 为 Barwick--Glasman--Haine 意义的 Galois category。则 $X$
的 hypercomplete pro-étale $\infty$-topos 等价于
$\operatorname{Gal}(X)$ 在 pyknotic spaces 中的连续表示范畴。

这个定理说明 pro-étale 与 pyknotic 的高阶接口已经存在，但它没有断言
$X_{\operatorname{proet}}$ 与 compact Hausdorff 测试站点相同，也不把任意
$G(U_S)$ 自动变成凝聚 sheaf。比较经过 $\operatorname{Gal}(X)$，并且 coherent 与
hypercomplete 两项假设都属于定理陈述。把该等价稳定化可以得到谱值接口；进一步要求
它保持 solid 或 analytic localization，则还须检查相应 Dirac--测度 cone 和幺半结构。

## 7.7 共同机制停在何处

本章得到的共同结论是形式性的：有限不交分支把 sheaf 值化为乘积；足够多的好测试
对象检测零 sheaf；带截面的覆盖使增广 Čech 复形收缩。这些命题在两个站点上都可
调用，因为证明只用覆盖与 sheaf 公理。两侧真正不同的内容，是好对象的几何构造、
pro-etale 逆极限的合法性以及可能的站点比较函子。附录 D 保留两个站点的对照表和
典型误用；本章的 finite worked example 与收缩证明则提供其数学主线。

## 练习

**练习 7.1.** 对 $I=\{0,1,2\}$ 写出 7.2 节等化子的每一个非空交项，并直接构造
$G(U_I)\to G(X)^3$ 的逆映射。

**练习 7.2.** 把命题 7.3.1 推广为：若 sheaf 态射在所有 $P\in\mathcal P$ 上为单射，
则它在所有对象上为单射。

**练习 7.3.** 对带截面覆盖写出 Čech 复形前四项，并验证额外退化给出的
$dh+hd=\operatorname{id}$。

**练习 7.4.** 令 $S=\mathbb N\cup\{\infty\}$。证明
$C(S,k)$ 是最终常值的 $k$-值序列环，并写出它作为有限函数环 $k^{S_i}$ 的滤过余极限。

**练习 7.5.** 说明从 $G(U_S)$ 定义凝聚 sheaf 至少需要验证哪四项结构，并为每一项
指出只检查有限集合为何不够。

**练习 7.6.** 逐项说明定理 7.6.1 为什么不是
$\operatorname{Shv}(X_{\operatorname{proet}})\simeq
\operatorname{Shv}(\mathbf{CHaus})$ 的断言。
