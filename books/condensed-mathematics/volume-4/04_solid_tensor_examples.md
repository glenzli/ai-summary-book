# 第四章：solid 张量积例子

有限自由群的张量由笛卡尔积上的基控制；把有限集合换成 Cantor 空间后，一个“自由
solid 元素”不再只是有限个点的线性组合，而是所有有限商上相容的整值测度。两个这样
的对象应通过测度外积落到 $S\times T$，这正是
$\otimes^{L,\square}$ 的结构定理。若误用普通张量，无限乘积和换底会立刻产生缺失
元素。

以下从有限集合的基双射开始，再对 Cantor 空间写出有限商、转移映射和相容系数，最后
用无统一分母的有理数列定位普通张量失败。每次计算都区分普通、派生与 solid 派生
张量；无限情形的比较同构保留为 Scholze 外部输入，而有限层和反例在正文完成。

## 4.1 有限集合

若 $S,T$ 有限，则

$$
\mathbb Z^\square[S]=\mathbb Z[S],
\qquad
\mathbb Z^\square[T]=\mathbb Z[T].
$$

因此

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z[S\times T].
$$

**证明。** 有限离散集合同时是 compact Hausdorff 和 profinite。此时自由凝聚阿贝尔群
$\mathbb Z[\underline S]$ 已经是 solid 对象，solidification 不改变它；对有限离散集
$T$ 应用同一事实，也有 $\mathbb Z^\square[T]=\mathbb Z[T]$。普通自由阿贝尔群满足

$$
\mathbb Z[S]\otimes_{\mathbb Z}\mathbb Z[T]
\cong
\mathbb Z[S\times T],
\qquad
[s]\otimes[t]\mapsto[(s,t)].
$$

因为两边都由基元素生成，且该映射把基双射到基，所以是同构。有限情形无高阶导出项，故导出 solid 张量积与普通张量积一致。证毕。

## 4.2 profinite 集合

若 $S,T$ profinite，则输入定理给出

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

这可视为测度外积。

更准确地说，$\mathbb Z^\square[S]$ 可理解为 $S$ 上的 solid 整系数测度对象。若

$$
S=\varprojlim_iS_i,\qquad T=\varprojlim_jT_j
$$

是有限集合的逆极限，则 $S\times T=\varprojlim_{i,j}(S_i\times T_j)$，而有限层上的公式与逆极限相容。深层输入是 solid 张量积正是使这种 profinite 极限计算成立的张量结构；它不是普通阿贝尔群张量积的形式推论。

## 4.3 Worked example：Cantor 测度的外积

令

$$
S=\{0,1\}^{\mathbb N},
\qquad
S_m=\{0,1\}^m,
\qquad
S=\varprojlim_mS_m.
$$

投影 $p_m:S_{m+1}\to S_m$ 遗忘最后一位。它诱导

$$
(p_m)_*:\mathbb Z[S_{m+1}]\to\mathbb Z[S_m],
\qquad
[(x,\varepsilon)]\mapsto[x].
$$

因此一个全局截面
$\mu\in\mathbb Z^\square[S](*)$ 可由相容族
$\mu\in\varprojlim_m\mathbb Z[S_m]$ 逐层写成

$$
\mu_m=\sum_{x\in S_m}a_x^{(m)}[x],
$$

相容条件为

$$
a_x^{(m)}=a_{(x,0)}^{(m+1)}+a_{(x,1)}^{(m+1)}.
$$

这正是 cylinder set 上有限可加整值测度的关系。对另一元素
$\nu_n=\sum_{y\in S_n}b_y^{(n)}[y]$，有限层外积定义为

$$
\mu_m\boxtimes\nu_n
=
\sum_{(x,y)\in S_m\times S_n}
a_x^{(m)}b_y^{(n)}[(x,y)].
$$

沿 $m,n$ 的转移映射推前时，系数求和分别使用上面的相容关系，所以这些有限层输出组成

$$
(\mu_m\boxtimes\nu_n)_{m,n}
\in
\varprojlim_{m,n}\mathbb Z[S_m\times S_n]
=\mathbb Z^\square[S\times S](*).
$$

外部 solid 张量定理断言该双线性构造由同构

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[S]
\simeq
\mathbb Z^\square[S\times S]
$$

表示。输入是两组相容有限层系数，步骤是逐层取外积并检查推前相容，输出是乘积 Cantor
空间上的 solid 测度全局截面。若有限层族不满足系数求和关系，它根本不是
$\mathbb Z^\square[S](*)$ 的元素，外积计算也没有合法输入。对象级同构比这项全局
截面计算更强；它仍由外部 solid 张量定理提供，不能由逐元素计算替代。

## 4.4 乘积型对象

若

$$
\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z},
\quad
\mathbb Z^\square[T]\cong\prod_J\underline{\mathbb Z},
$$

则

$$
\prod_I\underline{\mathbb Z}
\otimes^{L,\square}
\prod_J\underline{\mathbb Z}
\simeq
\prod_{I\times J}\underline{\mathbb Z}.
$$

这是上一节在 Stone 对偶视角下的一个常见计算：若 profinite 集合由布尔代数的极限给出，对应的 solid 自由对象表现为乘积型对象。该公式应读作 solid 范畴中的公式，而不是 $\mathbf{Ab}$ 中的公式。

**普通张量积反例。** 自然映射

$$
\left(\prod_{n\ge1}\mathbb Z\right)\otimes_{\mathbb Z}\mathbb Q
\longrightarrow
\prod_{n\ge1}\mathbb Q
$$

不是满射。左边等于把 $\prod_n\mathbb Z$ 对所有非零整数局部化，因而每个元素可写成

$$
\frac{(a_n)_n}{m}
$$

其中分母 $m$ 对所有坐标相同。右边元素

$$
\left(1,\frac12,\frac13,\frac14,\ldots\right)
$$

没有统一有界分母，因此不在像中。这个例子说明普通张量积不与无限乘积按坐标交换；solid 张量积公式的内容恰恰在于修正这个缺陷。

## 4.5 无统一分母定位了哪一步失败

有限集合上，普通自由对象已经 solid，且没有高阶 Tor，所以基元素计算足够。Cantor
空间上，有限商逆极限和 solidification 是输出的一部分；若只在 $\mathbf{Ab}$ 中取
普通张量，4.4 节的序列 $(1,1/2,1/3,\ldots)$ 就因没有统一分母而丢失。这个反例
精确说明普通 tensor 不保持相关无限乘积，不能据此推出 4.3 节的测度外积同构。正确
调用公式时，输入须先是 solid 对象，运算须写成 $\otimes^{L,\square}$，profinite
极限也须在该 localization 中解释。第五章将把这种“先指定局部对象、再局部化张量”
的机制推广到一般 analytic ring。

## 练习

**练习 4.1.** 对有限集合验证张量公式。

**练习 4.2.** 解释为什么 profinite 情形可理解为测度外积。

**练习 4.3.** 举例说明普通张量积不保持无限乘积。

**练习 4.4.** 证明上面的 $\left(1,\frac12,\frac13,\ldots\right)$ 不在 $\left(\prod_n\mathbb Z\right)\otimes\mathbb Q$ 的像中。
