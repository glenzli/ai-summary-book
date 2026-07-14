# 第十一章：派生张量积与 Tor

相对张量积 $M\otimes_RN$ 只右正合，因此它会抹去短正合列左端的核。这个缺失量正是
Tor 应记录的内容，但在凝聚语境中不能直接照搬逐点 $R(S)$-模计算：投射对象属于整个
sheaf 模范畴，张量积也在该范畴中取余等化子。第十章已经构造足够多的投射
$R[\underline E]$，使标准导出函子方法能够在正确范畴内运行。

我们选择 $M$ 的投射分解 $P_\bullet\to M$，以
$P_\bullet\otimes_RN$ 的同调定义 $\operatorname{Tor}^R_i(M,N)$，并通过比较定理
证明选择无关。随后从投射消失、短正合列和 horseshoe 引理推出基本计算规则。更深的
Tor 公式需要 solid 完备性；这里先把它们将要依赖的定义与长正合列完整建立起来。

## 11.1 张量积的右正合性

设 $R$ 是交换凝聚环。对固定的 $N\in\mathbf{CondMod}_R$，有函子

$$
-\otimes_R N:\mathbf{CondMod}_R\to\mathbf{CondMod}_R.
$$

**命题 11.1.** 函子 $-\otimes_R N$ 是右正合函子。

**证明.** 相对张量积 $M\otimes_R N$ 可由余等化子定义：

$$
M\otimes R\otimes N
\rightrightarrows
M\otimes N
\to
M\otimes_R N.
$$

更结构化地说，附录 E 定义了内部 Hom，并给出伴随

$$
\operatorname{Hom}_R(M\otimes_R N,P)
\cong
\operatorname{Hom}_R(M,\mathcal Hom_R(N,P)).
$$

因此 $-\otimes_RN$ 是左伴随。左伴随保持所有余极限，特别保持 cokernel。在阿贝尔范畴中，加性函子保持 cokernel 等价于右正合。故 $-\otimes_RN$ 右正合。证毕。

**注 11.2.** 本章只需要右正合性来定义左导出函子。内部 Hom 与闭幺半结构会在后续附录中系统处理。

## 11.2 派生张量积

由于 $\mathbf{CondMod}_R$ 有足够多投射对象，可以对右正合函子 $-\otimes_R N$ 取左导出函子。

**定义 11.3.** 设 $M,N\in\mathbf{CondMod}_R$。取 $M$ 的投射分解

$$
\cdots\to P_2\to P_1\to P_0\to M\to 0.
$$

定义派生张量复形为

$$
P_\bullet\otimes_R N.
$$

其同调对象记为

$$
\operatorname{Tor}^R_i(M,N)
=
H_i(P_\bullet\otimes_R N).
$$

特别地，

$$
\operatorname{Tor}^R_0(M,N)\cong M\otimes_R N.
$$

投射分解的选择无关性由附录 I 的比较定理和附录 H 的 K-flat 形式给出。

## 11.3 对称性

因为 $R$ 是交换凝聚环，张量积 $\otimes_R$ 是对称的。因此存在自然同构

$$
\operatorname{Tor}^R_i(M,N)
\cong
\operatorname{Tor}^R_i(N,M).
$$

**定理 11.4.** 对 $M,N\in\mathbf{CondMod}_R$，存在自然同构

$$
\operatorname{Tor}^R_i(M,N)
\simeq
\operatorname{Tor}^R_i(N,M).
$$

**证明.** 取 $M,N$ 的 K-flat 替换

$$
P^\bullet\to M,\qquad Q^\bullet\to N.
$$

由附录 H 的 K-flat 比较定理，

$$
M\otimes_R^LN\simeq P^\bullet\otimes_RQ^\bullet,
\qquad
N\otimes_R^LM\simeq Q^\bullet\otimes_RP^\bullet.
$$

交换凝聚环上的张量积有链级对称同构

$$
P^\bullet\otimes_RQ^\bullet
\to
Q^\bullet\otimes_RP^\bullet,
\qquad
p\otimes q\mapsto (-1)^{|p||q|}q\otimes p.
$$

Koszul 符号保证该映射与总微分相容。它是复形同构，故在导出范畴中给出自然同构

$$
M\otimes_R^LN\simeq N\otimes_R^LM.
$$

取同调即得

$$
\operatorname{Tor}^R_i(M,N)
\simeq
\operatorname{Tor}^R_i(N,M).
$$

证毕。

## 11.4 自由模的 Tor 消失

**命题 11.5.** 若 $P$ 是投射凝聚 $R$-模，则

$$
\operatorname{Tor}^R_i(P,N)=0
\qquad i>0.
$$

**证明.** 对 $P$ 可取长度为零的投射分解

$$
0\to P\to P\to 0
$$

或者更直接地，投射分解只有第 $0$ 项。代入 $-\otimes_R N$ 后，复形高阶项为零，因此高阶同调为零。证毕。

**推论 11.6.** 若 $E$ 极不连通，则

$$
\operatorname{Tor}^R_i(R[\underline E],N)=0
\qquad i>0.
$$

**证明.** 第十章定理 10.7 说明 $R[\underline E]$ 投射，代入命题 11.5。证毕。

## 11.5 短正合列与长正合 Tor 列

**定理 11.7.** 设

$$
0\to M'\to M\to M''\to 0
$$

是 $\mathbf{CondMod}_R$ 中短正合列。对任意 $N$，存在自然长正合列

$$
\cdots\to
\operatorname{Tor}^R_1(M'',N)
\to
M'\otimes_R N
\to
M\otimes_R N
\to
M''\otimes_R N
\to 0.
$$

更完整地，对所有 $i\ge 1$ 有连接同态

$$
\operatorname{Tor}^R_i(M'',N)
\to
\operatorname{Tor}^R_{i-1}(M',N).
$$

**证明.** 短正合列在导出范畴中给出 distinguished triangle

$$
M'\to M\to M''\to M'[1].
$$

由附录 H，$-\otimes_R^LN$ 是导出范畴上的三角函子；等价地，也可使用附录 I 定理 I.9。于是得到三角

$$
M'\otimes_R^LN
\to
M\otimes_R^LN
\to
M''\otimes_R^LN
\to
(M'\otimes_R^LN)[1].
$$

对该三角取同调，得到长正合列

$$
\cdots\to
H_i(M'\otimes_R^LN)
\to
H_i(M\otimes_R^LN)
\to
H_i(M''\otimes_R^LN)
\to
H_{i-1}(M'\otimes_R^LN)
\to\cdots.
$$

按定义 $H_i(-\otimes_R^LN)=\operatorname{Tor}_i^R(-,N)$。当 $i=0$ 时，$\operatorname{Tor}_0^R(-,N)\cong -\otimes_RN$，于是得到题中列出的尾部

$$
\operatorname{Tor}^R_1(M'',N)
\to
M'\otimes_RN
\to
M\otimes_RN
\to
M''\otimes_RN
\to0.
$$

最后的 $0$ 来自右正合性，即命题 11.1。证毕。

## 11.6 ED 测试对象与 Tor 的关系

极不连通空间给出大量投射模 $R[\underline E]$，因此实际构造投射分解时，可以从 ED 测试对象出发：

$$
\bigoplus_\alpha R[\underline{E_\alpha}]\to M.
$$

这说明凝聚语境中的 Tor 计算不是抽象存在性游戏，而是可以由测试空间上的截面逐步生成。

但是要注意：即使 $M(E)$ 在每个 ED 空间上容易描述，也不意味着 $\operatorname{Tor}$ 计算自动简单。核、关系和高阶 syzygy 仍然可能复杂。后续 solid 理论的作用之一，正是给某些对象提供更好的完备性与张量行为。

第一卷的基本 Tor 消失和长正合列规则整理在 [附录 G](G_basic_ext_and_tor_calculations.md)。

## 11.7 Tor 记录张量失去的核

右正合性给出 $\operatorname{Tor}^R_0(M,N)=M\otimes_RN$，投射分解的比较定理保证
高阶同调与选择无关，horseshoe 构造则把短正合列送到长正合 Tor 列。ED 自由模让这些
定义可由测试截面逐层生成，但一般 syzygy 仍可能复杂。下一章引入 solid 条件，目的
不是改写这套导出机制，而是挑出一类在无限乘积、测度与完成化下具有更好张量行为的
凝聚对象。

## 练习

**练习 11.1.** 证明 $\operatorname{Tor}^R_0(M,N)\cong M\otimes_R N$。

**练习 11.2.** 设 $E$ 极不连通。用第十章的表示公式证明 $R[\underline E]$ 投射，并据此证明推论 11.6。

**练习 11.3.** 写出定理 11.7 中长正合列的前六项。

**练习 11.4.** 查阅 horseshoe lemma，并说明其证明只使用阿贝尔范畴与足够多投射对象。
