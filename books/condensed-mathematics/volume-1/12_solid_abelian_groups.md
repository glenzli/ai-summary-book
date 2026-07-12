# 第十二章：固体阿贝尔群

## 本章目标

本章定义固体阿贝尔群（solid abelian group），并说明它们为什么是凝聚阿贝尔群中的“完备化”对象。核心对象是

$$
\mathbb Z^\square[S],
$$

即 profinite 集合 $S$ 上的自由固体阿贝尔群。

本章主要依据 Scholze 讲义第五、六讲。若某个定理证明过长，本章标为引用结果，并说明后续如何使用。

## 依赖前置知识

需要凝聚阿贝尔群、自由对象、投射对象、派生 Hom 和 profinite 集合。

本章所有测试对象均属于 \(\mathbf{ProFin}_\kappa\)，所有未注明大小的集合指标均满足
基数小于固定 \(\kappa\)。大凝聚范畴中的任意小指标版本需另用附录 A.5 过渡。

## 12.1 从自由凝聚群到自由固体群

设 $S\in\mathbf{ProFin}_\kappa$。写作逆极限

$$
S=\varprojlim_i S_i,
$$

其中 $S_i$ 是有限离散集合。

第七章中，$\mathbb Z[\underline S]$ 表示由凝聚集合 $\underline S$ 生成的自由凝聚阿贝尔群。固体理论引入另一个对象：

**定义 12.1.** profinite 集合 $S=\varprojlim_i S_i$ 上的自由固体阿贝尔群定义为

$$
\mathbb Z^\square[S]
=
\varprojlim_i \mathbb Z[\underline{S_i}]
$$

其中极限在 $\mathbf{CondAb}$ 中取。

自然映射

$$
S\to \mathbb Z^\square[S]
$$

诱导凝聚阿贝尔群态射

$$
\mathbb Z[\underline S]\to \mathbb Z^\square[S].
$$

**注 12.2.** 当 $S$ 是有限离散集合时，系统稳定，故

$$
\mathbb Z^\square[S]\cong \mathbb Z[\underline S].
$$

差异只在无限 profinite 集合上出现。

## 12.2 整值测度

令

$$
C(S,\mathbb Z)
$$

表示从 profinite 集合 $S$ 到离散群 $\mathbb Z$ 的连续函数构成的阿贝尔群。

**命题 12.3.** $\mathbb Z^\square[S]$ 的全局截面可理解为

$$
M(S,\mathbb Z)=\operatorname{Hom}(C(S,\mathbb Z),\mathbb Z),
$$

即 $S$ 上的整值测度群。

**证明说明.** 若 $S=\varprojlim_i S_i$，则

$$
C(S,\mathbb Z)=\varinjlim_i C(S_i,\mathbb Z)
$$

因为连续映射 $S\to\mathbb Z$ 在某个有限商 $S_i$ 上因子化。于是

$$
\varprojlim_i \mathbb Z[S_i]
\cong
\varprojlim_i \operatorname{Hom}(C(S_i,\mathbb Z),\mathbb Z)
\cong
\operatorname{Hom}(C(S,\mathbb Z),\mathbb Z).
$$

这里使用有限集合上 $\mathbb Z[S_i]\cong\operatorname{Hom}(C(S_i,\mathbb Z),\mathbb Z)$。证毕。

更一般地，若 $T$ 是 profinite 测试对象，则

$$
\mathbb Z^\square[S](T)
\cong
\operatorname{Hom}(C(S,\mathbb Z),C(T,\mathbb Z)).
$$

这是因为

$$
\mathbb Z[\underline{S_i}](T)
\cong
C(T,\mathbb Z[S_i])
\cong
\operatorname{Hom}(C(S_i,\mathbb Z),C(T,\mathbb Z)),
$$

再对 $i$ 取逆极限。这个版本比只看全局截面更重要；后文判断凝聚对象同构时使用的是所有测试对象上的取值。

## 12.3 固体阿贝尔群的定义

**定义 12.4.** 凝聚阿贝尔群 $A$ 称为固体阿贝尔群，如果对任意
\(S\in\mathbf{ProFin}_\kappa\)，自然映射

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z^\square[S],A)
\longrightarrow
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline S],A)
\cong A(S)
$$

是双射。

等价地，任意映射

$$
f:S\to A
$$

都唯一延拓为

$$
\tilde f:\mathbb Z^\square[S]\to A.
$$

**直观说明.** 凝聚阿贝尔群 $A$ 固体，表示它不仅能对点状 Dirac 测度取值，还能对所有整值测度取值，而且这种积分由 $f:S\to A$ 唯一决定。

## 12.4 复形的固体性

**定义 12.5.** 设 $C\in D(\mathbf{CondAb})$。若对任意 profinite 集合 $S$，自然映射

$$
R\operatorname{Hom}(\mathbb Z^\square[S],C)
\longrightarrow
R\operatorname{Hom}(\mathbb Z[\underline S],C)
$$

是同构，则称 $C$ 为固体复形。

右侧也可写作

$$
R\Gamma(S,C).
$$

**注 12.6.** 对普通对象 $A\in\mathbf{CondAb}$，$A[0]$ 的固体性与 $A$ 的固体性并非形式上立即等价；Scholze 的结构定理保证这些定义相容。

## 12.5 Nöbeling 定理

固体理论依赖一个重要代数事实。

**外部输入定理 12.7（Nöbeling）.** 对任意
\(S\in\mathbf{ProFin}_\kappa\)，阿贝尔群

$$
C(S,\mathbb Z)
$$

是自由阿贝尔群。

**来源与外部边界.** S26 Theorem 5.4 给出 Bergman 版本证明；A23 Theorem
`Nobeling` 给出可形式化版本。第一卷附录 F、P 证明有限、可数和过滤代数模块，但不承担
任意 profinite 情形的完整超限构造，因此本章按输入定理使用。

**推论 12.8.** 对任意 \(S\in\mathbf{ProFin}_\kappa\)，存在满足
\(|I|<\kappa\) 的集合 \(I\)，使得

$$
\mathbb Z^\square[S]\cong \prod_I \underline{\mathbb Z}
$$

作为凝聚阿贝尔群成立。

**证明.** 由 Nöbeling 定理，

$$
C(S,\mathbb Z)\cong \bigoplus_I \mathbb Z.
$$

还需检查基的大小没有越过截断。若 \(S\) 无限，则

$$
|C(S,\mathbb Z)|
\le |\mathbb Z|^{|S|}
\le 2^{|S|}<\kappa,
$$

最后一步使用 \(\kappa\) 为不可数强极限。若 \(S\) 有限，则
\(|C(S,\mathbb Z)|\le\aleph_0^{|S|}=\aleph_0<\kappa\)。因此任一基 \(I\) 的
基数也小于 \(\kappa\)。于是下面的乘积仍是固定层级允许的乘积。

于是对任意 profinite 测试对象 $T$，

$$
\mathbb Z^\square[S](T)
\cong
\operatorname{Hom}(C(S,\mathbb Z),C(T,\mathbb Z))
\cong
\operatorname{Hom}\left(\bigoplus_I\mathbb Z,C(T,\mathbb Z)\right)
\cong
\prod_I C(T,\mathbb Z)
\cong
\left(\prod_I\underline{\mathbb Z}\right)(T).
$$

这些同构与 $T$ 的拉回相容，因此给出凝聚阿贝尔群同构。
证毕。

## 12.6 固体范畴的结构定理

**外部输入定理 12.9（Scholze）.** 记 $\mathbf{Solid}$ 为固体阿贝尔群构成的全子范畴。则：

1. $\mathbf{Solid}$ 是 $\mathbf{CondAb}$ 的阿贝尔全子范畴。
2. $\mathbf{Solid}$ 在固定 universe 允许的所有极限、余极限和扩张下稳定。
3. 对所有满足 \(|I|<\kappa\) 的集合 $I$，对象
   $\prod_I\underline{\mathbb Z}$ 构成一族紧投射生成元。
4. 包含函子
   $$
   \mathbf{Solid}\hookrightarrow \mathbf{CondAb}
   $$
   有左伴随
   $$
   M\mapsto M^\square.
   $$
5. 该左伴随是保持余极限的函子，并满足
   $$
   \mathbb Z[\underline S]^\square\simeq \mathbb Z^\square[S].
   $$

**来源与外部边界.** 这是 S26 Theorem 5.8(i) 的固定 \(\kappa\)-层级版本；紧投射
对象的精确描述还见 Corollary 6.1(i)。证明使用 Nöbeling 定理、固体复形判别以及对
$\mathbb Z^\square[S]$ 的派生 Hom 分析。第一卷把该定理作为 solid 理论的外部输入；
第二卷只证明接受输入后的局部化形式后果，不重做 S26 Lecture VI 的核心计算。

## 12.7 固化函子

**定义 12.10.** 左伴随

$$
(-)^\square:\mathbf{CondAb}\to\mathbf{Solid}
$$

称为固化（solidification）函子。

对凝聚阿贝尔群 $M$，自然映射

$$
M\to M^\square
$$

是把 $M$ 映到最接近它的固体对象的普遍态射：对任意固体阿贝尔群 $A$，复合给出双射

$$
\operatorname{Hom}_{\mathbf{Solid}}(M^\square,A)
\cong
\operatorname{Hom}_{\mathbf{CondAb}}(M,A).
$$

## 12.8 本章小结

本章引入了固体阿贝尔群：

1. $\mathbb Z^\square[S]$ 是 profinite 集合上的自由固体阿贝尔群。
2. 固体对象允许从 $S$ 到 $A$ 的映射唯一延拓到整值测度。
3. Nöbeling 定理说明 $C(S,\mathbb Z)$ 是自由阿贝尔群，从而 $\mathbb Z^\square[S]$ 是某个 $\prod_I\mathbb Z$。
4. 固体阿贝尔群的阿贝尔范畴结构、固化函子和派生反射局部化性质作为第二卷 D.1-D.3 的输入定理使用；第一卷只建立定义、Nöbeling 背景和基本计算入口。

下一章讨论固体张量积。

## 练习

**练习 12.1.** 证明当 $S$ 为有限离散集合时，$\mathbb Z^\square[S]\cong\mathbb Z[\underline S]$。

**练习 12.2.** 设 $S=\varprojlim_iS_i$。说明为什么连续映射 $S\to\mathbb Z$ 必在某个有限商 $S_i$ 上因子化。

**练习 12.3.** 假设 Nöbeling 定理，证明推论 12.8。

**练习 12.4.** 用定义证明：若 $A$ 固体，则任意 $f:S\to A$ 和 $\mu\in M(S,\mathbb Z)$ 可定义“积分” $\int_S f\,d\mu$。
