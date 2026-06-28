# 第三章：Ext 与 Tor 的计算模板

## 本章目标

本章把第一卷附录 G 中的 Ext/Tor 计算整理成可证明的工作规则。默认使用如下输入定理：$\mathbf{CondAb}$ 是有足够投射对象的 Grothendieck 阿贝尔范畴；对凝聚交换环 $R$，$\mathbf{CondMod}_R$ 也是有足够投射对象的阿贝尔范畴，且张量积是右正合双函子。

## 3.1 Ext 的定义和合法性

设 $M,A\in\mathbf{CondAb}$。取投射分解

$$
\cdots\to P_2\to P_1\to P_0\to M\to0.
$$

定义

$$
\operatorname{Ext}^i_{\mathbf{CondAb}}(M,A)
=H^i\operatorname{Hom}_{\mathbf{CondAb}}(P_\bullet,A),
$$

其中 $\operatorname{Hom}(P_\bullet,A)$ 是上链复形

$$
0\to\operatorname{Hom}(P_0,A)
\to\operatorname{Hom}(P_1,A)
\to\operatorname{Hom}(P_2,A)\to\cdots .
$$

**命题 3.1.1（定义与分解无关）。** 上述群在自然同构意义下不依赖投射分解的选择。

**证明。** 在任意有足够投射对象的阿贝尔范畴中，两个投射分解 $P_\bullet\to M$ 与 $Q_\bullet\to M$ 之间存在提升恒等映射的链映射

$$
P_\bullet\to Q_\bullet,\qquad Q_\bullet\to P_\bullet,
$$

且任意两个这样的链映射链同伦。证明使用投射性逐阶提升：在第 $0$ 阶，由 $Q_0\to M$ 是满射且 $P_0$ 投射，$P_0\to M$ 提升到 $P_0\to Q_0$；假设已构造到第 $n$ 阶，则边界相容性把问题化为从 $P_{n+1}$ 到某个核对象的提升，仍由投射性解决。链同伦唯一性同理逐阶构造。对 $\operatorname{Hom}(-,A)$ 后，链同伦等价的复形映射诱导相同同调映射，于是得到自然同构。证毕。

## 3.2 投射对象的 Ext 消失

**命题 3.2.1。** 若 $P$ 是 $\mathbf{CondAb}$ 中投射对象，则对任意 $A$，

$$
\operatorname{Ext}^i(P,A)=0,\qquad i>0.
$$

**证明。** $P$ 的投射分解可取为

$$
0\to P\xrightarrow{\operatorname{id}}P\to0
$$

集中在 $0$ 阶。于是 $\operatorname{Hom}(P,A)$ 复形只在 $0$ 阶非零，高阶同调全为 $0$。证毕。

更具体地，若 $E$ 极不连通，第一卷证明自由凝聚阿贝尔群

$$
\mathbb Z[\underline E]
$$

是投射对象。因此

$$
\operatorname{Ext}^i(\mathbb Z[\underline E],A)=0
\quad(i>0).
$$

这里真正使用的是：对凝聚阿贝尔群的满射 $B\to C$，在极不连通 $E$ 上取值 $B(E)\to C(E)$ 仍满射；而

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline E],A)
\cong A(E).
$$

## 3.3 两项投射分解公式

设

$$
0\to P_1\xrightarrow{d}P_0\to M\to0
$$

是 $M$ 的投射分解。对任意 $A$，令

$$
d^\vee:\operatorname{Hom}(P_0,A)\to\operatorname{Hom}(P_1,A),
\qquad
d^\vee(f)=f\circ d .
$$

**命题 3.3.1。** 有自然同构

$$
\operatorname{Hom}(M,A)\cong\ker d^\vee,
$$

$$
\operatorname{Ext}^1(M,A)\cong\operatorname{coker}d^\vee,
$$

且

$$
\operatorname{Ext}^i(M,A)=0\qquad(i\ge2).
$$

**证明。** 对短正合列应用左正合函子 $\operatorname{Hom}(-,A)$，得到复形

$$
0\to\operatorname{Hom}(P_0,A)
\xrightarrow{d^\vee}
\operatorname{Hom}(P_1,A)\to0
$$

计算 $\operatorname{RHom}(M,A)$。第 $0$ 个同调是 $\ker d^\vee$，这正是经过 $P_0\to M$ 因子化的映射 $P_0\to A$，即 $\operatorname{Hom}(M,A)$。第 $1$ 个同调是余核，高阶没有项。证毕。

这个公式是手算 $\operatorname{Ext}^1$ 的基本入口：找到短投射分解后，问题变成显式计算两个 Hom 群之间的余核。

## 3.4 维数平移和长正合列

设

$$
0\to K\to P\to M\to0
$$

正合，且 $P$ 投射。对任意 $A$ 和 $n\ge1$，有维数平移同构

$$
\operatorname{Ext}^{n+1}(M,A)\cong
\operatorname{Ext}^n(K,A).
$$

**证明。** 短正合列给出 Ext 长正合列

$$
\cdots\to
\operatorname{Ext}^n(P,A)\to
\operatorname{Ext}^n(K,A)\to
\operatorname{Ext}^{n+1}(M,A)\to
\operatorname{Ext}^{n+1}(P,A)\to\cdots .
$$

由命题 3.2.1，$\operatorname{Ext}^m(P,A)=0$ 对 $m>0$ 成立。代入后，中间箭头成为同构。证毕。

Ext 长正合列本身来自右导出函子的一般理论；若需要完全初等地写出连接同态，可取三个对象的相容投射分解，用 snake lemma 作用到 Hom 复形的短正合列。

## 3.5 Tor 的定义和消失判别

设 $R$ 是凝聚交换环，$M,N\in\mathbf{CondMod}_R$。取 $M$ 的投射分解 $P_\bullet\to M$，定义

$$
\operatorname{Tor}_i^R(M,N)
=H_i(P_\bullet\otimes_RN).
$$

和 Ext 一样，定义与投射分解无关，因为投射分解之间的链映射唯一到链同伦。

**命题 3.5.1（平坦对象的 Tor 消失）。** 若 $M$ 是平坦 $R$-模，则

$$
\operatorname{Tor}_i^R(M,N)=0
\qquad(i>0)
$$

对任意 $N$ 成立。

**证明。** 平坦性定义为函子 $M\otimes_R-$ 保持短正合列。等价地，左导出函子 $L_i(M\otimes_R-)$ 在 $i>0$ 消失。由于 $\operatorname{Tor}_i^R(M,N)$ 正是该左导出函子在 $N$ 上的值，结论成立。若偏好用分解证明，可取 $N$ 的投射分解 $Q_\bullet\to N$，平坦性保证 $M\otimes_RQ_\bullet\to M\otimes_RN$ 是无高阶同调的分解，故高阶 Tor 为 $0$。证毕。

**命题 3.5.2（短投射分解的 Tor 公式）。** 若

$$
0\to P_1\xrightarrow{d}P_0\to M\to0
$$

是 $R$-模的投射分解，则

$$
\operatorname{Tor}_1^R(M,N)\cong
\ker(P_1\otimes_RN\to P_0\otimes_RN),
$$

$$
M\otimes_RN\cong
\operatorname{coker}(P_1\otimes_RN\to P_0\otimes_RN),
$$

且 $\operatorname{Tor}_i^R(M,N)=0$ 对 $i\ge2$ 成立。

**证明。** 复形 $P_\bullet\otimes_RN$ 只有 $1$ 阶和 $0$ 阶两项。其第 $1$ 个同调是核，第 $0$ 个同调是余核，高阶无项。第 $0$ 个同调与 $M\otimes_RN$ 的同构来自张量积的右正合性。证毕。

## 3.6 风险点

1. “在 ED 对象上取值容易”不等于“Ext 自动知道”。Ext 要求投射分解或导出范畴输入。
2. 投射分解必须在 $\mathbf{CondAb}$ 或 $\mathbf{CondMod}_R$ 中。普通拓扑向量空间的 projective resolution 不能直接替代凝聚范畴中的分解。
3. solid、analytic 或 liquid 语境中的张量积通常经过局部化或 solidification；不能把普通阿贝尔群张量积公式直接搬过去。

## 练习

**练习 3.1.** 设 $S$ 为有限集合，证明 $\operatorname{Ext}^i(\mathbb Z[\underline S],A)=0$ 对 $i>0$ 成立。

**练习 3.2.** 对两项投射分解写出 $\operatorname{Ext}^1(M,A)$ 中一个元素对应的短正合扩张。

**练习 3.3.** 设 $M$ 有长度为 $n$ 的投射分解，证明 $\operatorname{Ext}^i(M,A)=0$ 对 $i>n$ 成立。
