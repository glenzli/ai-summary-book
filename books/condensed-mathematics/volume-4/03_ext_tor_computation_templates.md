# 第三章：Ext 与 Tor 的投射分解计算

Ext 与 Tor 都不是从对象名称直接读出的数值。计算 Ext 要选择第一变量的投射分解、反向
施加 Hom 并取上同调；计算 Tor 要张量同一类分解并取同调。分解选择无关、投射对象的
高阶消失和连接同态分别由比较定理、投射性与 horseshoe/snake 机制保证。只有这些步骤
齐全，两项复形的 kernel 与 cokernel 才是合法输出。

以下接受第一卷已经建立的输入：$\mathbf{CondAb}$ 是有足够投射对象的 Grothendieck
阿贝尔范畴；对凝聚交换环 $R$，$\mathbf{CondMod}_R$ 有足够投射对象，且相对张量为
右正合双函子；用于计算 Tor 的投射 $R$-模为平坦对象。各节先证明一般规则，随后以
$0\to\underline{\mathbb Z}\xrightarrow{n}\underline{\mathbb Z}
\to\underline{\mathbb Z/n}\to0$ 完整计算 Ext、Tor 及 $n=0$ 时的失败。

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

且任意两个这样的链映射链同伦。证明使用投射性逐阶提升：在第 $0$ 阶，由 $Q_0\to M$ 是满射且 $P_0$ 投射，$P_0\to M$ 提升到 $P_0\to Q_0$；假设已构造到第 $n$ 阶，则边界相容性把问题化为从 $P_{n+1}$ 到某个核对象的提升，仍由投射性解决。

再设 $f,g:P_\bullet\to Q_\bullet$ 是两个这样的提升。令 $h_{-1}=0$，并假设
$h_0,\ldots,h_{n-1}$ 已使
$f_k-g_k=d_Qh_k+h_{k-1}d_P$ 对 $k<n$ 成立。则

$$
r_n=f_n-g_n-h_{n-1}d_P:P_n\longrightarrow Q_n
$$

满足 $d_Qr_n=0$。由于 $Q_\bullet\to M$ 正合，$r_n$ 的像落在
$\ker d_Q=\operatorname{im}(Q_{n+1}\to Q_n)$；又因 $P_n$ 投射，可把 $r_n$
提升为 $h_n:P_n\to Q_{n+1}$，使 $d_Qh_n=r_n$。归纳便给出
$f-g=d_Qh+hd_P$。对 $\operatorname{Hom}(-,A)$ 后，链同伦等价的复形映射诱导
相同同调映射，于是得到自然同构。证毕。

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

和 Ext 一样，定义与投射分解无关，因为投射分解之间的链映射唯一到链同伦；这里还用
到上述投射对象平坦输入，确保投射分解计算的确是导出张量。

**命题 3.5.1（平坦对象的 Tor 消失）。** 若 $M$ 是平坦 $R$-模，则

$$
\operatorname{Tor}_i^R(M,N)=0
\qquad(i>0)
$$

对任意 $N$ 成立。

**证明。** 先说明可在第二变量分解。取 $M,N$ 的投射分解 $P_\bullet,Q_\bullet$，考虑
第一象限双复形 $P_\bullet\otimes_RQ_\bullet$。按两个方向分别取同调：因投射模平坦，
一条谱序列先把 $Q_\bullet$ 压到 $N$，得到
$H_i(P_\bullet\otimes_RN)$；另一条先把 $P_\bullet$ 压到 $M$，得到
$H_i(M\otimes_RQ_\bullet)$。两者收敛到同一总复形同调，故 Tor 可由任一变量的投射
分解计算。

现在若 $M$ 平坦，函子 $M\otimes_R-$ 保持正合，所以分解
$Q_\bullet\to N$ 经张量后在正次数仍正合。于是
$H_i(M\otimes_RQ_\bullet)=0$ 对 $i>0$ 成立，由平衡性即得结论。证毕。

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

## 3.6 Worked example：$\underline{\mathbb Z/n}$

固定 $n\ne0$。单点 $*$ 极不连通，所以
$\underline{\mathbb Z}=\mathbb Z[\underline *]$ 在 $\mathbf{CondAb}$ 中投射。输入
投射分解

$$
0\to\underline{\mathbb Z}
\xrightarrow{\,n\,}\underline{\mathbb Z}
\to\underline{\mathbb Z/n}\to0.
$$

对 $A\in\mathbf{CondAb}$，表示公式
$\operatorname{Hom}(\underline{\mathbb Z},A)=A(*)$ 把 Hom 复形变成

$$
A(*)\xrightarrow{\,n\,}A(*).
$$

逐次取同调，输出

$$
\operatorname{Hom}(\underline{\mathbb Z/n},A)
=A(*)[n]:=\ker(n:A(*)\to A(*)),
$$

$$
\operatorname{Ext}^1(\underline{\mathbb Z/n},A)
=A(*)/nA(*),
\qquad
\operatorname{Ext}^i=0\quad(i\ge2).
$$

在 $\underline{\mathbb Z}$-模范畴中与 $N$ 张量，两个投射项分别输出 $N$，微分仍为
乘以 $n$。命题 3.5.2 因而给出凝聚阿贝尔群同构

$$
\operatorname{Tor}_1^{\underline{\mathbb Z}}
(\underline{\mathbb Z/n},N)
\cong N[n],
$$

其中 $N[n]:=\ker(n:N\to N)$。并且

$$
\underline{\mathbb Z/n}\otimes N
\cong N/nN,
\qquad
\operatorname{Tor}_i=0\quad(i\ge2).
$$

若 $A=N=\underline{\mathbb Z/m}$ 且 $g=\gcd(m,n)$，kernel 与 cokernel 都同构于
$\underline{\mathbb Z/g}$ 在相应类型中的版本；Ext 输出其全局截面
$\mathbb Z/g$，Tor 输出凝聚群 $\underline{\mathbb Z/g}$。当 $n=0$ 时第一支箭头不
单射，所写序列不再是投射分解，以上两项公式不能从它推出；这就是计算的明确失败条件。

## 3.7 分解所在范畴决定答案

极不连通对象上的取值可帮助构造投射生成元，但仅知道 $M(E)$ 并不会自动给出 Ext；
仍须在 $\mathbf{CondAb}$ 或 $\mathbf{CondMod}_R$ 中解析 kernel 和 relation。普通拓扑
向量空间的分解属于另一范畴，不能替换这里的投射分解。若对象已经进入 solid、analytic
或 liquid 子范畴，张量通常还要经过相应 localization；第四、五章将用普通无限乘积的
失败说明为何不能把本节的 $\otimes_R$ 原样搬过去。

## 练习

**练习 3.1.** 设 $S$ 为有限集合，证明 $\operatorname{Ext}^i(\mathbb Z[\underline S],A)=0$ 对 $i>0$ 成立。

**练习 3.2.** 对两项投射分解写出 $\operatorname{Ext}^1(M,A)$ 中一个元素对应的短正合扩张。

**练习 3.3.** 设 $M$ 有长度为 $n$ 的投射分解，证明 $\operatorname{Ext}^i(M,A)=0$ 对 $i>n$ 成立。
