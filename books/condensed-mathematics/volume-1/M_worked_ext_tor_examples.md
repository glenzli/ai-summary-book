# 附录 M：Ext 与 Tor 工作例题

## M.0 目标

附录 G 给出 Ext/Tor 的计算规则。本附录补充可手算的例题，训练读者区分：

1. 普通阿贝尔群中的分解。
2. 凝聚阿贝尔群中的投射分解。
3. 凝聚模范畴中的 Tor。
4. solid/analytic 局部化后的张量积。

本附录只做第一卷范围内的计算；不使用 Scholze 的深层 solid Ext 公式。

## M.1 有限离散空间的自由对象

设 $S$ 是有限离散空间，有 $n$ 个点。则

$$
\mathbb Z[\underline S]\cong\bigoplus_{s\in S}\mathbb Z[\underline *].
$$

**命题 M.1.** 对任意凝聚阿贝尔群 $A$，

$$
\operatorname{Hom}(\mathbb Z[\underline S],A)\cong A(*)^S,
$$

并且

$$
\operatorname{Ext}^i(\mathbb Z[\underline S],A)=0,\qquad i>0.
$$

**证明.** 有限离散空间是极不连通紧 Hausdorff 空间，因此 $\mathbb Z[\underline S]$ 投射。Hom 公式来自自由对象泛性质：

$$
\operatorname{Hom}(\mathbb Z[\underline S],A)\cong A(S).
$$

而 $S$ 有限离散，$A(S)\cong A(*)^S$。高阶 Ext 消失由投射性得到。证毕。

## M.2 一个两项分解的 Ext 计算模板

设 $M$ 有两项投射分解

$$
0\to P_1\xrightarrow dP_0\to M\to0.
$$

**命题 M.2.** 对任意 $A$，

$$
\operatorname{Ext}^1(M,A)
\cong
\operatorname{Hom}(P_1,A)/
d^\vee\operatorname{Hom}(P_0,A),
$$

其中 $d^\vee(f)=f\circ d$。

**证明.** 对分解应用 $\operatorname{Hom}(-,A)$，得到两项上链复形

$$
0\to\operatorname{Hom}(P_0,A)
\xrightarrow{d^\vee}
\operatorname{Hom}(P_1,A)\to0.
$$

第一同调就是所示余核。证毕。

**例 M.3（单关系对象）.** 若 $P_0=\mathbb Z[\underline E_0]$、$P_1=\mathbb Z[\underline E_1]$，其中 $E_0,E_1$ 极不连通，则

$$
\operatorname{Ext}^1(M,A)
\cong
A(E_1)/d^\vee A(E_0),
$$

其中 $d^\vee$ 是由 $d:\mathbb Z[\underline E_1]\to\mathbb Z[\underline E_0]$ 诱导的限制/线性组合映射。

**证明.** 用

$$
\operatorname{Hom}(\mathbb Z[\underline E_i],A)\cong A(E_i)
$$

代入命题 M.2。证毕。

## M.3 乘以 $n$ 的例子

令 $\mathbb Z_{\operatorname{cond}}=\mathbb Z[\underline *]$。对整数 $n\ne0$，考虑凝聚阿贝尔群

$$
\mathbb Z_{\operatorname{cond}}/n
$$

定义为乘法映射

$$
n:\mathbb Z_{\operatorname{cond}}\to\mathbb Z_{\operatorname{cond}}
$$

的 cokernel。

**命题 M.4.** 序列

$$
0\to\mathbb Z_{\operatorname{cond}}
\xrightarrow n
\mathbb Z_{\operatorname{cond}}
\to
\mathbb Z_{\operatorname{cond}}/n
\to0
$$

是 $\mathbf{CondAb}$ 中的投射分解。

**证明.** $\mathbb Z_{\operatorname{cond}}=\mathbb Z[\underline *]$ 投射，因为 $*$ 极不连通。乘以 $n$ 在每个测试对象上是整数值函数逐点乘以 $n$。若 $n f=0$，则 $f=0$，故单射。cokernel 按定义为 $\mathbb Z_{\operatorname{cond}}/n$。证毕。

**推论 M.5.** 对任意凝聚阿贝尔群 $A$，

$$
\operatorname{Ext}^1(\mathbb Z_{\operatorname{cond}}/n,A)
\cong
A(*)/nA(*),
$$

并且

$$
\operatorname{Ext}^i(\mathbb Z_{\operatorname{cond}}/n,A)=0,\qquad i\ge2.
$$

**证明.** 对命题 M.4 的两项分解应用命题 M.2。Hom 群

$$
\operatorname{Hom}(\mathbb Z_{\operatorname{cond}},A)\cong A(*),
$$

而 $d^\vee$ 为乘以 $n$。高阶 Ext 因分解长度为 $1$ 而消失。证毕。

**边界 M.6.** 这个计算只涉及 $\mathbb Z[\underline *]$ 和其 cokernel，不代表所有离散阿贝尔群凝聚化的 Ext 都退化为普通阿贝尔群 Ext。一般对象需要在 $\mathbf{CondAb}$ 中取投射分解。

## M.4 Tor 的两项计算

设 $R$ 是凝聚交换环，$M$ 有两项投射分解

$$
0\to P_1\xrightarrow dP_0\to M\to0.
$$

**命题 M.7.** 对任意凝聚 $R$-模 $N$，

$$
\operatorname{Tor}_1^R(M,N)
\cong
\ker(P_1\otimes_RN\to P_0\otimes_RN),
$$

$$
M\otimes_RN
\cong
\operatorname{coker}(P_1\otimes_RN\to P_0\otimes_RN),
$$

并且 $\operatorname{Tor}_i^R(M,N)=0$ 对 $i\ge2$。

**证明.** 张量分解后得到两项链复形

$$
P_1\otimes_RN\to P_0\otimes_RN.
$$

其第一同调为核，零次同调为余核。张量积右正合给出零次同调与 $M\otimes_RN$ 的同构。证毕。

**例 M.8（乘以 $n$ 的 Tor）.** 在 $R=\mathbb Z_{\operatorname{cond}}$ 上，

$$
\operatorname{Tor}_1^{\mathbb Z_{\operatorname{cond}}}
(\mathbb Z_{\operatorname{cond}}/n,N)
\cong
\ker(n:N\to N),
$$

并且

$$
(\mathbb Z_{\operatorname{cond}}/n)\otimes N
\cong
N/nN.
$$

**证明.** 使用命题 M.4 的分解并张量 $N$。所得复形为

$$
N\xrightarrow nN.
$$

第一同调是 $n$-torsion，零次同调是 $N/nN$。证毕。

## M.5 与 solid/analytic 的区别

**警告 M.9.** 例 M.8 是普通凝聚张量积中的计算。若进入 solid 或 analytic 范畴，张量积一般变为

$$
L(M\otimes^L N),
$$

其中 $L$ 是 solidification 或 analyticization。局部化可能改变普通 Tor 复形的同调对象。

**检查规则 M.10.** 每次使用 Tor 计算时，应写明：

1. 所在范畴是 $\mathbf{CondMod}_R$、solid $R$-模，还是 analytic $R$-模。
2. 张量积是普通派生张量，还是局部化后的 solid/analytic 张量。
3. 投射或平坦分解是否存在于该范畴中。

## M.6 练习

**练习 M.1.** 对有限二点集 $S$，写出 $\operatorname{Hom}(\mathbb Z[\underline S],A)$。

**练习 M.2.** 用命题 M.4 计算 $\operatorname{Ext}^1(\mathbb Z_{\operatorname{cond}}/2,A)$。

**练习 M.3.** 若 $N$ 没有 $n$-torsion，计算 $\operatorname{Tor}_1(\mathbb Z_{\operatorname{cond}}/n,N)$。

**练习 M.4.** 说明为什么例 M.8 不能直接替代 solid tensor product 的计算。
