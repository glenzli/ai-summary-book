# 序章：把定义变成可复核计算

前三卷建立的对象只有在具体输入上经过运算，读者才能看见各层结构的差异。有限覆盖的
sheaf 条件要落成等化子，$\operatorname{Ext}$ 和 $\operatorname{Tor}$ 要落成某个
Hom 或 tensor 复形的 kernel/cokernel，solid 张量要明确发生在 localization 之后，
liquid cohomology 则要检查连续满射是否局部可提升。本卷沿这些实际计算推进，而不是把
公式整理成操作清单。

一个计算至少要交代四件事：输入对象与所在范畴、逐步构造、输出的类型和数值、以及
哪项假设失败时结论不成立。序章先用 $\underline{\mathbb Z/n}$ 的两项分解展示这套
口径；后续各章再把同样的精度应用到站点、无限 profinite 测度、解析化、函数空间和
pro-étale 对照。

## 0.1 一个贯穿同调计算的两项分解

固定整数 $n\ne0$。单点空间 $*$ 极不连通，因此自由凝聚阿贝尔群

$$
\mathbb Z[\underline *]=\underline{\mathbb Z}
$$

是 $\mathbf{CondAb}$ 中的投射对象。乘以 $n$ 给出正合列

$$
0\longrightarrow\underline{\mathbb Z}
\xrightarrow{\,n\,}
\underline{\mathbb Z}
\longrightarrow
\underline{\mathbb Z/n}
\longrightarrow0.
$$

输入是凝聚范畴、非零整数 $n$ 与这个投射分解。对任意
$A\in\mathbf{CondAb}$ 施加 $\operatorname{Hom}(-,A)$，并使用

$$
\operatorname{Hom}(\underline{\mathbb Z},A)
\cong A(*),
$$

得到两项复形

$$
A(*)\xrightarrow{\,n\,}A(*).
$$

因此输出为

$$
\operatorname{Hom}(\underline{\mathbb Z/n},A)
\cong A(*)[n],
\qquad
\operatorname{Ext}^1(\underline{\mathbb Z/n},A)
\cong A(*)/nA(*),
$$

其中 $A(*)[n]:=\ker(n:A(*)\to A(*))$，且更高 Ext 为零。若改为与凝聚
$\underline{\mathbb Z}$-模 $N$ 张量，输出是

$$
\operatorname{Tor}_1^{\underline{\mathbb Z}}
(\underline{\mathbb Z/n},N)
\cong\ker(n:N\to N),
$$

$$
\underline{\mathbb Z/n}\otimes N
\cong\operatorname{coker}(n:N\to N).
$$

第三章会证明这些公式。这里已经能看见失败条件：$n=0$ 时开头的序列不再是分解；若
$\underline{\mathbb Z}$ 不在所选范畴中投射，Hom 复形也不能直接计算 derived Hom；
换到 solid 或 analytic 张量时，还必须先说明使用哪一个 localization。

## 0.2 站点计算提供输入对象

上述投射性最终来自 sheaf 计算。对有限联合满射覆盖
$\{S_i\to S\}$，匹配族组成等化子

$$
F(S)
\longrightarrow
\prod_iF(S_i)
\rightrightarrows
\prod_{i,j}F(S_i\times_SS_j).
$$

若 $S=E$ 极不连通，覆盖的联合满射有截面，局部提升便变成全局提升；这正是
$A\mapsto A(E)$ 正合、$\mathbb Z[\underline E]$ 投射的原因。第一、二章分别把这个
逻辑拆成可形式化规格和可手算等化子。

## 0.3 无限对象改变运算

有限集合 $S$ 上，普通自由对象与 solid 自由对象相同：

$$
\mathbb Z[\underline S]\cong\mathbb Z^\square[S].
$$

无限 profinite $S$ 上，右侧还含相容测度，Dirac 有限和不再穷尽全部元素。于是

$$
\mathbb Z^\square[S]
\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T]
$$

是 solid 结构定理，不是普通无限乘积张量的公式。第四章用 Cantor 空间的有限商逐层
展示这一差异；第五章再把 Dirac-to-measure 映射的 cofiber 用作 analytic
localization 的实际检测对象。

## 0.4 连续性决定输出是否仍可表示

Banach 或 Fréchet 空间 $V$ 的凝聚化

$$
\underline V(S)=\operatorname{Cont}(S,V)
$$

保留连续参数族。若连续线性映射的像不闭，拓扑 quotient 可能非 Hausdorff，复形
cohomology 便未必由另一个经典拓扑向量空间的凝聚化表示。第六章以
$\ell^1\hookrightarrow c_0$ 展示该失败，并以有连续截面的投影和 Hodge 分解展示可行
条件。第七章则比较另一种局部化技术：pro-étale 站点中的几何对象与凝聚站点的紧
Hausdorff 测试空间只共享证明模式，不共享对象类型。

## 0.5 从对象级运算到谱级问题

Ext 与 Tor 分别是 mapping complex 和 derived tensor 的同调。稳定化后，它们成为
mapping spectrum 与 tensor spectrum 的 homotopy groups；集合值 sheaf 等化子也升级
为 Čech totalization。第八章以
$H\underline{\mathbb Z/n}\otimes^L_{H\underline{\mathbb Z}}
H\underline{\mathbb Z/m}$ 为完整例子，说明对象级计算如何
进入凝聚谱与 pyknotic 开放问题。

## 练习

**练习 0.1.** 在 $A=\underline{\mathbb Z/m}$ 时计算
$A(*)[n]$ 和 $A(*)/nA(*)$，用 $\gcd(m,n)$ 表示答案。

**练习 0.2.** 指出序章的 Ext 与 Tor 计算分别在哪一步使用
$\underline{\mathbb Z}$ 的投射性和张量右正合性。

**练习 0.3.** 给出一个两项拓扑向量空间复形，分别写出其代数 cokernel 与拓扑
cokernel；说明闭像条件控制哪一项。
