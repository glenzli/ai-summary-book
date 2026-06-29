# 第二十七章：dg 范畴、稳定增强与导出 Morita 理论

## 本章目标

本章把稳定 $\infty$-范畴、富范畴、双模和紧生成理论连接到 dg 范畴。dg 范畴是链复形值富范畴；它们为三角范畴和稳定 $\infty$-范畴提供“增强”。导出 Morita 理论说明，对许多目的而言，dg 范畴的正确等价不是逐 Hom 复形的 quasi-isomorphism，而是导出模范畴的等价。

## 依赖前置知识

需要富范畴、闭幺半范畴、coend、模型范畴、稳定 $\infty$-范畴、compact generation、Morita $\infty$-范畴和 Bousfield localization。除非另说，本章固定一个交换环 $k$，并在链复形范畴 $\operatorname{Ch}(k)$ 中使用同调次数约定。

## 27.1 链复形富化与 dg 范畴

**定义 27.1.** 设 $\operatorname{Ch}(k)$ 为 $k$-模链复形的闭对称幺半范畴，张量积为复形张量积，内 Hom 记作 $\underline{\operatorname{Hom}}_k(-,-)$。一个 small dg category $\mathcal A$ 是一个 $\operatorname{Ch}(k)$-富范畴：它由对象集 $\operatorname{Ob}\mathcal A$、Hom 复形

$$
\mathcal A(x,y)\in\operatorname{Ch}(k)
$$

和复形态射

$$
\mathcal A(y,z)\otimes\mathcal A(x,y)\to\mathcal A(x,z),\qquad k\to\mathcal A(x,x)
$$

组成，满足富范畴的结合律和单位律。

**定义 27.2.** dg 函子 $F:\mathcal A\to\mathcal B$ 是 $\operatorname{Ch}(k)$-富函子，即对象函数 $x\mapsto Fx$ 和复形态射

$$
\mathcal A(x,y)\to\mathcal B(Fx,Fy)
$$

相容于复合和单位。

**定义 27.3.** dg category $\mathcal A$ 的同伦范畴 $H^0(\mathcal A)$ 具有同一对象，并定义

$$
H^0(\mathcal A)(x,y)=H^0(\mathcal A(x,y)).
$$

复合由 Hom 复形复合诱导。

**命题 27.4.** $H^0(\mathcal A)$ 是普通范畴，且 dg 函子 $F:\mathcal A\to\mathcal B$ 诱导普通函子 $H^0(F):H^0(\mathcal A)\to H^0(\mathcal B)$。

**证明.** Hom 复形复合是链映射，故把 cycle 送到 cycle，把 boundary 送到 boundary，因此诱导同调群上的复合

$$
H^0\mathcal A(y,z)\otimes H^0\mathcal A(x,y)\to H^0\mathcal A(x,z).
$$

富结合律和单位律在链复形层面成立，取 $H^0$ 后仍成立。dg 函子给出的 Hom 复形链映射与复合、单位相容，取 $H^0$ 得到普通函子。$\square$

**定义 27.5.** dg 函子 $F:\mathcal A\to\mathcal B$ 称为 quasi-equivalence，若：

1. 对任意 $x,y\in\mathcal A$，链映射
   $$
   \mathcal A(x,y)\to\mathcal B(Fx,Fy)
   $$
   是 quasi-isomorphism；
2. $H^0(F)$ 本质满。

**命题 27.6.** 若 $F$ 是 quasi-equivalence，则 $H^0(F)$ 是范畴等价。

**证明.** 第一条件给出每个 Hom 集映射

$$
H^0\mathcal A(x,y)\to H^0\mathcal B(Fx,Fy)
$$

为同构，所以 $H^0(F)$ 完全忠实。第二条件正是本质满。完全忠实且本质满的函子是范畴等价。$\square$

## 27.2 dg 模与导出范畴

**定义 27.7.** $\mathcal A$ 上的右 dg 模是 dg 函子

$$
M:\mathcal A^{op}\to\operatorname{Ch}(k).
$$

所有右 dg 模组成 dg 范畴 $\operatorname{Mod}_{\mathcal A}$，其中 Hom 复形由富自然变换的 end 给出：

$$
\underline{\operatorname{Hom}}_{\operatorname{Mod}_{\mathcal A}}(M,N)
=\int_{a\in\mathcal A}\underline{\operatorname{Hom}}_k(M(a),N(a)).
$$

**定义 27.8.** dg Yoneda 嵌入为

$$
h:\mathcal A\to\operatorname{Mod}_{\mathcal A},\qquad a\mapsto h_a=\mathcal A(-,a).
$$

**命题 27.9（dg Yoneda）.** 对任意 $a\in\mathcal A$ 和右 dg 模 $M$，有自然 quasi-isomorphism

$$
\underline{\operatorname{Hom}}_{\operatorname{Mod}_{\mathcal A}}(h_a,M)\simeq M(a).
$$

**证明.** 这是富 Yoneda 引理在富基 $\operatorname{Ch}(k)$ 中的特例。直接展开，左边是 end

$$
\int_x\underline{\operatorname{Hom}}_k(\mathcal A(x,a),M(x)).
$$

end 的元素是与 $\mathcal A$ 作用相容的一族链映射 $\mathcal A(x,a)\to M(x)$。由单位 $k\to\mathcal A(a,a)$，任意自然族给出 $k\to M(a)$，即 $M(a)$ 的元素；反过来，$m\in M(a)$ 沿右模作用

$$
M(a)\otimes\mathcal A(x,a)\to M(x)
$$

给出自然族。两构造互逆，并与微分相容。$\square$

**外部输入定理 27.10.** $\operatorname{Mod}_{\mathcal A}$ 上存在投射型模型结构，其弱等价为逐点 quasi-isomorphism。局部化得到稳定 presentable $\infty$-范畴

$$
D(\mathcal A).
$$

称为 $\mathcal A$ 的导出模 $\infty$-范畴。

**定义 27.11.** perfect $\mathcal A$-modules 组成 $D(\mathcal A)$ 的最小稳定、幂等完备全子范畴

$$
\operatorname{Perf}(\mathcal A)\subseteq D(\mathcal A)
$$

它包含所有可表模 $h_a$。

**命题 27.12.** 每个可表模 $h_a$ 在 $D(\mathcal A)$ 中 compact。

**证明.** 设 $\{M_i\}_{i\in I}$ 是滤过图形。由 dg Yoneda 和导出模范畴的逐点余极限，

$$
\operatorname{Map}_{D(\mathcal A)}(h_a,\operatorname{colim}_iM_i)
\simeq
(\operatorname{colim}_iM_i)(a)
\simeq
\operatorname{colim}_iM_i(a)
\simeq
\operatorname{colim}_i\operatorname{Map}_{D(\mathcal A)}(h_a,M_i).
$$

因此 $\operatorname{Map}(h_a,-)$ 保持滤过余极限，$h_a$ compact。$\square$

**外部输入定理 27.13.** 若 $\mathcal A$ small，则 $D(\mathcal A)$ 由可表模 $\{h_a\}_{a\in\mathcal A}$ compactly generated，并且

$$
D(\mathcal A)^\omega\simeq\operatorname{Perf}(\mathcal A).
$$

## 27.3 Pretriangulated dg 范畴与稳定增强

**定义 27.14.** dg category $\mathcal A$ 称为 pretriangulated，若在 $D(\mathcal A)$ 中，可表模的本质像对有限极限、有限余极限和悬挂闭合；等价地，$H^0(\mathcal A)$ 从 dg 模导出范畴继承三角结构。

**外部输入定理 27.15.** 若 $\mathcal A$ pretriangulated，则 dg nerve $N_{\operatorname{dg}}(\mathcal A)$ 是稳定 $\infty$-范畴，且

$$
hN_{\operatorname{dg}}(\mathcal A)\simeq H^0(\mathcal A)
$$

作为三角范畴。若 $\mathcal A$ 还幂等完备，则 $N_{\operatorname{dg}}(\mathcal A)$ 是 idempotent-complete stable $\infty$-category。

**定义 27.16.** 设 $T$ 为三角范畴。一个 dg enhancement 是 pretriangulated dg category $\mathcal A$ 连同三角等价

$$
H^0(\mathcal A)\simeq T.
$$

设 $C$ 为稳定 $\infty$-范畴。一个 dg enhancement 是 dg category $\mathcal A$ 连同稳定 $\infty$-范畴等价

$$
N_{\operatorname{dg}}(\mathcal A)\simeq C.
$$

**命题 27.17.** dg enhancement 比仅给出三角范畴包含更多信息。

**证明.** 三角范畴 $H^0(\mathcal A)$ 只记录 Hom 复形的 $0$ 次同调和由锥构造诱导的三角结构。dg category $\mathcal A$ 记录所有 Hom 复形 $\mathcal A(x,y)$，因而记录所有

$$
H^n\mathcal A(x,y)
$$

以及链级复合。稳定 $\infty$-范畴 $N_{\operatorname{dg}}(\mathcal A)$ 又保留映射空间或映射谱的高阶同伦信息。把这些数据传到 $H^0$ 会丢失高次同调和高阶相干，因此增强严格更丰富。$\square$

**例子 27.18.** 若 $R$ 是普通 $k$-代数，把 $R$ 看成一个单对象 dg category，且 Hom 复形集中在 $0$ 次，则 $D(R)$ 是通常的 $R$-模导出 $\infty$-范畴，$\operatorname{Perf}(R)$ 是 perfect complexes 的稳定 $\infty$-范畴。

## 27.4 Morita 等价与导出 Morita 理论

**定义 27.19.** dg 函子 $F:\mathcal A\to\mathcal B$ 诱导限制函子

$$
F^*:D(\mathcal B)\to D(\mathcal A).
$$

若 $F^*$ 是稳定 $\infty$-范畴等价，则称 $F$ 为 Morita equivalence。

**命题 27.20.** Quasi-equivalence 是 Morita equivalence。

**证明.** 设 $F:\mathcal A\to\mathcal B$ 是 quasi-equivalence。第一条件说明可表模的 Hom 复形在 $F$ 下保持 quasi-isomorphism；第二条件说明 $\mathcal B$ 的每个对象在 $H^0(\mathcal B)$ 中等价于某个 $Fa$。因此限制函子 $F^*$ 在 compact generators，即可表模生成的对象上给出等价。由外部输入定理 27.13，$D(\mathcal A)$ 与 $D(\mathcal B)$ 分别由可表模紧生成。保持小余极限的正合函子若在紧生成子上给出等价，则在整个紧生成稳定 presentable 范畴上给出等价，故 $F^*$ 为等价。$\square$

**例子 27.21.** Morita equivalence 不必是 quasi-equivalence。自然嵌入

$$
\mathcal A\to\operatorname{Perf}(\mathcal A)
$$

通常会加入有限锥、悬挂和 retract。它在导出模范畴上给出等价，但对象层面一般不是本质满的 quasi-equivalence。

**定义 27.22.** 一个 $\mathcal A$-$\mathcal B$ dg bimodule 是 dg 函子

$$
M:\mathcal A^{op}\otimes\mathcal B\to\operatorname{Ch}(k).
$$

若 $N$ 是 $\mathcal B$-$\mathcal C$ dg bimodule，则其导出复合为

$$
M\otimes^{\mathbb L}_{\mathcal B}N.
$$

可用 cofibrant replacement 或双边 bar 构造计算：

$$
\operatorname{Bar}_\bullet(M,\mathcal B,N).
$$

**命题 27.23.** 恒等 $\mathcal A$-$\mathcal A$ bimodule 是 Hom bimodule

$$
\mathcal A(-,-):\mathcal A^{op}\otimes\mathcal A\to\operatorname{Ch}(k).
$$

它对导出张量复合满足单位律。

**证明.** 对右 $\mathcal A$-模 $M$，导出张量 $M\otimes^{\mathbb L}_{\mathcal A}\mathcal A(-,a)$ 由 bar 构造

$$
\cdots\to\coprod_{x,y}M(y)\otimes\mathcal A(x,y)\otimes\mathcal A(x,a)\to\coprod_xM(x)\otimes\mathcal A(x,a)
$$

的几何实现给出。富 co-Yoneda 引理把该实现识别为 $M(a)$。对左单位同理。$\square$

**外部输入定理 27.24（导出 Morita 定理）.** dg categories 关于 Morita equivalences 的局部化可用 dg bimodules 描述：从 $\mathcal A$ 到 $\mathcal B$ 的导出映射对象由适当的右 quasi-representable $\mathcal A$-$\mathcal B$ bimodules 给出；复合由导出相对张量积给出。等价地，small dg categories、bimodules 和 bimodule maps 形成 Morita 型高阶范畴。

## 27.5 紧生成稳定范畴的代数模型

**定义 27.25.** 稳定 presentable $\infty$-范畴 $C$ 称为 $k$-linear，若它是 $D(k)$-模对象，且其张量作用

$$
D(k)\otimes C\to C
$$

分别保持小余极限。

**外部输入定理 27.26.** 若 $\mathcal A$ small，则 $D(\mathcal A)$ 是 compactly generated $k$-linear stable presentable $\infty$-category，compact objects 为 $\operatorname{Perf}(\mathcal A)$。反过来，在通常代数性假设下，compactly generated $k$-linear stable presentable $\infty$-categories 可由 small dg categories 的导出模范畴建模。

**命题 27.27.** 若 $C$ 是 compactly generated stable presentable $\infty$-category，则 $C\simeq\operatorname{Ind}(C^\omega)$，其中 $C^\omega$ 是 compact objects 的小稳定幂等完备子范畴。

**证明.** 由 compact generation，$C^\omega$ 的对象生成 $C$ 中的 localizing subcategory，即整个 $C$。Ind 完备化 $\operatorname{Ind}(C^\omega)$ 是由 $C^\omega$ 自由加入滤过余极限并再在 presentable 稳定语境中闭合得到的 compactly generated stable presentable $\infty$-category。Yoneda 型嵌入 $C^\omega\to C$ 延拓为保持小余极限的正合函子

$$
\operatorname{Ind}(C^\omega)\to C.
$$

该函子在 compact generators 上为恒等，因此完全忠实且本质满；由紧生成对象检测等价，得到等价。$\square$

**注 27.28.** 命题 27.27 是高阶 Morita 理论的分类口径：presentable 大范畴由其紧对象的小稳定范畴控制；若该小稳定范畴有 dg enhancement，则大范畴由相应 dg 模范畴控制。

## 27.6 Hochschild 型不变量与 Morita 不变性

**定义 27.29.** dg category $\mathcal A$ 的 Hochschild chains 可写为恒等 bimodule 的 trace：

$$
HH(\mathcal A)=\mathcal A\otimes^{\mathbb L}_{\mathcal A^{op}\otimes\mathcal A}\mathcal A.
$$

等价地，它由循环 bar 构造计算：

$$
\bigoplus_{a_0,\dots,a_n}
\mathcal A(a_n,a_0)\otimes\mathcal A(a_{n-1},a_n)\otimes\cdots\otimes\mathcal A(a_0,a_1)
$$

组成的单纯链复形给出。

**外部输入定理 27.30.** Hochschild homology、cyclic homology、topological Hochschild homology 的适当 dg 或谱版本是 Morita invariant：若 $\mathcal A\to\mathcal B$ 是 Morita equivalence，则相应不变量等价。

**命题 27.31.** 对普通 $k$-代数 $A$ 视为单对象 dg category，定义 27.29 恢复通常 Hochschild chains

$$
HH(A)\simeq A\otimes^{\mathbb L}_{A^{op}\otimes A}A.
$$

**证明.** 单对象 dg category 的对象和态射数据完全由 dg 代数 $A$ 给出。$\mathcal A^{op}\otimes\mathcal A$ 对应 enveloping dg algebra $A^{op}\otimes A$。恒等 bimodule 是 $A$ 作为 $(A,A)$-双模。代入定义 27.29 即得公式。$\square$

## 27.7 dg Yoneda 全忠实与单对象例子

**命题 27.32.** dg Yoneda 嵌入

$$
h:\mathcal A\to\operatorname{Mod}_{\mathcal A}
$$

在 Hom 复形上全忠实，即存在自然 quasi-isomorphism

$$
\mathcal A(a,b)\simeq
\underline{\operatorname{Hom}}_{\operatorname{Mod}_{\mathcal A}}(h_a,h_b).
$$

**证明.** 在命题 27.9 中取 $M=h_b$，得到

$$
\underline{\operatorname{Hom}}_{\operatorname{Mod}_{\mathcal A}}(h_a,h_b)
\simeq h_b(a)=\mathcal A(a,b).
$$

该等价对 $a,b$ 自然，并与复合相容，因为富 Yoneda 等价由模作用和单位给出。$\square$

**例子 27.33（单对象 dg 范畴）.** 单对象 dg category 等价于 dg algebra。唯一对象 $*$ 的 Hom 复形

$$
A=\mathcal A(*,*)
$$

带有由复合给出的乘法 $A\otimes A\to A$ 和单位 $k\to A$。反过来，任意 dg algebra $A$ 给出一个单对象 dg category。右 dg 模正是右 dg $A$-模。

**命题 27.34.** 若 dg category $\mathcal A$ 的 Hom 复形全都集中在 $0$ 次，且微分为零，则 $H^0(\mathcal A)$ 恢复底层普通 $k$-线性范畴。

**证明.** 此时

$$
H^0(\mathcal A(x,y))=\mathcal A(x,y)^0
$$

且没有非零边界需要取商。复合在 $H^0$ 上就是原来的 $k$-双线性复合，所以 $H^0(\mathcal A)$ 正是底层 $k$-线性范畴。$\square$

## 27.8 本章小结

dg 范畴是链复形富范畴；$H^0$ 给出普通同伦范畴，但会丢失链级和高阶信息。dg 模范畴 $D(\mathcal A)$ 是稳定 presentable $\infty$-范畴，可表模紧生成它，perfect modules 正是紧对象。Quasi-equivalence 强于 Morita equivalence；导出 Morita 理论表明 dg 双模与导出相对张量积才是组织 dg 范畴的自然广义态射。对于紧生成稳定 $k$-线性范畴，dg enhancement 把抽象稳定同伦论连接到显式代数模型。

## 练习

**练习 27.1.** 写出 small dg category 的定义。

**练习 27.2.** 证明 $H^0(\mathcal A)$ 的复合良定义。

**练习 27.3.** 定义 dg 函子和 quasi-equivalence。

**练习 27.4.** 证明 quasi-equivalence 诱导同伦范畴等价。

**练习 27.5.** 定义右 dg 模和可表模 $h_a$。

**练习 27.6.** 用富 Yoneda 证明 $\underline{\operatorname{Hom}}(h_a,M)\simeq M(a)$。

**练习 27.7.** 说明为什么可表模在 $D(\mathcal A)$ 中 compact。

**练习 27.8.** 定义 perfect module。

**练习 27.9.** 定义 pretriangulated dg category。

**练习 27.10.** 比较三角范畴的 dg enhancement 与稳定 $\infty$-范畴的 dg enhancement。

**练习 27.11.** 定义 Morita equivalence。

**练习 27.12.** 证明 quasi-equivalence 是 Morita equivalence。

**练习 27.13.** 写出 $\mathcal A$-$\mathcal B$ dg bimodule 的定义。

**练习 27.14.** 用富 co-Yoneda 解释恒等 bimodule 的单位律。

**练习 27.15.** 说明 $D(\mathcal A)^\omega\simeq\operatorname{Perf}(\mathcal A)$ 的意义。

**练习 27.16.** 对普通代数 $A$，写出 Hochschild chains 的导出张量公式。

**练习 27.17.** 用 dg Yoneda 证明 $h:\mathcal A\to\operatorname{Mod}_{\mathcal A}$ 在 Hom 复形上全忠实。

**练习 27.18.** 证明单对象 dg category 与 dg algebra 等价。

**练习 27.19.** 若 Hom 复形集中在 $0$ 次，说明 dg 函子就是普通 $k$-线性函子。
