# 第八章：凝聚谱中的六种运算与开放问题

Ext 与 Tor 分别记录映射的高阶障碍和张量的高阶核；进入稳定范畴后，它们不再是两套
分离的记号，而成为 mapping spectrum 的负同伦群与 derived tensor spectrum 的正同伦
群。与此同时，集合值 sheaf 的二重交等化子必须升级为谱值 totalization，solid 与
analytic 的 Dirac cone 也要升级为稳定 Bousfield localization。这样一来，前三章的
有限分解计算可以直接提供凝聚谱的低阶同伦群，而不能由这些计算推出的 monoidal 与
六函子相容性则有清楚边界。

本章先固定**超完备**谱值 sheaf 的约定，并说明普通 sheaf 条件为什么只在截断层面化为
等化子；随后完整计算 $H\underline{\mathbb Z}/n$ 与
$H\underline{\mathbb Z}/m$ 的 mapping 和 tensor spectra。最后把
$f^*,f_*,f_!,f^!,\otimes,\underline{\operatorname{Hom}}$ 排进同一个条件性框架，列出
从凝聚谱走向 spectral analytic geometry 时真正需要解决的数学问题，而不把外部高阶
定理写成书内已经得到的结论。

## 8.1 从等化子到 hyperdescent

设 $\mathcal C$ 是固定 universe 中的 compact Hausdorff 站点，
$\operatorname{Sp}$ 为谱的稳定 $\infty$-范畴。

**定义 8.1.1（本卷约定的超完备凝聚谱）。** 本卷称满足 hyperdescent 的函子

$$
E:\mathcal C^{op}\longrightarrow\operatorname{Sp}.
$$

为超完备凝聚谱，并把它们组成的范畴记为
$\operatorname{CondSp}^{\wedge}$。有些文献把只满足 Čech descent 的谱值 sheaf 也称
为 condensed spectrum，再另取 hypercompletion；两种约定不应混用。以下依靠
Postnikov 完备性或任意 hypercover 的陈述均发生在
$\operatorname{CondSp}^{\wedge}$ 中。

具体地，对每个 hypercover $U_\bullet\to U$，自然映射必须是等价

$$
E(U)\xrightarrow{\sim}
\operatorname{Tot}_{[r]\in\Delta}E(U_r).
$$

若只考察一个覆盖 $V\to U$，右侧是其 Čech nerve
$V^{\times_U(r+1)}$ 上的 totalization。谱中的匹配不仅要求两个限制相等，还包括同伦、
同伦之间的相容同伦以及全部更高 coherence，因此不能截断为前两项 equalizer。

**命题 8.1.2（零截断恢复普通 sheaf 条件）。** 若 $F$ 是集合值预 sheaf，把所有值看
成离散空间，则覆盖下降等价于通常的等化子条件

$$
F(U)\longrightarrow F(V)
\rightrightarrows F(V\times_UV).
$$

**证明。** 离散空间中两点之间若有同伦，则两点相等，且相等证明没有额外高阶选择。
因此 totalization 的一个点恰由 $x\in F(V)$ 给出，条件是它沿两个投影到
$F(V\times_UV)$ 的限制相同。sheaf 条件把这样的匹配截面唯一粘合为 $F(U)$ 的截面，
反之，全局截面的两个限制都来自同一截面的拉回，函子性使它们相等，因而给出匹配族。
证毕。

对谱值对象，命题证明中的“没有额外高阶选择”不再成立。若只检查
$\pi_0E$ 的 equalizer，就可能漏掉 $\pi_{-1},\pi_{-2},\ldots$ 中的下降障碍。

## 8.2 Eilenberg--Mac Lane 对象与导出截面

凝聚阿贝尔群 $A$ 给稳定 sheaf 范畴 heart 中的 Eilenberg--Mac Lane 对象 $HA$，其
homotopy sheaves 满足

$$
\pi_0(HA)=A,
\qquad
\pi_i(HA)=0\quad(i\ne0).
$$

这里必须区分 sheaf 的 homotopy sheaf 与某个对象上的导出截面。一般而言
$R\Gamma(U,HA)$ 不必等于离散谱 $H(A(U))$。按 sheaf cohomology 的导出定义，有
自然同构

$$
\pi_{-s}R\Gamma(U,HA)
\cong H^s(U,A),
\qquad s\ge0.
$$

只有当 $U$ 对 $A$ 无高阶同调时，右侧才集中在次数零。第七章的分裂 Čech 收缩解释了
一种 Čech 障碍消失机制，但从该机制到全部导出截面仍需要 Čech-to-derived 比较输入。

在导出凝聚阿贝尔群范畴中，mapping spectrum 把 Ext 统一为

$$
\pi_{-r}\operatorname{Map}(HA,HB)
\cong
\operatorname{Ext}^r_{\mathbf{CondAb}}(A,B),
\qquad r\ge0.
$$

若 $A,B$ 是 $\underline{\mathbb Z}$-模，导出张量则满足

$$
\pi_r(HA\otimes^L_{H\underline{\mathbb Z}}HB)
\cong
\operatorname{Tor}^{\underline{\mathbb Z}}_r(A,B),
\qquad r\ge0.
$$

这两式只是稳定范畴对导出 Hom 与导出张量的重新编码；Ext/Tor 的具体值仍须由第三章
的投射分解计算。这里不带下划线的 $\operatorname{Map}$ 是全局 mapping spectrum，
其 homotopy groups 是普通阿贝尔群；内部 mapping 对象记为
$\underline{\operatorname{Map}}$，其 homotopy sheaves 才是凝聚阿贝尔群。

## 8.3 Worked example：两个循环凝聚谱

取整数 $n,m\ge1$，并写

$$
A=\underline{\mathbb Z}/n,
\qquad
B=\underline{\mathbb Z}/m,
\qquad
g=\gcd(n,m).
$$

$A$ 有长度一自由分解

$$
0\longrightarrow\underline{\mathbb Z}
\xrightarrow{n}
\underline{\mathbb Z}
\longrightarrow A\longrightarrow0.
$$

### Mapping spectrum

对分解施加 $\underline{\operatorname{Hom}}(-,B)$，得到两项复形

$$
B\xrightarrow{n}B.
$$

乘 $n$ 在 $\mathbb Z/m$ 上的核与 cokernel 都是阶数 $g$ 的循环群：核由
$m/g$ 的倍数生成，像的阶数为 $m/g$，所以 cokernel 的阶数也为 $g$。因此

$$
\operatorname{Hom}_{\mathbf{CondAb}}(A,B)\cong\mathbb Z/g,
\qquad
\operatorname{Ext}^1_{\mathbf{CondAb}}(A,B)\cong\mathbb Z/g,
$$

且 $\operatorname{Ext}^r_{\mathbf{CondAb}}(A,B)=0$ 对 $r\ge2$。输出的全局 mapping
spectrum 于是满足

$$
\pi_0\operatorname{Map}(HA,HB)
\cong\mathbb Z/g,
\qquad
\pi_{-1}\operatorname{Map}(HA,HB)
\cong\mathbb Z/g,
$$

其余 homotopy groups 为零。若改取内部对象
$\underline{\operatorname{Map}}(HA,HB)$，同一个两项内部 Hom 复形说明其第
$0,-1$ 个 homotopy sheaves 都是 $\underline{\mathbb Z}/g$。

### Tensor spectrum

把同一分解与 $B$ 张量，仍得到两项复形

$$
B\xrightarrow{n}B,
$$

但现在左项位于同调次数 $1$。因此

$$
\operatorname{Tor}_0(A,B)\cong\underline{\mathbb Z}/g,
\qquad
\operatorname{Tor}_1(A,B)\cong\underline{\mathbb Z}/g,
$$

且更高 Tor 消失。等价地，余纤维列

$$
H B\xrightarrow{n}H B
\longrightarrow
HA\otimes^L_{H\underline{\mathbb Z}}HB
$$

的长正合 homotopy 列给

$$
\pi_0(HA\otimes^L HB)\cong\underline{\mathbb Z}/g,
\qquad
\pi_1(HA\otimes^L HB)\cong\underline{\mathbb Z}/g,
$$

其余为零。

这项计算的输入是非零整数 $n,m$ 和长度一自由分解；步骤是分别取 Hom 与 tensor 后求
乘 $n$ 的核、cokernel；输出是全局 mapping 的 $0,-1$ 次 homotopy groups、内部
mapping 的相应 homotopy sheaves，以及 tensor 的 $0,1$ 次 homotopy sheaves。若用
普通张量，只会保留 $\pi_0$ 而漏掉 $\pi_1$；若令 $n=0$，上面的两项列
不再是 $\mathbb Z/0$ 的分解，计算从输入处即失效。

## 8.4 Dirac cone 的稳定化

对 analytic ring $(A,\mathcal M)$，第五章定义

$$
K_S^{\mathcal M}
=\operatorname{cofib}\bigl(A[\underline S]\to\mathcal M[S]\bigr).
$$

在稳定范畴中，对象 $E$ 为 $K_S^{\mathcal M}$-local 的条件可以直接写成 mapping
spectrum 收缩：

$$
\underline{\operatorname{Map}}_A(K_S^{\mathcal M},E)\simeq0.
$$

**命题 8.4.1（local 对象对 Dirac 映射不变）。** 若上式对所有测试 $S$ 成立，则

$$
\underline{\operatorname{Map}}_A(\mathcal M[S],E)
\xrightarrow{\sim}
\underline{\operatorname{Map}}_A(A[\underline S],E).
$$

**证明。** 对定义 $K_S^{\mathcal M}$ 的余纤维列施加反变内部 mapping spectrum，
得到纤维列。第一项为零对象时，另外两项之间的映射为等价。证毕。

由所有这些 cone 生成的反射 localization 是否存在、是否 accessible、其 kernel 是否
为 tensor ideal，以及 localization 是否与 hyperdescent 相容，采用 solid/analytic
稳定化的外部输入。命题 8.4.1 是该输入一旦成立后的完整形式后果；它不需要再调用
solid 结构定理。

## 8.5 六种运算进入稳定局部化

设 $f:\mathcal X\to\mathcal Y$ 是由站点或几何对象诱导的 $\infty$-topos 态射。稳定化
后首先有伴随

$$
f^*: \operatorname{Sp}(\mathcal Y)
\rightleftarrows
\operatorname{Sp}(\mathcal X):f_*.
$$

在适当紧支撑与对偶性假设下，所寻求的六函子形式为

$$
f^*\dashv f_* ,
\qquad
f_!\dashv f^!,
\qquad
-\otimes E\dashv\underline{\operatorname{Hom}}(E,-).
$$

$f^*$ 与 $f_*$ 来自几何 topos 态射；$f_!$、$f^!$ 的存在以及 proper base change、
projection formula 不是任意站点映射的形式后果，必须作为具体几何理论的输入。

现在再施加 solid 或 analytic localization $L$，并记 local 子范畴的全忠实包含为
$i$。复合 $L\circ T\circ i$ 对任何环境函子 $T$ 都是 local 子范畴之间的函子；真正
需要检验的是 $T$ 是否**下降穿过局部化**，即是否把 local equivalence 送到 local
equivalence，从而存在满足 $\overline T\circ L\simeq L\circ T$ 的诱导函子
$\overline T$。在适用的伴随条件下，这等价于其伴随保持 local 对象。例如要让 $f^*$
具有这种下降性质，可检查每个生成 cone 的像 $f^*K_S^{\mathcal M}$ 在目标 localization
中为零。检查完成后可无歧义地写

$$
f^*_{\mathrm{an}}(LE):=L(f^*E).
$$

若检查失败，$L\circ T\circ i$ 仍然存在，但它不一定由环境 localization 的商泛性质
诱导，也不一定保留原六函子的伴随、复合或 base-change 结构。这给六函子构造一个
可操作的失败条件，而不只是“需要更多高阶理论”的笼统说明。

## 8.6 数学上的开放问题

下面各问都由本章已经出现的明确对象提出。

**问题 8.6.1（hyperdescent 检测）。** 在选定的 compact Hausdorff 或 extremally
disconnected 基上，哪些有界性或 hypercompleteness 条件允许只用一类显式 Čech nerve
检测谱值 hyperdescent？第七章的分裂收缩控制单个覆盖，但尚不自动控制任意 hypercover。

**问题 8.6.2（紧生成与对偶对象）。** Dirac cone 稳定化后，哪些
$A[\underline S]$ 或 $\mathcal M[S]$ 保持 compact？它们是否生成 local 范畴；其中哪些
对象 dualizable？这些性质决定 mapping spectrum 和 tensor 是否能由有限层计算检测。

**问题 8.6.3（monoidal analytic localization）。** 对给定 spectral theory of
measures，怎样验证所有 $K_S^{\mathcal M}$ 生成 tensor ideal？第五章的普通换底反例
说明，无限乘积相容性必须进入证明，而不能由有限集合上的 cone 消失推出。

**问题 8.6.4（六函子相容）。** 对 analytic、pro-etale 或 pyknotic 几何中的映射
$f$，哪些几何条件保证 $f^*,f_*,f_!,f^!$ 保持相应 local 对象，并满足 base change 与
projection formula？8.5 节把这一问化为生成 cone 的像与伴随保持性检查。

**已知定理与问题 8.6.5（pro-étale--pyknotic 比较）。** Wolf 已证明：若 $X$ 是
coherent scheme，则 $X$ 的 hypercomplete pro-étale $\infty$-topos 等价于其 Galois
category $\operatorname{Gal}(X)$ 在 pyknotic spaces 中的连续表示范畴。因此“是否存在
任何 pro-étale--pyknotic 接口”不是开放问题。仍需分别研究的是：该等价的谱值稳定化
怎样与本卷的 compact-Hausdorff 站点约定比较，solid/analytic localization 是否被
保持，以及去掉 coherent 或 hypercomplete 假设后哪些结论仍成立。

每一问都化为关于特定 localization、生成元和伴随函子的可判定命题。附录 E 给
pyknotic 背景，附录 G 保留谱值 sheaf 与稳定 localization 的参考接口；本章给出进入
这些问题前可直接复核的 Ext/Tor 与 cone 计算。

## 8.7 从低阶计算到稳定几何

循环群例子表明，第三章的一条长度一分解已经同时决定 mapping spectrum 和 tensor
spectrum 的全部 homotopy sheaves；第五章的 cone 判别在稳定化后也只需一条纤维列
就能证明 local 对象不区分 Dirac 与测度。由此得到的可靠边界是：稳定范畴中的形式
后果可以完整证明，localization 的存在、monoidal 性以及六函子几何仍须分别输入并
检查。四卷主线因此收束在一个具体接口上：从 sheaf 匹配族出发，经 Ext/Tor、solid
与 analytic cone，最终抵达可计算但仍有明确开放条件的凝聚谱。

## 练习

**练习 8.1.** 对一个集合值 sheaf 完成命题 8.1.2 的双向构造，并指出唯一性在哪一步
使用。

**练习 8.2.** 取 $n=6,m=15$，逐元素计算乘 $6$ 在 $\mathbb Z/15$ 上的核与 cokernel，
再写出相应 mapping 与 tensor spectrum 的非零 homotopy groups。

**练习 8.3.** 由余纤维列直接推导 8.3 节 tensor spectrum 的长正合列，不使用 Tor
记号。

**练习 8.4.** 证明若左伴随 $T$ 把一组生成 local equivalence 送到 local
equivalence，则 $LTL$ 在 local 对象上的值与未局部化代表无关。

**练习 8.5.** 为第七章有限对象 $U_I$ 写出谱值 Čech totalization 前三层，并说明当
覆盖由不交分支组成时为何它仍收缩到有限乘积。
