# 第十六章：Trace Formula 与 Endoscopy

## 本章目标

本章解释 trace formula 和 endoscopy 在 Langlands 纲领中的作用。第十五章把函子性表述为 L 群同态诱导的自守表示转移；本章说明为什么证明这种转移通常需要比较不同群的 trace formula。Trace formula 的基本形态是：同一个测试函数作用在自守谱上的 trace，可以按表示论谱侧展开，也可以按共轭类几何侧展开。Endoscopy 的作用是稳定化几何侧，并把一个群的稳定谱与另一个群的端oscopic 数据联系起来。

## 依赖前置知识

需要第四章的 Hecke 代数，第十三章的自守表示和自守谱，第十五章的函子性。需要知道局部紧群上的测试函数、orbital integral、parabolic subgroup、Levi subgroup 和稳定共轭的基本概念。本章把 Arthur-Selberg trace formula、稳定 trace formula、transfer factors、fundamental lemma 和端oscopic classification 作为外部输入。

## 16.1 测试函数与卷积算子

设 $K$ 为整体域，$G/K$ 为 connected reductive group。令
$$
[G]=G(K)\backslash G(\mathbb A_K).
$$

**定义 16.1.** 设 $f\in C_c^\infty(G(\mathbb A_K))$。右卷积算子 $R(f)$ 在自守函数 $\phi$ 上定义为
$$
(R(f)\phi)(x)=\int_{G(\mathbb A_K)}f(g)\phi(xg)\,dg.
$$
这里 $dg$ 是 $G(\mathbb A_K)$ 上固定的 Haar 测度。

**命题 16.2.** 若 $\phi$ 右 $K_f$-有限且 $f$ 紧支撑光滑，则 $R(f)\phi$ 仍为右有限的自守函数。

**证明.** 左 $G(K)$-不变性由
$$
(R(f)\phi)(\gamma x)=\int f(g)\phi(\gamma xg)\,dg=\int f(g)\phi(xg)\,dg
$$
得到。右有限性来自 $f$ 在有限 adele 方向局部常值且紧支撑，因此存在开紧子群 $J$ 使 $f(j_1gj_2)=f(g)$；若 $\phi$ 对某个开紧子群不变，则 $R(f)\phi$ 对交子群不变。Archimedean 光滑性由卷积保持。$\square$

**定义 16.3.** 若 $R(f)$ 在某个离散自守空间上为 trace class，则记其 trace 为
$$
\operatorname{tr}R(f).
$$
Trace formula 的任务是用两种方式计算该 trace。

## 16.2 紧商情形的 Selberg trace formula

先考虑 $G(K)\backslash G(\mathbb A_K)$ modulo center 紧的情形。此时没有连续谱，公式形式最清楚。

**定义 16.4.** 对 $\gamma\in G(K)$，其 orbital integral 定义为
$$
O_\gamma(f)=\int_{G_\gamma(\mathbb A_K)\backslash G(\mathbb A_K)}
f(x^{-1}\gamma x)\,dx,
$$
其中 $G_\gamma$ 为 $\gamma$ 的 centralizer。测度由 $G_\gamma(\mathbb A_K)$ 和 $G(\mathbb A_K)$ 的 Haar 测度诱导。

**外部输入定理 16.5（紧商 trace formula，接口形式）.** 若 $[G]$ modulo center 紧，且 $f$ 满足合适正则性条件，则
$$
\sum_{\pi}m(\pi)\operatorname{tr}\pi(f)
=
\sum_{\{\gamma\}}a(\gamma)O_\gamma(f).
$$
左侧按离散自守表示 $\pi$ 求和，$m(\pi)$ 为 multiplicity；右侧按 $G(K)$ 的共轭类求和，$a(\gamma)$ 为由体积和中心化子给出的权重。

**注 16.6.** 紧商公式已经显示 trace formula 的本质：谱侧是表示，几何侧是共轭类和 orbital integrals。非紧商情形需要截断、连续谱、Eisenstein series 和加权 orbital integrals。

## 16.3 Arthur-Selberg trace formula 的一般形态

一般 reductive group 的自守商非紧，$L^2([G])$ 含有连续谱。

**外部输入定理 16.7（Arthur trace formula，粗略接口）.** 对适当测试函数 $f$，有恒等式
$$
J_{\operatorname{spec}}^G(f)=J_{\operatorname{geom}}^G(f),
$$
其中：

1. 谱侧 $J_{\operatorname{spec}}^G(f)$ 由离散谱、连续谱、Eisenstein series、intertwining operators 和 Levi subgroups 的诱导数据组成。
2. 几何侧 $J_{\operatorname{geom}}^G(f)$ 由 semisimple conjugacy classes、unipotent contributions 和 weighted orbital integrals 组成。
3. 两侧都依赖截断参数，但完整公式中依赖被组织为规范分布。

**注 16.8.** 定理 16.7 是接口陈述。完整 Arthur trace formula 的陈述需要大量关于截断、加权 characters、加权 orbital integrals 和 $(G,M)$-families 的技术。本书只使用它作为比较不同群自守谱的工具。

**定义 16.9.** Trace formula 的稳定化是把
$$
J_{\operatorname{geom}}^G(f)
$$
分解为稳定 orbital integrals，并把非稳定部分解释为 endoscopic groups 的贡献。相应谱侧被重写为 stable distributions 和 endoscopic transfer 的组合。

## 16.4 稳定共轭与稳定轨道积分

设 $F$ 为局部域，$G/F$ 为 connected reductive group。

**定义 16.10.** 两个 semisimple 元素 $\gamma,\gamma'\in G(F)$ 称为稳定共轭，若它们在 $G(\overline F)$ 中共轭。

**定义 16.11.** 设 $\gamma$ 为 strongly regular semisimple element。稳定轨道积分形式上定义为
$$
SO_\gamma(f)=\sum_{\gamma'}e(G_{\gamma'})O_{\gamma'}(f),
$$
其中 $\gamma'$ 遍历 $\gamma$ 的稳定共轭类中的 $G(F)$-共轭类，$e(G_{\gamma'})$ 为适当符号或 Kottwitz sign。

**注 16.12.** 稳定轨道积分不是单个轨道积分。它把同一稳定共轭类中的多个 rational conjugacy classes 组合起来。Endoscopy 正是研究这些组合如何从另一个群的稳定轨道积分转移而来。

**命题 16.13.** 若 $G=\operatorname{GL}_n$，regular semisimple 元素的稳定共轭与普通 $G(F)$-共轭一致。

**证明.** Regular semisimple 元素由其特征多项式和相应的 étale $F$-代数作用决定。对 $\operatorname{GL}_n$，具有相同特征多项式并在 $\overline F$ 中共轭的 regular semisimple 元素给出同构的 $F[t]$-模结构，因此在 $\operatorname{GL}_n(F)$ 中共轭。$\square$

## 16.5 Endoscopic Data

Endoscopy 的输入不是任意群同态，而是与 $\widehat G$ 中 semisimple 元素相关的对偶群数据。

**定义 16.14.** 一个 elliptic endoscopic datum 的接口形式是四元组
$$
(H,\mathcal H,s,\eta),
$$
其中：

1. $H$ 是 $K$ 上的 quasi-split reductive group。
2. $\mathcal H$ 是 ${}^LG$ 的子 L 群或扩张数据。
3. $s\in\widehat G$ 是 semisimple element，其 centralizer 的恒等分量与 $\widehat H$ 相关。
4. $\eta:{}^LH\to{}^LG$ 是 L embedding，满足与 Galois 作用相容的条件。

**注 16.15.** 完整定义包含 $z$-extensions、outer automorphisms、Kottwitz-Shelstad cocycles 和 equivalence relation。本书在第十五章只需知道 endoscopy 给出某类特殊 L 同态；本章进一步说明它要求轨道积分转移。

## 16.6 Transfer Factors 与匹配函数

设 $H$ 是 $G$ 的 endoscopic group。要比较 $H$ 与 $G$ 的 trace formula，必须比较它们的 orbital integrals。

**定义 16.16.** Transfer factor 是函数
$$
\Delta(\gamma_H,\gamma_G)
$$
定义在相互匹配的 strongly regular semisimple 元素对上，用于修正 orbital integrals，使 $G$ 上的加权组合与 $H$ 上的稳定轨道积分可比较。

**定义 16.17.** 局部测试函数 $f^G\in C_c^\infty(G(F))$ 与 $f^H\in C_c^\infty(H(F))$ 称为匹配，若对所有 strongly regular semisimple $\gamma_H\in H(F)$，
$$
SO_{\gamma_H}(f^H)
=
\sum_{\gamma_G}\Delta(\gamma_H,\gamma_G)O_{\gamma_G}(f^G),
$$
其中 $\gamma_G$ 遍历与 $\gamma_H$ 匹配的 $G(F)$-共轭类。

**外部输入定理 16.18（transfer 的存在，接口形式）.** 在合适的函数空间和归一化下，对 $G$ 上的测试函数 $f^G$ 存在 $H$ 上的匹配函数 $f^H$，反之在稳定分布意义下也有相应转移。

**注 16.19.** Transfer factor 的归一化是 endoscopy 中最精细的部分之一。不同归一化会改变局部字符恒等式中的符号和 packet 参数化。

## 16.7 Fundamental Lemma

**外部输入定理 16.20（Fundamental lemma，Ngô）.** 对非 Archimedean 局部域上的非分歧 endoscopic datum，单位元测试函数
$$
\mathbf 1_{G(\mathcal O_F)}
$$
与相应的
$$
\mathbf 1_{H(\mathcal O_F)}
$$
在合适 transfer factor 归一化下匹配。更一般的 weighted fundamental lemma 也在稳定 trace formula 中使用。

**注 16.21.** Fundamental lemma 是稳定 trace formula 可用的关键局部输入。没有它，无法在几乎所有非分歧位置把几何侧的 Euler product 型比较拼接成全局恒等式。

## 16.8 稳定 Trace Formula 与谱转移

**外部输入定理 16.22（稳定 trace formula，接口形式）.** 对适当测试函数，Arthur trace formula 可稳定化为
$$
I^G(f)
=
S^G(f)
+
\sum_{H} \iota(G,H)S^H(f^H),
$$
其中：

1. $S^G$ 是 $G$ 的稳定分布。
2. $H$ 遍历 proper endoscopic data。
3. $f^H$ 是 $f$ 的 transfer。
4. $\iota(G,H)$ 是由 Tamagawa 数、中心和 component groups 给出的系数。

**注 16.23.** 公式的精确形式依赖是否使用 invariant trace formula、stable trace formula、twisted trace formula 或 simple trace formula。本书当前只使用它解释 endoscopic classification 的结构。

**命题 16.24（trace formula 比较的函子性含义）.** 若 $H$ 与 $G$ 的稳定 trace formula 可通过 transfer factors 匹配，并且谱侧稳定分布可分解为 L-packets 的 stable characters，则可从几何侧恒等式推出 $H$ 与 $G$ 之间的自守表示转移。

**证明草图.** 几何侧匹配给出对所有匹配测试函数的分布恒等式。由 trace formula，几何侧等于谱侧，因此得到稳定谱分布恒等式。若 stable characters 在线性无关性意义下可分离不同 packets，则分布恒等式迫使谱侧 packets 按 L embedding 对应。该过程给出 endoscopic transfer 或其 multiplicity formula。完整证明依赖稳定 trace formula、局部字符恒等式和 packet 参数化。$\square$

## 16.9 Twisted Endoscopy 与 Base Change

Base change 和 automorphic induction 常通过 twisted trace formula 证明。

**定义 16.25.** 若 $\theta$ 是 $G$ 的 automorphism，twisted orbital integral 形式上为
$$
O_{\gamma,\theta}(f)=
\int_{G_{\gamma,\theta}(F)\backslash G(F)}
f(x^{-1}\gamma\theta(x))\,dx.
$$
其中 $G_{\gamma,\theta}$ 是 twisted centralizer。

**外部输入定理 16.26（twisted trace formula 的 base change 接口）.** 对 cyclic base change，比较 $\operatorname{GL}_n$ 的 twisted trace formula 与目标群的 ordinary trace formula，可证明 base change lift 的存在，并与非分歧位置的 Weil 群限制相容。

**注 16.27.** Arthur-Clozel 的 solvable base change 和 automorphic induction 使用了 twisted trace formula。第十五章把它们作为函子性的例子；本章说明其技术来源。

## 16.10 本章小结

Trace formula 把自守表示问题转化为测试函数分布恒等式。谱侧记录自守表示和 Eisenstein series，几何侧记录共轭类、轨道积分和加权轨道积分。Endoscopy 通过稳定共轭、transfer factors、matching functions 和 fundamental lemma 把不同群的 trace formula 连接起来。稳定 trace formula 是 Arthur 分类、classical groups 到 `GL(N)` 的转移、base change 和许多函子性结果的核心工具。

## 练习

**练习 16.1.** 在紧商情形下，解释 trace formula 中谱侧和几何侧各自的对象。

**练习 16.2.** 对 $G=\operatorname{GL}_n$，说明 regular semisimple 稳定共轭为何等于普通共轭。

**练习 16.3.** 写出匹配函数定义中 transfer factor 的位置，并说明没有 transfer factor 时公式为什么不具有不变量意义。

**练习 16.4.** 解释 fundamental lemma 在几乎所有非分歧位置上的作用。

**练习 16.5.** 说明稳定 trace formula 如何从几何侧恒等式导出谱侧转移。

**练习 16.6.** 比较 ordinary orbital integral 与 twisted orbital integral。
