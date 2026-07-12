# 第二十一章：谱侧、局部系统和范畴化对应

## 本章目标

本章把第二十章的 Hecke eigensheaf 形式升级为范畴化的几何 Langlands 对应。朴素表述给每个 $\widehat G$-local system 一个 Hecke eigensheaf；现代形式要求一个范畴等价：谱侧是 $\widehat G$-local systems 模栈上的 sheaves，特征零 de Rham 自动侧是 $\operatorname{Bun}_G$ 上的 half-twisted D-modules，其他系数理论则使用来源定义的相应 automorphic sheaf category。

## 依赖前置知识

需要第十八章的 $\operatorname{Bun}_G$，第十九章的几何 Satake，第二十章的 Hecke eigensheaves。需要派生代数几何、D-modules、ind-coherent sheaves、quasi-coherent sheaves、singular support 和 stack 上的范畴。特征零 unramified de Rham/Betti 版本按 Gaitsgory-Raskin 合作项目的五篇 preprint theorem 记录为外部输入；正特征 $\ell$-adic 情形只调用 21.8.1 的部分结果，unrestricted、ramified 和 quantum 版本仍列为研究边界。附录 O 给出本章所用 D-module、IndCoh 和 singular support 的技术索引，附录 AB 给出 derived stacks、cotangent complex、six functors、renormalization 和 spectral action 的更细接口。

收口归一化回指：本章只固定范畴化接口；若与有限域函数迹或局部几何 Langlands 比较，必须使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 9 节。

## 21.1 谱侧局部系统模栈

本章定理性主口径固定 $X/\mathbb C$ 为 smooth projective connected curve，$G/\mathbb C$ 为 connected reductive group，$\widehat G$ 为复对偶群。正特征 $\ell$-adic 口径只在外部输入 21.8.1、紧随其后的研究边界或明确的 sheaf-function 比较中出现，不与特征零 D-module 公式混写。

**定义 21.1.** Betti 或 de Rham 口径下，$\widehat G$-local systems 的模栈记为
$$
\operatorname{LocSys}_{\widehat G}(X).
$$
当 $k=\mathbb C$ 时，可把它理解为主 $\widehat G$-丛连同 flat connection 的 derived moduli stack。

**注 21.2.** $\operatorname{LocSys}_{\widehat G}(X)$ 必须按 derived stack 理解。原因是 reducible local systems 有 automorphisms 和 obstruction theory；若只取经典点集，会丢失范畴等价所需的高阶结构。

**定义 21.3.** 谱侧范畴的常见候选为
$$
\operatorname{QCoh}(\operatorname{LocSys}_{\widehat G}(X))
$$
或
$$
\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X)),
$$
其中 $\mathcal N$ 是 nilpotent singular support 条件。

**注 21.4.** 对一般 $G$，正确谱侧通常不是全部 quasi-coherent sheaves，而是带 nilpotent singular support 的 ind-coherent sheaves。该修正反映 Eisenstein series、非半单 local systems 和连续谱现象。

**注 21.4.1.** Nilpotent singular support 是现代范畴几何 Langlands 中控制谱侧大小的关键条件。附录 O 将其作为外部输入理论给出，并说明它如何进入 $\operatorname{IndCoh}_{\mathcal N}$。

**注 21.4.2.** $\operatorname{LocSys}_{\widehat G}(X)$ 的 derived structure 不是形式修饰。其 cotangent complex 记录 infinitesimal deformations 与 obstructions，singularity stack 则给出 singular support 的载体。附录 AB.1--AB.3 固定这些对象，使 $\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X))$ 成为可检验的范畴表达式，而不是点集上的记号。

## 21.2 自动侧范畴

**定义 21.5.** 令 $\det_{\operatorname{Bun}_G}$ 为 Gaitsgory-Raskin proof series 按伴随丛上同调归一化的 determinant line bundle，并令 $\det_{\operatorname{Bun}_G}^{1/2}$ 表示其平方根的 $\mu_2$-gerbe。特征零 de Rham 普通自动侧定义为相应的 ordinary half-twisted D-module category
$$
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G).
$$
本章后文的 $\mathcal D(\operatorname{Bun}_G)$ 在 de Rham 主口径下是这个范畴的简写。在 $\ell$-adic 口径下，必须取来源定理定义的 automorphic sheaf category；“相应 constructible derived category”只作对象类型提示，不能替代 21.8.1 的 nilpotent singular-support 和 renormalization 条件。

**注 21.5.1（half-twist 归一化）.** 选择 $\omega_X^{1/2}$ 后，来源给出
$$
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
\simeq
\operatorname{DMod}(\operatorname{Bun}_G)
$$
的识别，但这不允许在 Hecke 公式中删除下标 $1/2$：未扭曲写法会把通常的 $\operatorname{Rep}(\widehat G)$ Hecke action 改成带 canonical central gerbe 的版本。因而本章定理性陈述始终保留 half-twist。

**注 21.6.** $\operatorname{Bun}_G$ 非 quasi-compact，因此普通 $\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)$ 的 compact generation、$!$-extension 和沿非 proper correspondence 的函子需要 truncatable/co-truncative 理论。这是普通范畴自身的构造问题，不意味着定义 21.5 已把它替换为 renormalized category。

**定义 21.6.1.** 普通自动侧的 renormalized companion 另记为
$$
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}},
$$
并配备伴随对
$$
\operatorname{ren}:
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
\rightleftarrows
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}}
:\operatorname{un\text{-}ren}.
$$
来源还构造 companion functor
$$
\mathbb L_{G,\operatorname{ren}}:
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}}
\longrightarrow
\operatorname{IndCoh}(\operatorname{LocSys}_{\widehat G}(X)).
$$
注意其目标为全部 $\operatorname{IndCoh}$。它与下文普通自动侧到
$\operatorname{IndCoh}_{\mathcal N}$ 的主函子是两个可比较但不同型的版本。

## 21.3 范畴化几何 Langlands

**外部输入定理 21.7（unramified categorical geometric Langlands，特征零 preprint theorem）.** 在本章固定的 $X/\mathbb C$ 与 $G/\mathbb C$ 设定下，并按来源固定 half-twisting gerbe、中心、普通 D-module category、nilpotent singular support 及 de Rham stack conventions，Gaitsgory-Raskin 合作项目构造从自动侧到谱侧的 Langlands functor，并证明它是范畴等价：
$$
\mathbb L_G:
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
\xrightarrow{\sim}
\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X))
$$
以及相应 Betti 版本；该等价与项目中精确定义的 spectral/Hecke action 和 parabolic induction 兼容。

**注 21.8（来源状态与函子方向）.** 定理 21.7 的来源是 2024 年起发布的五篇 proof-series preprints，而非本书证明；本书把它登记为“外部 preprint theorem”，且不让任何算术主线证明依赖其出版或审稿状态。Proof I 构造的 $\mathbb L_G$ 的方向是 automorphic $\to$ spectral；正文若从谱侧对象构造自动侧对象，一律写逆等价 $\mathbb L_G^{-1}$。公式左端是定义 21.5 的 ordinary half-twisted category，不是定义 21.6.1 的 renormalized companion；右端是 derived 且带 nilpotent singular support 的范畴，不能替换为 naive bounded derived category。

**外部输入定理 21.8.1（正特征中的部分 $\ell$-adic 几何 Langlands；preprint theorem）.** 在 Gaitsgory-Raskin 2025 预印本精确定义的正特征 unramified $\ell$-adic 设定中，带 nilpotent singular support 的 automorphic sheaves 范畴与 Langlands 参数栈的**某些连通分支之并**上的适当 $\operatorname{IndCoh}$ 范畴等价。这里“某些连通分支之并”和“适当定义的范畴”都是定理的一部分；本书不把该结果扩张为整个参数栈上的 unrestricted 等价。

**研究边界 21.8.2.** 定理 21.7 与 21.8.1 合在一起仍不自动覆盖：正特征中整个参数栈所有连通分支上的 unrestricted $\ell$-adic categorical equivalence、ramified/level-structure 版本、integral coefficients、quantum geometric Langlands，或 Fargues-Fontaine 曲线上的 local geometric Langlands。需要这些版本时必须另立外部定理或猜想，不能引用 21.7 或 21.8.1 代替。

**推论 21.9.** 设 $\mathcal E\in\operatorname{LocSys}_{\widehat G}(X)$，并假设其 residual gerbe 上所取点对象属于
$\operatorname{IndCoh}_{\mathcal N}$ 且定理 21.7 的 spectral action 对该对象可取值。若把该点对象记为 $\delta_{\mathcal E}$，则自动侧对象
$$
\mathcal F_{\mathcal E}:=\mathbb L_G^{-1}(\delta_{\mathcal E})
\in\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
$$
满足第二十章的 Hecke eigencondition。

**证明.** 谱侧 residual gerbe 上的点对象在 $\operatorname{Rep}(\widehat G)$ 的 tautological local system 作用下取值 $V_{\mathcal E}$。定理 21.7 的 Hecke/spectral compatibility 说明 $\mathbb L_G^{-1}$ 把这个作用送回自动侧 Hecke functor，所以 $\mathcal F_{\mathcal E}$ 满足
$$
\mathsf H_V(\mathcal F_{\mathcal E})
\cong
\mathcal F_{\mathcal E}\boxtimes V_{\mathcal E}.
$$
这些同构继承 tensor compatibility，正是定义 20.4 的 Hecke eigensheaf 条件。存在的深内容全部位于外部输入定理 21.7；本推论只作函子相容性的形式推导。$\square$

## 21.4 Eisenstein Series 与 Constant Term

设 $P\subset G$ 为 parabolic subgroup，Levi quotient 为 $M$。几何 Langlands 也要求与 parabolic functors 相容。

**定义 21.10.** 几何 Eisenstein functor 是由 $\operatorname{Bun}_P$ correspondence 给出的函子
$$
\operatorname{Eis}_P:\mathcal D(\operatorname{Bun}_M)\to\mathcal D(\operatorname{Bun}_G).
$$
Constant term functor 是反向 correspondence 给出的函子
$$
\operatorname{CT}_P:\mathcal D(\operatorname{Bun}_G)\to\mathcal D(\operatorname{Bun}_M).
$$

**注 21.11.** 谱侧对应于 $\operatorname{LocSys}_{\widehat M}\to\operatorname{LocSys}_{\widehat G}$ 的推拉函子。Nilpotent singular support 条件正是为了使这些 functors 与范畴等价相容。

## 21.5 Functoriality 的几何形式

设有对偶群同态
$$
\widehat H\to\widehat G.
$$
谱侧有诱导映射
$$
\operatorname{LocSys}_{\widehat H}(X)\to\operatorname{LocSys}_{\widehat G}(X).
$$

**条件 21.12（几何函子性接口）.** 几何 Langlands 等价应把谱侧沿
$$
\operatorname{LocSys}_{\widehat H}\to\operatorname{LocSys}_{\widehat G}
$$
的推拉操作，对应到自动侧 $\operatorname{Bun}_H$ 与 $\operatorname{Bun}_G$ 之间的 kernel functor 或 theta/Eisenstein 型函子。

**注 21.13.** 这比数论函子性更范畴化：数论侧期待自守表示转移；几何侧期待范畴之间的函子与谱侧映射相容。

**注 21.13.1.** 附录 Y 说明 Hecke action 的 factorization 结构为何是范畴化几何 Langlands 的基础输入。若没有多点 Hecke 修改和 tensor compatibility，就不能从 Hecke eigensheaves 恢复完整的 $\widehat G$-local system。

## 21.6 本章小结

特征零 unramified 几何 Langlands 的核心不是“点到对象”的对应，而是外部输入定理 21.7 的范畴等价：
$$
\mathbb L_G:
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
\xrightarrow{\sim}
\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X)).
$$
正特征 $\ell$-adic 情形只按 21.8.1 记录某些参数连通分支之并上的部分 preprint theorem；它不改变 21.8.2 的 unrestricted 边界。
Hecke eigensheaves 是谱侧点对象在逆等价 $\mathbb L_G^{-1}$ 下的像。该等价使用普通 $\operatorname{DMod}_{1/2}$ 与 $\operatorname{IndCoh}_{\mathcal N}$；renormalized companion 与全部 $\operatorname{IndCoh}$ 的公式另行记录，不与它混同。Half-twist、nilpotent singular support、Eisenstein functors 和 constant term functors 是完整理论不可省略的结构。

## 练习

**练习 21.1.** 解释为什么 $\operatorname{LocSys}_{\widehat G}(X)$ 应作为 stack 而不是集合处理。

**练习 21.2.** 说明 Hecke eigensheaf 如何由谱侧 skyscraper sheaf 推出。

**练习 21.3.** 写出 $\operatorname{Bun}_P$ 如何同时映到 $\operatorname{Bun}_M$ 和 $\operatorname{Bun}_G$。

**练习 21.4.** 解释 nilpotent singular support 修正的动机。

**练习 21.5.** 比较数论函子性和几何函子性的对象类型差异。
