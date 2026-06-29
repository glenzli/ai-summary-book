# 第二十一章：谱侧、局部系统和范畴化对应

## 本章目标

本章把第二十章的 Hecke eigensheaf 形式升级为范畴化的几何 Langlands 对应。朴素表述给每个 $\widehat G$-local system 一个 Hecke eigensheaf；现代形式要求一个范畴等价：谱侧是 $\widehat G$-local systems 模栈上的 sheaves，自动侧是 $\operatorname{Bun}_G$ 上的 D-modules 或 $\ell$-adic sheaves。

## 依赖前置知识

需要第十八章的 $\operatorname{Bun}_G$，第十九章的几何 Satake，第二十章的 Hecke eigensheaves。需要派生代数几何、D-modules、ind-coherent sheaves、quasi-coherent sheaves、singular support 和 stack 上的范畴。完整理论采用 Gaitsgory-Lurie、Arinkin-Gaitsgory 等框架；本章只给出严格接口。

## 21.1 谱侧局部系统模栈

设 $X/k$ 为光滑射影曲线，$\widehat G$ 为对偶群。

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

## 21.2 自动侧范畴

**定义 21.5.** 自动侧范畴记为
$$
\mathcal D(\operatorname{Bun}_G)
$$
或更精确地，在特征零 de Rham 口径下取
$$
\operatorname{DMod}(\operatorname{Bun}_G).
$$
在 $\ell$-adic 口径下，取相应 constructible derived category。

**注 21.6.** $\operatorname{Bun}_G$ 非紧且不是有限型，因此 naive D-module category 往往不足以支撑所有 functorial operations。现代处理使用 renormalized categories、compact generation 和 !-extension 的精细版本。

## 21.3 范畴化几何 Langlands

**猜想 21.7（几何 Langlands，范畴形式）.** 存在自然等价
$$
\mathbb L_G:
\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X))
\xrightarrow{\sim}
\operatorname{DMod}(\operatorname{Bun}_G)
$$
或在适当情形下的相应 $\ell$-adic/Betti 版本。该等价应与 Hecke 作用、Eisenstein series、constant term、Verdier duality 和 functoriality 相容。

**注 21.8.** 对某些群和某些开子范畴，定理形式已知；完整一般形式依赖深层 derived algebraic geometry。这里将其作为现代几何 Langlands 的目标陈述。

**命题 21.9.** 若范畴等价 21.7 存在，则第二十章的 Hecke eigensheaf 形式应由谱侧 skyscraper sheaf 推出。

**证明草图.** 设 $\mathcal E\in\operatorname{LocSys}_{\widehat G}(X)$。谱侧点 $\mathcal E$ 的 skyscraper 或残余 gerbe 上的对象在 $\operatorname{Rep}(\widehat G)$ 的 tautological local system 作用下取本征值 $V_{\mathcal E}$。若 $\mathbb L_G$ 与 Hecke 作用相容，则其像 $\mathcal F_{\mathcal E}$ 满足
$$
\mathsf H_V(\mathcal F_{\mathcal E})
\cong
\mathcal F_{\mathcal E}\boxtimes V_{\mathcal E}.
$$
这正是 Hecke eigensheaf 条件。$\square$

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

## 21.6 本章小结

现代几何 Langlands 的核心不是“点到对象”的对应，而是范畴等价：
$$
\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X))
\simeq
\operatorname{DMod}(\operatorname{Bun}_G).
$$
Hecke eigensheaves 是谱侧点对象的像。Nilpotent singular support、Eisenstein functors 和 constant term functors 是完整理论不可省略的结构。

## 练习

**练习 21.1.** 解释为什么 $\operatorname{LocSys}_{\widehat G}(X)$ 应作为 stack 而不是集合处理。

**练习 21.2.** 说明 Hecke eigensheaf 如何由谱侧 skyscraper sheaf 推出。

**练习 21.3.** 写出 $\operatorname{Bun}_P$ 如何同时映到 $\operatorname{Bun}_M$ 和 $\operatorname{Bun}_G$。

**练习 21.4.** 解释 nilpotent singular support 修正的动机。

**练习 21.5.** 比较数论函子性和几何函子性的对象类型差异。
