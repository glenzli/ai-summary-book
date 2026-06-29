# 第二十章：Hecke Eigensheaves

## 本章目标

本章定义 Hecke eigensheaf，并给出几何 Langlands 对应的基本形式。第十九章给出 $\operatorname{Rep}(\widehat G)$ 对 $\mathcal D(\operatorname{Bun}_G)$ 的 Hecke 作用。Hecke eigensheaf 是这些 Hecke 函子的共同本征对象，其本征值不是数字，而是曲线 $X$ 上的 $\widehat G$-local system。

## 依赖前置知识

需要第十八章的 $\operatorname{Bun}_G$ 和 Hecke correspondence，第十九章的几何 Satake。需要 local systems、D-modules 或 $\ell$-adic sheaves、张量函子和 derived categories。完整几何 Langlands 需要 derived stacks、renormalized sheaf categories 和 spectral side；本章只陈述基础接口。

## 20.1 对偶群局部系统

设 $X/k$ 为光滑射影连通曲线，$G/k$ 为 split connected reductive group，对偶群为 $\widehat G$。

**定义 20.1.** 若 $k=\mathbb C$，一个 $\widehat G$-local system 是 $X$ 上的主 $\widehat G$-丛 $\mathcal E$，带 flat connection
$$
\nabla:\mathcal E\to\mathcal E\otimes\Omega_X^1.
$$

若 $k$ 为特征 $p$ 域并使用 $\ell$-adic sheaves，则 $\widehat G$-local system 可理解为连续表示
$$
\rho:\pi_1^{\operatorname{et}}(X)\to\widehat G(\overline{\mathbb Q}_\ell)
$$
的几何对象。

**定义 20.2.** 设 $V\in\operatorname{Rep}(\widehat G)$。由 $\widehat G$-local system $\mathcal E$ 关联的 $V$-局部系统记为
$$
V_{\mathcal E}=\mathcal E\times^{\widehat G}V.
$$

**命题 20.3.** 映射
$$
V\mapsto V_{\mathcal E}
$$
是张量函子
$$
\operatorname{Rep}(\widehat G)\to\operatorname{Loc}(X).
$$

**证明.** 关联丛构造与直和、张量积和对偶相容：
$$
(V\oplus W)_{\mathcal E}\cong V_{\mathcal E}\oplus W_{\mathcal E},
\qquad
(V\otimes W)_{\mathcal E}\cong V_{\mathcal E}\otimes W_{\mathcal E}.
$$
Flat connection 或 étale monodromy 也按这些线性代数运算诱导。因此得到张量函子。$\square$

## 20.2 Hecke 本征条件

设 $\mathcal D(\operatorname{Bun}_G)$ 为选定 sheaf theory 下的 derived category。

**定义 20.4.** 设 $\mathcal E$ 为 $\widehat G$-local system。对象
$$
\mathcal F\in\mathcal D(\operatorname{Bun}_G)
$$
称为 Hecke eigensheaf with eigenvalue $\mathcal E$，若对每个 $V\in\operatorname{Rep}(\widehat G)$，给定同构
$$
\alpha_V:\mathsf H_V(\mathcal F)
\xrightarrow{\sim}
\mathcal F\boxtimes V_{\mathcal E}
$$
作为 $\operatorname{Bun}_G\times X$ 上的对象，并且这些同构对 $V$ 的张量积、直和、单位对象满足相容条件。

**注 20.5.** 该定义中本征值是局部系统 $V_{\mathcal E}$，而不是标量。经典 Hecke eigenform 的本征值 $a_v$ 在几何化后被所有表示 $V$ 的局部系统纤维和 Frobenius trace 统一编码。

**命题 20.6.** 若 $\mathcal F$ 是 eigenvalue 为 $\mathcal E$ 的 Hecke eigensheaf，则
$$
\mathsf H_{V\otimes W}(\mathcal F)
\cong
\mathcal F\boxtimes (V_{\mathcal E}\otimes W_{\mathcal E})
$$
与先作用 $\mathsf H_W$ 再作用 $\mathsf H_V$ 的结果相容。

**证明.** 由第十九章，Hecke 函子满足
$$
\mathsf H_V\circ\mathsf H_W\simeq\mathsf H_{V\otimes W}
$$
在 factorization 意义下成立。由定义 20.4，
$$
\mathsf H_W(\mathcal F)\cong\mathcal F\boxtimes W_{\mathcal E}.
$$
再作用 $\mathsf H_V$，并使用局部系统张量相容性，得到
$$
\mathcal F\boxtimes V_{\mathcal E}\otimes W_{\mathcal E}.
$$
这正是 $\mathsf H_{V\otimes W}(\mathcal F)$ 的本征同构。$\square$

## 20.3 几何 Langlands 的基本形式

**猜想 20.7（几何 Langlands，朴素本征层形式）.** 对适当的 $\widehat G$-local system $\mathcal E$ on $X$，应存在 $\operatorname{Bun}_G$ 上的 Hecke eigensheaf $\mathcal F_{\mathcal E}$，使得对所有 $V\in\operatorname{Rep}(\widehat G)$ 有
$$
\mathsf H_V(\mathcal F_{\mathcal E})
\cong
\mathcal F_{\mathcal E}\boxtimes V_{\mathcal E}.
$$

**注 20.8.** 猜想 20.7 是朴素形式。完整几何 Langlands 不是简单地给每个 local system 一个 sheaf；它应是 spectral side 上 quasi-coherent 或 ind-coherent sheaves 与 automorphic side 上 D-modules 的范畴等价。Hecke eigensheaf 是该范畴等价在 skyscraper sheaf 或点对象上的影子。

## 20.4 `GL(1)` 的几何 Langlands

`GL(1)` 情形由几何类域论描述。

**外部输入定理 20.9（几何类域论，接口形式）.** 设 $G=\mathbb G_m$。Rank-one local systems on $X$ 与 $\operatorname{Pic}(X)$ 上满足 Hecke eigensheaf 条件的 rank-one sheaves 对应。换言之，几何 Langlands for $\mathbb G_m$ 退化为 Picard stack 上的 Fourier-Mukai/Abel-Jacobi 型对应。

**注 20.10.** 这对应数论侧的 `GL(1)` Langlands，即类域论。Hecke 修改 line bundle 相当于张量 $\mathcal O(x)$，本征条件编码 local system 在点 $x$ 的 monodromy 或 Frobenius trace。

## 20.5 Sheaf-Function Dictionary 下的解释

设 $k=\mathbb F_q$。若 $\mathcal F$ 是 $\operatorname{Bun}_G$ 上的 $\ell$-adic sheaf，可取 Frobenius trace 得到函数
$$
f_{\mathcal F}:\operatorname{Bun}_G(\mathbb F_q)\to\overline{\mathbb Q}_\ell.
$$

**命题 20.11.** 若 $\mathcal F$ 是 eigenvalue 为 $\mathcal E$ 的 Hecke eigensheaf，则 $f_{\mathcal F}$ 是函数域自守意义下的 Hecke eigenfunction，其 Hecke eigenvalues 由 $\mathcal E$ 的 Frobenius conjugacy classes 给出。

**证明草图.** Hecke 函子经 sheaf-function dictionary 对应 Hecke 算子。等式
$$
\mathsf H_V(\mathcal F)\cong\mathcal F\boxtimes V_{\mathcal E}
$$
取 Frobenius trace 后变为
$$
T_{V,x}f_{\mathcal F}
=
\operatorname{tr}(\operatorname{Frob}_x\mid V_{\mathcal E,x})f_{\mathcal F}.
$$
右侧就是由 $\widehat G$-local system 的 Frobenius 共轭类给出的 Satake 本征值。$\square$

## 20.6 存在性与唯一性的技术边界

**注 20.12.** Hecke eigensheaf 的存在性依赖 $\mathcal E$ 的性质。若 $\mathcal E$ reducible、有非平凡 automorphisms 或落在 proper parabolic 的 dual data 中，则 automorphic side 可能不是单个 sheaf，而是有 derived 或 categorical multiplicity 的对象。

**注 20.13.** 对 ramified 几何 Langlands，还需加入 marked points、level structures、parabolic bundles、Iwahori 或 deeper level Hecke categories。本书当前先处理 unramified 形式。

## 20.7 本章小结

Hecke eigensheaf 是几何 Langlands 的核心对象。几何 Satake 把 $\operatorname{Rep}(\widehat G)$ 转化为 Hecke 函子；$\widehat G$-local system $\mathcal E$ 把每个表示 $V$ 转化为局部系统 $V_{\mathcal E}$；Hecke eigensheaf 是同时满足
$$
\mathsf H_V(\mathcal F)\cong\mathcal F\boxtimes V_{\mathcal E}
$$
的对象。完整理论应升级为 automorphic category 与 spectral category 的等价。

## 练习

**练习 20.1.** 证明 $V\mapsto V_{\mathcal E}$ 是张量函子。

**练习 20.2.** 对 $G=\mathbb G_m$，描述 Hecke 修改 line bundle 的操作。

**练习 20.3.** 说明 Hecke eigensheaf 的本征值为什么是 local system 而不是标量。

**练习 20.4.** 在有限域情形，用 Frobenius trace 解释 Hecke eigensheaf 如何给出 Hecke eigenfunction。

**练习 20.5.** 解释朴素 eigensheaf 形式与范畴等价形式的差别。
