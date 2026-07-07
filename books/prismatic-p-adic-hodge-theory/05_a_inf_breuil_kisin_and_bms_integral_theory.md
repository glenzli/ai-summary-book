# 第五章：$A_{\inf}$、Breuil-Kisin theory 与 BMS 积分比较

## 本章目标

本章说明 Bhatt-Morrow-Scholze integral $p$-adic Hodge theory 在 prismatic theory 出现前解决的问题：构造 $A_{\inf}$-cohomology 和 Breuil-Kisin 型上同调，使同一个积分对象同时控制 de Rham、crystalline 和 etale specialization。Prismatic cohomology 后来把这些对象解释为特定 prism 上的 cohomology。

## 依赖前置知识

依赖第一章的 perfectoid interface、第二章的 perfect prism 与 Breuil-Kisin prism，以及第四章的 classical comparison interface。需要熟悉 derived category、almost mathematics 和 pro-etale site 的基本语言；大型构造作为外部输入。

## 5.1 Fontaine's $A_{\inf}$

**约定 5.1.** 令 $C/\mathbf Q_p$ 为代数闭完备非阿基米德域，$\mathcal O_C$ 为整数环。记
$$
A_{\inf}=A_{\inf}(\mathcal O_C)=W(\mathcal O_C^\flat).
$$
Fontaine map 记为
$$
\theta:A_{\inf}\to\mathcal O_C.
$$

**定义 5.2.** 选取 $\xi\in A_{\inf}$ 生成 $\ker(\theta)$。若 $\mathcal O_C$ 中选定兼容的 $p$-power roots of unity $\epsilon=(1,\zeta_p,\zeta_{p^2},\ldots)$，则记
$$
\mu=[\epsilon]-1.
$$

**警告 5.3.** $\xi$ 和 $\mu$ 的选择依赖生成元或 roots of unity。最终公式中的 twists、Nygaard filtration 和 syntomic maps 不得在未说明选择的情况下混用。

## 5.2 $L\eta$ functor

**定义 5.4.** 令 $A$ 为环，$f\in A$。对一个可由 $f$-torsionfree complex 表示的对象 $K\in D(A)$，$L\eta_fK$ 是 Berthelot-Ogus 型 truncation functor 的导出版。直观地，它保留满足 differential 后多一个 $f$-整除性的元素。

**警告 5.5.** $L\eta_f$ 不是普通 truncation，也不是 derived tensor product。它与 modulo $f$、completion 和 Frobenius 的相互作用是 BMS 理论的核心技术之一。本书当前版本只记录它的功能，不重建完整定义。

**外部输入定理 5.6.** $L\eta_f$ 满足 BMS 所需的 functoriality、modulo $f$ comparison 和 compatibility with Frobenius，用于从 pro-etale cohomology 构造 $A_{\inf}$-cohomology。

## 5.3 BMS 的 $A_{\inf}$-cohomology

**外部输入定义 5.7.** 令 $\mathfrak X$ 为 proper smooth formal scheme over $\mathcal O_C$，generic fibre 为 $X$. BMS 构造一个 perfect complex
$$
R\Gamma_{A_{\inf}}(\mathfrak X)
$$
over $A_{\inf}$，带 Frobenius-semilinear endomorphism。该 complex 可通过 pro-etale period sheaves 和 $L\eta_\mu$ 构造。

**外部输入定理 5.8（BMS integral comparison）.** 在定义 5.7 的假设下，$R\Gamma_{A_{\inf}}(\mathfrak X)$ 同时具有以下 specialization：

1. after $\theta:A_{\inf}\to\mathcal O_C$，得到 de Rham cohomology of $\mathfrak X/\mathcal O_C$；
2. after suitable crystalline specialization，得到 special fibre 的 crystalline cohomology；
3. after inverting $\mu$ and taking Frobenius-compatible comparison，得到 generic fibre 的 $p$-adic etale cohomology；
4. cohomology groups 给出 Breuil-Kisin-Fargues module 型积分结构。

**说明 5.9.** 定理 5.8 的真正强度在于“积分”：它不是只在 $\mathbf Q_p$-向量空间层面比较，而是控制 torsion、lattices 和 Frobenius module 结构。

## 5.4 Prismatic reinterpretation

**外部输入定理 5.10.** 令 $(A_{\inf},\ker\theta)$ 为 perfect prism。则对 proper smooth $\mathfrak X/\mathcal O_C$，Bhatt-Scholze prismatic cohomology 与 BMS 的 $A_{\inf}$-cohomology 存在自然 $\phi$-相容同构：
$$
R\Gamma_\Delta(\mathfrak X/A_{\inf})
\simeq
R\Gamma_{A_{\inf}}(\mathfrak X).
$$

**形式推论 5.11.** 定理 5.10 与第三章的 comparison theorem 组合后，BMS 的 de Rham、crystalline 和 etale specialization 可由 prismatic comparison theorem 统一解释。

**证明.** 定理 5.10 把 BMS complex 识别为 perfect prism 上的 prismatic cohomology。第三章的 Hodge-Tate/de Rham/crystalline/etale comparison 分别给出该 prismatic object 的 specialization。因此 BMS 的各 specialization 被统一为同一 prismatic object 的不同 specialization。证毕。

## 5.5 Breuil-Kisin cohomology

**约定 5.12.** 令 $K/\mathbf Q_p$ 为完全离散赋值域，剩余域 $k$ 完美，uniformizer 为 $\pi$。设
$$
\mathfrak S=W(k)[[u]],\qquad \phi(u)=u^p,
$$
并令 $E(u)$ 为 $\pi$ 的 Eisenstein polynomial。

**外部输入定义 5.13.** 对合适的 proper smooth formal scheme $\mathfrak X/\mathcal O_K$，BMS2 构造 Breuil-Kisin cohomology complex
$$
R\Gamma_{\mathfrak S}(\mathfrak X)
$$
over $\mathfrak S$，带 Frobenius，并通过 relative topological Hochschild homology 给出。在 affine smooth $\operatorname{Spf}(A)/\mathcal O_K$ 情形，BMS2 的 local object 可写为一个 $(p,z)$-complete $E_\infty$-$\mathfrak S$-algebra
$$
\widehat{\Prism}_{A/\mathfrak S},
$$
带 $\varphi$-linear Frobenius，且 Frobenius after inverting $E(u)$ 成为同构。源码级 locator 为 `BMS2-BKLOCAL`。

**外部输入定理 5.14.** 在 Breuil-Kisin prism $(\mathfrak S,(E(u)))$ 上，prismatic cohomology 回收 BMS2 的 Breuil-Kisin cohomology：
$$
R\Gamma_\Delta(\mathfrak X/\mathfrak S)
\simeq
R\Gamma_{\mathfrak S}(\mathfrak X).
$$
在 local affine formulation 中，该对象有三个基本出口：

1. base change to $A_{\inf}$ recovers $A\Omega$ of $\mathfrak X_{\mathcal O_C}$；
2. base change along $\theta:\mathfrak S\to\mathcal O_K$ recovers completed de Rham cohomology over $\mathcal O_K$；
3. base change to $W(k)$ recovers crystalline cohomology of the special fibre。

**说明 5.15.** 这说明 prismatic site 不是只重写 $A_{\inf}$ 情形，而是把 crystalline prism、perfect prism、Breuil-Kisin prism 和 $q$-crystalline prism 放入同一形式。BMS2 的 THH construction 还解释了 Nygaard filtration、syntomic sheaves 和 Breuil-Kisin twists 的来源；这些结构在第七章和第十一章中通过 BMS2-SYN locator 使用。

## 5.6 积分结构与 torsion 控制

**命题 5.16（形式层 torsion 约束）.** 若 $C\in D(A)$ 是 perfect complex，$A\to B$ 是 derived base change，且 $C\otimes_A^L B$ 的 cohomology 在某范围内无 torsion，则 $C$ 的 cohomology torsion 受到 base change spectral sequence 的限制。

**证明草图.** 使用 Tor spectral sequence
$$
E_2^{i,j}=\operatorname{Tor}_{-i}^A(H^j(C),B)\Rightarrow H^{i+j}(C\otimes_A^L B).
$$
若目标 cohomology 的 torsion 消失或被有界控制，则 $E_\infty$ 页给出源 cohomology 中相应 torsion 的约束。精确结论依赖 $A\to B$ 和 $C$ 的 Tor-amplitude，因此本命题只作为形式机制，不替代 BMS 的积分比较定理。证毕。

## 5.7 BMS object 的四个出口

**说明 5.17.** 对 proper smooth $\mathfrak X/\mathcal O_C$，$R\Gamma_{A_{\inf}}(\mathfrak X)$ 应被看成一个有四个出口的对象：

| 出口 | 操作 | 目标 |
| --- | --- | --- |
| de Rham | $\otimes_{A_{\inf},\theta}^L\mathcal O_C$ | $R\Gamma_{\mathrm{dR}}(\mathfrak X/\mathcal O_C)$ |
| crystalline | crystalline specialization | special fibre crystalline cohomology |
| etale | invert $\mu$ and Frobenius comparison | $R\Gamma_{\mathrm{et}}(X,\mathbf Z_p)$ |
| lattice | cohomology with Frobenius | BKF module 型对象 |

**命题 5.18.** 若一个 construction 只给出 de Rham 出口，则它不足以替代 $A_{\inf}$-cohomology。

**证明.** $A_{\inf}$-cohomology 的作用是同时控制 de Rham、crystalline、etale 和 integral lattice 信息。只给出 de Rham 出口会遗忘 Frobenius、torsion 和 etale comparison 所需的结构。因此它不能替代完整对象。证毕。

## 本章小结

BMS 理论在 prismatic theory 前已经构造了强大的积分 cohomology object。BMS1 给出 $A_{\inf}$-cohomology 和 $A\Omega$，BMS2 用 THH/TC refinement 产生 Breuil-Kisin cohomology、Nygaard filtration 和 syntomic sheaves。Prismatic theory 的作用不是废弃 BMS，而是把 $A_{\inf}$ 和 Breuil-Kisin cohomology 解释为特定 prism 上的统一 cohomology，并把 comparison theorem 放入一个 site-theoretic 框架。

## 练习

**练习 5.1.** 说明为什么 $R\Gamma_{A_{\inf}}(\mathfrak X)$ 不能只看成 $\mathbf Z_p$-complex。

**练习 5.2.** 比较 perfect prism $(A_{\inf},\ker\theta)$ 与 Breuil-Kisin prism $(\mathfrak S,(E(u)))$ 的 quotient ring。

**练习 5.3.** 写出形式推论 5.11 中用到的两个外部输入定理，并说明它们分别属于 integral layer 还是 prismatic layer。
