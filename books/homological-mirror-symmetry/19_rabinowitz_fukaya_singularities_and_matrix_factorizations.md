# 第十九章：Rabinowitz Fukaya categories、singularities 与 matrix factorizations

## 本章目标

本章介绍奇点、Milnor fibers、Rabinowitz Fukaya categories 和 matrix factorizations 在 HMS 中的关系。该方向属于现代研究边界，但已有若干证明型结果。

## 依赖前置知识

需要第二章 matrix factorizations、第五章 curved structures、第六章 wrapped categories。

## 19.1 奇点与 Milnor fiber

**定义 19.1.** 设 $f:\mathbb C^{n+1}\to\mathbb C$ 是有孤立临界点的 holomorphic function。对足够小的 $\epsilon,\delta$，Milnor fiber 定义为
$$
F_f=f^{-1}(\epsilon)\cap B_\delta(0).
$$
它带有 exact symplectic structure 的自然模型。

**定义 19.2.** Brieskorn-Pham polynomial 是
$$
f(x_0,\ldots,x_n)=x_0^{a_0}+\cdots+x_n^{a_n}
$$
形式的多项式。

## 19.2 Matrix factorizations

**定义 19.3.** 对 polynomial $f$，B-side singularity category 常由
$$
\operatorname{MF}(\mathbb A^{n+1},f)
$$
或带群作用的 equivariant matrix factorizations 表示。

**外部输入定理 19.4（Orlov singularity 关系）.** 在合适假设下，matrix factorizations 与 hypersurface singularity category 等价或紧密相关。该结果给出 singularity HMS 的 B-side 标准模型。

## 19.3 Rabinowitz Fukaya category

**定义 19.5.** Rabinowitz Floer theory 研究 contact-type hypersurface 上的 Reeb dynamics 和 action functional 带 Lagrange multiplier 的 Floer theory。Rabinowitz Fukaya category 是把这种 Floer theory 范畴化后得到的 A-side 对象；具体模型依赖文献构造。

**警告 19.6.** Rabinowitz Fukaya category 不是 ordinary wrapped Fukaya category 的同义词。其对象、morphisms 和 grading 需按具体文献定义；本书当前只把它作为研究专题对象。

**外部输入定理 19.7（Lekili-Ueda Brieskorn-Pham 结果）.** 对非 Calabi-Yau 型 Brieskorn-Pham singularities 的 Milnor fibers，Rabinowitz Fukaya categories 与 equivariant matrix factorizations 之间存在 HMS 型结果，并可用 Hochschild homology 计算 Rabinowitz Floer homology。  
来源：Lekili-Ueda, *Homological mirror symmetry for Rabinowitz Fukaya categories of Milnor fibers of Brieskorn-Pham singularities*。

## 19.4 Categorical consequence

**命题 19.8.** 假设 Rabinowitz HMS 等价
$$
\mathcal R\mathcal F(F_f)\simeq\operatorname{MF}^{G}(\mathbb A^{n+1},f)
$$
在 Morita 意义下成立。则
$$
HH_\ast(\mathcal R\mathcal F(F_f))\cong
HH_\ast(\operatorname{MF}^{G}(\mathbb A^{n+1},f)).
$$

**证明.** 由 Hochschild homology 的 Morita invariance 直接得到。证毕。

**解释 19.9.** 若某个 Rabinowitz Floer invariant 可被 open-closed 或 Hochschild 结构识别，则 B-side matrix factorization 的 Hochschild homology 给出可计算模型。

## 本章小结

奇点版本 HMS 把 Milnor fiber 和 Rabinowitz/Fukaya 型 categories 与 matrix factorizations 连接起来。该方向高度依赖具体模型和外部输入，本书当前把它放在研究边界，但保留严格的数据包格式和 Morita 后果。

## 练习

**练习 19.1.** 对 $f=x^a+y^b$，写出其 Milnor fiber 的定义。

**练习 19.2.** 说明 matrix factorization 中 $d^2=f$ 与普通复形 $d^2=0$ 的差异。

**练习 19.3.** 证明命题 19.8。

**练习 19.4.** 解释为什么 Rabinowitz Fukaya category 需要单独声明模型，而不能直接写作 $\mathcal W(F_f)$。
