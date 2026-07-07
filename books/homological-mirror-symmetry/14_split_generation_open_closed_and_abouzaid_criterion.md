# 第十四章：split-generation、open-closed map 与 Abouzaid criterion

## 本章目标

本章建立 HMS 证明中最常用的生成工具：split-generation、Hochschild homology、open-closed map、closed-open map 和 Abouzaid criterion。核心结论多为外部输入，但其范畴逻辑可以内部证明。

## 依赖前置知识

需要第一章 Morita 理论、第四章 Fukaya category、第八章不变量检查。

## 14.1 Split-generation

**定义 14.1.** 设 $\mathcal A$ 是 pretriangulated idempotent-complete $A_\infty$ category。对象集合 $\mathcal G$ split-generates $\mathcal A$，若包含 $\mathcal G$ 的最小厚子范畴等于 $H^0(\mathcal A)$。

**命题 14.2.** 若 $\mathcal G$ split-generates $\mathcal A$，则 restriction functor
$$
\operatorname{Perf}(\mathcal A)\to\operatorname{Perf}(\mathcal A_{\mathcal G})
$$
在 Morita 意义下由 $\mathcal G$ 的 endomorphism category 决定。

**证明.** $\operatorname{Perf}(\mathcal A)$ 由 representables 生成。每个 representable 属于 $\mathcal G$ 的厚闭包，所以 perfect module category 由 $\mathcal G$ 对应的 representables 生成。Morita 类型只依赖这些生成对象的 endomorphism $A_\infty$ category。证毕。

## 14.2 Hochschild homology 与 open-closed map

**定义 14.3.** 对 $A_\infty$ category $\mathcal A$，Hochschild chain complex $CC_\ast(\mathcal A)$ 由循环 composable morphism tensors 组成，微分由内部 $\mu^d$ 和循环插入运算给出。其 homology 记为 $HH_\ast(\mathcal A)$。

**定义 14.4.** 对 Fukaya category $\mathcal F(M)$，open-closed map 是映射
$$
\mathcal{OC}:HH_\ast(\mathcal F(M))\to QH^{\ast+n}(M)
$$
或在 exact/noncompact 情况下到 symplectic cohomology
$$
\mathcal{OC}:HH_\ast(\mathcal W(M))\to SH^{\ast+n}(M).
$$
它由带一个 interior marked point 的 holomorphic disks 计数定义。

**定义 14.5.** closed-open map 是反向方向的环映射
$$
\mathcal{CO}:QH^\ast(M)\to HH^\ast(\mathcal F(M))
$$
或 $SH^\ast(M)\to HH^\ast(\mathcal W(M))$。

**外部输入定理 14.6（open-closed/closed-open 构造）.** 在适当 compactness、transversality、orientation 和 monotone/exact/Novikov 假设下，open-closed 与 closed-open maps 可构造，并与 product、module structures 和 Cardy-type relations 相容。

## 14.3 Abouzaid criterion

**外部输入定理 14.7（Abouzaid generation criterion）.** 设 $\mathcal G\subset\mathcal W(M)$ 是对象集合。若 open-closed map restricted to $HH_\ast(\mathcal G)$ 命中 symplectic cohomology $SH^\ast(M)$ 的单位，则 $\mathcal G$ split-generates $\mathcal W(M)$。compact/monotone 版本中，目标替换为 quantum cohomology 的相应单位或幂等分量。

**解释 14.8.** 这个准则把一个范畴生成问题转化为 closed-string invariant 中单位是否被 open-string Hochschild class 命中的问题。它是 wrapped HMS 中证明生成性的主要工具。

**命题 14.9.** 假设定理 14.7。若 $\mathcal G$ 命中单位且 $\mathcal H$ 为 B-side 生成对象，并且
$$
\operatorname{End}_{\mathcal W(M)}(\mathcal G)\simeq
\operatorname{End}_{\mathcal B}(\mathcal H)
$$
为 $A_\infty$ quasi-isomorphism，则得到 wrapped HMS 的 Morita 等价。

**证明.** 定理 14.7 给出 $\mathcal G$ split-generates $\mathcal W(M)$。假设 $\mathcal H$ split-generates $\mathcal B$。endomorphism quasi-isomorphism 给出 full subcategories quasi-equivalent。由命题 8.9 得到 Morita equivalence。证毕。

## 14.4 幂等分解

**定义 14.10.** 若 quantum cohomology 或 symplectic cohomology 分解为幂等分量
$$
QH^\ast(M)=\bigoplus_\lambda QH^\ast(M)_\lambda,
$$
则 Fukaya category 常相应分解为
$$
\mathcal F(M)=\bigoplus_\lambda \mathcal F(M)_\lambda,
$$
其中 $\lambda$ 通常由 curvature/potential value 标记。

**命题 14.11.** 若对象 $(L,b)$ 的 curvature value 为 $\lambda$，则它只可能属于 $\lambda$ 对应的 Fukaya summand；不同 $\lambda$ 的 summands 之间 morphism cohomology 消失。

**证明草图.** 由命题 5.14，两个不同 potential value 的对象不属于同一个非弯曲 Fukaya fiber category；若放入 curved 或 matrix-factorization 型总模型，则可逆曲率差使相应 morphism object contractible。幂等分解把这种正交性组织为范畴直和分解。证毕。

## 本章小结

Split-generation 是 HMS 证明的核心瓶颈。Open-closed map 和 Abouzaid criterion 提供了可操作的生成性判据：若生成对象的 Hochschild classes 命中 closed-string 单位，则它们生成 Fukaya category。结合 endomorphism algebra 比较即可得到 Morita 版本 HMS。

## 练习

**练习 14.1.** 证明厚子范畴在 direct summands 下闭合。

**练习 14.2.** 解释 open-closed map 中 interior marked point 的几何含义。

**练习 14.3.** 用命题 14.9 写出一个完整 HMS 证明的形式骨架。

**练习 14.4.** 说明 quantum cohomology 幂等分解为什么会对应 Fukaya category 的分块。
