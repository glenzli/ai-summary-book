# 第十八章：Hochschild invariants、closed-open maps 与 categorical enumerative checks

## 本章目标

本章说明 HMS 等价应保持的高阶范畴不变量：Hochschild homology/cohomology、Serre functor、Calabi-Yau structure、closed-open/open-closed maps 和枚举几何检查。它们不能替代 HMS 证明，但能提供强约束。

## 依赖前置知识

需要第一章 Morita 理论、第十四章 open-closed map，以及第二章 B-side 导出几何。

## 18.1 Hochschild invariants

**定义 18.1.** 对 dg 或 $A_\infty$ category $\mathcal A$，Hochschild homology $HH_\ast(\mathcal A)$ 是 Hochschild chains 的 homology；Hochschild cohomology $HH^\ast(\mathcal A)$ 是 Hochschild cochains 的 cohomology，并带 Gerstenhaber algebra 结构。

**外部输入定理 18.2（Morita invariance）.** Hochschild homology 和 cohomology 在 Morita equivalence 下不变。

**推论 18.3.** 若 HMS 以 Morita equivalence 形式成立：
$$
\mathcal A_A\simeq_{\mathrm{Morita}}\mathcal B_B,
$$
则
$$
HH_\ast(\mathcal A_A)\cong HH_\ast(\mathcal B_B),\qquad
HH^\ast(\mathcal A_A)\cong HH^\ast(\mathcal B_B).
$$

**证明.** 直接由定理 18.2 应用于 HMS Morita equivalence。证毕。

## 18.2 HKR 与 Hodge 检查

**外部输入定理 18.4（HKR）.** 若 $X$ 是特征零域上的光滑 proper variety，则存在 Hochschild-Kostant-Rosenberg 型同构
$$
HH_i(\operatorname{Perf}(X))\cong
\bigoplus_{p-q=i}H^q(X,\Omega_X^p)
$$
在适当修正后与乘法、Todd class 和 Mukai pairing 相容。

**警告 18.5.** HKR 的乘法相容不是裸同构自动给出的，需要 Todd class 修正。HMS 文献中比较 pairings 时必须说明采用的规范化。

**命题 18.6.** 若 $X,Y$ 为 Calabi-Yau mirror pair 且 HMS 成立，则 A-side Fukaya category 的 Hochschild homology 维数与 $X$ 的 Hodge numbers 的相应组合匹配。

**证明.** HMS 给出 $HH_\ast(\mathcal F(Y))\cong HH_\ast(\operatorname{Perf}(X))$。由 HKR 外部输入，右边维数等于 Hodge cohomology 的相应直和维数。证毕。

## 18.3 Closed-open map 与 quantum/symplectic cohomology

**定义 18.7.** closed-open map
$$
\mathcal{CO}:SH^\ast(M)\to HH^\ast(\mathcal W(M))
$$
把 closed-string operations 映到 open-string category 的 Hochschild cochains。compact monotone 情况下 $SH^\ast$ 替换为 $QH^\ast$ 或其幂等分量。

**外部输入定理 18.8（closed-open 同构现象）.** 在许多 Weinstein 或 generation 已知的情形中，closed-open map 在合适条件下是同构或检测生成性的关键映射。具体成立范围依赖几何假设。

**解释 18.9.** B-side 上，$HH^\ast(\operatorname{Perf}(X))$ 与 polyvector fields 相关。镜像对称预期把 A-side quantum/symplectic cohomology 与 B-side polyvector/Hochschild cohomology 匹配。

## 18.4 Categorical enumerative checks

**定义 18.10.** 一个 categorical enumerative check 是从 HMS 等价推出的数值或结构匹配，例如：

- Euler pairing 与 intersection pairing 匹配；
- Serre functor 与 Calabi-Yau dimension 匹配；
- Hochschild homology 与 Hodge diamond 匹配；
- disk potential critical values 与 quantum cohomology eigenvalues 匹配；
- open-closed map 命中单位与生成性匹配。

**命题 18.11.** 若某个候选 HMS 数据包无法通过 Euler pairing 检查，则不存在保持给定对象字典的增强等价。

**证明.** 增强等价保持 morphism complexes 的 quasi-isomorphism type，因此保持其上同调维数和 Euler pairing。若对象字典下 Euler pairing 不匹配，则不存在这样的等价。证毕。

**警告 18.12.** 通过所有已知 categorical checks 不推出 HMS。它们是必要条件，不是充分条件。

## 本章小结

Hochschild invariants、HKR、closed-open maps 和 categorical checks 是 HMS 的强约束。它们能排除错误字典，验证维数和 pairing，但不能替代生成性和 endomorphism $A_\infty$ algebra 的比较。

## 练习

**练习 18.1.** 证明 Morita 等价保持 Hochschild homology 的形式原因。

**练习 18.2.** 对椭圆曲线，写出 HKR 右边的各项维数。

**练习 18.3.** 解释 Todd class 修正在 HKR pairing 比较中的作用。

**练习 18.4.** 给出一个候选 HMS 字典，并列出三项 categorical checks。
