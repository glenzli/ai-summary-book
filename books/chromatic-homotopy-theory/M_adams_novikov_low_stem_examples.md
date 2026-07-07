# 附录 M：Adams-Novikov 低阶样例与 hidden extension

## M.1 从 Adams 到 Adams-Novikov

**定义 M.1.** Classical Adams spectral sequence 使用 Steenrod algebra：
$$
\operatorname{Ext}_{\mathcal A}^{s,t}(\mathbb F_p,\mathbb F_p)\Rightarrow \pi_{t-s}\mathbb S^\wedge_p.
$$
Adams-Novikov spectral sequence 使用 $MU$ 或 $BP$：
$$
\operatorname{Ext}_{BP_*BP}^{s,t}(BP_*,BP_*)\Rightarrow \pi_{t-s}\mathbb S^\wedge_p.
$$

**警告 M.2.** 两个 spectral sequences 的 filtration 不同。相同稳定同伦元素在两个谱序列中的表示和 differential 可能不同。

## M.2 低阶元素的定位

**例 M.3.** Hopf map $\eta\in\pi_1\mathbb S$ 在 $p=2$ 的稳定 stems 中非零。它在 Adams-Novikov 语境中属于高度 $1$ 相关的低阶周期现象。

**边界 M.4.** 本附录不直接给出完整 Ext 表，因为表格需要固定 prime、resolution convention 和 differential。当前只记录正式教材需要的检查格式。

**样例表 M.5.** 低阶计算表应至少有如下列：

| stem | ANSS class | filtration | differential | hidden extension | resulting group |
| --- | --- | --- | --- | --- | --- |
| $1$ | 待定位 | 待定位 | 待定位 | 待定位 | $\mathbb Z/2$ at $p=2$ |
| $3$ | 待定位 | 待定位 | 待定位 | 待定位 | 包含 $\nu$ 相关信息 |
| $7$ | 待定位 | 待定位 | 待定位 | 待定位 | 包含 $\sigma$ 相关信息 |

**警告 M.6.** 表 M.5 是模板，不是计算结果。填表前必须引用 Ravenel 或机器可复现 Ext 计算。

## M.3 Hidden additive extension

**定义 M.7.** 若 $E_\infty$ 页给出
$$
\operatorname{gr}G\cong \mathbb Z/2\oplus\mathbb Z/2
$$
但实际 filtered group $G$ 可能是 $\mathbb Z/4$，则决定 $G$ 的问题称为 hidden additive extension。

**例 M.8.** 稳定 stems 表中的 cyclic summand 阶数常需要 hidden extension 判定。只列 $E_\infty$ 页不能确定最终群。

## M.4 Hidden multiplicative extension

**定义 M.9.** 若 $E_\infty$ 页上两个 associated graded classes 的乘积为零，但在实际 homotopy ring 中代表元乘积落入更高 filtration 的非零元素，则称为 hidden multiplicative extension。

**警告 M.10.** Multiplicative hidden extensions 会影响 nilpotence、periodicity 和 ring structure 结论。任何乘法结构表都必须单独校验。

## M.5 机器校验协议

**协议 M.11.** 若使用软件或数据库给出 Adams-Novikov 表，必须记录：

1. prime；
2. Hopf algebroid convention；
3. resolution 长度；
4. stem 范围；
5. differential 输入；
6. hidden extension 输入；
7. 输出版本；
8. 与文献表格的逐项比对。

## 本附录小结

Adams-Novikov 低阶计算是正式教材必需内容，但不能用未经定位的表格填充。当前附录先固定计算表格式和 hidden extension 风险，下一轮应填入 Ravenel/Isaksen-Wang-Xu 等来源的可定位低阶样例。
