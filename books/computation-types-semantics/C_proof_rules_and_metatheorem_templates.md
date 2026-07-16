# 附录 C：推导规则和元定理证明模板

本附录把正文反复使用的证明形状集中写出，避免每章重新发明 preservation、progress、soundness 的元语言。

## C.1 推导规则格式

推导规则写作：
$$
\frac{J_1\quad\cdots\quad J_n}{J}.
$$
其含义是：若前提判断 $J_1,\ldots,J_n$ 可推导，则结论判断 $J$ 可推导。无前提规则称为公理规则。

**定义 C.1（推导树高度）。** 推导树高度为从根到叶的最长规则边数。对推导树高度归纳可处理按规则生成的判断。

## C.2 Preservation 模板

要证明：
$$
\Gamma\vdash e:A\land e\to e'\Rightarrow \Gamma\vdash e':A,
$$
通常按 $e\to e'$ 的推导归纳。

检查表：

1. 每个上下文规约规则用归纳假设重建原类型规则；
2. 每个计算 redex 规则需要 inversion 引理；
3. 有绑定的 redex 需要替换引理；
4. 有类型变量的 redex 需要类型替换引理；
5. 有 store 的语言需额外维护 store typing。

## C.3 Progress 模板

要证明闭良类型项要么是值要么可步进，通常按类型推导归纳。

检查表：

1. 空上下文排除自由变量；
2. 对应用、投影、case 等消去形式，先对主表达式用归纳假设；
3. 若主表达式是值，用 canonical forms 引理识别值形状；
4. 若语言有错误或异常，结论必须把错误状态纳入，而不能仍写“值或可步进”。

**引理 C.2（canonical forms 示例）。** 在 STLC 中，若 $\emptyset\vdash v:A\to B$ 且 $v$ 是值，则 $v=\lambda x:A.e$。

**证明。** 第 4 章 STLC 的值语法只有项抽象 $\lambda x:C.e$。对其类型判断使用抽象反演，
再由箭头类型单射性得到 $C=A$ 及主体结果类型 $B$。证毕。

## C.4 Soundness 模板

Hoare soundness、指称 soundness 和类型 soundness 形式不同：

- 类型 soundness：语法推导保持运行不出错；
- 指称 soundness：操作运行结果被语义函数正确反映；
- 逻辑 soundness：证明系统推导出的三元组语义有效。

证明时必须先写清楚 soundness 的方向。若写成双向，则通常是在证明 adequacy 或 completeness。

## C.5 模板附录的证明责任

本附录不使用外部输入。它提供证明模板，不新增正文主线定理。

## 练习

**练习 EC.1.** 为 STLC 的应用规则写出 inversion 引理。

**练习 EC.2.** 说明为什么有异常的语言中 progress 结论需要包含“抛出未捕获异常”或把异常视为结果。
