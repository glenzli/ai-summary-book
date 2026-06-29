# 第六章：函数外延性、命题外延性与单值性

## 本章目标

本章引入 HoTT 的核心原则：函数外延性（function extensionality）、命题外延性和单值性（univalence）。我们明确区分公理化 HoTT 口径和 cubical type theory 口径，并证明函数外延性下的若干基础定理。

## 依赖前置知识

本章依赖前五章，尤其是等价、路径代数和同伦层级。单值性从本章开始可用，之前章节不得逆向依赖它。

## 6.1 函数外延性

**定义 6.1.** 对函数 $f,g:\prod_{x:A}B(x)$，有规范映射
$$
\mathsf{happly}_{f,g}:(f=g)\to\prod_{x:A}(f(x)=g(x))
$$
由对路径 $p:f=g$ 作路径归纳定义；在 $p\equiv\mathsf{refl}_f$ 时取逐点反身路径。

**公理 6.2（函数外延性）.** 函数外延性断言 $\mathsf{happly}_{f,g}$ 是等价：
$$
\mathsf{funext}_{f,g}:\left(\prod_{x:A}(f(x)=g(x))\right)\simeq(f=g).
$$
本书把从逐点同伦到函数路径的方向记为
$$
\mathsf{funext}:\left(\prod_{x:A}f(x)=g(x)\right)\to(f=g).
$$

**验证状态。** 在 HoTT Book 的公理化口径中，函数外延性可由单值性推出，也可单独作为公理。在 cubical type theory 口径中，函数外延性由路径类型的区间结构给出计算性解释。

**命题 6.3（命题值函数类型是命题）.** 若每个 $B(x)$ 是命题，则 $\prod_{x:A}B(x)$ 是命题。

**证明（书内证明，使用函数外延性）.** 见附录 F.5。给定 $f,g:\prod_{x:A}B(x)$。由 $B(x)$ 是命题，对每个 $x:A$ 得 $f(x)=g(x)$。由函数外延性得到 $f=g$。$\square$

## 6.2 命题外延性

**定义 6.4.** 对命题 $P,Q:\mathcal U$，命题外延性断言若 $P\to Q$ 且 $Q\to P$，则 $P=Q$。

**命题 6.5（单值性推出命题外延性）.** 假设单值性。若 $P,Q$ 是命题，并且有 $P\to Q$ 与 $Q\to P$，则 $P=Q$。

**证明（书内证明，使用单值性）.** 见附录 F.4。两个方向的函数在命题性假设下给出 fiber 可收缩意义下的等价 $P\simeq Q$；再由单值性把该等价转为类型路径。这个证明不需要函数外延性。$\square$

## 6.3 从类型路径到等价

**定义 6.6.** 对 $A,B:\mathcal U$，定义
$$
\mathsf{idtoequiv}_{A,B}:(A=B)\to(A\simeq B)
$$
为对路径 $p:A=B$ 作路径归纳。在 $p\equiv\mathsf{refl}_A$ 时取恒等等价 $\mathsf{id}_A$。

**命题 6.7.** $\mathsf{idtoequiv}_{A,A}(\mathsf{refl}_A)$ judgmentally computes 为恒等等价。

**证明.** 这是定义 6.6 的路径归纳计算规则。$\square$

## 6.4 单值性

**公理 6.8（单值性）.** 对任意 $A,B:\mathcal U$，映射
$$
\mathsf{idtoequiv}_{A,B}:(A=B)\to(A\simeq B)
$$
是等价。

因此有反向函数
$$
\mathsf{ua}_{A,B}:(A\simeq B)\to(A=B).
$$

**计算原则 6.9.** 对任意 $e:A\simeq B$，有路径
$$
\mathsf{idtoequiv}(\mathsf{ua}(e))=e.
$$
这是 $\mathsf{idtoequiv}$ 为等价的一个三角同伦。若采用 cubical type theory，相关计算可具有更强的 judgmental 或计算行为；若采用公理化 HoTT，一般只得到路径。

**警告 6.10.** 单值性不是说所有等价类型 judgmentally equal。它给出类型路径 $A=B$。沿该路径 transport 可把结构从 $A$ 转移到 $B$，但计算行为取决于基础系统。

## 6.5 单值性推出函数外延性

**定理 6.11.** 单值性推出函数外延性。

**验证状态：精确外部输入。** 见附录 T。附录 T 记录标准数学路线：由单值性得到等价预合成保持函数空间，再经 path space 投影、可收缩族函数空间和 contractible cone 推出 $\mathsf{happly}$ 是等价。

本书当前正文采用“函数外延性与单值性均可作为第六章后原则”的清晰口径；若要最小化公理，则可用附录 T 的外部定理把函数外延性从公理表中删去。

## 本章小结

本章正式引入函数外延性和单值性。函数外延性把逐点路径提升为函数路径；单值性把类型等价提升为类型路径。二者使 HoTT 成为“结构可运输”的基础系统。

## 练习

**练习 6.1.** 用函数外延性证明常值族上的 transport 函数等于恒等函数。

**练习 6.2.** 设 $P,Q$ 是命题，证明 $(P\simeq Q)$ 是命题。

**练习 6.3.** 写出 $\mathsf{idtoequiv}$ 在 $\mathsf{refl}$ 上的完整路径归纳定义。

**练习 6.4.** 说明为什么公理化单值性通常不给出 judgmental computation。
