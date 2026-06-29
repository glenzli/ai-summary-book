# 第二章：恒等类型、路径归纳与路径代数

## 本章目标

本章引入恒等类型 $\mathsf{Id}_A(a,b)$，也记为 $a=_A b$。我们给出形成、引入、消去和计算规则，并从路径归纳推出 transport、逆路径、路径复合和函数作用于路径。所有结果都只使用第一章的规则和本章的恒等类型规则。

## 依赖前置知识

本章依赖第一章的语境、替换、$\Pi$ 型、$\Sigma$ 型和 judgmental equality。尚不使用函数外延性、单值性或高阶归纳类型。

## 2.1 恒等类型规则

**规则 2.1（恒等类型形成）.** 若 $\Gamma\vdash A:\mathcal U_i$，并且 $\Gamma\vdash a:A$、$\Gamma\vdash b:A$，则

$$
\Gamma\vdash \mathsf{Id}_A(a,b):\mathcal U_i.
$$

本书也写作 $a=_A b$。若 $A$ 可由上下文恢复，则写作 $a=b$。

**规则 2.2（反身路径）.** 若 $\Gamma\vdash a:A$，则

$$
\Gamma\vdash \mathsf{refl}_a:a=_A a.
$$

**规则 2.3（路径归纳，J）.** 设

$$
\Gamma,x:A,y:A,p:x=y\vdash C(x,y,p):\mathcal U_k
$$

是依赖于两端点和路径的类型族。若有

$$
\Gamma,x:A\vdash d(x):C(x,x,\mathsf{refl}_x),
$$

则对任意 $x,y:A$ 和 $p:x=y$，有

$$
\Gamma,x:A,y:A,p:x=y\vdash
\mathsf J_C(d,x,y,p):C(x,y,p).
$$

**规则 2.4（J 的计算规则）.** 在上述条件下，

$$
\mathsf J_C(d,x,x,\mathsf{refl}_x)\equiv d(x):C(x,x,\mathsf{refl}_x).
$$

这是恒等类型最关键的 judgmental computation。

**解释 2.5.** 要证明一个依赖于路径 $p:x=y$ 的命题，只需证明它在 $p\equiv\mathsf{refl}_x$ 时成立。这不是说所有路径 judgmentally equal 于反身路径；它是恒等类型的消去原则。

## 2.2 使用 J 的模式

**说明 2.6（固定端点形式）.** 许多教材会把“固定一个端点的路径归纳”（based path induction）作为 J 的等价形式来使用：

$$
y:A,\ p:a=y\vdash C(y,p):\mathcal U_k,\qquad
c:C(a,\mathsf{refl}_a)
$$

推出对任意 $y:A$ 和 $p:a=y$ 有 $C(y,p)$ 的项。

本书在本章不把这一形式当作新的原始规则。下面的 transport、逆路径、路径复合和 $\mathsf{ap}$ 都直接用完整 J 定义。固定端点形式在附录 D.1 中由完整 J、路径复合和单位律推出。

## 2.3 Transport

**定义 2.7（transport）.** 设 $P:A\to\mathcal U_j$，并设 $p:x=y$。定义

$$
\mathsf{transport}^{P}(p):P(x)\to P(y)
$$

如下：对路径 $p$ 作路径归纳。若 $p\equiv\mathsf{refl}_x$，则取恒等函数

$$
\mathsf{id}_{P(x)}:P(x)\to P(x).
$$

因此有计算规则

$$
\mathsf{transport}^{P}(\mathsf{refl}_x,u)\equiv u.
$$

这里也写 $\mathsf{transport}^{P}(p,u)$ 表示函数 $\mathsf{transport}^{P}(p)$ 作用在 $u:P(x)$ 上。

**命题 2.8（transport 沿反身路径）.** 对任意 $u:P(x)$，

$$
\mathsf{transport}^{P}(\mathsf{refl}_x,u)\equiv u.
$$

**证明.** 这正是定义 2.7 中由 J 给出的 judgmental 计算规则。$\square$

**例 2.9.** 若 $P$ 是常值族 $P(z)\equiv B$，则 $\mathsf{transport}^{P}(p):B\to B$ 是沿路径 $p$ 的常值族运输。仅凭本章规则，我们可以证明它 propositionally equal 于恒等函数；若要证明函数本身相等，通常需要函数外延性，当前尚不可用。

## 2.4 逆路径与复合

**定义 2.10（逆路径）.** 若 $p:x=y$，定义

$$
p^{-1}:y=x
$$

为对 $p$ 作路径归纳后的结果。在 $p\equiv\mathsf{refl}_x$ 时，取

$$
(\mathsf{refl}_x)^{-1}\coloneqq \mathsf{refl}_x.
$$

**定义 2.11（路径复合）.** 若 $p:x=y$ 且 $q:y=z$，定义

$$
p\cdot q:x=z
$$

为对 $q$ 作路径归纳后的结果。在 $q\equiv\mathsf{refl}_y$ 时，取

$$
p\cdot \mathsf{refl}_y\coloneqq p.
$$

由此有 judgmental 计算

$$
p\cdot \mathsf{refl}_y\equiv p.
$$

若改为对 $p$ 归纳，也可得到另一种定义；两种定义 propositionally equal，但不必 judgmentally equal。

**命题 2.12（左单位律）.** 若 $p:x=y$，则

$$
\mathsf{refl}_x\cdot p=p.
$$

**证明（书内证明）.** 对 $p$ 作路径归纳。反身情形中目标为

$$
\mathsf{refl}_x\cdot \mathsf{refl}_x=\mathsf{refl}_x.
$$

由路径复合的计算规则，左边 judgmentally equal 于 $\mathsf{refl}_x$，故取 $\mathsf{refl}_{\mathsf{refl}_x}$。$\square$

**命题 2.13（右单位律）.** 若 $p:x=y$，则

$$
p\cdot \mathsf{refl}_y=p.
$$

**证明（书内证明）.** 按定义 2.11，这一等式左边 judgmentally equal 于 $p$。因此取 $\mathsf{refl}_p$。$\square$

**命题 2.14（逆路径的左逆律）.** 若 $p:x=y$，则

$$
p^{-1}\cdot p=\mathsf{refl}_y.
$$

**证明（书内证明）.** 对 $p$ 作路径归纳。反身情形中，目标化为

$$
(\mathsf{refl}_x)^{-1}\cdot\mathsf{refl}_x=\mathsf{refl}_x.
$$

按逆路径和复合的计算规则，左边 judgmentally equal 于 $\mathsf{refl}_x$，故取 $\mathsf{refl}_{\mathsf{refl}_x}$。$\square$

**命题 2.15（逆路径的右逆律）.** 若 $p:x=y$，则

$$
p\cdot p^{-1}=\mathsf{refl}_x.
$$

**证明（书内证明）.** 对 $p$ 作路径归纳。反身情形同上，取 $\mathsf{refl}_{\mathsf{refl}_x}$。$\square$

## 2.5 函数作用于路径

**定义 2.16（ap）.** 若 $f:A\to B$ 且 $p:x=y$，定义

$$
\mathsf{ap}_f(p):f(x)=f(y)
$$

为对 $p$ 作路径归纳后的结果。在 $p\equiv\mathsf{refl}_x$ 时，取

$$
\mathsf{ap}_f(\mathsf{refl}_x)\coloneqq \mathsf{refl}_{f(x)}.
$$

**命题 2.17（ap 保持反身路径）.** 对任意 $x:A$，

$$
\mathsf{ap}_f(\mathsf{refl}_x)\equiv \mathsf{refl}_{f(x)}.
$$

**证明.** 这是定义 2.16 的计算规则。$\square$

**命题 2.18（ap 保持逆，命题形式）.** 若 $p:x=y$，则

$$
\mathsf{ap}_f(p^{-1})=(\mathsf{ap}_f(p))^{-1}.
$$

**证明（书内证明）.** 对 $p$ 作路径归纳。反身情形中，两边都按定义计算为 $\mathsf{refl}_{f(x)}$，因此取反身路径。$\square$

## 2.6 路径代数的派生公式

**命题 2.19（逆的逆）.** 若 $p:x=y$，则
$$
(p^{-1})^{-1}=p.
$$

**证明（书内证明）.** 对 $p$ 作路径归纳。反身情形中，左边按逆路径定义计算为 $\mathsf{refl}_x$，目标为 $\mathsf{refl}_x=\mathsf{refl}_x$，取反身路径。$\square$

**命题 2.20（复合结合律）.** 若 $p:w=x$、$q:x=y$、$r:y=z$，则
$$
(p\cdot q)\cdot r=p\cdot(q\cdot r).
$$

**证明（书内证明）.** 对 $r$ 作路径归纳。反身情形中，左边按复合的计算规则化为 $p\cdot q$，右边中的 $q\cdot\mathsf{refl}_y$ judgmentally equal 于 $q$，因此右边也化为 $p\cdot q$。取反身路径。$\square$

**命题 2.21（ap 保持复合）.** 若 $f:A\to B$，$p:x=y$，$q:y=z$，则
$$
\mathsf{ap}_f(p\cdot q)=\mathsf{ap}_f(p)\cdot\mathsf{ap}_f(q).
$$

**证明（书内证明）.** 对 $q$ 作路径归纳。反身情形中，$p\cdot\mathsf{refl}_y\equiv p$，且 $\mathsf{ap}_f(\mathsf{refl}_y)\equiv\mathsf{refl}_{f(y)}$。右边化为 $\mathsf{ap}_f(p)\cdot\mathsf{refl}_{f(y)}\equiv\mathsf{ap}_f(p)$，故取反身路径。$\square$

**命题 2.22（transport 与路径复合）.** 设 $P:A\to\mathcal U$，$p:x=y$，$q:y=z$，$u:P(x)$。则
$$
\mathsf{transport}^{P}(q,\mathsf{transport}^{P}(p,u))
=
\mathsf{transport}^{P}(p\cdot q,u).
$$

**证明（书内证明）.** 对 $q$ 作路径归纳。反身情形中，左边为
$$
\mathsf{transport}^{P}(\mathsf{refl}_y,\mathsf{transport}^{P}(p,u))
\equiv \mathsf{transport}^{P}(p,u),
$$
右边为
$$
\mathsf{transport}^{P}(p\cdot\mathsf{refl}_y,u)
\equiv \mathsf{transport}^{P}(p,u).
$$
取反身路径。$\square$

本节公式构成后续 $\Sigma$ 路径、fiber 收缩和等价稳定性的证明基础。更完整的证明核见附录 D。

## 2.7 路径空间与同伦直觉

恒等类型让每个类型 $A$ 带有一族路径类型 $x=y$。进一步，若 $p,q:x=y$，则还可以形成路径类型 $p=q$。因此“相等的相等”仍是类型。这是 HoTT 与集合式基础的关键差异。

**定义 2.23.** 对 $A:\mathcal U_i$ 和 $x,y:A$，类型 $x=y$ 称为 $A$ 的路径空间（path space）或恒等类型。

**例 2.24.** 在尚未引入 $\mathbb N$ 的具体规则前，我们不能证明自然数的路径空间是集合式的。类似地，对于任意类型 $A$，不能默认任意两条路径 $p,q:x=y$ 相等。

**警告 2.25.** 本章已经能证明路径满足群胚律的命题版本，但不能证明所有类型都是集合。事实上，HoTT 的核心思想之一正是允许类型具有非平凡高阶路径结构。

## 本章小结

本章引入了恒等类型和路径归纳，并从中构造 transport、逆路径、路径复合和函数作用于路径。所有群胚律都以路径类型中的项表达，而不是 judgmental equality。下一步可以引入自然数、空类型、单位类型和和类型，也可以继续发展路径代数与同伦层级。

## 练习

**练习 2.1.** 用路径归纳证明 $(p^{-1})^{-1}=p$。

**练习 2.2.** 用路径归纳证明路径复合的结合律：
$$
(p\cdot q)\cdot r=p\cdot(q\cdot r).
$$

**练习 2.3.** 设 $P:A\to\mathcal U$，$p:x=y$，$q:y=z$，证明 transport 满足
$$
\mathsf{transport}^{P}(q,\mathsf{transport}^{P}(p,u))
=
\mathsf{transport}^{P}(p\cdot q,u).
$$

**练习 2.4.** 解释为什么命题 2.18 的结论是路径类型中的等式，而不是 judgmental equality。

**练习 2.5.** 若 $f:A\to B$、$g:B\to C$，证明
$$
\mathsf{ap}_{g\circ f}(p)=\mathsf{ap}_g(\mathsf{ap}_f(p)).
$$
