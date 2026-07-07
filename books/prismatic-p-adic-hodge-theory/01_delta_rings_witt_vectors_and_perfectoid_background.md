# 第一章：$\delta$-环、Witt vectors 与 perfectoid 背景

## 本章目标

本章建立 prism 定义所需的算术微积分语言：$\delta$-环、Frobenius lift、distinguished elements、Witt vectors 和 perfectoid interface。重点是把 $\delta$-结构写成可检查的代数公理，而不是把它当作 Frobenius 的直觉替身。

## 依赖前置知识

需要交换代数、模、理想、完备化和 Witt vectors 的基本背景。Witt vectors 的完整构造不在本章重建，只固定本书使用的符号和外部输入。

## 1.1 $\delta$-环

**定义 1.1.** 令 $A$ 为交换环。一个 $p$-derivation 或 $\delta$-结构是映射
$$
\delta:A\to A
$$
满足 $\delta(0)=\delta(1)=0$，并且对任意 $x,y\in A$ 有
$$
\delta(x+y)=\delta(x)+\delta(y)+\frac{x^p+y^p-(x+y)^p}{p},
$$
$$
\delta(xy)=x^p\delta(y)+y^p\delta(x)+p\delta(x)\delta(y).
$$
带有 $\delta$-结构的环称为 $\delta$-环。$\delta$-环态射是保持 $\delta$ 的环同态。

**说明 1.2.** 上式中的分式是整数系数组合，因为对 $1\le i\le p-1$，二项式系数 $\binom pi$ 可被 $p$ 整除：
$$
\frac{x^p+y^p-(x+y)^p}{p}
=-\sum_{i=1}^{p-1}\frac{1}{p}\binom pi x^iy^{p-i}\in\mathbf Z[x,y].
$$

**命题 1.3.** 若 $(A,\delta)$ 是 $\delta$-环，则
$$
\phi_A(x)=x^p+p\delta(x)
$$
定义一个环同态 $\phi_A:A\to A$，并且 $\phi_A$ 模 $p$ 化为绝对 Frobenius。

**证明.** 首先 $\phi_A(0)=0$ 且 $\phi_A(1)=1$。对加法，
$$
\begin{aligned}
\phi_A(x+y)
&=(x+y)^p+p\delta(x+y)\\
&=(x+y)^p+p\delta(x)+p\delta(y)+x^p+y^p-(x+y)^p\\
&=x^p+p\delta(x)+y^p+p\delta(y)\\
&=\phi_A(x)+\phi_A(y).
\end{aligned}
$$
对乘法，
$$
\begin{aligned}
\phi_A(xy)
&=x^py^p+p\delta(xy)\\
&=x^py^p+p x^p\delta(y)+p y^p\delta(x)+p^2\delta(x)\delta(y)\\
&=(x^p+p\delta(x))(y^p+p\delta(y))\\
&=\phi_A(x)\phi_A(y).
\end{aligned}
$$
模 $p$ 后有 $\phi_A(x)\equiv x^p\pmod p$。证毕。

**命题 1.4.** 若 $A$ 是 $p$-torsionfree，则在 $A$ 上给出 $\delta$-结构等价于给出环同态 $\phi:A\to A$，使得 $\phi(x)\equiv x^p\pmod p$。

**证明.** 由命题 1.3，$\delta$-结构给出这样的 $\phi$。反向，若有 Frobenius lift $\phi$，令
$$
\delta(x)=\frac{\phi(x)-x^p}{p}.
$$
因为 $A$ 无 $p$-torsion，除以 $p$ 的结果唯一。$\phi$ 的加法性给出定义 1.1 的加法恒等式，$\phi$ 的乘法性给出乘法恒等式。两边计算与命题 1.3 的证明完全相反。证毕。

## 1.2 $\delta$-环中的 distinguished element

**定义 1.5.** 令 $A$ 为 $\delta$-环。元素 $d\in A$ 称为 distinguished，如果 $\delta(d)\in A^\times$。

**命题 1.6.** 若 $d$ 是 distinguished element，则在商环 $A/(d)$ 中，$p$ 属于由 $\phi(d)$ 的像生成的理想。更精确地，
$$
\phi(d)=d^p+p\delta(d)
$$
推出
$$
p\equiv \delta(d)^{-1}\phi(d)\pmod d.
$$

**证明.** 由 Frobenius lift 的定义有 $\phi(d)=d^p+p\delta(d)$。模 $(d)$ 后 $d^p$ 消失，因此
$$
\phi(d)\equiv p\delta(d)\pmod d.
$$
由于 $\delta(d)$ 是单位，乘以其逆元即得结论。证毕。

**警告 1.7.** Distinguished element 的定义依赖所选 $\delta$-结构。相同的底层环配上不同 Frobenius lift 时，distinguished elements 的集合可能改变。

## 1.3 Witt vectors 与初始 $\delta$-环

**外部输入定理 1.8.** 对任意 $\mathbf F_p$-代数 $R$，$p$-typical Witt vector 环 $W(R)$ 带自然 Frobenius 和 Verschiebung。若 $R$ 完美，则 Witt Frobenius 为同构，且 $W(R)$ 是 $p$-torsionfree、$p$-adically complete 的 $\delta$-环。

本书使用该定理作为 Witt vectors 的基础输入，不在正文中重建 Witt 多项式的完整理论。

**定义 1.9.** 若 $R$ 是 characteristic $p$ 的完美环，则把 $W(R)$ 上由 Witt Frobenius 给出的 $\delta$-结构记作标准 $\delta$-结构。

**例 1.10.** 取完美域 $k$。则 $W(k)$ 是 $p$-torsionfree 的 $p$-完备 $\delta$-环，Frobenius lift 是 Witt vector Frobenius。理想 $(p)$ 生成 Cartier divisor，且
$$
p\in(p)+\phi(p)W(k)
$$
由 $p\in(p)$ 成立。因此 $(W(k),(p))$ 将在第二章给出 crystalline prism 的基本例子。

## 1.4 Perfectoid interface

**定义 1.11.** 令 $R$ 为 $p$-adic 完备环。其 tilt 记为
$$
R^\flat=\varprojlim_{x\mapsto x^p}R/p.
$$
若 $R$ 是 perfectoid ring，则定义
$$
A_{\inf}(R)=W(R^\flat).
$$
存在 Fontaine map
$$
\theta:A_{\inf}(R)\to R.
$$

**警告 1.12.** 本书不把 perfectoid ring 的所有等价定义作为基础定义。Perfectoid 条件、tilting equivalence 和 Fontaine map 的基本性质作为外部输入使用。后续只在需要构造 perfect prism 例子时调用。

**外部输入定理 1.13.** 若 $R$ 是 perfectoid ring，则 $\ker(\theta)$ 是 principal ideal，由 distinguished nonzerodivisor 生成，并且
$$
(A_{\inf}(R),\ker\theta)
$$
是 perfect prism。反之，perfect prism 与 perfectoid rings 在适当范畴中等价。

**说明 1.14.** 该定理是 prismatic theory 把 perfectoid geometry 向非完美对象“去完美化”的入口。本书后续不会用它证明 prism 的一般定义，而只用它识别 perfect prism 例子和 $A_{\inf}$ specialization。

## 1.5 Breuil-Kisin 型 Frobenius lift

**定义 1.15.** 令 $K/\mathbf Q_p$ 为完全离散赋值域，剩余域 $k$ 完美，选定 uniformizer $\pi\in\mathcal O_K$。令
$$
\mathfrak S=W(k)[[u]]
$$
带 Frobenius lift，限制在 $W(k)$ 上为 Witt Frobenius，并满足 $\phi(u)=u^p$。若 $E(u)$ 是 $\pi$ 在 $W(k)$ 上的 Eisenstein polynomial，则理想 $(E(u))$ 是 Breuil-Kisin prism 的候选 Cartier divisor。

**命题 1.16.** 在定义 1.15 的情形中，$E(u)$ 是 nonzerodivisor，$\mathfrak S$ 是 $(p,E(u))$-完备，且
$$
\mathfrak S/(E(u))\cong \mathcal O_K.
$$

**证明.** $W(k)[[u]]$ 是二维正则局部环；Eisenstein polynomial 非零且首一，因此不是零因子。$(p,u)$-adic 完备性来自形式幂级数环定义，而 $E(u)$ Eisenstein 给出 $(p,E(u))$-adic topology 与 $(p,u)$-adic topology 的相容性。商同构由 $u\mapsto \pi$ 和 $E(\pi)=0$ 给出。证毕。

**外部输入定理 1.17.** 在定义 1.15 的情形中，$(\mathfrak S,(E(u)))$ 是 prism。更具体地，$E(u)$ 在上述 $\delta$-结构下为 distinguished element。

**说明 1.18.** 第二章会把该对象作为 Breuil-Kisin prism 的标准例子。其 deep arithmetic 内容不在命题 1.16，而在 distinguished 条件与后续 Galois representation 分类中。

## 1.6 $\delta$-计算的低阶公式

**命题 1.19.** 在任意 $\delta$-环 $A$ 中，
$$
\delta(x^2)=2x^p\delta(x)+p\delta(x)^2.
$$

**证明.** 由乘法公式取 $y=x$：
$$
\delta(x^2)=x^p\delta(x)+x^p\delta(x)+p\delta(x)\delta(x).
$$
合并同类项即得。证毕。

**命题 1.20.** 若 $A$ 是 $p$-torsionfree，$\phi$ 为对应 Frobenius lift，则对任意 $n\ge1$，
$$
\delta(x^n)=\frac{\phi(x)^n-x^{np}}{p}.
$$

**证明.** 在 $p$-torsionfree 情形中 $\phi(x)=x^p+p\delta(x)$，且 $\delta(z)=(\phi(z)-z^p)/p$。令 $z=x^n$，并用 $\phi$ 是环同态：
$$
\delta(x^n)=\frac{\phi(x^n)-(x^n)^p}{p}
=\frac{\phi(x)^n-x^{np}}{p}.
$$
证毕。

**例 1.21.** 当 $n=p$ 时，
$$
\delta(x^p)=\frac{(x^p+p\delta(x))^p-x^{p^2}}{p}.
$$
展开可见右侧每项均含足够 $p$-幂，因此该公式在 $p$-adic 完备计算中控制 Frobenius 迭代的可整性。

## 本章小结

本章建立了 $\delta$-环的可检查公理，证明了 $\delta$-结构与 Frobenius lift 的基本关系，说明 distinguished element 如何产生 prism 条件中的 $p\in I+\phi(I)A$，并固定了 Witt vectors、perfectoid interface 和 Breuil-Kisin 型 Frobenius lift 的符号。

## 练习

**练习 1.1.** 直接验证定义 1.1 中 $\delta(-x)$ 的表达式，并写成 $\delta(x)$ 与 $x$ 的多项式。

**练习 1.2.** 设 $A$ 为 $p$-torsionfree 环，$\phi$ 为 Frobenius lift。用命题 1.4 的公式验证 $\delta(x^n)$ 对 $n=2,3$ 的表达式。

**练习 1.3.** 令 $A=\mathbf Z_p[[q-1]]$，$\phi(q)=q^p$。计算 $[p]_q=(q^p-1)/(q-1)$ 在 $q=1$ 处的像，并解释为什么它是 $q$-crystalline prism 的候选 distinguished element。
