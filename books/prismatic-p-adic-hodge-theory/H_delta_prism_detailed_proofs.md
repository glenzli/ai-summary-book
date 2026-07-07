# 附录 H：$\delta$-环与 prism 的逐项证明

## 本附录目标

本附录补齐第一、二章中最常用的代数计算。它把 $\delta$-恒等式、Frobenius lift、distinguished element、boundedness 和 prism morphism 的细节集中到一个可查位置。

## H.1 $\delta$-恒等式的整数性

**命题 H.1.** 多项式
$$
C_p(X,Y)=\frac{X^p+Y^p-(X+Y)^p}{p}
$$
属于 $\mathbf Z[X,Y]$。

**证明.** 由二项式定理，
$$
(X+Y)^p=X^p+Y^p+\sum_{i=1}^{p-1}\binom piX^iY^{p-i}.
$$
对 $1\le i\le p-1$，素数 $p$ 整除 $\binom pi$。故
$$
C_p(X,Y)=-\sum_{i=1}^{p-1}\frac{1}{p}\binom piX^iY^{p-i}
$$
有整数系数。证毕。

## H.2 Frobenius lift 的等价

**命题 H.2.** 若 $A$ 是 $p$-torsionfree，则 $\delta$-结构与 Frobenius lift $\phi:A\to A$ 一一对应。

**证明.** 第一章命题 1.4 已给出证明框架。这里补乘法恒等式的反向计算。给定 Frobenius lift $\phi$，定义
$$
\delta(x)=\frac{\phi(x)-x^p}{p}.
$$
则
$$
\begin{aligned}
p\delta(xy)
&=\phi(xy)-x^py^p\\
&=\phi(x)\phi(y)-x^py^p\\
&=(x^p+p\delta(x))(y^p+p\delta(y))-x^py^p\\
&=p x^p\delta(y)+p y^p\delta(x)+p^2\delta(x)\delta(y).
\end{aligned}
$$
由于 $A$ 无 $p$-torsion，可消去 $p$，得
$$
\delta(xy)=x^p\delta(y)+y^p\delta(x)+p\delta(x)\delta(y).
$$
加法同理。证毕。

## H.3 Distinguished element 与 prism 条件

**命题 H.3.** 令 $A$ 为 $\delta$-环，$d\in A$ 为 distinguished element。则
$$
p\in(d,\phi(d)).
$$

**证明.** 因为 $d$ distinguished，$\delta(d)$ 是单位。由
$$
\phi(d)=d^p+p\delta(d)
$$
得到
$$
p=\delta(d)^{-1}\phi(d)-\delta(d)^{-1}d^p.
$$
右侧第一项在 $(\phi(d))$ 中，第二项在 $(d)$ 中。故 $p\in(d,\phi(d))$。证毕。

**推论 H.4.** 若 $d$ 是 nonzerodivisor，$A$ derived $(p,d)$-complete，且 $d$ distinguished，则 $(A,(d))$ 满足 prism 定义中除 boundedness 之外的条件。

**证明.** Nonzerodivisor 给出 Cartier divisor；derived completeness 为假设；命题 H.3 给出 $p\in(d,\phi(d))=(d)+\phi((d))A$。证毕。

## H.4 Boundedness 的基本检查

**命题 H.5.** 若 $p^N(A/I)=0$，则 $(A/I)[p^\infty]$ 有界。

**证明.** 对任意 $x\in A/I$，有 $p^Nx=0$，故 $x\in(A/I)[p^N]$。于是
$$
(A/I)[p^\infty]=A/I=(A/I)[p^N].
$$
证毕。

**推论 H.6.** Crystalline prism $(A,(p))$ 自动 bounded。

**证明.** 此时 $A/I=A/p$，由 $p(A/p)=0$，命题 H.5 对 $N=1$ 适用。证毕。

## H.5 Prism morphism 的强制相容

**命题 H.7.** 若 $f:(A,I)\to(B,J)$ 是 $\delta$-环态射并满足 $f(I)\subset J$，则 $f(\phi_A(I))\subset\phi_B(J)B$。

**证明.** 对 $x\in I$，有 $f(x)\in J$。由 $\delta$-相容，
$$
f(\phi_A(x))=\phi_B(f(x)).
$$
右侧属于 $\phi_B(J)$。故包含成立。证毕。

## 本附录小结

Prism 条件的核心计算只有两类：$\delta$-结构等价于 Frobenius lift，以及 distinguished element 强迫 $p\in(d,\phi(d))$。后续所有基本例子都应回到这两类计算。

## 练习

**练习 H.1.** 对 $p=2,3$ 写出 $C_p(X,Y)$。

**练习 H.2.** 在命题 H.2 的反向证明中补全加法恒等式。

**练习 H.3.** 设 $d$ distinguished，证明 $\phi(d)$ 在 $A/(d)$ 中与 $p$ 只差一个单位倍。

