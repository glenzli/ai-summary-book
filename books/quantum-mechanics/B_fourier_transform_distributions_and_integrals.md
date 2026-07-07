# 附录 B：Fourier 变换、分布与常用积分

## 本章目标

本附录固定 Fourier 变换规范，并列出动量表象、delta 分布和 Gaussian 积分的常用公式。

## 依赖前置知识

需要多元积分、复指数和基本分布记号。

## B.1 Fourier 变换

**定义 B.1.** 本书采用
$$
\widehat f(p)=\frac1{(2\pi)^{d/2}}\int_{\mathbb R^d}e^{-ip\cdot x}f(x)\,dx,
$$
反变换为
$$
f(x)=\frac1{(2\pi)^{d/2}}\int_{\mathbb R^d}e^{ip\cdot x}\widehat f(p)\,dp.
$$

**命题 B.2.** 在 Schwartz 空间上，
$$
\widehat{-i\partial_j f}(p)=p_j\widehat f(p).
$$

**证明.** 分部积分得
$$
\widehat{\partial_jf}(p)=\frac1{(2\pi)^{d/2}}\int e^{-ip\cdot x}\partial_jf(x)\,dx
=ip_j\widehat f(p),
$$
边界项因 Schwartz 衰减为零。乘以 $-i$ 得结论。$\square$

## B.2 Delta 分布

**定义 B.3.** delta 分布由
$$
\int \delta(x-a)f(x)\,dx=f(a)
$$
刻画。形式恒等式
$$
\frac1{2\pi}\int_{\mathbb R}e^{ip(x-y)}\,dp=\delta(x-y)
$$
按分布意义理解。

## B.3 Gaussian 积分

**公式 B.4.** 若 $a>0$，
$$
\int_{\mathbb R}e^{-ax^2}\,dx=\sqrt{\frac\pi a}.
$$
振荡版本通过解析延拓或阻尼极限解释。

## 本章小结

Fourier 变换把动量算子变为乘法算子，是自由粒子、散射和路径积分计算的基础。delta 分布和 Gaussian 积分必须按分布或极限意义使用。

## 练习

**练习 B.1.** 验证上述 Fourier 规范下 Plancherel 公式的形式。

**练习 B.2.** 用 Fourier 变换求自由 Schrodinger 方程的形式解。

