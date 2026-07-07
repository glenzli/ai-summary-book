# 附录 E：D-module convention、left/right 转换和 twist

## 本章目标

本附录集中管理 D-module convention，避免 Beilinson-Bernstein localization、Riemann-Hilbert 和 Schubert constructible objects 之间出现 shift 或 variance 错误。

## E.1 Left modules

**约定 E.1.** 本书默认 $\mathcal D_X$-module 指 left $\mathcal D_X$-module。right module 必须写作 right $\mathcal D_X$-module。

**约定 E.2.** left module 的 de Rham functor 使用
$$
\operatorname{DR}_X(\mathcal M)=\Omega_X^\bullet\otimes_{\mathcal O_X}\mathcal M[\dim X].
$$
该 shift 使 $\mathcal O_X$ 对应 $\mathbb C_X[\dim X]$。

## E.2 Right module 转换

**定义 E.3.** left-to-right 转换为
$$
\mathcal M\mapsto\omega_X\otimes_{\mathcal O_X}\mathcal M.
$$
right-to-left 转换为
$$
\mathcal N\mapsto\mathcal Hom_{\mathcal O_X}(\omega_X,\mathcal N).
$$

**命题 E.4.** 若 $X$ 光滑，则上述两个构造在适当 module category 上互为等价。

**证明.** $\omega_X$ 是 invertible $\mathcal O_X$-module，其逆为 $\omega_X^{-1}=\mathcal Hom_{\mathcal O_X}(\omega_X,\mathcal O_X)$。张量 $\omega_X$ 与张量 $\omega_X^{-1}$ 在 $\mathcal O_X$-module 层面互逆。$\mathcal D_X$ 的 right action 由 Lie derivative 和 canonical bundle 的自然 right action 定义；局部坐标检查表明转换后的 action 满足 $\mathcal D_X$ 关系，并且反向转换恢复原 action。$\square$

## E.3 Twisted differential operators

**约定 E.5.** $\mathcal D_\lambda$ 的参数 $\lambda$ 必须与以下三项同时登记：

1. 线丛 $\mathcal L_\lambda$ 的 character convention；
2. Harish-Chandra isomorphism 中是否使用 $\rho$ shift；
3. Beilinson-Bernstein theorem 中 regular dominant 条件的写法。

**警告 E.6.** 不同文献的 $\mathcal D_\lambda$ 可能相差 $\rho$ 或符号。引用定理时不得只写“按通常 convention”。

## 本章小结

本附录锁定本书默认使用 left D-modules 和 shifted de Rham functor。twist convention 仍需在 Beilinson-Bernstein locator 阶段最终固定。

