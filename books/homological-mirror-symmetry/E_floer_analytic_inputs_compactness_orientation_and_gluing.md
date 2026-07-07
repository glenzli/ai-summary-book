# 附录 E：Floer analytic inputs、compactness、orientation 与 gluing

## E.1 分析输入清单

**清单 E.1.** 任何使用 holomorphic curve 计数的 Fukaya theory 都必须处理：

1. Gromov compactness；
2. transversality 或 virtual perturbation；
3. orientation lines 和 signs；
4. gluing theorem；
5. energy/action estimates；
6. bubbling analysis；
7. continuation maps 与选择独立性。

## E.2 Exact 情况

**命题 E.2.** 在 exact Lagrangian 边界条件下，非平凡 holomorphic disk bubble 的 symplectic area 与边界 primitive 的积分相等，因此在适当 exact 假设下被排除。

**证明.** 若 $u:(D,\partial D)\to(M,L)$，则
$$
\int_D u^\ast\omega=\int_D u^\ast d\lambda=\int_{\partial D}u^\ast\lambda
=\int_{\partial D}d(f_L\circ u)=0.
$$
holomorphic curve energy 非负，面积为零时曲线常值。证毕。

## E.3 Orientation

**定义 E.3.** 交点 $p\in L_0\cap L_1$ 的 orientation line 是相应 Cauchy-Riemann operator determinant line 的一维 $k$-向量空间。Spin/Pin 结构用于使这些 determinant lines 的相干取向可选。

**警告 E.4.** 若忽略 orientation lines，$\mu^d$ 的符号无法严格确定。只在 $\mathbb Z/2$ 系数下可以暂时避开部分符号问题。

## E.4 Gluing

**外部输入定理 E.5.** 一维 Floer/polygon moduli spaces 的紧化边界由 broken curves 组成，并且 gluing theorem 给出边界局部结构与 broken configurations 的对应。

**解释 E.6.** $A_\infty$ 方程的几何证明正是“紧一维定向流形边界计数为零”在 moduli spaces 上的应用。

## 本附录小结

Floer theory 的分析部分是 HMS 中最大的外部输入之一。本书正文只在 exact 或已标明外部输入的条件下使用 holomorphic curve 计数，不把 regularity、orientation 和 gluing 当作形式事实。
