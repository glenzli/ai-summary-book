# 附录 D：Fourier-Mukai transforms 与导出代数几何接口

## D.1 Kernel 形式

**定义 D.1.** 对光滑适当 schemes $X,Y$，kernel $K\in\operatorname{Perf}(X\times Y)$ 定义 Fourier-Mukai transform
$$
\Phi_K(E)=\mathbf R p_{Y*}(p_X^\ast E\otimes^{\mathbf L}K).
$$

**命题 D.2.** 若 $K\in\operatorname{Perf}(X\times Y)$、$L\in\operatorname{Perf}(Y\times Z)$，则复合 $\Phi_L\circ\Phi_K$ 由 convolution kernel
$$
K\star L=\mathbf R p_{XZ*}(p_{XY}^\ast K\otimes^{\mathbf L}p_{YZ}^\ast L)
$$
给出。

**证明.** 将 $\Phi_L(\Phi_K(E))$ 展开，使用投影公式、base change 和 derived tensor product 的结合性，把两次推出合并为 $X\times Y\times Z\to X\times Z$ 的推出。所得 kernel 正是 $K\star L$。证毕。

## D.2 Adjoints

**外部输入定理 D.3.** 对 smooth proper varieties，Fourier-Mukai transform 的左右伴随仍由显式 dual kernel 给出，涉及相对 dualizing sheaves 和维数 shift。

**解释 D.4.** 在 HMS 中，B-side functor 若由 kernel 给出，其 adjunction、spherical functor 性质和 twist 可通过 kernel convolution 计算。

## D.3 Derived geometry 接口

**定义 D.5.** derived fiber product 用 derived tensor product 修正普通 fiber product 中的 Tor 信息。B-side categories 中的 $\operatorname{Perf}$ 和 kernels 自然适合 derived intersections。

**警告 D.6.** 本书不把 derived algebraic geometry 全部纳入主线。只有当 Fourier-Mukai kernels、matrix factorizations、singularity categories 或 moduli of objects 需要时，才引入相应接口。

## 本附录小结

Fourier-Mukai transforms 是 B-side functors 的标准几何模型。HMS 的 functorial 版本常要求 A-side 几何 functor 与 B-side kernel transform 对应，因此 kernel convolution 和 adjunction 是必须掌握的形式工具。
