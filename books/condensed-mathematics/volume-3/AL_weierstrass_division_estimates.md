# 附录 AL：Weierstrass 除法的估计形式

## AL.0 目标

附录 AF 使用 Weierstrass division 推出 Oka coherence。本附录补充 division theorem 中被压缩的分析部分：在 Banach 空间上如何得到商、余数和收敛估计。

本附录不把多复变分析的全部细节重写为自足教材；它给出可核对的 Banach 估计框架，并把唯一性、连续依赖和 coherence 中需要的有限生成结论分开证明。

## AL.1 Banach 幂级数范数

设 \(z=(z',w)\)，其中 \(z'=(z_1,\ldots,z_{n-1})\)。取正实数

$$
r=(r_1,\ldots,r_{n-1}),\qquad \rho>0.
$$

记 \(A(r,\rho)\) 为在闭多圆盘

$$
\overline D(r,\rho)=\{|z_i|\le r_i,\ |w|\le\rho\}
$$

的邻域上全纯，并在该闭多圆盘上连续的函数构成的 Banach 代数，范数为

$$
\|g\|_{r,\rho}=\sup_{\overline D(r,\rho)}|g|.
$$

对

$$
g(z',w)=\sum_{k\ge0}g_k(z')w^k
$$

定义截断算子

$$
R_d(g)=\sum_{0\le k<d}g_k(z')w^k,\qquad
H_d(g)=\sum_{k\ge d}g_k(z')w^{k-d}.
$$

在收敛幂级数 germ 上有恒等式

$$
g=R_d(g)+w^dH_d(g).
$$

**引理 AL.1（截断估计）.** 对任意 \(0<\rho'<\rho\)，存在常数 \(C=C(d,\rho',\rho)\)，使得

$$
\|R_d(g)\|_{r,\rho'}+\|H_d(g)\|_{r,\rho'}\le C\|g\|_{r,\rho}.
$$

**证明.** 对每个 \(z'\) 固定后，用一变量 Cauchy 积分公式

$$
g_k(z')=\frac{1}{2\pi i}\int_{|\zeta|=\rho}
\frac{g(z',\zeta)}{\zeta^{k+1}}\,d\zeta
$$

得到 \(\|g_k\|_r\le \rho^{-k}\|g\|_{r,\rho}\)。于是

$$
\|R_d(g)\|_{r,\rho'}\le
\sum_{k<d}(\rho'/\rho)^k\|g\|_{r,\rho}.
$$

对 \(H_d\)，在 \(|w|\le\rho'\) 上估计

$$
\sum_{k\ge d}\|g_k\|_r |w|^{k-d}
\le
\rho^{-d}\sum_{m\ge0}(\rho'/\rho)^m\|g\|_{r,\rho}.
$$

右端有限。证毕。

## AL.2 除法算子的收缩形式

设

$$
f=w^d+a_{d-1}(z')w^{d-1}+\cdots+a_0(z')
$$

是 distinguished polynomial，并令

$$
a=f-w^d.
$$

在 germ 层面，欲求

$$
g=qf+r,\qquad \deg_w r<d.
$$

把等式写为

$$
g=qw^d+qa+r.
$$

应用 \(H_d\) 得到

$$
H_d(g)=q+H_d(qa).
$$

因此商 \(q\) 应满足不动点方程

$$
q=H_d(g)-H_d(qa).
$$

**输入定理 AL.2（Banach Weierstrass 估计）.** 经过缩小 \(r\) 和选择 \(0<\rho'<\rho\)，算子

$$
\Phi(q)=H_d(qa)
$$

在 \(A(r,\rho')\) 上有算子范数

$$
\|\Phi\|<1.
$$

更具体地，因 \(a_i(0)=0\)，可缩小 \(r\) 使 \(\sum_i\|a_i\|_r(\rho')^i\) 足够小；再由 AL.1 控制 \(H_d\) 的范数。

**定理 AL.3（估计形式的 Weierstrass division）.** 在 AL.2 的半径选择下，对每个 \(g\in A(r,\rho)\)，存在唯一

$$
q\in A(r,\rho'),\qquad r_0\in A(r)[w]_{<d},
$$

使

$$
g=qf+r_0
$$

在 \(\overline D(r,\rho')\) 上成立，并且存在常数 \(C_f\)，满足

$$
\|q\|_{r,\rho'}+\|r_0\|_{r,\rho'}\le C_f\|g\|_{r,\rho}.
$$

**证明.** 由 AL.2，\(I+\Phi\) 在 Banach 空间上可逆，逆为 Neumann 级数

$$
(I+\Phi)^{-1}=\sum_{m\ge0}(-\Phi)^m.
$$

令

$$
q=(I+\Phi)^{-1}H_d(g).
$$

再定义

$$
r_0=g-qf.
$$

由方程 \(H_d(g)=q+H_d(qa)\) 得

$$
H_d(g-qa)=q.
$$

因此

$$
H_d(r_0)=H_d(g-qa-qw^d)=0,
$$

所以 \(r_0=R_d(r_0)\) 是 \(w\)-次数小于 \(d\) 的多项式。范数估计来自 AL.1、\(\|(I+\Phi)^{-1}\|\le(1-\|\Phi\|)^{-1}\) 和 Banach 代数乘法连续性。

唯一性如下。若

$$
qf+r=0,\qquad \deg_w r<d,
$$

则 \(q=-H_d(qa)\)，即 \((I+\Phi)q=0\)。由于 \(I+\Phi\) 可逆，\(q=0\)，进而 \(r=0\)。证毕。

## AL.3 连续依赖与关系模有限生成

**推论 AL.4（商与余数连续依赖）.** 映射

$$
g\mapsto q,\qquad g\mapsto r_0
$$

是连续线性映射。

**证明.** 商映射为 \((I+\Phi)^{-1}H_d\)，由 AL.1 与 AL.2 是连续线性算子。余数映射 \(g\mapsto g-qf\) 是连续线性算子组合。证毕。

**命题 AL.5（有限关系提升的估计口径）.** 设 \(M\) 是 \(\mathbb C\{z',w\}/(f)\)-有限模。若 \(M\) 作为 \(\mathbb C\{z'\}\)-模由有限组 \(e_1,\ldots,e_N\) 生成，则这些生成元之间的 \(\mathbb C\{z',w\}\)-关系由有限组关系生成。

**证明.** 由于 \(w\) 在 \(M\) 上由一个 \(\mathbb C\{z'\}\)-线性端omorphism \(T\) 表示，任意 \(\mathbb C\{z',w\}\)-线性组合

$$
\sum_j b_j(z',w)e_j
$$

可用 AL.3 把每个 \(b_j\) 除以 \(f\)，余数次数 \(<d\)。因此关系由有限个 \(\mathbb C\{z'\}\)-系数函数控制。

设

$$
\psi:\mathbb C\{z'\}^{Nd}\to M
$$

把标准基送到 \(w^ke_j\)。关系模为 \(\ker\psi\)。若 \(\mathbb C\{z'\}\) coherent，则 \(\ker\psi\) 有限生成。把 \(w\)-作用矩阵 \(T\) 加入关系组，所得有限组生成全部 \(\mathbb C\{z',w\}\)-关系。证毕。

## AL.4 Oka coherence 中的使用位置

**定理 AL.6（除法估计到 coherence 的局部闭合步骤）.** 假设 \(\mathbb C\{z'\}\) coherent。则对 distinguished polynomial \(f\)，商环

$$
\mathbb C\{z',w\}/(f)
$$

作为 \(\mathbb C\{z',w\}\)-模 coherent。

**证明.** 由 AL.3，每个类有唯一次数 \(<d\) 的代表，所以商环作为 \(\mathbb C\{z'\}\)-模由 \(1,w,\ldots,w^{d-1}\) 生成。用 AL.5 控制有限生成元之间的关系，得到有限 presentation。coherent 的定义要求有限自由模态射的 kernel 有限生成；这由 \(\mathbb C\{z'\}\) coherent 和 AL.5 给出。证毕。

**边界 AL.7.** Oka coherence 的完整归纳还需要把任意非零理想中的元素化为 distinguished polynomial，并证明模 \(f\) 的有限关系可提升到原关系。AL.3-AL.6 只处理 division theorem 提供后的 Banach 估计和商环 coherence 步骤。

## 练习

1. 在 \(f=w^d\) 情形下，直接写出 \(\Phi\)、\(q\) 和 \(r_0\)。
2. 证明 AL.1 中 \(H_d\) 的估计常数可取
   $$
   C=\rho^{-d}(1-\rho'/\rho)^{-1}.
   $$
3. 说明为什么 \(a_i(0)=0\) 允许缩小 \(r\) 使 AL.2 中的算子范数小于一。
4. 在 AL.5 中写出 \(w\)-作用矩阵 \(T\) 给出的关系方程。
