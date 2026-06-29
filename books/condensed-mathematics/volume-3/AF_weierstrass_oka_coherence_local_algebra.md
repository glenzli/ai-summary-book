# 附录 AF：Weierstrass 与 Oka coherence 的局部代数

## AF.0 目标

附录 AB 把 Oka coherence 作为输入。本附录把其局部代数证明拆得更细：Weierstrass division 如何控制关系，为什么 $\mathcal O$ 的有限生成关系 sheaf 仍有限生成。

完整证明仍依赖 Weierstrass preparation/division。本附录证明从 division theorem 到 coherence 的归纳骨架。

## AF.1 Distinguished power series

设

$$
R_{n-1}=\mathbb C\{z_1,\ldots,z_{n-1}\}.
$$

**定义 AF.1.** 元素

$$
f\in R_{n-1}\{z_n\}
$$

称为关于 $z_n$ 的 distinguished polynomial，若

$$
f=z_n^d+a_{d-1}z_n^{d-1}+\cdots+a_0,
$$

且每个 $a_i\in R_{n-1}$ 在原点消失。

**输入定理 AF.2（Weierstrass division）.** 若 $f$ 是 distinguished polynomial，则对每个

$$
g\in\mathbb C\{z_1,\ldots,z_n\}
$$

存在唯一

$$
q\in\mathbb C\{z_1,\ldots,z_n\},\qquad
r\in R_{n-1}[z_n],\quad \deg_{z_n}r<d,
$$

使

$$
g=qf+r.
$$

## AF.2 有限模的关系控制

**命题 AF.3.** 设 $R=\mathbb C\{z_1,\ldots,z_n\}$，$f\in R$ 是 distinguished polynomial。则 $R/(f)$ 作为 $R_{n-1}$-模由

$$
1,z_n,\ldots,z_n^{d-1}
$$

生成。

**证明.** 对 $g\in R$ 用 AF.2 写 $g=qf+r$，其中 $\deg_{z_n}r<d$。在 $R/(f)$ 中，$g$ 的类等于 $r$ 的类，而 $r$ 是 $1,\ldots,z_n^{d-1}$ 的 $R_{n-1}$-线性组合。证毕。

**命题 AF.4.** 若 $R_{n-1}$ coherent，则 $R/(f)$ 作为 $R$-模 coherent。

**证明.** 由 AF.3，$R/(f)$ 作为 $R_{n-1}$-有限模。任意有限个生成元之间的关系可看作 $R_{n-1}$-有限模之间的 kernel；由 $R_{n-1}$ coherent，该 kernel 有限生成。再把 $R$-作用通过 $z_n$ 的 companion matrix 表示，关系仍由有限个 $R$-关系生成。证毕。

## AF.3 Oka coherence 归纳骨架

**输入定理 AF.5（Noether normalization for convergent power series，局部形式）.** 对 $R=\mathbb C\{z_1,\ldots,z_n\}$ 中非零理想 $I$，经过线性坐标变化后，可取 $f\in I$ 为关于 $z_n$ 的 distinguished polynomial。

**定理 AF.6（Oka coherence 归纳骨架）.** 若 $R_{n-1}$ coherent，则 $R_n=\mathbb C\{z_1,\ldots,z_n\}$ coherent。

**证明.** 需证明任意有限自由模态射

$$
R_n^a\to R_n^b
$$

的 kernel 有限生成。局部化到一个非零关系理想后，用 AF.5 取 distinguished polynomial $f$ 控制商。模 $R_n/(f)$ 的 coherence 由 AF.4 和归纳假设给出。再用短正合列

$$
0\to R_n\xrightarrow{\cdot f}R_n\to R_n/(f)\to0
$$

和 Artin-Rees 型有限生成传递，把 modulo $f$ 的有限关系提升为 $R_n$ 上的有限关系。证毕。

**边界 AF.7.** AF.6 的最后一步包含 Oka 原证明中的主要技术：需要证明关系提升过程在收敛幂级数范畴内终止，并保持收敛性。该点依赖 Weierstrass division 的范数估计。

## AF.4 Sheaf coherence

**推论 AF.8.** 复流形 $X$ 上 $\mathcal O_X$ 是 coherent sheaf of rings。

**证明.** coherence 是局部性质。每个点有坐标邻域，其局部环模型为收敛幂级数环。由 AF.6 对 $n$ 归纳，局部关系 sheaf 有限生成。证毕。

## 练习

1. 对 $f=z_n^d$ 写出 AF.2 的商和余数。
2. 证明 AF.3。
3. 解释 AF.6 中为什么需要坐标变化。
4. 指出 AF.7 中收敛性问题与形式幂级数情形的区别。
