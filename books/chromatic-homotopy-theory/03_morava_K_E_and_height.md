# 第三章：Morava K/E theories 与高度

## 本章目标

本章把第二章的形式群高度转化为两个核心谱：Morava K-theory $K(n)$ 和 Morava E-theory $E_n$。前者像高度 $n$ 的残差域，后者像高度 $n$ 形式群的完备局部变形环。

## 依赖前置知识

需要第二章的复定向、$BP$、$K(n)$ 和形式群高度。Lubin-Tate deformation theory、Goerss-Hopkins-Miller theorem 和 Morava stabilizer group action 作为外部输入。

## 3.1 Morava K-theory 的检测角色

**定义 3.1.** 对 $n\ge1$，Morava K-theory $K(n)$ 是系数为
$$
K(n)_*\cong \mathbb F_p[v_n^{\pm1}]
$$
的复定向同调理论，其关联形式群在代数闭包上高度为 $n$。约定 $K(0)=H\mathbb Q$。

**命题 3.2.** 若 $X\simeq 0$，则对所有 $n\ge0$，$K(n)_*X=0$。

**证明.** $K(n)_*X=\pi_*(K(n)\otimes X)$。若 $X\simeq0$，则 $K(n)\otimes X\simeq0$，同伦群为零。证毕。

**外部输入定理 3.3 (Morava K 检测有限谱).** 若 $X$ 是有限 $p$-局部谱且对所有 $n\ge0$ 有 $K(n)_*X=0$，则 $X\simeq0$。

**说明.** 这是 nilpotence/thick subcategory 体系的推论。它对任意非有限谱不按此形式成立，因此后续使用时必须检查 $X$ 是否有限。

**例 3.4.** 若 $X$ 是非零有限 torsion $p$-local spectrum，则 $K(0)_*X=H\mathbb Q_*X=0$，所以 type 至少为 $1$。它是否 type $1$ 取决于 $K(1)_*X$ 是否非零。

## 3.2 Morava E-theory

**定义 3.5.** 设 $k$ 是特征 $p$ 的完美域，$\Gamma$ 是 $k$ 上高度 $n$ 的一维形式群。Lubin-Tate deformation ring 为
$$
R_\Gamma\cong W(k)[[u_1,\ldots,u_{n-1}]].
$$
对应的 Morava E-theory $E_\Gamma$ 是偶周期谱，其 homotopy groups 为
$$
(E_\Gamma)_*\cong R_\Gamma[u^{\pm1}],\qquad |u|=2.
$$
若 $k=\mathbb F_{p^n}$ 且 $\Gamma$ 取标准高度 $n$ 形式群，本书简记为 $E_n$。

**外部输入 3.6 (Lubin-Tate).** 高度 $n$ 形式群的 deformation functor 由 $W(k)[[u_1,\ldots,u_{n-1}]]$ pro-represent。完整证明属于形式群变形理论。

**外部输入 3.7 (Goerss-Hopkins-Miller).** $E_n$ 可提升为 $\mathbb E_\infty$-ring spectrum，且 extended Morava stabilizer group $\mathbb G_n$ 的作用可在 $\mathbb E_\infty$ 层面实现。

**警告 3.8.** 仅给出系数环 $W(k)[[u_i]][u^{\pm1}]$ 不足以构造 $\mathbb E_\infty$-ring spectrum，也不足以构造 $\mathbb G_n$-action。这些是高度非平凡的 obstruction theory 结果。

## 3.3 $K(n)$ 与 $E_n$ 的关系

**定义 3.9.** 设 $\mathfrak m=(p,u_1,\ldots,u_{n-1})\subset (E_n)_0$。则 $E_n$ 的 residue theory 满足
$$
(E_n)_*/\mathfrak m\cong \mathbb F_{p^n}[u^{\pm1}].
$$
它与 $K(n)$ 的 Bousfield 类相同。

**外部输入 3.10.** $\langle E_n/\mathfrak m\rangle=\langle K(n)\rangle$。这里的 $E_n/\mathfrak m$ 需要作为 iterated cofiber 或 quotient module spectrum 构造，Bousfield 等价是 Morava theory 标准输入。

**命题 3.11.** 若 $X$ 是 $K(n)$-acyclic，则 $E_n/\mathfrak m\otimes X\simeq0$。

**证明.** 由外部输入 3.10，$K(n)$ 与 $E_n/\mathfrak m$ Bousfield 等价。按 Bousfield 等价定义，$K(n)\otimes X\simeq0$ 当且仅当 $(E_n/\mathfrak m)\otimes X\simeq0$。证毕。

## 3.4 高度作为局部几何

**定义 3.12.** 形式群模栈 $\mathcal M_{fg}$ 的高度 $n$ 点，在本书中作为如下启发式但可精确化的数据包使用：特征 $p$ 域 $k$、$k$ 上高度 $n$ 的形式群 $\Gamma$、以及其 automorphism group。$E_n$ 是这个点的形式邻域的谱实现。

**警告 3.13.** “谱是 $\mathcal M_{fg}$ 上的 sheaf”是现代导出代数几何中的强表述。基础章节只使用复定向诱导的形式群律和 Landweber exactness；sheaf-theoretic 解释进入后续几何章节。

**解释 3.14（结构图像，非定理）.** 高度 $0$、高度 $1$ 和一般高度的
第一近似分别对应有理同调、$p$-adic K-theory 型信息和 Morava
theories。

高度 $0$ 对应特征 $0$ 情形，局部化由 $H\mathbb Q$ 检测；高度 $1$
的典型形式群是乘法形式群，其周期性反映在复 K-theory 的 Bott 元素
中；高度 $n$ 的残差信息由 $K(n)$ 检测，形式邻域由 Lubin--Tate/Morava
$E_n$ 描述。这里没有一个单独、无附加数据的“对应定理”：严格使用时
必须分别调用 Quillen、Landweber、Lubin--Tate 和 Morava theory 输入，
本段不进入后续证明链。

## 3.5 Morava Kunneth 性质

**外部输入定理 3.15（Morava Künneth，CHT-P1-18）.** 对每个
$n\ge1$ 及任意 $X,Y\in\mathbf{Sp}_{(p)}$，自然外积给出同构
$$
K(n)_*(X\otimes Y)\cong K(n)_*X\otimes_{K(n)_*}K(n)_*Y
$$
；没有有限性、connectivity 或 dualizability 假设。来源定位为
Hopkins--Smith II, Proposition 1.5；其 Proposition 1.4 给出
$K(n)\otimes X$ 分解为若干悬挂 $K(n)$ 的 field-spectrum 版本。高度
$0$ 的对应式是普通有理 Künneth 同构。

**命题 3.16.** 若 $K(n)_*X=0$，则对任意谱 $Y$ 有 $K(n)_*(X\otimes Y)=0$。

**证明.** 用外部输入 3.15：
$$
K(n)_*(X\otimes Y)\cong K(n)_*X\otimes_{K(n)_*}K(n)_*Y=0.
$$
证毕。

**警告 3.17.** 命题 3.16 只说明 $K(n)$-acyclics 是 tensor ideal。它不说明 $X=0$，也不说明 $X$ 在其他高度不可见。

## 3.6 Residue field 与 completed local ring

**定义 3.18.** Morava E-theory 的系数环
$$
(E_n)_0=W(k)[[u_1,\ldots,u_{n-1}]]
$$
是 complete local ring，极大理想为
$$
\mathfrak m=(p,u_1,\ldots,u_{n-1}).
$$
其 residue field 是 $k$。

**命题 3.19.** 若 $M$ 是有限生成 $(E_n)_0$-模且 $M/\mathfrak m M=0$，则 $M=0$。

**证明.** 这是 Nakayama lemma。因为 $(E_n)_0$ 是 local ring，有限生成模 $M$ 若满足 $M=\mathfrak m M$，则 $M=0$。证毕。

**警告 3.20.** Nakayama lemma 只适用于有限生成模。对一般 $(E_n)_*X$，不能只由 mod $\mathfrak m$ 消失推出整体消失，除非有完备性和有限性假设。

## 本章小结

$K(n)$ 是高度 $n$ 的残差检测器，$E_n$ 是高度 $n$ 的完备局部变形对象。二者通过 residue field/Bousfield 类联系起来。$E_n$ 的 $\mathbb E_\infty$ 结构和 stabilizer group 作用是外部输入，后续 $K(n)$-local descent 全部依赖这些结构。

## 练习

**练习 3.1.** 证明若两个谱 Bousfield 等价，则它们定义相同的 acyclic 类。

**练习 3.2.** 写出 $n=1$ 时 $E_1$ 的系数环，并说明它为什么与 $p$-adic K-theory 有关。

**练习 3.3.** 解释为什么 $E_n/\mathfrak m$ 的系数域是 $\mathbb F_{p^n}$ 上的 Laurent 多项式环，而 $K(n)_*$ 通常写作 $\mathbb F_p[v_n^{\pm1}]$。它们的 Bousfield 类为什么仍可相同？
