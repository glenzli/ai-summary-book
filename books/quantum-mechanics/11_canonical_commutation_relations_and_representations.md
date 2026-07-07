# 第十一章：正则对易关系与表示

## 本章目标

本章研究位置、动量、Weyl 关系和正则对易关系的表示问题。

## 依赖前置知识

需要无界算子、Stone 定理、Fourier 变换和酉群。

## 11.1 正则对易关系

**定义 11.1.** 在共同不变稠密定义域 $\mathcal D$ 上，位置和动量算子满足正则对易关系，若
$$
[X_j,P_k]\psi=i\delta_{jk}\psi
$$
对所有 $\psi\in\mathcal D$ 成立。

**例子 11.2.** 在 $\mathcal S(\mathbb R^d)$ 上，
$$
(X_jf)(x)=x_jf(x),\qquad (P_kf)(x)=-i\partial_kf(x)
$$
满足正则对易关系。

**证明.** 计算
$$
X_jP_kf=-ix_j\partial_kf,
$$
而
$$
P_kX_jf=-i\partial_k(x_jf)=-i\delta_{jk}f-ix_j\partial_kf.
$$
相减得 $[X_j,P_k]f=i\delta_{jk}f$。$\square$

## 11.2 Weyl 关系

**定义 11.3.** Weyl 形式的正则对易关系为
$$
e^{-iaP}e^{-ibX}=e^{iab}e^{-ibX}e^{-iaP}
$$
对 $a,b\in\mathbb R$ 成立。

**命题 11.4.** 在 Schrodinger 表示中，Weyl 关系成立。

**证明.** 有
$$
(e^{-iaP}f)(x)=f(x-a),\qquad (e^{-ibX}f)(x)=e^{-ibx}f(x).
$$
因此
$$
(e^{-iaP}e^{-ibX}f)(x)=e^{-ib(x-a)}f(x-a),
$$
而
$$
(e^{-ibX}e^{-iaP}f)(x)=e^{-ibx}f(x-a).
$$
二者相差因子 $e^{iab}$，等价于定义中的关系。$\square$

## 11.3 唯一性

**外部输入定理 11.5（Stone-von Neumann，QM-EXT-4）.** 有限自由度下，满足适当不可约性和强连续性的 Weyl 关系表示酉等价于 Schrodinger 表示。

**边界 11.6.** 无限自由度中 Stone-von Neumann 唯一性失败。这是量子场论出现非等价表示的根源之一，本书不展开该理论。

## 11.4 为什么不能用有界算子满足 CCR

**命题 11.7.** 不存在有界算子 $X,P\in\mathcal B(\mathcal H)$ 满足
$$
[X,P]=iI.
$$

**证明.** 若 $[X,P]=iI$，则由归纳可得
$$
[X,P^n]=inP^{n-1}.
$$
归纳步使用
$$
[X,P^{n+1}]=[X,P^n]P+P^n[X,P].
$$
对两边取范数：
$$
n\|P^{n-1}\|=\|[X,P^n]\|
\le 2\|X\|\,\|P^n\|
\le 2\|X\|\,\|P\|\,\|P^{n-1}\|.
$$
若 $\|P^{n-1}\|\ne0$ 对无限多个 $n$ 成立，则得 $n\le2\|X\|\|P\|$ 对无限多个无界的 $n$ 成立，矛盾。若存在最小 $N\ge1$ 使 $P^N=0$，则把公式用于 $n=N$ 得
$$
0=[X,P^N]=iN P^{N-1},
$$
这与 $P^{N-1}\ne0$ 矛盾。故不可能。$\square$

**说明 11.8.** 该命题解释了为什么位置和动量必须作为无界算子处理。把 $[X,P]=iI$ 写在全 Hilbert 空间上而不谈定义域，是数学上不合法的。

## 本章小结

正则对易关系的微分形式需要定义域控制；Weyl 形式把无界算子关系转化为酉群关系。有限自由度下 Stone-von Neumann 定理给出标准表示的唯一性。

## 练习

**练习 11.1.** 在 $\mathcal S(\mathbb R)$ 上计算 $[X,P^2]$。

**练习 11.2.** 用 Fourier 变换说明动量算子在动量表象中成为乘法算子。
