# 附录 G：Warsaw flavor/Fierz 计数审计

## G.1 目标

第二十章给出通用 flavor 张量计数。本附录说明如何把通用计数推进到 Warsaw basis 的 exact 三代计数。这里给出可审计算法和关键例子；完整逐项证明仍是出版候选前的高级附表任务。

## G.2 线性关系的来源

Warsaw flavor 计数的减少来自四类关系：

1.  Hermiticity：$C_i=C_{\iota(i)}^\ast$；
2.  相同双线性交换：$(\bar\psi_p\Gamma\psi_r)(\bar\psi_s\Gamma\psi_t)$ 在交换两因子下不变；
3.  Lorentz Fierz：不同 spinor 收缩之间的线性关系；
4.  gauge-index Fierz：$SU(2)$ 或 $SU(3)$ 生成元完备性导致的关系。

## G.3 Burnside 计数模板

设 flavor 指标集合为
$$
I=\{(p,r,s,t):p,r,s,t=1,\ldots,n_g\}.
$$
若存在交换
$$
E(p,r,s,t)=(s,t,p,r),
$$
则独立复张量分量数为 $E$ 在 $I$ 上的轨道数。由 Burnside 引理，
$$
N_E={1\over2}\left(|I|+|I^E|\right)
={1\over2}(n_g^4+n_g^2).
$$
对 $n_g=3$ 得
$$
N_E={81+9\over2}=45.
$$
这解释了同种 current-current 四费米子结构的基本 $45$ 计数。

## G.4 Hermiticity 与交换的合并

Hermitian conjugation 对 flavor 指标作用为
$$
H(p,r,s,t)=(r,p,t,s).
$$
若先对 $E$ 取轨道，再用 $H$ 施加复共轭约束，则实参数数等于 $E$ 轨道数。理由与引理 20.3 相同：$H$ 的不动轨道贡献一个实数，二元轨道贡献一个复数。

## G.5 需要额外审计的扇区

通用计数给出 $2508$，标准 Warsaw 三代 baryon-number conserving exact 计数为 $2499$。差异说明某些结构还有额外线性关系。审计重点不在 bosonic、Yukawa-like 或 dipole 扇区，而在四费米子扇区，特别是含相同 $SU(2)_L$ doublet 或相同颜色结构的 current-current 算符。

出版级证明需要逐项列出：

1.  每个四费米子结构的 flavor 指标集合；
2.  Lorentz Fierz 关系；
3.  $SU(2)$ 完备性
    $$
    \tau^I_{ij}\tau^I_{kl}
    =2\delta_{il}\delta_{kj}-\delta_{ij}\delta_{kl};
    $$
4.  $SU(3)$ 完备性
    $$
    T^A_{\alpha\beta}T^A_{\gamma\delta}
    ={1\over2}\left(
    \delta_{\alpha\delta}\delta_{\gamma\beta}
    -{1\over3}\delta_{\alpha\beta}\delta_{\gamma\delta}
    \right);
    $$
5.  由这些关系生成的矩阵秩。

## G.6 机器无关审计算法

完整计数可按以下算法完成：

1.  生成所有 flavor 分量标签；
2.  为 Hermiticity、交换、Lorentz Fierz、gauge Fierz 写出线性方程；
3.  把复系数拆为实部和虚部；
4.  构造有理数矩阵；
5.  计算矩阵秩；
6.  实参数数等于变量数减去秩；
7.  将每个 operator class 的结果相加。

**原则 G.1.** 若不给出线性方程和矩阵秩，只引用 $2499$ 是文献输入；若给出上述矩阵秩审计，才算本书内部证明。

## G.7 当前状态

本附录把 flavor/Fierz 计数从“只引用总数”推进到“可审计算法”。尚未逐项列出所有四费米子关系矩阵，因此 exact $2499$ 仍保留为外部输入，而不是本书内定理。
