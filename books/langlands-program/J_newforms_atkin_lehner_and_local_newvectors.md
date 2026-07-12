# 附录 J：Newforms、Atkin-Lehner 理论和局部 Newvectors

本附录补充第六至十章使用的 newform theory。目标是说明经典 old/new 分解、局部导子和 adelic newvector 之间的精确接口。完整 Atkin-Lehner-Li 理论和 Casselman newvector theorem 作为外部输入；本附录证明它们如何支撑正文中的级、导子和 Hecke 本征数据。

收口归一化回指：本附录连接 classical level、adelic conductor、local newvector 和局部 L 因子；归一化按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、5、6、8 节处理。

## J.1 Degeneracy Maps 和 Oldforms

设 $M\mid N$，$k\ge2$，$\varepsilon$ 为模 $M$ 的 Dirichlet 特征，并通过自然投影视为模 $N$ 的特征。

**定义 J.1.** 对正整数 $d\mid N/M$，定义 degeneracy operator
$$
\iota_d:S_k(\Gamma_0(M),\varepsilon)\to S_k(\Gamma_0(N),\varepsilon)
$$
为
$$
(\iota_df)(z)=d^{k/2}f(dz)
$$
即按第六章 slash convention 有
$\iota_df=f|_k\begin{pmatrix}d&0\\0&1\end{pmatrix}$。本附录固定这个标量；若另行把 degeneracy map 定义为 $f(dz)$，只改变像的非零标量，不改变 old subspace。

**命题 J.2.** $\iota_d f$ 属于 $S_k(\Gamma_0(N),\varepsilon)$。

**证明.** 令 $\alpha_d=\operatorname{diag}(d,1)$。若
$\gamma=\begin{pmatrix}a&b\\c&e\end{pmatrix}\in\Gamma_0(N)$，则
$$
\alpha_d\gamma\alpha_d^{-1}
=\begin{pmatrix}a&db\\c/d&e\end{pmatrix}.
$$
因为 $d\mid N/M$，有 $c/d\in M\mathbb Z$，故该矩阵属于 $\Gamma_0(M)$；其右下角仍为 $e$。由 slash 右作用和 $\varepsilon$ 从模 $M$ 到模 $N$ 的提升，
$$
(\iota_df)|_k\gamma
=f|_k(\alpha_d\gamma\alpha_d^{-1})|_k\alpha_d
=\varepsilon(e)\iota_df.
$$
所以 $\iota_df$ 具有所需变换律。它全纯，因为 $z\mapsto dz$ 保持上半平面且 $f$ 全纯。

最后取任一 $\Gamma_0(N)$-cusp $c$。矩阵 $\alpha_d$ 把 $c\in\mathbb P^1(\mathbb Q)$ 送到一个
$\Gamma_0(M)$-cusp；在相应 cusp parameters 中，$\iota_d$ 只把局部参数替换为某个正整数次幂并乘非零常数。$f$ 的展开没有常数项，作这种替换后仍没有常数项。因此 $\iota_df$ 在每个 cusp 消失，属于
$S_k(\Gamma_0(N),\varepsilon)$。$\square$

**定义 J.3.** Old subspace 定义为
$$
S_k(\Gamma_0(N),\varepsilon)_{\operatorname{old}}
=
\sum_{\substack{M\mid N,\ M<N\\ d\mid N/M}}
\iota_d S_k(\Gamma_0(M),\varepsilon_M),
$$
其中 $\varepsilon_M$ 运行于能提升为 $\varepsilon$ 的低级 character。New subspace 定义为 Petersson 正交补：
$$
S_k(\Gamma_0(N),\varepsilon)_{\operatorname{new}}
=
S_k(\Gamma_0(N),\varepsilon)_{\operatorname{old}}^\perp.
$$

**外部输入定理 J.4（Atkin-Lehner-Li old/new decomposition）.** 有 Hecke 稳定分解
$$
S_k(\Gamma_0(N),\varepsilon)
=
S_k(\Gamma_0(N),\varepsilon)_{\operatorname{old}}
\oplus
S_k(\Gamma_0(N),\varepsilon)_{\operatorname{new}}
$$
在本附录固定的 Petersson inner product 和 nebentypus convention 下成立。New subspace 有归一化 Hecke eigenform 基；每个 newform 对应导子为 $N$ 的 cuspidal automorphic representation of $\operatorname{GL}_2(\mathbb A_\mathbb Q)$。

**注 J.5.** 第七章把“newform 生成的自守表示”作为外部输入。J.4 是其经典模形式侧来源；局部表示侧来源是 Casselman newvector theorem。

## J.2 Atkin-Lehner Involutions

**定义 J.6.** 设 $Q\mid N$ 且 $(Q,N/Q)=1$。Atkin-Lehner operator $W_Q$ 由任取矩阵
$$
w_Q=\begin{pmatrix}Qa&b\\Nc&Qd\end{pmatrix}
$$
满足 $\det(w_Q)=Q$ 定义，通过
$$
W_Qf=f|_kw_Q
$$
作用于 $S_k(\Gamma_0(N),\varepsilon)$，并带有依赖 nebentypus 的归一化因子。

**外部输入定理 J.7（Atkin-Lehner operators）.** $W_Q$ 在 Atkin-Lehner-Li normalization 下给出 $S_k(\Gamma_0(N),\varepsilon)$ 的自同构，保持 new subspace，并与 Hecke 算子满足标准交换关系。若 $f$ 为 newform，则 $W_Nf$ 为 $f$ 的共轭或 contragredient 相关 newform，特征值进入函数方程的 root number。

**命题 J.8.** 若 $f$ 为权 $k$、级 $N$ newform，则第六章完成 L 函数函数方程中的常数 $\eta_f$ 可由全局 Atkin-Lehner involution 和 nebentypus 归一化表达。

**证明路线（外部输入）.** Mellin 变换把 $f(iy)$ 与 $L(f,s)$ 联系起来。Fricke/Atkin-Lehner 变换 $W_N$ 把 $y$ 替换为 $1/(Ny)$，从而在 Mellin 积分中产生 $s\leftrightarrow k-s$。若 $f$ 是 $W_N$ 的本征向量，所得本征值与 nebentypus 标量共同给出 root number。完整常数依赖 slash operator、nebentypus 与完成因子 convention；本段不给出这些外部输入的完整证明。

## J.3 局部导子和 Newvector

设 $F$ 为非 Archimedean 局部域，$\mathcal O=\mathcal O_F$，$\mathfrak p$ 为极大理想，$\varpi$ 为一致化元。

**定义 J.9.** 对 $m\ge0$，定义
$$
K_0(\mathfrak p^m)=
\left\{
\begin{pmatrix}a&b\\c&d\end{pmatrix}\in\operatorname{GL}_2(\mathcal O):
c\in\mathfrak p^m
\right\},
$$
$$
K_1(\mathfrak p^m)=
\left\{
g\in K_0(\mathfrak p^m):d\equiv1\pmod{\mathfrak p^m}
\right\}.
$$

**定义 J.10.** 设 $\pi$ 为 $\operatorname{GL}_2(F)$ 的 irreducible admissible generic representation。其导子指数 $a(\pi)$ 定义为最小 $m\ge0$，使
$$
\pi^{K_1(\mathfrak p^m)}\ne0
$$
成立。Casselman 定理保证该最小值存在；定义之前不预设它。非零向量
$$
v_{\operatorname{new}}\in\pi^{K_1(\mathfrak p^{a(\pi)})}
$$
称为 local newvector。

**外部输入定理 J.11（Casselman newvector theorem）.** 对 $\operatorname{GL}_2(F)$ 的 irreducible admissible generic representation $\pi$：

1. 最小整数 $a(\pi)$ 存在。
2. Newvector line $\pi^{K_1(\mathfrak p^{a(\pi)})}$ 一维。
3. 若 $m<a(\pi)$，则 $\pi^{K_1(\mathfrak p^m)}=0$。
4. 对 $m\ge a(\pi)$，
   $$
   \dim\pi^{K_1(\mathfrak p^m)}=m-a(\pi)+1.
   $$
5. 固定 conductor 为 $\mathcal O$ 的非平凡加法特征 $\psi$，取
   $\operatorname{vol}(\mathcal O^\times,d^\times a)=1$。在 $\psi$-Whittaker model 中存在唯一满足
   $W_{\operatorname{new}}(1)=1$ 的 newvector，且
   $$
   \int_{F^\times}
   W_{\operatorname{new}}\!\begin{pmatrix}a&0\\0&1\end{pmatrix}
   |a|^{s-1/2}\,d^\times a
   =L(s,\pi)
   $$
   先在绝对收敛半平面成立，随后作为 $q^{-s}$ 的有理函数恒等。

第 5 项中的 $s-1/2$ 与 normalized induction/unitary L-variable 配套；改用 $|a|^s$ 的资料会把变量平移 $1/2$。

**命题 J.12.** 若全局 cuspidal automorphic representation
$$
\pi=\otimes_v'\pi_v
$$
在每个有限位置 $v$ 的局部导子指数为 $a(\pi_v)$，则只有有限多个指数非零，并定义其全局导子为
$$
N(\pi)=\prod_{v<\infty}\mathfrak p_v^{a(\pi_v)}.
$$

**证明.** 几乎所有 $\pi_v$ spherical；由 J.11，此时 $a(\pi_v)=0$，故乘积有限。张量积 newvector
$$
\otimes_{v<\infty}v_{\operatorname{new},v}
$$
在
$$
\prod_vK_1(\mathfrak p_v^{a(\pi_v)})
$$
下不变。若把任一局部指数降到 $a(\pi_v)$ 以下，J.11(3) 使该局部不变量空间消失，restricted tensor product 的相应全局不变量也为零。因此该 ideal 同时记录逐处最小 $K_1$-level。$\square$

**推论 J.13.** 若 $f$ 是级 $N$ 的 classical newform，且对应 $\pi_f$，则 $N$ 等于 $\pi_f$ 的全局导子。

**证明路线（外部输入）.** Atkin-Lehner-Li 理论把 classical primitive line 与 adelic 表示中的 primitive tensor line 识别；Casselman 定理逐处把该线的最小 $K_1$-level 识别为 $a(\pi_v)$。因此级等于 J.12 的导子。经典-adelic 识别与 Atkin-Lehner-Li 分解均为外部输入，本段不重证。

## J.4 Oldforms 是同一表示的高层向量

**命题 J.14.** 若 $f$ 是级 $M$ 的 newform，$M\mid N$，则每个非零 degeneracy image $\iota_df$ 位于 $\pi_f$ 的 automorphic isotypic component；它改变的是 $v\mid N/M$ 处的 level vector，不改变全局不可约表示同构类。

**证明.** 经典-adelic 对应把 $\iota_d$ 解释为在 $v\mid d$ 处平移局部向量，其他位置保持不变，所以所得向量属于 $\pi_f$ 的 restricted tensor product。反过来，即使只使用 Hecke 数据，$\iota_df$ 在所有
$r\nmid N$ 处仍有与 $f$ 相同的本征值；任何含它的 cuspidal irreducible constituent 因而在几乎所有位置与 $\pi_f$ 同构。`GL(2)` 强重数一定理推出该 constituent 就是 $\pi_f$。这里经典-adelic 对应和强重数一是已登记的外部输入，逻辑推导至此完整。$\square$

**注 J.15.** 这就是第七章注 7.17 的精确含义。Newform 是表示的导子级别新向量；oldforms 是同一表示在更高 level 的向量。

## J.5 费马应用中的级 `2`

**命题 J.16.** 若 Ribet 降层把 Frey 曲线的残余表示降到权 $2$、级 $2$ 的 newform，则它给出非零空间
$$
S_2(\Gamma_0(2))_{\operatorname{new}}\ne0.
$$

**证明.** “来自权 $2$、级 $2$ 的 newform”按定义给出
$$
f\in S_2(\Gamma_0(2))_{\operatorname{new}}
$$
非零。New subspace 是 $S_2(\Gamma_0(2))$ 的子空间。若附录 D 证明 $S_2(\Gamma_0(2))=0$，则 new subspace 也为 $0$，矛盾。$\square$

**注 J.17.** 费马应用章只需要 J.16 的逻辑形式。Wiles-Taylor-Wiles 和 Ribet 定理的深处在于证明 Frey 曲线残余表示确实满足模性和降层假设。

## J.6 本附录小结

本附录说明：

1. Oldforms 由低级形式经 degeneracy maps 拉到高级。
2. Newforms 是 old subspace 的 Petersson 正交补中的 primitive Hecke eigenforms。
3. Atkin-Lehner involutions 控制函数方程 root number。
4. Casselman newvector theorem 把局部导子解释为最小不变量层。
5. Classical newform 的级等于 adelic representation 的导子。
6. 费马应用中的级 $2$ 矛盾使用的是 newform 存在推出 $S_2(\Gamma_0(2))$ 非零。

## 练习

**练习 J.1.** 证明若 $f$ 是级 $M$ 的 cusp form 且 $M\mid N$，则 $f(dz)$ 在每个 cusp 处仍消失。

**练习 J.2.** 对 $N=pM$ 且 $p\nmid M$，写出由级 $M$ 到级 $N$ 的两个 degeneracy maps。

**练习 J.3.** 解释为什么 oldforms 与 newforms 的区分不是由好素数 Hecke eigenvalues 决定的。

**练习 J.4.** 设 $\pi_p$ 非分歧。用 J.10 说明 $a(\pi_p)=0$。

**练习 J.5.** 用 J.16 重写费马应用章中“级 $2$ 无 newform”的矛盾。
