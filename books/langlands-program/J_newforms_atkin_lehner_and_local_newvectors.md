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
或等价地 $f|_k\begin{pmatrix}d&0\\0&1\end{pmatrix}$，具体标量归一化依 Petersson 内积 convention 而定。

**命题 J.2.** $\iota_d f$ 属于 $S_k(\Gamma_0(N),\varepsilon)$。

**证明草图.** 若 $\gamma\in\Gamma_0(N)$，则
$$
\begin{pmatrix}d&0\\0&1\end{pmatrix}\gamma
\begin{pmatrix}d^{-1}&0\\0&1\end{pmatrix}
$$
在适当分母清理后落入 $\Gamma_0(M)$ 对应的 commensurator，且 $M\mid N$ 保证下左项满足模 $M$ 条件。尖点消失性由有限映射 $X_0(N)\to X_0(M)$ 下 holomorphic differentials 的 pullback 保持。完整证明通常用模曲线 degeneracy maps 表述。$\square$

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
在合适 Petersson inner product 和 nebentypus convention 下成立。New subspace 有归一化 Hecke eigenform 基；每个 newform 对应导子为 $N$ 的 cuspidal automorphic representation of $\operatorname{GL}_2(\mathbb A_\mathbb Q)$。

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

**外部输入定理 J.7（Atkin-Lehner operators）.** $W_Q$ 在合适归一化下给出 $S_k(\Gamma_0(N),\varepsilon)$ 的自同构，保持 new subspace，并与 Hecke 算子满足标准交换关系。若 $f$ 为 newform，则 $W_Nf$ 为 $f$ 的共轭或 contragredient 相关 newform，特征值进入函数方程的 root number。

**命题 J.8.** 若 $f$ 为权 $k$、级 $N$ newform，则第六章完成 L 函数函数方程中的常数 $\eta_f$ 可由全局 Atkin-Lehner involution 和 nebentypus 归一化表达。

**证明草图.** Mellin 变换把 $f(iy)$ 与 $L(f,s)$ 联系起来。Fricke/Atkin-Lehner 变换 $W_N$ 把 $y$ 替换为 $1/(Ny)$，从而在 Mellin 积分中产生 $s\leftrightarrow k-s$。若 $f$ 是 $W_N$ 的本征向量，所得本征值与 nebentypus 标量共同给出 root number。完整常数依赖 slash operator 与完成因子 convention。$\square$

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
在中心特征约束下成立。非零向量
$$
v_{\operatorname{new}}\in\pi^{K_1(\mathfrak p^{a(\pi)})}
$$
称为 local newvector。

**外部输入定理 J.11（Casselman newvector theorem）.** 对 $\operatorname{GL}_2(F)$ 的 irreducible admissible generic representation $\pi$：

1. 最小整数 $a(\pi)$ 存在。
2. Newvector line $\pi^{K_1(\mathfrak p^{a(\pi)})}$ 一维。
3. 若 $m<a(\pi)$，则 $\pi^{K_1(\mathfrak p^m)}=0$。
4. 局部 Whittaker newvector 的 zeta integral 给出局部标准 L 因子。

**命题 J.12.** 若全局 cuspidal automorphic representation
$$
\pi=\otimes_v'\pi_v
$$
在每个有限位置 $v$ 的局部导子指数为 $a(\pi_v)$，则其全局导子为
$$
N(\pi)=\prod_{v<\infty}\mathfrak p_v^{a(\pi_v)}.
$$

**证明草图.** 几乎所有 $\pi_v$ 非分歧，故 $a(\pi_v)=0$。张量积 newvector
$$
\otimes_{v<\infty}v_{\operatorname{new},v}
$$
在
$$
\prod_vK_1(\mathfrak p_v^{a(\pi_v)})
$$
下不变。若降低任何一个指数，J.11 给出相应局部不变量消失，因此全局最小级正是上述乘积。$\square$

**推论 J.13.** 若 $f$ 是级 $N$ 的 classical newform，且对应 $\pi_f$，则 $N$ 等于 $\pi_f$ 的全局导子。

**证明草图.** Atkin-Lehner-Li 理论说明 classical newform 是不来自低级 degeneracy maps 的 primitive vector。Casselman newvector theorem 说明 adelic 表示中的 primitive vector 的级由局部导子精确控制。经典-adelic 对应把二者识别，故级等于导子。$\square$

## J.4 Oldforms 是同一表示的高层向量

**命题 J.14.** 若 $f$ 是级 $M$ 的 newform，$M\mid N$，则其 degeneracy images $\iota_df$ 在级 $N$ 空间中通常对应同一全局表示 $\pi_f$ 的不同有限处 level vectors，而不是新的全局自守表示。

**证明草图.** Adelic 表示 $\pi_f=\otimes_v'\pi_{f,v}$ 由几乎所有 Hecke 本征值和强重数一确定。Degeneracy maps 不改变几乎所有好素数处的 Hecke eigenvalues；它们只改变 $v\mid N/M$ 处选择的 $K_0(N)$-不变量向量。强重数一排除它们给出不同 cuspidal representation 的可能。$\square$

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
