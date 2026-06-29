# 附录 H：Hecke 双陪集、Fourier 系数和 Adelic 比较

本附录补充第六、七章中 Hecke 算子的代数计算。核心目标是把三种对象严格连起来：

1. 经典双陪集 $\Gamma\alpha\Gamma$。
2. Fourier 展开上的系数公式。
3. 有限 adelic Hecke algebra 对 $K$-不变量的作用。

## H.1 双陪集算子

设 $\Gamma\subset\operatorname{SL}_2(\mathbb Z)$ 为同余子群，$\alpha\in\operatorname{GL}_2^+(\mathbb Q)$，并假设
$$
\Gamma_\alpha=\Gamma\cap \alpha^{-1}\Gamma\alpha
$$
在 $\Gamma$ 中有限指数。则双陪集有有限分解
$$
\Gamma\alpha\Gamma=\bigsqcup_{i=1}^r\Gamma\alpha_i.
$$

**定义 H.1.** 对权 $k$ 函数 $f:\mathfrak H\to\mathbb C$，定义双陪集算子
$$
[\Gamma\alpha\Gamma]f=\sum_{i=1}^r f|_k\alpha_i.
$$
若需要经典归一化，可再乘以依赖 $\det\alpha$ 的标量。

**命题 H.2（代表无关性）.** 定义 H.1 与右陪集代表 $\alpha_i$ 的选择无关。若 $f$ 满足
$$
f|_k\gamma=f,\qquad \gamma\in\Gamma,
$$
则 $[\Gamma\alpha\Gamma]f$ 仍满足同一 $\Gamma$-不变性。

**证明.** 若 $\alpha_i'=\gamma_i\alpha_i$，$\gamma_i\in\Gamma$，则
$$
f|_k\alpha_i'=f|_k(\gamma_i\alpha_i)=(f|_k\gamma_i)|_k\alpha_i=f|_k\alpha_i,
$$
其中使用 slash 算子的右作用律和 $f|_k\gamma_i=f$。

再取 $\gamma\in\Gamma$。右乘 $\gamma$ 置换右陪集集合，因为
$$
\Gamma\alpha\Gamma\gamma=\Gamma\alpha\Gamma.
$$
于是存在置换 $\sigma$ 和 $\delta_i\in\Gamma$ 使
$$
\alpha_i\gamma=\delta_i\alpha_{\sigma(i)}.
$$
因此
$$
([\Gamma\alpha\Gamma]f)|_k\gamma
=\sum_i f|_k(\alpha_i\gamma)
=\sum_i f|_k(\delta_i\alpha_{\sigma(i)})
=\sum_i f|_k\alpha_{\sigma(i)}
=[\Gamma\alpha\Gamma]f.
$$
$\square$

**注 H.3.** 带 nebentypus 的情形中，若 $f|_k\gamma=\varepsilon(d_\gamma)f$，则双陪集算子需在代表变换时追踪右下角模 $N$ 的 character。第六章命题 6.12 中的 $\varepsilon(\ell)$ 正来自该追踪。

## H.2 `Γ0(N)` 的素数双陪集

本节令
$$
\Gamma=\Gamma_0(N).
$$

**命题 H.4（好素数双陪集分解）.** 若素数 $\ell\nmid N$，则
$$
\Gamma
\begin{pmatrix}1&0\\0&\ell\end{pmatrix}
\Gamma
=
\left(\bigsqcup_{b=0}^{\ell-1}
\Gamma
\begin{pmatrix}1&b\\0&\ell\end{pmatrix}\right)
\sqcup
\Gamma
\begin{pmatrix}\ell&0\\0&1\end{pmatrix}.
$$

**证明草图.** 双陪集右陪集可由指数 $\ell$ 的子格分类。矩阵
$$
\begin{pmatrix}1&0\\0&\ell\end{pmatrix}
$$
把标准格 $\mathbb Z^2$ 送到子格 $\mathbb Z e_1+\ell\mathbb Z e_2$；右乘 $\Gamma$ 改变基。指数 $\ell$ 子格等价于 $\mathbb F_\ell^2$ 中的一维商，故由
$$
\mathbb P^1(\mathbb F_\ell)=\mathbb F_\ell\cup\{\infty\}
$$
参数化。有限点 $b\in\mathbb F_\ell$ 给出代表
$$
\begin{pmatrix}1&b\\0&\ell\end{pmatrix},
$$
无穷点给出代表
$$
\begin{pmatrix}\ell&0\\0&1\end{pmatrix}.
$$
条件 $\ell\nmid N$ 保证 $\Gamma_0(N)$ 在模 $\ell$ 层面与 $\operatorname{SL}_2(\mathbb Z)$ 有相同的 $\mathbb P^1(\mathbb F_\ell)$ 轨道计算，不额外合并这些代表。$\square$

**命题 H.5（坏素数 $U_\ell$ 分解）.** 若 $\ell\mid N$，则第六章定义的 $U_\ell$ 对应代表族
$$
\begin{pmatrix}1&b\\0&\ell\end{pmatrix},\qquad 0\le b<\ell.
$$
即
$$
U_\ell f=\ell^{k/2-1}\sum_{b=0}^{\ell-1}
f|_k
\begin{pmatrix}1&b\\0&\ell\end{pmatrix}.
$$

**证明.** 第六章把 $U_\ell$ 定义为该代表族的算子。它与完整双陪集
$$
\Gamma_0(N)\begin{pmatrix}1&0\\0&\ell\end{pmatrix}\Gamma_0(N)
$$
中的保级部分相对应。若加入代表 $\begin{pmatrix}\ell&0\\0&1\end{pmatrix}$，其右作用会改变 $\Gamma_0(N)$ 的局部级结构；newform theory 中该方向与 degeneracy maps 和 oldforms 相关，而不是本书第六章固定级空间上的 $U_\ell$ 算子。$\square$

**注 H.6.** 若要完全描述 $\ell\mid N$ 的 Hecke correspondence，需要同时记录 $\Gamma_0(N)$ 与 $\Gamma_0(N/\ell)$ 或 $\Gamma_0(N\ell)$ 之间的 degeneracy maps。费马应用章只需要级 $2$ 权 $2$ 空间为零，不需要该完整理论。

## H.3 Fourier 系数计算

设
$$
f(z)=\sum_{n\ge0}a_nq^n,\qquad q=e^{2\pi iz}.
$$

**命题 H.7.** 对
$$
\alpha_b=\begin{pmatrix}1&b\\0&\ell\end{pmatrix}
$$
有
$$
\ell^{k/2-1}\sum_{b=0}^{\ell-1}(f|_k\alpha_b)(z)
=\sum_{n\ge0}a_{\ell n}q^n.
$$

**证明.** 因 $\det\alpha_b=\ell$ 且 $cz+d=\ell$，
$$
(f|_k\alpha_b)(z)
=\ell^{k/2}\ell^{-k}f\left(\frac{z+b}{\ell}\right)
=\ell^{-k/2}\sum_{m\ge0}a_m
\exp\left(2\pi im\frac{z+b}{\ell}\right).
$$
乘以 $\ell^{k/2-1}$ 并对 $b$ 求和得
$$
\ell^{-1}\sum_{m\ge0}a_m e^{2\pi imz/\ell}
\sum_{b=0}^{\ell-1}e^{2\pi imb/\ell}.
$$
内层和为 $\ell$ 当 $\ell\mid m$，否则为 $0$。令 $m=\ell n$ 得
$$
\sum_{n\ge0}a_{\ell n}q^n.
$$
$\square$

**命题 H.8.** 对
$$
\beta=\begin{pmatrix}\ell&0\\0&1\end{pmatrix}
$$
有
$$
\ell^{k/2-1}(f|_k\beta)(z)=\ell^{k-1}\sum_{n\ge0}a_nq^{\ell n}.
$$

**证明.** 因 $\det\beta=\ell$ 且 $cz+d=1$，
$$
(f|_k\beta)(z)=\ell^{k/2}f(\ell z)
=\ell^{k/2}\sum_{n\ge0}a_nq^{\ell n}.
$$
乘以 $\ell^{k/2-1}$ 得结论。$\square$

**命题 H.9（好素数 Fourier 公式）.** 若 $\ell\nmid N$ 且 $f\in M_k(\Gamma_0(N),\varepsilon)$，则
$$
T_\ell f=\sum_{n\ge0}
\left(a_{\ell n}+\varepsilon(\ell)\ell^{k-1}a_{n/\ell}\right)q^n,
$$
其中 $a_{n/\ell}=0$ 当 $\ell\nmid n$。

**证明草图.** 由 H.4，$T_\ell$ 的代表由 $\alpha_b$ 和 $\beta$ 给出。H.7 给出 $\alpha_b$ 部分的贡献为 $a_{\ell n}$。$\beta$ 部分需与 nebentypus 变换律比较：在 $\Gamma_0(N)$ 层面，代表 $\beta$ 的作用对应下三角方向的 degree $\ell$ isogeny，其右下角在模 $N$ 单位群中的贡献为 $\ell$，故乘上 $\varepsilon(\ell)$。H.8 给出其 Fourier 指标从 $n/\ell$ 来，贡献
$$
\varepsilon(\ell)\ell^{k-1}a_{n/\ell}.
$$
两部分相加。$\square$

**命题 H.10（坏素数 Fourier 公式）.** 若 $\ell\mid N$，则
$$
U_\ell f=\sum_{n\ge0}a_{\ell n}q^n.
$$

**证明.** 由 H.5，$U_\ell$ 只含 H.7 中的 $\alpha_b$ 代表族。直接应用 H.7。$\square$

## H.4 Petersson 内积和 Hecke 算子

**定义 H.11.** 对 $f,g\in S_k(\Gamma_0(N),\varepsilon)$，Petersson inner product 定义为
$$
\langle f,g\rangle
=
\int_{\Gamma_0(N)\backslash\mathfrak H}
f(z)\overline{g(z)}y^k\frac{dx\,dy}{y^2},
\qquad z=x+iy.
$$

**命题 H.12.** Petersson inner product 绝对收敛。

**证明草图.** 在基本域的紧部分上，$f,g$ 连续且有界。每个 cusp 邻域中，用局部参数 $q=e^{2\pi iz/h}$，尖点形式满足
$$
f(z)=O(e^{-2\pi y/h}),\qquad g(z)=O(e^{-2\pi y/h})
$$
当 $y\to\infty$。因此 integrand 至多为指数衰减乘以多项式 $y^k$，在 cusp 邻域可积。有限多个 cusp 相加仍收敛。$\square$

**外部输入定理 H.13（Hecke 算子的正规性）.** 对 unitary nebentypus，Hecke 算子在 Petersson inner product 下满足标准伴随关系。特别地，当 nebentypus 平凡且 $\ell\nmid N$ 时，合适归一化的 $T_\ell$ 为 self-adjoint，并可在有限维空间 $S_k(\Gamma_0(N))$ 上对角化。

**注 H.14.** 第六章使用共同 Hecke eigenforms；其存在可由有限维性、Hecke 算子交换性和 Petersson inner product 下的正规性推出。完整 newform theory 还需 old/new 分解和 Atkin-Lehner-Li 理论。

## H.5 Adelic Hecke 代数比较

设 $p\nmid N$，令
$$
K_p=\operatorname{GL}_2(\mathbb Z_p).
$$
局部球 Hecke 代数为
$$
\mathcal H_p=C_c^\infty(K_p\backslash\operatorname{GL}_2(\mathbb Q_p)/K_p).
$$

**定义 H.15.** 令
$$
\mathbf T_p=\mathbf 1_{K_p\begin{pmatrix}1&0\\0&p\end{pmatrix}K_p}\in\mathcal H_p.
$$
在全局有限 adelic Hecke algebra 中，把 $\mathbf T_p$ 放在 $p$ 分量，其余好位置取单位元 $\mathbf 1_{K_q}$。

**命题 H.16.** 在第七章经典-adelic 对应的归一化下，$\mathbf T_p$ 对右 $K_0(N)$-有限 adelic 函数的作用对应第六章的经典 Hecke 算子 $T_p$。

**证明草图.** 局部双陪集
$$
K_p\begin{pmatrix}1&0\\0&p\end{pmatrix}K_p
$$
的右陪集代表由
$$
\begin{pmatrix}1&b\\0&p\end{pmatrix}\quad b\in\mathbb Z/p\mathbb Z,
\qquad
\begin{pmatrix}p&0\\0&1\end{pmatrix}
$$
给出。把这些代表嵌入有限 adeles，并用 strong approximation 把右平移转回无穷处 slash action，得到 H.4 的经典双陪集代表。第六章中因子 $p^{k/2-1}$ 正是匹配权 $k$ slash normalization 与有限处卷积测度的归一化因子。Nebentypus 由右 $K_0(N)$ 变换律给出。$\square$

**命题 H.17.** 若 $f$ 为归一化 Hecke eigenform，$p\nmid N$，且 $T_pf=a_pf$，则 $\pi_{f,p}^{K_p}$ 是一维，$\mathbf T_p$ 在该线上作用为 $a_p$。

**证明草图.** Newform theory 给出好素数处 $\pi_{f,p}$ 为 spherical representation 且 $K_p$-不变量一维。由 H.16，$\mathbf T_p$ 作用与经典 $T_p$ 作用相同，因此特征值为 $a_p$。$\square$

## H.6 Satake 参数和 Hecke 多项式

**命题 H.18.** 设 $p\nmid N$，$f$ 为归一化 Hecke eigenform。若 Satake 参数 $(\alpha_p,\beta_p)$ 满足
$$
\alpha_p+\beta_p=a_p,\qquad \alpha_p\beta_p=\varepsilon(p)p^{k-1},
$$
则 Hecke 多项式为
$$
1-a_pX+\varepsilon(p)p^{k-1}X^2=(1-\alpha_pX)(1-\beta_pX).
$$

**证明.** 展开右侧：
$$
(1-\alpha_pX)(1-\beta_pX)
=1-(\alpha_p+\beta_p)X+\alpha_p\beta_pX^2.
$$
代入定义即得。$\square$

**注 H.19.** 在 unitary automorphic normalization 中，常把参数改写为
$$
\alpha_p p^{-(k-1)/2},\qquad \beta_p p^{-(k-1)/2}.
$$
这样 temperedness 预期对应这些归一化参数的复绝对值为 $1$。第七至九章使用算术归一化，因此 Frobenius trace 直接等于 $a_p$。

## H.7 本附录小结

本附录证明或定位了下列事实：

1. 双陪集算子与代表选择无关。
2. $\ell\nmid N$ 时 $T_\ell$ 有 $\ell+1$ 个标准代表。
3. $\ell\mid N$ 时 $U_\ell$ 由 $\ell$ 个保级代表给出。
4. Fourier 系数公式来自有限指数根的求和。
5. Adelic 球 Hecke 代数作用与经典 Hecke 算子相同。
6. Hecke 多项式就是 Satake 参数的 characteristic polynomial。

## 练习

**练习 H.1.** 证明命题 H.2 中右乘 $\gamma\in\Gamma$ 确实置换右陪集集合。

**练习 H.2.** 对 $N=1$ 和 $\ell=2$，写出命题 H.4 中三个代表，并直接验证它们给出不同右陪集。

**练习 H.3.** 用 H.7 和 H.8 重新证明第六章命题 6.12。

**练习 H.4.** 设 $f$ 为归一化 Hecke eigenform。由 H.9 推出当 $(m,n)=1$ 且 $(mn,N)=1$ 时 $a_{mn}=a_ma_n$。

**练习 H.5.** 在 trivial nebentypus 情形，说明 Petersson self-adjointness 如何推出不同 $T_p$ 本征值的 cusp forms 彼此正交。
