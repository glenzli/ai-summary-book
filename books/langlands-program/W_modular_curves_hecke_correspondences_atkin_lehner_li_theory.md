# 附录 W：模曲线、Hecke Correspondences 和 Atkin-Lehner-Li 理论接口

收口归一化回指：本附录支撑 classical Hecke operators、old/new 分解、Atkin-Lehner signs 和费马级 `2` 矛盾；与 adelic 和 Galois convention 比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、8 节。

## W.1 Modular curves as stacks and coarse curves

**定义 W.1.** 对 congruence subgroup $\Gamma\subset\operatorname{SL}_2(\mathbb Z)$，复模曲线
$$
Y(\Gamma)=\Gamma\backslash\mathfrak H
$$
的紧化记为 $X(\Gamma)$。若 $\Gamma=\Gamma_0(N)$，写 $X_0(N)$。

**外部输入定理 W.2（模曲线代数化）.** $Y_0(N)$ 和 $X_0(N)$ 可定义为 $\mathbb Q$ 上的代数曲线或 Deligne-Mumford stack 的 coarse space。其 cusp、elliptic points、line bundles of modular forms 和 Hecke correspondences 均有代数定义。

**注 W.3.** 本书在费马应用中只需 $X_0(2)$ 的 genus 和权 $2$ cusp forms 的消失；完整模曲线理论作为外部输入。

## W.2 Line bundles and cusp forms

**定义 W.4.** 令 $\omega$ 为 universal generalized elliptic curve 的 Hodge bundle。权 $k$ modular forms 可解释为
$$
H^0(X_0(N),\omega^{\otimes k})
$$
的截面，cusp forms 为在 cusp divisor 处消失的截面：
$$
S_k(\Gamma_0(N))\simeq H^0(X_0(N),\omega^{\otimes k}(-\operatorname{cusps})).
$$

**外部输入定理 W.5（权二 cusp forms 与微分）.** 存在自然同构
$$
S_2(\Gamma)\simeq H^0(X(\Gamma),\Omega^1)
$$
在 torsion-free 或 stack 修正后的情形成立。

**命题 W.6.** 若 $X_0(N)$ 的 genus 为 $0$，则 $S_2(\Gamma_0(N))=0$。

**证明.** genus 为 $0$ 的光滑射影曲线没有非零全纯微分：
$$
H^0(X_0(N),\Omega^1)=0.
$$
由定理 W.5 得 $S_2(\Gamma_0(N))=0$。$\square$

## W.3 Genus formula and $X_0(2)$

**外部输入定理 W.7（genus formula）.** 对 $\Gamma_0(N)$，
$$
g(X_0(N))=
1+\frac{\mu_N}{12}-\frac{e_2(N)}4-\frac{e_3(N)}3-\frac{c_N}2,
$$
其中 $\mu_N=[\operatorname{SL}_2(\mathbb Z):\Gamma_0(N)]$，$e_2,e_3$ 为 elliptic point 数，$c_N$ 为 cusp 数。

**命题 W.8.** $X_0(2)$ 的 genus 为 $0$。

**证明.** 对 $N=2$，
$$
\mu_2=3,\qquad c_2=2,\qquad e_2=1,\qquad e_3=0.
$$
代入定理 W.7：
$$
g=1+\frac{3}{12}-\frac14-0-\frac22=0.
$$
$\square$

**推论 W.9.** $S_2(\Gamma_0(2))=0$。

**证明.** 由命题 W.8 和命题 W.6 得到。$\square$

## W.4 Hecke correspondences

**定义 W.10.** 对素数 $\ell\nmid N$，Hecke correspondence $T_\ell$ 参数化 cyclic isogenies
$$
E\to E'
$$
of degree $\ell$，保持 $\Gamma_0(N)$-level structure。它给出
$$
X_0(N)\xleftarrow{p_1}C_\ell\xrightarrow{p_2}X_0(N)
$$
和算子
$$
T_\ell=p_{2,*}p_1^*.
$$

**命题 W.11.** $T_\ell$ 在 Fourier expansion 上满足
$$
a_n(T_\ell f)=a_{\ell n}(f)+\ell^{k-1}a_{n/\ell}(f)
$$
在 trivial character 且 $\ell\nmid N$ 的情形成立。

**证明草图.** 双陪集分解
$$
\Gamma_0(N)\begin{pmatrix}1&0\\0&\ell\end{pmatrix}\Gamma_0(N)
=\bigsqcup_{b\bmod\ell}\Gamma_0(N)\begin{pmatrix}1&b\\0&\ell\end{pmatrix}
\sqcup
\Gamma_0(N)\begin{pmatrix}\ell&0\\0&1\end{pmatrix}
$$
作用在 $q$-expansion 上。前 $\ell$ 个代表通过 roots of unity averaging 选出 $a_{\ell n}$ 项，最后一个代表给出 $\ell^{k-1}a_{n/\ell}$ 项。$\square$

## W.5 Degeneracy maps and oldforms

设 $M\mid N$。

**定义 W.12.** Degeneracy maps
$$
\alpha_d:X_0(N)\to X_0(M),\qquad d\mid N/M
$$
在复上半平面模型中由 $z\mapsto dz$ 或相应 subgroup inclusions 给出。它们诱导
$$
\alpha_d^*:S_k(\Gamma_0(M))\to S_k(\Gamma_0(N)).
$$

**定义 W.13.** Old subspace 定义为所有 proper divisors $M\mid N$ 的 degeneracy images 张成的子空间：
$$
S_k(\Gamma_0(N))_{\operatorname{old}}
=\sum_{M\mid N,\ M<N}\sum_{d\mid N/M}\alpha_d^*S_k(\Gamma_0(M)).
$$
New subspace 是 Petersson inner product 下 old subspace 的正交补。

**外部输入定理 W.14（Atkin-Lehner-Li old/new decomposition）.** 有 Hecke-stable 正交分解
$$
S_k(\Gamma_0(N))=
S_k(\Gamma_0(N))_{\operatorname{old}}
\oplus
S_k(\Gamma_0(N))_{\operatorname{new}},
$$
且 new subspace 有归一化 simultaneous Hecke eigenbasis。Newforms 对应导子正好为 $N$ 的 cuspidal automorphic representations。

**命题 W.15.** 若 $S_k(\Gamma_0(N))=0$，则 new subspace 为 $0$。

**证明.** New subspace 是 $S_k(\Gamma_0(N))$ 的子空间。若后者为零，则所有子空间均为零。$\square$

## W.6 Atkin-Lehner involutions and functional equations

**定义 W.16.** 若 $Q\mid N$ 且 $(Q,N/Q)=1$，Atkin-Lehner operator $W_Q$ 由 determinant $Q$ 的矩阵作用给出，其在 $S_k(\Gamma_0(N))$ 上诱导 involution up to scalar normalization。

**外部输入定理 W.17（Atkin-Lehner signs）.** 对 newform $f$，Atkin-Lehner operators $W_Q$ 作用为标量 $\pm1$ 或相应 root of unity，并与局部 epsilon factors 和 completed L-function 的 functional equation 相容。

**命题 W.18.** Atkin-Lehner sign 是局部 root number 的 classical shadow。

**证明草图.** Newform $f$ 的 Mellin transform 给出 completed L-function。Fricke/Atkin-Lehner involution 在积分变量 $z\mapsto-1/(Nz)$ 下把 $s$ 与 $k-s$ 交换，产生 functional equation 的符号。Adelic 解释中该符号分解为局部 epsilon factors，故 classical Atkin-Lehner eigenvalue 记录坏素数处的局部 root number 数据。$\square$

## W.7 费马应用中的级 $2$ 矛盾

**命题 W.19.** 若 Ribet 降层给出权 $2$、级 $2$ newform，则与 $S_2(\Gamma_0(2))=0$ 矛盾。

**证明.** 降层结论给出
$$
S_2(\Gamma_0(2))_{\operatorname{new}}\ne0.
$$
由推论 W.9，$S_2(\Gamma_0(2))=0$，由命题 W.15，new subspace 也为 $0$。矛盾。$\square$

## 练习

**练习 W.1.** 用 genus formula 重新计算 $X_0(2)$ 的 genus。

**练习 W.2.** 解释为什么权二 cusp forms 等同于全纯微分。

**练习 W.3.** 从双陪集代表推导 $T_\ell$ 的 Fourier 系数公式。

**练习 W.4.** 说明 oldforms 为什么不改变几乎所有好素数 Hecke eigenvalues。

**练习 W.5.** 用 W.19 重述费马应用中的最终矛盾。
