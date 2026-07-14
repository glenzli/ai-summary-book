# 第八章：椭圆曲线、导子和 Hasse-Weil L 函数

模形式的 Fourier 系数看似来自解析函数，椭圆曲线的点数则来自代数几何；模性定理断言这两组数据可以完全相同。要精确表达这一点，必须先从 Weierstrass 模型提取判别式和约化类型，再由 Tate module 得到 Galois 表示，并把各素数处的 Frobenius 特征多项式装配成 Hasse--Weil L 函数。导子在其中记录分歧程度，也决定应与哪一级的 newform 比较。

本章假定有限域上曲线、局部域与基本 Galois 表示知识。Neron 模型、Tate 算法、Neron--Ogg--Shafarevich 判别和导子公式作为外部输入，其详细接口集中在附录 AD。点计数 Frobenius、Tate module 与 Euler 因子的归一化按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 6、7、8 节处理。

## 8.1 椭圆曲线和 Weierstrass 方程

**定义 8.1.** 设 $K$ 为域。$K$ 上的椭圆曲线是二元组 $(E,O)$，其中 $E/K$ 是光滑、射影、几何连通的 genus $1$ 曲线，$O\in E(K)$ 是指定的 $K$-有理点。点 $O$ 作为群结构的零元。

**外部输入定理 8.2（Weierstrass 方程）.** 若 $\operatorname{char}K\ne2,3$，则每条椭圆曲线可由短 Weierstrass 方程
$$
E:\quad y^2=x^3+Ax+B
$$
给出，其中 $A,B\in K$ 且
$$
\Delta=-16(4A^3+27B^2)\ne0.
$$
一般特征或整数模型中，需要使用长 Weierstrass 方程
$$
y^2+a_1xy+a_3y=x^3+a_2x^2+a_4x+a_6.
$$

**定义 8.3.** 对长 Weierstrass 方程，定义标准不变量
$$
\begin{aligned}
b_2&=a_1^2+4a_2,\\
b_4&=2a_4+a_1a_3,\\
b_6&=a_3^2+4a_6,\\
b_8&=a_1^2a_6+4a_2a_6-a_1a_3a_4+a_2a_3^2-a_4^2,
\end{aligned}
$$
判别式定义为
$$
\Delta=-b_2^2b_8-8b_4^3-27b_6^2+9b_2b_4b_6.
$$
方程定义光滑曲线当且仅当 $\Delta\ne0$。

## 8.2 整模型和约化

本节设 $E/\mathbb Q$ 为椭圆曲线。取整数系数 Weierstrass 方程
$$
\mathcal E:\quad y^2+a_1xy+a_3y=x^3+a_2x^2+a_4x+a_6,\qquad a_i\in\mathbb Z.
$$

**定义 8.4.** 对素数 $p$，若 $\mathcal E$ 的判别式 $\Delta(\mathcal E)$ 的 $p$-adic 赋值在所有整数 Weierstrass 方程中最小，则称 $\mathcal E$ 在 $p$ 处是最小模型。相应最小判别式的 $p$-adic 赋值记为
$$
v_p(\Delta_E).
$$

**外部输入定理 8.5（最小模型存在性）.** 对任意椭圆曲线 $E/\mathbb Q$ 和任意素数 $p$，存在 $p$-局部最小 Weierstrass 模型。全局最小模型也存在，但其唯一性需要按允许的整变量变换理解。

**定义 8.6.** 设 $\mathcal E$ 是 $p$-局部最小模型。将其系数模 $p$ 化，得到 $\mathbb F_p$ 上的平面曲线 $\widetilde E/\mathbb F_p$。若 $\widetilde E$ 光滑，则称 $E$ 在 $p$ 处有好约化（good reduction）。若 $\widetilde E$ 奇异，则称 $E$ 在 $p$ 处有坏约化（bad reduction）。

坏约化进一步分为：

1. 若奇点是 node，则称为乘法约化（multiplicative reduction）。
2. 若奇点是 cusp，则称为加法约化（additive reduction）。
3. 乘法约化按 node 的两条切线是否在 $\mathbb F_p$ 上定义分为 split 和 nonsplit。

**命题 8.7.** 若 $p\nmid\Delta(\mathcal E)$，则 $E$ 在 $p$ 处有好约化。

**证明.** 对 Weierstrass 方程，判别式非零等价于相应平面三次曲线光滑。若 $p\nmid\Delta(\mathcal E)$，则 $\Delta(\mathcal E)$ 模 $p$ 非零，所以模 $p$ 曲线 $\widetilde E$ 光滑。$\square$

## 8.3 好约化处的局部因子

设 $E/\mathbb Q$ 在素数 $p$ 处有好约化。

**定义 8.8.** 定义
$$
a_p(E)=p+1-\#\widetilde E(\mathbb F_p).
$$
好约化处的局部 Hasse-Weil 因子定义为
$$
L_p(E,s)=\left(1-a_p(E)p^{-s}+p^{1-2s}\right)^{-1}.
$$

**外部输入定理 8.9（Hasse 界）.** 若 $E/\mathbb Q$ 在 $p$ 处有好约化，则
$$
|a_p(E)|\le2\sqrt p.
$$

**命题 8.10.** 好约化局部因子可写为
$$
L_p(E,s)=\left((1-\alpha_pp^{-s})(1-\beta_pp^{-s})\right)^{-1}
$$
其中
$$
\alpha_p+\beta_p=a_p(E),\qquad \alpha_p\beta_p=p.
$$

**证明.** 与命题 7.19 相同，展开
$$
(1-\alpha_pX)(1-\beta_pX)=1-a_p(E)X+pX^2
$$
并令 $X=p^{-s}$。$\square$

## 8.4 坏约化处的局部因子

**定义 8.11.** 对坏素数 $p$，定义
$$
a_p(E)=
\begin{cases}
1,&\text{若 }E\text{ 在 }p\text{ 处 split multiplicative},\\
-1,&\text{若 }E\text{ 在 }p\text{ 处 nonsplit multiplicative},\\
0,&\text{若 }E\text{ 在 }p\text{ 处 additive}.
\end{cases}
$$
相应局部因子定义为
$$
L_p(E,s)=
\begin{cases}
(1-a_p(E)p^{-s})^{-1},&\text{若 }E\text{ 在 }p\text{ 处 multiplicative},\\
1,&\text{若 }E\text{ 在 }p\text{ 处 additive}.
\end{cases}
$$

**注 8.12.** 该定义是椭圆曲线 Hasse-Weil L 函数的标准局部因子口径。若用 $\ell$-adic 表示统一定义，则坏约化局部因子来自惯性不变量空间；见 8.7 节。

## 8.5 Hasse-Weil L 函数

**定义 8.13.** 椭圆曲线 $E/\mathbb Q$ 的 Hasse-Weil L 函数定义为 Euler 乘积
$$
L(E,s)=\prod_p L_p(E,s)
$$
在绝对收敛半平面中成立。

**命题 8.14.** Euler 乘积 $L(E,s)$ 在 $\operatorname{Re}(s)>3/2$ 中绝对收敛。

**证明.** 除有限多个坏素数外，局部因子为
$$
\left(1-a_p(E)p^{-s}+p^{1-2s}\right)^{-1}.
$$
由 Hasse 界，$|a_p(E)|\le2p^{1/2}$。因此 Euler 乘积对数的主项由
$$
\sum_p |a_p(E)|p^{-\operatorname{Re}(s)}
\le
2\sum_p p^{1/2-\operatorname{Re}(s)}
$$
控制。该素数和在 $\operatorname{Re}(s)>3/2$ 时收敛；二次项由 $\sum_p p^{1-2\operatorname{Re}(s)}$ 控制，也在该半平面收敛。有限多个坏素数不影响绝对收敛。$\square$

**定义 8.15.** 若 $N_E$ 是 $E$ 的导子，完成 L 函数定义为
$$
\Lambda(E,s)=N_E^{s/2}(2\pi)^{-s}\Gamma(s)L(E,s).
$$

解析延拓和函数方程不是定义的一部分；它们由模性定理或更一般的自守理论提供。

## 8.6 导子

**定义 8.16.** 椭圆曲线 $E/\mathbb Q$ 的导子是正整数
$$
N_E=\prod_p p^{f_p(E)}
$$
其中 $f_p(E)$ 是 $E$ 在 $p$ 处的局部导子指数。它可由 $\ell$-adic Tate module 的 Artin 导子定义，且与辅助素数 $\ell\ne p$ 无关。

**外部输入定理 8.17（导子和约化类型）.** 局部导子指数满足：

1. $f_p(E)=0$ 当且仅当 $E$ 在 $p$ 处有好约化。
2. 若 $E$ 在 $p$ 处有乘法约化，则 $f_p(E)=1$。
3. 若 $E$ 在 $p$ 处有加法约化，则 $f_p(E)\ge2$。
4. $f_p(E)$ 可由 Tate 算法从局部最小 Weierstrass 方程计算。

**注 8.17.1.** 附录 AD.3--AD.6 把本定理拆成 Kodaira-Neron 分类、Tate curve 乘法约化、Ogg conductor formula 和 Tate algorithm 输出。对 residue characteristic 不为 $2,3$ 的局部域，附录 AD.10 的表可直接读出 $v(\Delta_E)$、components 数和 conductor exponent。

**定义 8.18.** 椭圆曲线 $E/\mathbb Q$ 称为半稳定的（semistable），若它在每个素数处只有好约化或乘法约化。等价地，它没有加法约化。

**命题 8.19.** 若 $E/\mathbb Q$ 半稳定，则
$$
N_E=\prod_{p\text{ bad}}p.
$$

**证明.** 半稳定意味着每个坏素数处都是乘法约化。由外部输入定理 8.17，坏素数处 $f_p(E)=1$，好素数处 $f_p(E)=0$。因此导子正是所有坏素数的一次乘积。$\square$

## 8.7 Tate module 和 Galois 表示

**定义 8.20.** 设 $\ell$ 为素数。椭圆曲线的 $\ell$-power torsion 定义为
$$
E[\ell^n]=\{P\in E(\overline{\mathbb Q}):\ell^nP=O\}.
$$
$\ell$-adic Tate module 定义为逆极限
$$
T_\ell(E)=\varprojlim_n E[\ell^n],
$$
其中过渡映射由乘以 $\ell$ 给出。

绝对 Galois 群 $G_\mathbb Q$ 作用在 torsion points 上，因而给出连续表示
$$
\rho_{E,\ell}:G_\mathbb Q\to\operatorname{Aut}_{\mathbb Z_\ell}(T_\ell(E))\cong\operatorname{GL}_2(\mathbb Z_\ell).
$$

**外部输入定理 8.21（Tate module 基本性质）.** $T_\ell(E)$ 是自由 $\mathbb Z_\ell$-模，秩为 $2$。并且 $\det\rho_{E,\ell}$ 等于 $\ell$-adic cyclotomic character
$$
\chi_\ell:G_\mathbb Q\to\mathbb Z_\ell^\times.
$$

**证明路线（外部输入）.** Weil pairing 给出非退化交替配对
$$
e_{\ell^n}:E[\ell^n]\times E[\ell^n]\to\mu_{\ell^n}.
$$
Galois 作用满足
$$
e_{\ell^n}(\sigma P,\sigma Q)=\sigma(e_{\ell^n}(P,Q)).
$$
右端由 cyclotomic character 描述。通过 $\operatorname{GL}_2$ 对标准交替形式的作用取行列式，得到 $\det\rho_{E,\ell}=\chi_\ell$。$\square$

**外部输入定理 8.22（Neron-Ogg-Shafarevich 判别准则）.** 设 $p\ne\ell$。椭圆曲线 $E/\mathbb Q$ 在 $p$ 处有好约化，当且仅当 $\rho_{E,\ell}$ 在 $p$ 处非分歧，即惯性群 $I_p$ 在 $T_\ell(E)$ 上平凡。

**外部输入定理 8.23（好约化处的 Frobenius 多项式）.** 若 $p\ne\ell$ 且 $E$ 在 $p$ 处有好约化，则 $\rho_{E,\ell}$ 在 $p$ 处非分歧，并且算术 Frobenius $\operatorname{Frob}_p^{\operatorname{arith}}$ 的特征多项式为
$$
X^2-a_p(E)X+p.
$$

这里 $\operatorname{Frob}_p^{\operatorname{arith}}$ 是在剩余域上诱导 $x\mapsto x^p$ 的元素。若要与第五章的几何 Frobenius 归一化比较，必须取逆或使用对偶/Tate twist 归一化。

**定义 8.24.** 对 $p\ne\ell$，$\ell$-adic 表示给出的局部因子定义为
$$
L_p(E,s)=
\det\left(1-\rho_{E,\ell}(\operatorname{Frob}_p^{\operatorname{arith}})p^{-s}
\mid V_\ell(E)^{I_p}\right)^{-1},
$$
其中
$$
V_\ell(E)=T_\ell(E)\otimes_{\mathbb Z_\ell}\mathbb Q_\ell.
$$

**外部输入定理 8.25（局部因子相容）.** 定义 8.24 与 8.3 和 8.4 节给出的好约化、乘法约化和加法约化局部因子一致，并且与辅助素数 $\ell\ne p$ 的选择无关。

## 8.8 模性接口

**外部输入定理 8.26（模性定理，接口形式）.** 对任意椭圆曲线 $E/\mathbb Q$，存在权 $2$、级 $\Gamma_0(N_E)$ 的归一化 newform
$$
f_E(q)=\sum_{n\ge1}a_n(f_E)q^n
$$
使得
$$
L(E,s)=L(f_E,s).
$$
等价地，对所有好素数 $p\nmid N_E$，
$$
a_p(f_E)=a_p(E)=p+1-\#E(\mathbb F_p),
$$
并且坏素数处的局部因子也与 newform 的局部因子相容。

**注 8.27.** Wiles 和 Taylor-Wiles 首先证明了半稳定情形中足以推出费马大定理的模性；完整有理数域上椭圆曲线模性后来由 Breuil-Conrad-Diamond-Taylor 等工作完成。本书在费马应用章只需要半稳定模性。

**椭圆曲线的算术输出 8.A.** 与 `GL(2)` 自守数据比较时，需要从椭圆曲线提取以下对象：

| 椭圆曲线侧 | Langlands 侧 | 使用位置 |
|---|---|---|
| $a_p(E)=p+1-\#E(\mathbb F_p)$ | 好素数 Frobenius trace | 与模形式 $a_p(f)$ 比较 |
| $T_\ell(E)$ 和 $\rho_{E,\ell}$ | 二维 $\ell$-adic Galois 表示 | 第九章模性和第十章 residual 表示 |
| conductor $N_E$ | newform level 和 automorphic conductor | 第七至十章级结构比较 |
| Neron-Ogg-Shafarevich | 好约化与非分歧性 | 局部-整体相容 |
| Frey 曲线局部导子 | residual level 降到 $2$ | 第十章和第九十章 |

## 8.9 与前后章节的关系

本章把椭圆曲线和 Langlands 主线连接起来：

1. 椭圆曲线 $E/\mathbb Q$ 给出二维 $\ell$-adic Galois 表示 $\rho_{E,\ell}$。
2. 好素数处，$\rho_{E,\ell}$ 的 Frobenius trace 等于 $a_p(E)$。
3. Hasse-Weil L 函数由这些局部 Frobenius 数据组成。
4. 模性定理把同一个 L 函数识别为权 $2$ newform 的 L 函数。
5. 第七章把该 newform 进一步解释为 `GL(2,\mathbb A_\mathbb Q)` 的 cuspidal automorphic representation。

因此，椭圆曲线模性是 `GL(2)/\mathbb Q` Langlands 对应的一个具体实例：几何对象产生的二维 Galois 表示对应自守侧的 cuspidal representation。

## 8.10 椭圆曲线的局部与整体数据

椭圆曲线的局部约化决定 Hasse-Weil L 函数的局部因子和导子。Tate module 给出二维 $\ell$-adic Galois 表示，其好约化处 Frobenius 多项式为 $X^2-a_p(E)X+p$。模性定理断言这些数据来自权 $2$ newform，从而把椭圆曲线放入 `GL(2)` Langlands 框架。

## 练习

**练习 8.1.** 对短 Weierstrass 方程 $y^2=x^3+Ax+B$，说明为什么 $\Delta\ne0$ 等价于右端三次多项式无重根。

**练习 8.2.** 设 $E/\mathbb Q$ 在 $p$ 处有好约化。证明命题 8.10 的局部因子分解公式。

**练习 8.3.** 设 $E$ 半稳定。用定义证明 $N_E$ 是所有坏素数的乘积。

**练习 8.4.** 使用 Weil pairing 证明 $\det\rho_{E,\ell}=\chi_\ell$ 的有限层版本。

**练习 8.5.** 解释为什么模性定理中权必须是 $2$：从完成 L 函数的 Gamma 因子或 Hodge 结构角度给出说明。
