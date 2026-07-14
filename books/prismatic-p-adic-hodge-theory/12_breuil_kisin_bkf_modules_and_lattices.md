# 第十二章：Breuil-Kisin、Breuil-Kisin-Fargues modules 与 lattices

Breuil--Kisin module、Breuil--Kisin--Fargues module、prismatic $F$-crystal 与 Galois lattice 都携带 Frobenius，却生活在不同底环并满足不同的反演与下降条件。把它们统称为“带 Frobenius 的模”会抹去积分分类真正依赖的 height、torsion 和 lattice 信息。本章以第二章的 Breuil--Kisin prism、第五章的 BMS comparison、第六章的 $F$-crystal 和附录 J 的 semilinear linearization 为基础，逐项固定四类对象的类型，并说明 cohomology 输出在何种有限性条件下能够落入相应模范畴以及怎样与 crystalline Galois lattice 比较。

## 12.1 Breuil-Kisin prism

**约定 12.1.** 令 $K/\mathbf Q_p$ 为 complete discretely valued field，
剩余域 $k$ 完美，选定 uniformizer $\pi$。令
$$
\mathfrak S=W(k)[[u]],\qquad \phi_{\mathfrak S}(u)=u^p,
$$
并令 $E(u)$ 为 $\pi$ 的 Eisenstein polynomial。记
$$
\widetilde\theta:\mathfrak S\twoheadrightarrow\mathcal O_K,
\qquad u\longmapsto\pi.
$$

**定义 12.2.** Breuil-Kisin prism 是 bounded prism
$$
(\mathfrak S,(E(u))).
$$

**说明 12.3.** 该 presentation 依赖 $\pi$。改变 uniformizer 会同时改变
$\mathfrak S$ 的坐标、$E(u)$ 和到 $\mathcal O_K$ 的 quotient map；坐标无关
对象是 absolute prismatic site 上的 crystal，而不是某个裸
$\mathfrak S$-module。

## 12.2 Breuil-Kisin modules

**定义 12.4（BMS2 convention）.** Breuil-Kisin module 是有限生成
$\mathfrak S$-module $M$ 与 $\mathfrak S[1/E(u)]$-linear 同构
$$
\varphi_M^{\mathrm{lin}}:
(\phi_{\mathfrak S}^\ast M)[1/E(u)]
\xrightarrow{\sim}M[1/E(u)],
\qquad
\phi_{\mathfrak S}^\ast M=
\mathfrak S\otimes_{\mathfrak S,\phi_{\mathfrak S}}M.
$$
若该同构由 integral map
$\phi_{\mathfrak S}^\ast M\to M$ 诱导，则称这里的 Frobenius structure
effective。一般定义只要求 localized isomorphism，不自动给出 integral map。

**警告 12.5.** 定义 12.4 不蕴含 $M$ finite projective、$p$-torsionfree、
$E(u)$-torsionfree 或具有有限 height。任何使用 dual、cokernel height 或
Galois lattice 的 theorem 都必须另列这些假设。

**命题 12.6.** 若 $M$ finite projective，且配有定义 12.4 的 Frobenius
isomorphism，则 $M[1/E(u)]$ 是 $\mathfrak S[1/E(u)]$ 上的 finite
projective Frobenius module。

**证明.** Finite projectivity 在 localization 下保持。定义 12.4 已给出
$$
(\phi_{\mathfrak S}^\ast M)[1/E(u)]
\xrightarrow{\sim}M[1/E(u)],
$$
故 localized module 带可逆的 linearized Frobenius。证毕。

## 12.3 Breuil-Kisin-Fargues modules

**定义 12.7（BMS1 convention）.** 固定第五章的 $A_{\inf}$、$\xi$ 与
Frobenius $\phi$。Breuil-Kisin-Fargues module 是有限呈示
$A_{\inf}$-module $N$，配有 $\phi$-semilinear 同构
$$
\varphi_N:N[1/\xi]\xrightarrow{\sim}N[1/\phi(\xi)],
$$
并满足 $N[1/p]$ 是 finite free $A_{\inf}[1/p]$-module。若 $N$ 本身
finite free over $A_{\inf}$，称其为 finite free BKF module。

**外部输入定理 12.8（Fargues classification）.** 设 $C$ 为 characteristic
$0$ 的 algebraically closed complete nonarchimedean field。Finite free BKF
modules 的范畴等价于 pairs $(T,\Xi)$ 的范畴，其中 $T$ 是 finite free
$\mathbf Z_p$-module，且
$$
\Xi\subset T\otimes_{\mathbf Z_p}B_{\mathrm{dR}}
$$
是 $B_{\mathrm{dR}}^+$-lattice。来源为 BMS1, Theorem 4.28。本书不重证该分类。

**警告 12.8A.** 定理 12.8 只分类 finite free BKF modules。一般 BKF
module 可含 $p$-power torsion 或 closed-point contribution，不能直接读出
finite free $\mathbf Z_p$-lattice。即使在 finite free 情形，输出也是 pair
$(T,\Xi)$，不是单独的 rational representation。

## 12.4 Cohomology 输出与 torsion 门槛

**外部输入定理 12.9（BMS1 cohomology modules）.** 令 $\mathfrak X$ 为
proper smooth formal scheme over $\mathcal O_C$，$k$ 为其 residue field。
BMS1, Theorem 14.3 断言每个
$$
H^i_{A_{\inf}}(\mathfrak X)
$$
都是定义 12.7 的 BKF module，无需先假设 crystalline cohomology
torsionfree。进一步，BMS1, Theorem 14.5 (iii) 给出以下精确门槛：

1. 若 $H^i_{\mathrm{crys}}(\mathfrak X_k/W(k))$ 无 $p$-torsion，则
   $H^i_{A_{\inf}}(\mathfrak X)$ 是 finite free BKF module，并与由
   $H^i_{\mathrm{et}}(X,\mathbf Z_p)$ 及其 $B_{\mathrm{dR}}^+$-lattice 构造的
   BKF module 同构；
2. 在第一项之外，若
   $H^{i+1}_{\mathrm{crys}}(\mathfrak X_k/W(k))$ 也无 $p$-torsion，则
   crystalline specialization 给出积分等式
   $$
   H^i_{A_{\inf}}(\mathfrak X)\otimes_{A_{\inf}}W(k)
   \cong H^i_{\mathrm{crys}}(\mathfrak X_k/W(k)).
   $$

只假设 degree $i$ torsionfree 时，第二项一般只有一个 inclusion；不能把
derived complex comparison 静默降成 cohomology-group base change。

**外部输入定理 12.10（BMS2 cohomology modules）.** 令 $\mathfrak X$ 为
proper smooth formal scheme over $\mathcal O_K$。BMS2, Theorem 1.2 给出
perfect $\mathfrak S$-complex $R\Gamma_{\mathfrak S}(\mathfrak X)$，并断言
每个 $H^i_{\mathfrak S}(\mathfrak X)$ 是定义 12.4 的 Breuil-Kisin module。
这不等于断言它在无额外 torsion 假设时 finite projective。其 de Rham 与
crystalline specializations 是 derived tensor comparisons；逐次
cohomology 的误差由命题 5.16 的 Tor 项度量。

**说明 12.11.** 选择 $\pi$ 的兼容 $p$-power roots 后，BMS1 §4.4 的
embedding $\mathfrak S\to A_{\inf}$ 给出从 Breuil-Kisin modules 到 BKF
modules 的 scalar-extension functor。该 functor 依赖选择，且不是把两种
定义认作同一个范畴。

## 12.5 与 crystalline Galois lattices 的接口

**外部输入定理 12.12（Bhatt-Scholze classification）.** 设
$K/\mathbf Q_p$ complete discretely valued，residue field perfect。令
$X=\operatorname{Spf}(\mathcal O_K)$。Etale realization 给出范畴等价
$$
\mathrm{Vect}^{\varphi}(X_\Delta,\mathcal O_\Delta)
\xrightarrow{\sim}
\mathrm{Rep}^{\mathrm{crys}}_{\mathbf Z_p}(G_K),
$$
其中右侧对象是 finite free $\mathbf Z_p$-modules $T$ 配 continuous
$G_K$-action，且 $T[1/p]$ crystalline。来源为 Bhatt-Scholze, Theorem
5.6。

**外部输入定理 12.13（Breuil-Kisin evaluation 与 descent boundary）.**
Breuil-Kisin prism $(\mathfrak S,(E(u)))$ 是 $X_\Delta$ 中覆盖 final object
的 probe。Evaluation 给出
$$
\operatorname{ev}_{\mathfrak S}:
\mathrm{Vect}^{\varphi}(X_\Delta,\mathcal O_\Delta)
\longrightarrow \mathrm{Vect}^{\varphi}(\mathfrak S),
$$
右侧是 finite projective Breuil-Kisin modules。一个在 essential image 中的
module 还带有其在 prismatic Cech nerve
$\mathfrak S^{(\bullet)}$ 上的 descent isomorphism，并满足 cocycle 条件。
在定理 12.12 下，该 composite 回收 Kisin 的 crystalline-lattice functor；
其 full faithfulness 与 descent 描述是外部输入，来源为
`BS-FCRYS-BK`（尤其 Bhatt-Scholze §7）。

因此，“一个裸 Breuil-Kisin module 自动给出 crystalline lattice”是错误的：
必须证明它落在 evaluation functor 的 essential image，或另行调用带完整
height/descent 假设的 Kisin classification theorem。

## 12.6 Height 条件的代数读法

**定义 12.14.** 设 $(M,\varphi_M)$ 是 effective Breuil-Kisin module，
$M$ finite projective。若 integral linearization
$$
\varphi_M^{\mathrm{lin}}:\phi_{\mathfrak S}^\ast M\to M
$$
的 cokernel 被 $E(u)^h$ 杀死，则称 $M$ 的 height at most $h$。

**命题 12.15.** 若 $M=\mathfrak S e$ 且
$\varphi_M^{\mathrm{lin}}(1\otimes e)=E(u)^he$，则 $M$ 的 height at most
$h$。

**证明.** 在所选 bases 下，linearization 是乘以 $E(u)^h$。其 cokernel
为 $\mathfrak S/(E(u)^h)$，故被 $E(u)^h$ 杀死。证毕。

**警告 12.16.** Height convention 与 Hodge-Tate weights 的正负号和区间
normalization 有关。未声明 normalization 时，不能从 “height $\le h$”
直接读出 weights 属于 $[0,h]$ 或 $[-h,0]$。

## 12.7 Integral、after-inversion 与 rational 层级

**说明 12.17.** 下表中的 operations 互不等价：

| 对象 | 操作 | 保留/遗忘的信息 |
| --- | --- | --- |
| BK module $M$ | 不反演 | $\mathfrak S$-integral module、可能的 torsion、effective/height data |
| $M[1/E(u)]$ | invert prism ideal | Frobenius 可逆；遗忘 $E(u)$-power torsion；仍未 invert $p$ |
| $M[1/p]$ | rationalize in $p$ | 遗忘 $p$-power torsion；仍保留 $E(u)$-divisor |
| $M[1/pE(u)]$ | 两者都反演 | 只剩 generic Frobenius isocrystal 型信息 |
| BKF module $N$ | 不反演 | $A_{\inf}$-integral modification，可能非 finite free |
| $N[1/\xi]$ | invert prism divisor | Frobenius comparison locus；不等于 $N[1/p]$ |
| $N[1/p]$ | rationalize in $p$ | 遗忘 $p$-torsion，但 $\xi$-modification 尚在 |
| lattice $T\subset V$ | $T[1/p]=V$ | 忘掉 lattice；不同 $T$ 可给同一 $V$ |

最后一行的非唯一性由命题 J.11 的例子
$\mathbf Z_p,p\mathbf Z_p\subset\mathbf Q_p$ 已在书内证明。

## 12.8 四类 Frobenius 对象的类型边界

Breuil-Kisin 与 BKF modules 都是积分 Frobenius modules，但底环、反演
divisor 与 finiteness 条件不同。BMS cohomology 总先给 module 型输出；
finite-free lattice statement 需要明确的 torsion 门槛。Prismatic
$F$-crystal 在 Breuil-Kisin prism 上的 evaluation 还携带 Cech descent，
这正是裸 Breuil-Kisin module 与坐标无关 crystalline lattice 之间的边界。

## 练习

**练习 12.1.** 说明定义 12.4 中 invert $E(u)$ 后 linearized Frobenius
为何可逆，并解释这不蕴含 $M[1/p]$ 上的任何结论。

**练习 12.2.** 比较 Breuil-Kisin module 与 BKF module 的底环、反演元素
和 finiteness 条件。

**练习 12.3.** 解释为什么 rational filtered $\varphi$-module 不能唯一
恢复 integral Galois lattice，并给出 rank-one 例子。
