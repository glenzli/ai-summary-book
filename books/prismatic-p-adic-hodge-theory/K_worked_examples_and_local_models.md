# 附录 K：Worked examples 与局部模型

## 本附录目标

本附录提供低维、局部、可计算的例子，帮助读者把 prism、specialization、Frobenius fixed points 和 Breuil-Kisin module 的定义落到具体公式。

## K.1 Crystalline prism 的零维模型

**例 K.1.** 令 $k$ 为完美域，$A=W(k)$，$I=(p)$，$X=\operatorname{Spf}(k)$。则 crystalline prism 为 $(W(k),(p))$。在该零维情形，预期
$$
R\Gamma_\Delta(X/A)
$$
与 $W(k)$ 本身比较，Frobenius 为 Witt Frobenius。

**检查 K.2.** Prism 条件逐项为：

1. $W(k)$ 是 $p$-torsionfree $\delta$-环；
2. $(p)$ 由 nonzerodivisor 生成；
3. $W(k)$ 是 $p$-complete；
4. $p\in(p)$；
5. quotient $W(k)/(p)=k$ 的 $p^\infty$-torsion 由 $p$ 杀死。

## K.2 Hodge-Tate associated graded 的一维光滑模型

**例 K.3.** 令 $R=(A/I)\langle T\rangle$。若 $R$ over $A/I$ formally smooth of relative dimension $1$，则
$$
\Omega^0_{R/(A/I)}=R,\qquad
\Omega^1_{R/(A/I)}=R\,dT,\qquad
\Omega^i=0\ (i\ge2).
$$
因此 Hodge-Tate comparison 的 associated graded 只有两项：
$$
\operatorname{gr}^0\simeq R,\qquad
\operatorname{gr}^1\simeq R\,dT[-1]\{-1\}.
$$

**说明 K.4.** 该计算不证明 comparison theorem；它只说明一旦 comparison theorem 成立，graded pieces 的形状如何读出。

## K.3 Derived fixed point 的两项计算

**例 K.5.** 令 $C=\mathbf Z_p$，$\varphi=\operatorname{id}$。则
$$
C^{\varphi=1}=\operatorname{fib}(\mathbf Z_p\xrightarrow{0}\mathbf Z_p).
$$
因此
$$
H^0(C^{\varphi=1})=\mathbf Z_p,\qquad H^1(C^{\varphi=1})=\mathbf Z_p.
$$

**警告 K.6.** 这个例子说明 derived fixed point 可能有高一阶 cohomology。普通 fixed subgroup 只给出 $H^0$。

## K.4 $q$-difference 计算

**例 K.7.** 对 $f(T)=T^n$，
$$
\nabla_q(T^n)=\frac{(qT)^n-T^n}{qT-T}
=\frac{q^n-1}{q-1}T^{n-1}
=[n]_qT^{n-1}.
$$
当 $q\to1$ 时，$[n]_q\to n$，得到普通导数 $nT^{n-1}$。

## K.5 Breuil-Kisin module 的 rank-one 模型

**例 K.8.** 令 $M=\mathfrak S e$ 为自由 rank-one module。一个 Frobenius-semilinear map 由
$$
\varphi_M(1\otimes e)=a e,\qquad a\in\mathfrak S
$$
决定。它在 invert $E(u)$ 后为同构，当且仅当 $a$ 在 $\mathfrak S[1/E(u)]$ 中为单位。

**命题 K.9.** 若 $a=E(u)^h\cdot b$，其中 $b\in\mathfrak S^\times$，则例 K.8 给出 height $h$ 型 rank-one Breuil-Kisin module 的基本模型。

**证明.** Localizing at $E(u)$ 后，$E(u)^h$ 成为单位，$b$ 已为单位，因此 $a$ 为单位，Frobenius linearization 为同构。Height 的精确定义依赖文献 convention，但该模型展示了 $E(u)$-power 如何度量 Frobenius cokernel。证毕。

## K.6 Syntomic fibre 的两项模型

**例 K.10.** 令 $C$ 为 degree $0$ 的 $p$-complete ring，令 $N^{\ge i}C\subset C$ 为子模，且给定 map
$$
\varphi_i:N^{\ge i}C\to C.
$$
则 convention form 的 syntomic fibre
$$
\operatorname{fib}(N^{\ge i}C\xrightarrow{\varphi_i-1}C)
$$
由两项 complex 控制，其 cohomology 满足
$$
H^0=\ker(\varphi_i-1),\qquad
H^1=\operatorname{coker}(\varphi_i-1).
$$

**警告 K.11.** 例 K.10 只适用于 degree $0$ 模型。实际 prismatic syntomic complex 处在 derived category 中，且 $N^{\ge i}$、twist、mod $p^r$ 和 truncation 需要按第七章、十一章和附录 F 的 convention 解释。
在带 Tate twist 的正式公式中，本书写作 $\varphi_i-\operatorname{can}_i$；例 K.10 的 $-1$ 只表示无 twist、源目标已相同的玩具情形。

## K.7 二维 Hodge-Tate graded 模型

**例 K.12.** 令 $R=(A/I)\langle T_1,T_2\rangle$。若 $R$ over $A/I$ formally smooth of relative dimension $2$，则
$$
\Omega^1_{R/(A/I)}=R\,dT_1\oplus R\,dT_2,
$$
且
$$
\Omega^2_{R/(A/I)}=R\,dT_1\wedge dT_2.
$$
Hodge-Tate associated graded 只有 $i=0,1,2$ 三层：
$$
R,\qquad
(R\,dT_1\oplus R\,dT_2)[-1]\{-1\},\qquad
R\,dT_1\wedge dT_2[-2]\{-2\}.
$$

## 本附录小结

这些 worked examples 覆盖六个高频计算：crystalline prism 检查、Hodge-Tate graded pieces、derived fixed points、$q$-difference、syntomic fibre 和 Breuil-Kisin rank-one modules。它们是后续阅读大型比较定理的局部模型。

## 练习

**练习 K.1.** 对 $R=(A/I)\langle T_1,T_2\rangle$ 计算 Hodge-Tate associated graded 的形状。

**练习 K.2.** 对 $C=\mathbf Z_p$、$\varphi=p$，计算 $C^{\varphi=1}$ 的 cohomology。

**练习 K.3.** 在例 K.8 中判断 $a=p$ 时是否 invert $E(u)$ 后为单位，需区分 $p$ 与 $E(u)$ 的关系。
