# 第九章：Borel-Weil-Bott、translation functors 与 wall crossing

## 本章目标

本章把第一章的 line bundles on $G/B$ 与第二章的 highest weight representations 连接起来，陈述 Borel-Weil-Bott theorem，并给出 translation functors 和 wall crossing 的代数与几何接口。核心定理作为外部输入，内部证明集中在 functorial construction 和类型检查。

## 依赖前置知识

需要第一章的 $\mathcal L_\lambda$、第二章的 category $\mathcal O$ 和第八章的 localization 语言。

## 9.1 Dominant weights 和 line bundle cohomology

**定义 9.1.** 权 $\lambda\in X^\ast(T)$ 称为 dominant，若
$$
\langle\lambda,\alpha^\vee\rangle\ge0
$$
对所有 simple coroot $\alpha^\vee$ 成立。若所有不等式严格，则称为 strictly dominant。

**定义 9.2.** 对 $\lambda\in X^\ast(T)$，令
$$
\mathcal L_\lambda=G\times^B k_{-\lambda}
$$
如定义 1.15。其 sheaf cohomology 记为
$$
H^i(\mathcal B,\mathcal L_\lambda).
$$
$G$ 左作用在 $\mathcal B$ 和 $\mathcal L_\lambda$ 上，因此每个 $H^i$ 是 rational $G$-representation。

**命题 9.3.** $H^i(\mathcal B,\mathcal L_\lambda)$ 上的 $G$-作用由 functoriality 给出。

**证明.** 对 $g\in G$，左乘给出 automorphism $L_g:\mathcal B\to\mathcal B$ 和 line bundle 同构 $L_g^\ast\mathcal L_\lambda\simeq\mathcal L_\lambda$。sheaf cohomology 对 automorphism 和 sheaf isomorphism functorial，因此得到线性自同构
$$
H^i(\mathcal B,\mathcal L_\lambda)\to H^i(\mathcal B,\mathcal L_\lambda).
$$
群乘法相容性来自 $L_{g_1g_2}=L_{g_1}\circ L_{g_2}$ 和 associated bundle 构造的自然性。$\square$

**外部输入定理 9.4.** Borel-Weil theorem：若 $\lambda$ dominant，则
$$
H^0(\mathcal B,\mathcal L_\lambda)^\vee
$$
是最高权为 $\lambda$ 的 irreducible representation，且 $H^i(\mathcal B,\mathcal L_\lambda)=0$ for $i>0$。  
归一化依赖 $\mathcal L_\lambda=G\times^B k_{-\lambda}$ 的符号约定。

**外部输入定理 9.5.** Borel-Weil-Bott theorem：若 $\lambda+\rho$ singular，则所有 $H^i(\mathcal B,\mathcal L_\lambda)$ 为零。若 $\lambda+\rho$ regular，存在唯一 $w\in W$ 使
$$
w(\lambda+\rho)-\rho
$$
dominant，且唯一非零 cohomology 出现在 $i=\ell(w)$，并给出相应 highest weight representation。

**例 9.5.1.** 对 $G=SL_2$，$\mathcal B\simeq\mathbb P^1$，线丛可写为 $\mathcal O_{\mathbb P^1}(n)$。经典计算给出
$$
H^0(\mathbb P^1,\mathcal O(n))\simeq \operatorname{Sym}^n(E^2)^\vee,\qquad n\ge0,
$$
且 $H^1(\mathbb P^1,\mathcal O(n))=0$ for $n\ge -1$。若 $n\le -2$，Serre duality 给出
$$
H^1(\mathbb P^1,\mathcal O(n))\simeq H^0(\mathbb P^1,\mathcal O(-n-2))^\vee.
$$

**命题 9.5.2.** 上述 $SL_2$ 计算与 dot action 公式相容。

**证明.** $SL_2$ 的 Weyl group 为 $\{e,s\}$，$\rho$ 为 fundamental weight。若权用整数 $n$ 表示，则 dot action 为
$$
s\cdot n=-n-2.
$$
当 $n\ge0$ 时，$n$ 已 dominant，非零 cohomology 在 degree $0$。当 $n=-1$ 时，$n+\rho=0$ singular，cohomology 全消失。若 $n\le-2$，则 $s\cdot n=-n-2\ge0$，非零 cohomology 出现在 $\ell(s)=1$，并由 Serre duality 识别为最高权 $-n-2$ 的表示。$\square$

## 9.2 Translation functors

**定义 9.6.** 设 $\chi,\chi'$ 为两个中心 characters，$V$ 为有限维 $G$-representation。translation functor 的基本形式为
$$
T_{\chi}^{\chi'}(M)=\operatorname{pr}_{\chi'}(V\otimes M),
$$
其中 $\operatorname{pr}_{\chi'}$ 表示投影到 $\mathcal O_{\chi'}$ block。

**命题 9.7.** 若 $M\in\mathcal O_\chi$ 且 $V$ 有限维，则 $V\otimes M\in\mathcal O$。

**证明.** $V\otimes M$ 是 finitely generated，因为 $V$ 有限维且 $M$ 由有限多个元素生成。它是 weight module，weight spaces 为有限直和
$$
(V\otimes M)_\nu=\bigoplus_{\alpha+\beta=\nu}V_\alpha\otimes M_\beta,
$$
其中对固定 $\nu$ 只有有限个 $\alpha$ 出现。$\mathfrak n$ locally finite：对 $v\otimes m$，$U(\mathfrak n)v$ 有限维，因为 $V$ 有限维，$U(\mathfrak n)m$ 有限维，因为 $M\in\mathcal O$；Leibniz rule 下生成空间包含在有限维张量积中。因此 $V\otimes M\in\mathcal O$。$\square$

**外部输入定理 9.8.** Translation functors 控制 regular、singular blocks 之间的关系，并与 wall crossing functors、projective functors 和 Hecke algebra action 相容。  
来源：BGG、Jantzen、Soergel。

## 9.3 Wall crossing 的几何解释

**定义 9.9.** 对 simple reflection $s\in S$，wall crossing functor 可表示为
$$
\theta_s=T_{\mu}^{\lambda}T_{\lambda}^{\mu}
$$
的形式，其中 $\lambda$ regular，$\mu$ 位于 $s$-wall 上。具体权的选择和 block projection 需要中心 character convention。

**外部输入定理 9.10.** 在 Beilinson-Bernstein localization 下，wall crossing functors 对应 flag variety 上沿 minimal parabolic projection
$$
\pi_s:G/B\to G/P_s
$$
的 pull-push functor
$$
\pi_s^\ast\pi_{s\ast}
$$
的适当 shifted/twisted 版本。

**命题 9.11.** $\pi_s:G/B\to G/P_s$ 是 proper。

**证明.** $G/B$ 和 $G/P_s$ 都是 projective varieties，$\pi_s$ 是 projective varieties 之间的 morphism。projective varieties 到 $\operatorname{Spec}\mathbb C$ proper，且 proper morphisms 在复合和基变换下稳定。由 $G/B\to G/P_s$ 可由 quotient map 诱导并作为 projective morphism 处理，故 proper。$\square$

## 本章小结

本章建立 line bundle cohomology 与表示的接口，补充 $SL_2$ 的 Borel-Weil-Bott 计算，定义 translation functors，并说明 wall crossing 与 minimal parabolic projection 的 pull-push 关系。一般 Borel-Weil-Bott、translation theory 和 wall crossing-localization 对应均为外部输入。

## 练习

**练习 9.1.** 对 $G=SL_2$，计算 $\mathcal O_{\mathbb P^1}(n)$ 的 cohomology，并与 Borel-Weil-Bott 对照。

**练习 9.2.** 证明 $V\otimes-$ 是 exact functor on $\mathcal O$。

**练习 9.3.** 对 $SL_2$ 写出 $\pi_s:G/B\to G/P_s$ 的含义，并解释为什么这个例子退化。
