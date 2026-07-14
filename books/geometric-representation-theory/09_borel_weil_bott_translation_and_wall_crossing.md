# 第九章：Borel-Weil-Bott、translation functors 与 wall crossing

一条权 $\lambda$ 在旗簇上给出线丛 $\mathcal L_\lambda$，但其上同调不总出现在次数零，也不总产生表示。真正控制答案的是 $\lambda+\rho$ 相对 Weyl chambers 的位置：落在墙上时全部消失，离开墙时唯一的 Weyl 元素把它送回 dominant chamber，并把长度记录为上同调次数。$SL_2/B\simeq\mathbb P^1$ 上的 $\mathcal O(n)$ 可把这一规律逐项算出。随后将权跨过一堵墙，代数侧表现为 translation functor，几何侧则表现为沿 $G/B\to G/P_s$ 的 pull--push；这使 Borel--Weil--Bott 的静态参数与 category $\mathcal O$ 的函子结构相连。

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

射影直线上的线丛上同调让定理的三个分支全部出现：$n\ge0$ 对应恒等 Weyl 元素，$n=-1$ 位于墙上，$n\le-2$ 则由唯一反射送回 dominant 区域。

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

张量再投影是纯代数构造；要看见“跨墙”为什么对应一个几何方向，必须回到相邻 Borel flags 忘掉一步所得的 minimal parabolic projection。

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

$SL_2$ 的计算把 dot action、上同调次数与 Serre duality 放在同一个公式中；一般情形由 Borel--Weil--Bott 外部输入控制。Translation functor 通过有限维表示张量并投影到另一个 block，minimal parabolic pull--push 则给出其几何影像。下一章转向 annihilator 与 associated variety，考察一个表示在 cotangent 与 nilpotent 几何中留下的微局部支撑。

## 练习

**练习 9.1.** 对 $G=SL_2$，计算 $\mathcal O_{\mathbb P^1}(n)$ 的 cohomology，并与 Borel-Weil-Bott 对照。

**练习 9.2.** 证明 $V\otimes-$ 是 exact functor on $\mathcal O$。

**练习 9.3.** 对 $SL_2$ 写出 $\pi_s:G/B\to G/P_s$ 的含义，并解释为什么这个例子退化。
