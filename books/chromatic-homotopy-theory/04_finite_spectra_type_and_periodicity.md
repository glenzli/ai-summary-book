# 第四章：有限谱的 type、nilpotence 与 periodicity

## 本章目标

本章进入 chromatic theory 的核心定理包：有限谱按 Morava K-theory 的首次非消失高度分层；有限谱的 thick 子范畴完全由 type 分类；type $n$ 有限谱携带 $v_n$-周期自映射。大型定理作为 Hopkins-Smith/Devinatz-Hopkins-Smith 外部输入处理。

## 依赖前置知识

需要第一章的有限谱和第二、三章的 $K(n)$。nilpotence theorem、periodicity theorem 和 thick subcategory theorem 不预设证明，但要理解其陈述。

## 4.1 Type

**定义 4.1.** 非零有限 $p$-局部谱 $X$ 的 type 为 $n$，若
$$
K(n-1)_*X=0,\qquad K(n)_*X\ne0,
$$
其中 $K(0)=H\mathbb Q$，并且对 $n=0$ 只要求 $K(0)_*X\ne0$。零谱的 type 记作 $\infty$。

**外部输入 4.2 (finite detection).** 每个非零有限 $p$-局部谱有有限 type。

**命题 4.3.** 若有限谱 $X$ 有 type $n$，则对所有 $m<n$ 有 $K(m)_*X=0$。

**证明.** 对 $m=n-1$ 是定义。对更小的 $m$，该结论属于 Morava K-theory 对有限谱的厚子范畴嵌套性质；严格证明依赖 thick subcategory theorem 或 Ravenel conjectures 定理包。因此本命题在完整形式下记为外部输入的直接推论。证毕。

**警告 4.4.** 不能把定义 4.1 改写成“只要 $K(n)_*X\ne0$ 就 type $n$”。同一个有限谱可能被更高的 $K(m)$ 检测到；type 记录的是第一个非消失高度。

## 4.2 厚子范畴

**定义 4.5.** 有限谱范畴的 thick 子范畴是全稳定子范畴 $\mathcal T\subseteq\mathbf{Sp}^{\omega}_{(p)}$，对 cofiber、悬挂、脱悬挂和 retract 封闭。

**定义 4.6.** 对 $n\ge0$，定义
$$
\mathcal C_n=\{X\in\mathbf{Sp}^{\omega}_{(p)}\mid K(n-1)_*X=0\},
$$
其中 $\mathcal C_0=\mathbf{Sp}^{\omega}_{(p)}$。

**命题 4.7.** $\mathcal C_n$ 是 thick 子范畴。

**证明.** $K(n-1)_*(-)$ 是同调理论，因此把 cofiber 序列送到长正合列。若两项的 $K(n-1)_*$ 为零，第三项也为零。悬挂只平移次数。retract 情形中，若 $X$ 是 $Y$ 的 retract，则 $K(n-1)_*X$ 是 $K(n-1)_*Y$ 的 retract；若后者为零，前者为零。证毕。

**外部输入定理 4.8 (Hopkins-Smith thick subcategory theorem).** 有限 $p$-局部谱范畴中的每个非零 proper thick 子范畴都等于某个 $\mathcal C_n$。因此 thick 子范畴链为
$$
\mathcal C_0\supsetneq\mathcal C_1\supsetneq\mathcal C_2\supsetneq\cdots.
$$

**使用说明.** 本书把定理 4.8 作为外部输入。任何用 type 分类 thick 子范畴的证明必须引用该定理，不能只引用 $K(n)_*$ 的系数环形式。

## 4.3 Nilpotence

**定义 4.9.** 设 $R$ 是 ring spectrum。元素 $\alpha\in\pi_dR$ 称为 nilpotent，若存在 $N>0$ 使得乘法意义下
$$
\alpha^N=0\in\pi_{Nd}R.
$$

**外部输入定理 4.10 (Devinatz-Hopkins-Smith nilpotence).** 对 ring spectrum $R$，若 $\alpha\in\pi_*R$ 在 $MU_*R$ 中为零，则 $\alpha$ nilpotent。等价表述需要按文献模型精确定位。

**推论 4.11.** 球谱正次数稳定同伦中的许多元素的 nilpotence 可由 complex cobordism 检测。

**证明草图.** 将 $R$ 取为球谱或相关 endomorphism ring spectrum，应用 nilpotence theorem。Nishida nilpotence 是早期特例。完整比较需定位 DHS 定理表述。证毕。

## 4.4 Periodicity 和 $v_n$ self-map

**定义 4.12.** 设 $X$ 是 type $n$ 有限谱。一个 $v_n$ self-map 是某个正次数映射
$$
f:\Sigma^dX\to X
$$
使得 $K(n)_*f$ 是同构，并且对 $m\ne n$ 的 Morava K-theory 作用满足 Hopkins-Smith periodicity theorem 中的相应 nilpotence/zero condition。精确版本见附录 I；正文在 locator 完成前只使用存在性、$K(n)$-周期性和唯一到幂这三个后果。

**外部输入定理 4.13 (periodicity theorem).** 每个 type $n$ 有限谱存在 $v_n$ self-map，并且任意两个 $v_n$ self-maps 在取正幂后相容。更精确地说，$v_n$-周期自映射在适当意义下唯一到幂。

**定义 4.14.** 给定 $v_n$ self-map $f:\Sigma^dX\to X$，令 $\tilde f:X\to\Sigma^{-d}X$ 为脱悬挂伴随。其 telescope 定义为
$$
f^{-1}X=\operatorname*{colim}\left(X\xrightarrow{\tilde f}\Sigma^{-d}X\xrightarrow{\Sigma^{-d}\tilde f}\Sigma^{-2d}X\to\cdots\right).
$$
其 Bousfield 类记作 $T(n)$，在 periodicity theorem 保证下与选择的 type $n$ 有限谱和 self-map 在适当范围内无关。

**警告 4.15.** $T(n)$ 与 $K(n)$ 在 Bousfield 局部化上不能默认相同。telescope conjecture 断言的正是这类比较；2023 年之后高度至少 $2$ 的一般等同性已被反例否定。

## 4.5 Type 的低阶例子

**例 4.16.** 球谱 $\mathbb S_{(p)}$ 是 type $0$。

**证明.** $K(0)_*\mathbb S_{(p)}=H\mathbb Q_*\mathbb S_{(p)}\cong\mathbb Q$ 集中在 degree $0$，非零。按 type $0$ 定义，球谱为 type $0$。证毕。

**例 4.17.** Moore spectrum $M(p)$ 不是 type $0$。

**证明.** $M(p)$ 是 cofiber
$$
\mathbb S_{(p)}\xrightarrow{p}\mathbb S_{(p)}\to M(p).
$$
张量 $H\mathbb Q$ 后，乘以 $p$ 在 $H\mathbb Q$ 上为等价，故 cofiber 为零。因此 $K(0)_*M(p)=0$，所以不是 type $0$。证毕。

**外部输入 4.18.** $M(p)$ 在合适素数范围和模型下是 type $1$ 的基本例子；完整陈述和 $v_1$ self-map 存在性属于 periodicity theorem 的低高度实例。

## 4.6 Thick theorem 的使用格式

**命题 4.19.** 若 $\mathcal T$ 是有限 $p$-局部谱的非零 thick 子范畴，则存在唯一 $n$ 使
$$
\mathcal T=\mathcal C_n.
$$

**证明.** 这是 Hopkins-Smith thick subcategory theorem 的直接表述。唯一性来自严格包含链
$$
\mathcal C_0\supsetneq\mathcal C_1\supsetneq\cdots.
$$
严格包含本身也是外部输入的一部分，因为需要存在每个 type 的有限谱。证毕。

**使用规则 4.20.** 调用 thick theorem 时必须说明：

1. 对象在有限 $p$-局部谱范畴；
2. 子范畴对 cofiber、悬挂、脱悬挂和 retract 封闭；
3. 结论分类的是 thick 子范畴，不是 arbitrary full subcategory；
4. 若涉及 tensor ideal，应额外说明张量封闭是否已知。

## 本章小结

有限谱是 chromatic theory 最刚性的对象。type 给出高度分层，thick subcategory theorem 说明这个分层穷尽所有 thick 子范畴，periodicity theorem 给出 $v_n$ self-map 和 telescopes。三大定理均是外部输入，本书后续会频繁使用，但不会伪装成内部证明。

## 练习

**练习 4.1.** 证明命题 4.7 中 retract 封闭的细节。

**练习 4.2.** 若 $X$ 是 type $n$ 有限谱，解释为什么 $K(n)_*X\ne0$ 不推出 $L_{K(n)}X\simeq X$。

**练习 4.3.** 写出 telescope $f^{-1}X$ 的 colimit 定义中每个箭头的次数，检查它们都变成从同一悬挂规范下的谱到下一项的映射。
