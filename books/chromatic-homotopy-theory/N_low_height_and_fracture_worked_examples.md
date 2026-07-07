# 附录 N：低高度与 fracture worked examples

本附录给出可直接检查的低高度样例。它不替代 Ravenel/Hovey-Strickland 等外部计算，只把本书定义链落到具体对象。

## N.1 高度 0 fracture

**例 N.1.** 对任意谱 $X$，$L_0X=L_{H\mathbb Q}X$ 是有理化。若 $X$ 是有限 $p$-局部谱，则 $\pi_*L_0X\cong \pi_*X\otimes\mathbb Q$。

**证明草图.** $K(0)=H\mathbb Q$。有理化 localization 对有限谱同伦群逐次张量 $\mathbb Q$；有限性保证没有 completion/pathology。一般谱需用 derived rationalization 表述。证毕。

**例 N.2.** 若 $X=M(p)$，则 $L_0M(p)\simeq0$。

**证明.** 见附录 F.7。$H\mathbb Q\otimes M(p)\simeq0$，故其 $H\mathbb Q$-localization 为零。证毕。

## N.2 高度 1 fracture square

**例 N.3.** 对有限 $p$-局部谱 $X$，高度 $1$ fracture square 写作
$$
\begin{CD}
L_1X @>>> L_{K(1)}X\\
@VVV @VVV\\
L_0X @>>> L_0L_{K(1)}X.
\end{CD}
$$
若 $L_0X\simeq0$，则该 square 退化为
$$
L_1X\simeq \operatorname{fib}\left(L_{K(1)}X\to L_0L_{K(1)}X\right)
$$
而不是自动等于 $L_{K(1)}X$。

**证明.** 在 pullback square 中，若左下角为零，则左上角是右上角到右下角的 fiber。除非右下角也为零，否则不能删去该 fiber。证毕。

**警告 N.4.** 许多初学者会在 $L_0X=0$ 时把 $L_1X$ 误认为 $L_{K(1)}X$。右下角 $L_0L_{K(1)}X$ 仍可能携带 rational information inside the $K(1)$-local object。

## N.3 Moore spectrum 与 type

**命题 N.5.** $M(p)$ 的 type 至少为 $1$。

**证明.** $K(0)_*M(p)=H\mathbb Q_*M(p)=0$，见例 N.2。若 $M(p)$ 非零有限谱，则 finite detection 给出某个 $n\ge1$ 使 $K(n)_*M(p)\ne0$。因此 type 至少为 $1$。证毕。

**外部输入 N.6.** 在标准 chromatic theory 中，$M(p)$ 是 type $1$ 的基本例子，但 $v_1$ self-map 的具体存在和周期需要按素数和模型引用 periodicity theorem 或 Toda-Smith 复形相关结果。

## N.4 $K(1)$ 与 Adams operations

**外部输入 N.7.** 在高度 $1$，$K(1)$-local sphere 可通过 $p$-adic K-theory 和 Adams operations 描述。奇素数下常用 Adams summand，$p=2$ 时需要区分 $KO/KU$ 和实结构。

**使用规则 N.8.** 写 $K(1)$-local sphere 的 fiber 公式前必须指定：

1. $p$ 是否为奇素数；
2. 使用 $KU_p^\wedge$、Adams summand 还是 $KO$；
3. Adams operation $\psi^q$ 中 $q$ 的 topological generator 选择；
4. fiber 的悬挂 convention。

**警告 N.9.** 不同 convention 会把公式中的悬挂平移一位。正式教材在 locator 完成前不写具体 fiber 公式。

## N.5 Supersingular local model 的最小例子格式

**模板 N.10.** 一个 $K(2)$-local tmf worked example 必须记录：

| 项 | 内容 |
| --- | --- |
| prime | 例如 $p=3$ |
| moduli problem | 无 level 或指定 level |
| supersingular points | 点数和 automorphism group |
| local Morava theory | 对应 $E_2$ |
| descent groupoid | 单群或多点 groupoid |
| spectral sequence | $H^s(G;(E_2)_t)$ 或 groupoid cohomology |

**警告 N.11.** 在没有填表前，不能把 $TMF_{K(2)}$ 写成某个单一 $E_2^{hG}$。

## 本附录小结

低高度样例的价值在于暴露方向错误：$L_0$ 不是可忽略项，$L_1$ 不是自动 $K(1)$-localization，$M(p)$ 的 type 需要 finite detection，$K(2)$-local tmf 需要 supersingular groupoid 数据。
