# 附录 N：低高度与 fracture worked examples

本附录给出可直接检查的低高度样例。它不替代 Ravenel/Hovey-Strickland 等外部计算，只把本书定义链落到具体对象。

## N.1 高度 0 fracture

**例 N.1.** 对任意谱 $X$，$L_0X=L_{H\mathbb Q}X$ 是有理化，并且
$$
\pi_*L_0X\cong \pi_*X\otimes_{\mathbb Z}\mathbb Q.
$$

**证明.** 命题 F.1 已证明
$$
L_{K(0)}X\simeq H\mathbb Q\otimes X.
$$
把有理球谱写成沿正整数乘法映射的 filtered colimit
$$
H\mathbb Q\simeq\mathbb S_{\mathbb Q}simeq
\operatorname*{colim}_{n\mid m}\mathbb S,
$$
其中结构映射在同伦群上实现乘以 $m/n$。Smash product 与稳定同伦群都与 filtered colimit 相容，故
$$
\pi_*(H\mathbb Q\otimes X)\cong\pi_*X\otimes_{\mathbb Z}\mathbb Q.
$$
这对任意谱成立，不需要有限性假设。证毕。

**例 N.2.** 若 $X=M(p)$，则 $L_0M(p)\simeq0$。

**证明.** 见附录 F.7。$H\mathbb Q\otimes M(p)\simeq0$，故其 $H\mathbb Q$-localization 为零。证毕。

## N.2 高度 1 fracture square

**例 N.3（任意输入的高度一方块）.** 对任意
$X\in\mathbf{Sp}_{(p)}$，高度 $1$ fracture square 写作
$$
\begin{array}{ccc}
L_1X & \longrightarrow & L_{K(1)}X\\
\downarrow & & \downarrow\\
L_0X & \longrightarrow & L_0L_{K(1)}X
\end{array}
$$
若 $L_0X\simeq0$，则该 square 退化为
$$
L_1X\simeq \operatorname{fib}\left(L_{K(1)}X\to L_0L_{K(1)}X\right)
$$
而不是自动等于 $L_{K(1)}X$。

**证明.** 外部输入定理 5.11 对每个 $p$-局部谱成立，不要求有限性。
在 pullback square 中令左下角为零，左上角即为右边竖直映射的 fiber。
只有再证明右下角为零，才能把该 fiber 等同于右上角。证毕。

**警告 N.4（缺失的假设）.** 若不假设 $X$ 有限 dualizable，条件
$L_0X\simeq0$ 本身不推出 $L_0L_{K(1)}X\simeq0$：$K(1)$-localization
不是 smashing，不能擅自把 $L_{K(1)}$ 与 $L_0$ 或 smash product 交换。
因此例 N.3 的 fiber 一般不能删去。下一例说明有限 dualizable 输入为何
不同。

**例 N.4A（Moore spectrum 的 overlap 消失）.** 对 Moore spectrum
$M(p)$，有
$$
L_0M(p)\simeq0,
\qquad
L_0L_{K(1)}M(p)\simeq0,
$$
从而
$$
M_1M(p)\simeq L_1M(p)\simeq L_{K(1)}M(p).
$$

**证明.** 例 N.2 给出第一项。$M(p)$ 是有限谱，故命题 1.14C 给出
$$
L_{K(1)}M(p)\simeq M(p)\otimes L_{K(1)}\mathbb S_{(p)}.
$$
$L_0$ 是 smashing（事实上是有理化），所以
$$
\begin{aligned}
L_0L_{K(1)}M(p)
&\simeq L_0\mathbb S_{(p)}\otimes M(p)
 \otimes L_{K(1)}\mathbb S_{(p)}\\
&\simeq L_0M(p)\otimes L_{K(1)}\mathbb S_{(p)}\simeq0.
\end{aligned}
$$
例 N.3 的 fiber 因而等于 $L_{K(1)}M(p)$。命题 N.5 将直接验证
$M(p)$ 为 type $1$；也可随后把结论视为命题 5.14A 的 $n=1$ 特例。
定义 5.4 再给出 $M_1M(p)\simeq L_1M(p)$。证毕。

## N.3 Moore spectrum 与 type

**命题 N.5.** 对每个素数 $p$，$M(p)$ 的 type 恰为 $1$。

**证明.** $K(0)_*M(p)=H\mathbb Q_*M(p)=0$，见例 N.2。对 cofiber
sequence
$$
\mathbb S_{(p)}\xrightarrow p\mathbb S_{(p)}\longrightarrow M(p)
$$
施加 $K(1)_*(-)$。由于 $K(1)_*=\mathbb F_p[v_1^{\pm1}]$，长正合列
中的乘 $p$ 映射为零，于是 $K(1)_*\to K(1)_*M(p)$ 单射。特别地
$K(1)_*M(p)\ne0$。按定义 4.1 的首次非消失高度，type 恰为 $1$。
这个证明不调用 finite detection 或 periodicity theorem。证毕。

**外部输入 N.6（只负责周期性）.** Hopkins--Smith periodicity theorem
保证 type $1$ 有限谱 $M(p)$ 存在某个 $v_1$-self-map。若要写最小周期、
具体 Adams map 或 point-set 模型，仍须按素数分别引用低高度计算；这些
细节不参与命题 N.5 的 type 判定。

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

低高度样例的价值在于暴露假设边界：对任意谱，$L_0X\simeq0$ 仍不能
自动删除 overlap；对有限 type $1$ 谱，dualizability 才使该 overlap
消失。$M(p)$ 的 type 可由 $K(0)$/$K(1)$ 长正合列直接判定，不需要把
finite detection 当作替代计算。$K(2)$-local tmf 仍需要 supersingular
groupoid 数据。
