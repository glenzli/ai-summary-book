# 附录 AN：Grauert 定理的 Banach 复形证明模块

## AN.0 目标

附录 AC 把 Grauert direct image theorem 作为输入，并证明它推出紧复空间相干上同调有限性。本附录补充 Grauert 定理证明中最常用的局部模型：把 proper family 上的相干上同调化为有限 Banach 复形的上同调，再由有限 presentation 得到直接像相干性。

Grauert 定理本身仍是输入；本附录证明输入定理的形式后果，并把证明义务拆成可检查模块。

## AN.1 Privileged covering 与 Banach 截面

设

$$
f:X\to Y
$$

为 proper holomorphic map，\(\mathcal F\) 为 \(X\) 上相干解析层。固定 \(y_0\in Y\)。取 \(y_0\) 的小 Stein 邻域 \(V\)，并取 \(f^{-1}(V)\) 的有限 Stein 覆盖

$$
\mathfrak U=\{U_i\}_{i=1}^N
$$

使所有有限交 \(U_I\) Stein。

**输入定理 AN.1（Grauert privileged covering）.** 可选择 \(V\)、\(\mathfrak U\) 和 \(\mathcal F\) 在每个 \(U_I\) 上的有限自由 presentation，使 Čech 复形

$$
C^\bullet(\mathfrak U,\mathcal F)
$$

可由有限个 Banach \( \mathcal O_Y(V)\)-模和连续 \( \mathcal O_Y(V)\)-线性映射表示，并且其 fiber 复形计算

$$
H^q(X_y,\mathcal F|_{X_y})
$$

对 \(y\in V\) 成立。

这里 Banach 结构来自紧包含的 Stein 多圆柱上全纯函数的 sup 范数。

## AN.2 有限 Banach 复形的相干上同调

本节证明 Grauert 证明中的代数部分。

设 \(A=\mathcal O_Y(V)\)。考虑有界复形

$$
E^\bullet:\quad
0\to E^0\xrightarrow{d^0}E^1\xrightarrow{d^1}\cdots\xrightarrow{d^{m-1}}E^m\to0,
$$

其中每个 \(E^i\) 是有限自由 \(A\)-模。

**命题 AN.2（有限自由复形的上同调有限 presentation）.** 若每个

$$
\operatorname{im}d^{i-1}\subset\ker d^i
$$

在解析意义下由有限个截面局部生成，则

$$
H^i(E^\bullet)=\ker d^i/\operatorname{im}d^{i-1}
$$

是相干 \(A\)-模。

**证明.** 有限自由 \(A\)-模对应有限秩自由解析层。相干解析层范畴对 kernel 封闭，所以 \(\ker d^i\) 相干。有限生成假设给 \(\operatorname{im}d^{i-1}\) 是相干子层。相干层对 cokernel 封闭，故商 \(\ker d^i/\operatorname{im}d^{i-1}\) 相干。证毕。

**输入定理 AN.3（Grauert finite generation step）.** 在 AN.1 的 privileged covering 下，\(\operatorname{im}d^{i-1}\) 在 \(\ker d^i\) 中局部有限生成。更精确地，经过缩小 \(V\)，存在有限个 cocycle 代表，使每个 fiber 上同调类由这些代表生成，并且生成关系随参数全纯变化。

AN.3 是 Grauert 证明的核心分析步骤；它使用紧性、Banach 复形扰动、开映射定理和半连续性。

**定理 AN.4（局部相干直接像模型）.** 在 AN.1 与 AN.3 下，

$$
R^qf_\ast\mathcal F|_V
$$

由有限 presentation 的解析层给出，因而相干。

**证明.** 由 AN.1，\(R^qf_\ast\mathcal F|_V\) 的截面由 \(C^\bullet(\mathfrak U,\mathcal F)\) 的第 \(q\) 上同调给出。AN.3 给有限生成的 image 条件，AN.2 给该上同调相干。证毕。

## AN.3 半连续性与 base change

设 \(E^\bullet\) 是有限自由 \(A\)-复形。对 \(y\in V\)，记

$$
E^\bullet_y=E^\bullet\otimes_A\mathbb C(y).
$$

**命题 AN.5（有限复形维数上半连续）.** 函数

$$
y\mapsto \dim_{\mathbb C}H^q(E^\bullet_y)
$$

上半连续。

**证明.** 在局部平凡化中，微分 \(d^i_y\) 是矩阵，其元素为全纯函数在 \(y\) 的取值。矩阵秩函数下半连续，因为某个 \(r\times r\) minor 非零是开条件。于是

$$
\dim H^q(E_y^\bullet)=
\dim E^q_y-\operatorname{rank}d_y^q-\operatorname{rank}d_y^{q-1}
$$

是上半连续函数。证毕。

**命题 AN.6（base change 的代数判别）.** 若 \(H^q(E^\bullet)\) 在 \(y_0\) 附近局部自由，且 \(H^{q+1}(E^\bullet)\) 无 \(y_0\)-邻域上的相关 torsion 阻碍，则自然映射

$$
H^q(E^\bullet)\otimes_A\mathbb C(y)
\to
H^q(E^\bullet_y)
$$

在 \(y_0\) 的邻域内为同构。

**证明.** 对复形 \(E^\bullet\) 截断，取短正合列

$$
0\to Z^q\to E^q\to B^{q+1}\to0
$$

和

$$
0\to B^q\to Z^q\to H^q\to0.
$$

张量 \(\mathbb C(y)\) 后的正合性由 Tor 项控制。若 \(H^q\) 局部自由且下一阶阻碍消失，则 Tor 项为零，fiber 上的 cycles、boundaries 与先取 cohomology 后取 fiber 相容。证毕。

## AN.4 到紧空间有限性的闭合

**推论 AN.7（Grauert 到点）.** 若 \(Y=\{*\}\)，则 AN.4 给

$$
H^q(X,\mathcal F)
$$

有限维。

**证明.** 点上的 \(A=\mathbb C\)。有限 presentation 的 \(\mathbb C\)-模就是有限维向量空间。AN.4 的局部模型在唯一点处给 \(R^qf_\ast\mathcal F=H^q(X,\mathcal F)\)。证毕。

## AN.5 凝聚/analytic 语言中的记录方式

在第三卷正文中使用 Grauert 时，应记录四项数据：

1. proper map \(f:X\to Y\)；
2. 相干层 \(\mathcal F\) 或 bounded coherent complex；
3. privileged covering 输入给出的有限 Banach 复形；
4. 由 finite presentation 得到的 \(R^qf_\ast\mathcal F\) 相干性。

这些数据进入 condensed/analytic 范畴后，有限 Banach 复形应被替换为对应的 liquid/analytic 对象；上同调有限 presentation 则对应 compact 或 perfect 性质。

## 练习

1. 对两项复形 \(E^0\to E^1\)，写出 \(H^0\) 与 \(H^1\) 的有限 presentation。
2. 证明矩阵秩函数下半连续。
3. 在 AN.6 中找出 Tor 项出现的位置。
4. 解释 properness 在 AN.1 中为何不可删除。
