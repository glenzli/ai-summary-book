# 第二十二章：Betti、etale、real etale 与 analytic realization

Realization 把 motivic 对象送入另一套同伦论，使几何问题可以由拓扑、Galois 作用或
层论不变量检测。不同 realization 忘掉的信息并不相同：复点不再区分 motivic weight，
etale homotopy type 必须保留逆系统，实点与带复共轭的复点又落入不同目标范畴。因此
“存在一个 realization”从不等于“它保守”，也不自动包含六操作相容性。

构造 realization 时有三道独立门槛。首先，几何函子要满足所选拓扑的下降；其次，它
必须反演 `\mathbb A^1`-投影；最后，稳定 realization 必须把 Tate sphere 送到可逆
对象。本章先用泛性质分离这三步，再逐一说明各经典构造的定义域、目标与局部化。

## 22.1 从几何函子到稳定 realization

**命题 22.1（因子化判据）.** 设 `\mathcal C` 是 presentable infinity-category，
`F:\operatorname{Sm}_S\to\mathcal C` 为函子。若 `F` 满足 Nisnevich descent 且

$$
F(X\times\mathbb A^1)\longrightarrow F(X)
$$

为等价，则 `F` 唯一地通过 unstable motivic category `\mathbf H(S)` 因子化。若
`\mathcal C` 还是稳定对称幺半范畴、该因子化保持相应余极限与幺半结构，并且
`F(T)` 可逆，则它进一步唯一地通过 `\mathbf{SH}(S)` 因子化。

**证明.** 第一项先使用 Nisnevich 层化的泛性质，再使用 `\mathbb A^1`-Bousfield
局部化的泛性质。第二项使用 pointed category 中对 Tate sphere `T` 的对称幺半反演
泛性质。每一步的唯一性均指相应函子 infinity-groupoid 可缩。`\square`

**注 22.2.** 命题 22.1 只构造单个基 `S` 上的函子。要得到随 `S` 变化并与
`f_*`、`f_!` 或 `f^!` 交换的系统，还必须验证 mate transformations；这些不由局部化
泛性质给出。

## 22.2 复 Betti realization

对光滑复概形 `X`，取解析拓扑空间 `X(\mathbb C)`。Nisnevich distinguished square
在复点上给出拓扑 excision，而

$$
(X\times\mathbb A^1)(\mathbb C)
=X(\mathbb C)\times\mathbb C\simeq X(\mathbb C).
$$

此外

$$
\operatorname{Re}_{B}(T)\simeq
\operatorname{hocofib}\bigl(\mathbb C^*\longrightarrow\mathbb C\bigr)
\simeq S^2.
$$

这里取的是 pointed spaces 中的 homotopy cofiber，而不是集合论商空间；其同伦型
等价于复直线在原点处的 Thom space。

因此命题 22.1 给出对称幺半复 Betti realization

$$
\operatorname{Re}_{B}:\mathbf{SH}(\mathbb C)\longrightarrow\mathbf{Sp},
\qquad
\Sigma_T^\infty X_+\longmapsto\Sigma^\infty X(\mathbb C)_+.
$$

**例子 22.3.** `\mathbb P^1(\mathbb C)\cong S^2`，故 pointed
`\mathbb P^1` 的 motivic suspension 在 Betti realization 下成为二重拓扑 suspension。
这同时核对了 `T\simeq(\mathbb P^1,\infty)` 的像。

绝对构造与相对六操作相容性必须分开陈述。

**外部输入定理 22.4（Ayoub）.** 对有限型复代数概形的相对 motivic categories，
Ayoub 构造到相应复解析层论范畴的 Betti realization system，并证明它与
Grothendieck 六操作及 nearby-cycle formalism 相容。这里引用的是
*Note sur les operations de Grothendieck et la realisation de Betti*, J. Inst. Math.
Jussieu 9 (2010), Theorem 3.19 及该文的主相容性结论。其对象模型和 constructibility
假设应按原文保留；这项定理不能被改写为“任意复栈上的 `\mathbf{SH}` realization
无条件交换所有六操作”。

## 22.3 Etale realization

Etale covers 形成的是一个余过滤近似系统；把它压成单一空间通常会忘掉 fundamental
group 的有限商与高阶 etale cohomology。因此 etale realization 的自然目标是
pro-spaces、profinite spaces 或相应稳定范畴。

**外部输入定理 22.5（Isaksen）.** 设 `S` 为概形。Etale topological type 给出从
etale site 上 simplicial presheaves 到 pro-spaces 的同伦不变函子。完成于
`S` 的各剩余域特征之外以后，它通过 Morel--Voevodsky 的 unstable
`\mathbb A^1`-homotopy category 因子化。来源为 Isaksen,
*Etale realization on the A1-homotopy theory of schemes*, arXiv:math/0106158。

**外部输入定理 22.6（Quick）.** 对任意特征的基域，存在稳定 profinite homotopy
theory，以及从光滑概形的 stable motivic homotopy theory 到该理论的 etale
topological realization。来源为 Quick,
*Stable etale realization and etale cobordism*, arXiv:math/0608313。

这两个定理的目标和假设不同：前者明确给出完成后的 unstable pro-space 构造；后者
使用 profinite stable target。不能把它们合并成一个取值于 ordinary spectra 的
无条件函子。Ayoub 的 etale motives realization 还具有六操作相容结果，但那是另一
套系数与目标范畴，也须单独引用。

## 22.4 两种实 realization 与 real-etale 局部化

对 `\mathbb R` 上的概形至少有两种拓扑输入：

1. `X(\mathbb R)` 给出非等变 real-points realization；
2. `X(\mathbb C)` 带复共轭作用，给出取值于 genuine `C_2`-spectra 的 equivariant
   Betti realization。

二者不能互换。例如 `X=\operatorname{Spec}\mathbb C` 作为实概形没有实点，但
`X(\mathbb C)` 非空且带交换两个嵌入的 `C_2`-作用。

Real-etale theory 又是第三种构造。记
`\rho:\mathbb 1\to\mathbb G_m` 为由单位 `-1` 给出的稳定 motivic 类（符号按
Bachmann 的约定）。

**外部输入定理 22.7（Bachmann）.** 若 `S` 是有限维 Noetherian 概形，则有典范等价

$$
\mathbf{SH}(S)[\rho^{-1}]
\simeq
\mathbf{SH}(S_{\mathrm{ret}}),
$$

右端是小 real-etale site 上的局部 stable homotopy category。特别地，

$$
\mathbf{SH}(\mathbb R)[\rho^{-1}]\simeq\mathbf{Sp}.
$$

来源为 Bachmann, *Motivic and real etale stable homotopy theory*, 主定理。结论是
`\rho`-局部化后的范畴等价，不是 integral `\mathbf{SH}(S)` 与普通拓扑谱范畴的等价。

## 22.5 六操作相容意味着什么

设 `R_X:\mathcal D(X)\to\mathcal E(X^{\mathrm{an}})` 随 `X` 变化。称 `R` 与指定
六操作相容，是指在允许的态射类上给出相干等价，例如

$$
R_Xf^*\simeq(f^{\mathrm{an}})^*R_Y,
\qquad
R_Yf_!\simeq(f^{\mathrm{an}})_!R_X,
$$

以及与 `f_*`、`f^!`、tensor、internal Hom、base change 和 projection formula
对应的等价。右伴随相容还可能只在 constructible objects 上成立，必须保留来源中的
限制。

**命题 22.8.** 与 inverse image 相容不推出与 exceptional pushforward 相容。

**证明.** `f^*` 的相容可由几何 pullback 直接产生；`f_!` 还编码 compact support、
开浸入延零和 proper compactification。即使 `R_Xf^*\simeq(f^{\mathrm{an}})^*R_Y`，
其伴随 mate 也只有在 `R` 保持相应伴随、紧支撑与粘合数据时才可逆。因此需要像定理
22.4 那样的独立相容性定理。`\square`

**命题 22.9.** 若 `R` 正合，并与闭开对 `(i,j)` 的 `j_!,j^*,i_*,i^*` 相容，则它把
motivic localization cofiber sequence 送到目标理论的 localization sequence。

**证明.** 对

$$
j_!j^*E\longrightarrow E\longrightarrow i_*i^*E
$$

施加正合函子 `R`，再用四个相容等价逐项替换，即得到

$$
j_!j^*R(E)\longrightarrow R(E)\longrightarrow i_*i^*R(E).
$$

正合性保证它仍为 cofiber sequence。`\square`

## 22.6 目标、局部化与可见信息

| 构造 | 典型目标 | 必要修正 | 主要保留的信息 |
| --- | --- | --- | --- |
| complex Betti | `\mathbf{Sp}` 或解析层论 | 复嵌入/相对有限型假设 | 复点拓扑 |
| equivariant Betti over `\mathbb R` | `\mathbf{Sp}^{C_2}` | 保留复共轭 | 实结构与共轭固定点 |
| Isaksen etale | pro-spaces | 远离剩余特征完成 | etale homotopy type |
| Quick stable etale | profinite spectra | profinite 模型 | 稳定 etale 信息 |
| real-etale | `\mathbf{SH}(S_{\mathrm{ret}})` | 反演 `\rho` | 排序与实谱信息 |

表中的每一行都压缩了一部分 motivic 信息。例如 complex Betti 把 motivic bigrading 的
weight 方向折叠进普通拓扑分次。由此可以看出 realization 不可能仅凭存在性就允许
从目标完整重建源；真正的 conservativity 只能在明确子范畴、完成或附加假设下另证。

## 22.7 Analytic stacks 的新进展

**研究边界 22.10.** Magen 2025/2026 的 pullback-formalism 与 complex analytic
stacks 预印本构造 analytic motivic homotopy theory，并讨论 localization 及
Betti/analytification maps 的六操作相容性。本书只在第二十四章登记其已核查版本；
在这些新定理完成独立比较之前，不用它们扩张定理 22.4 的定义域。

Betti、etale 与 real-etale realization 因而不是同一函子的不同名字。一个严格的比较
定理至少要写明源范畴、目标范畴、基、完成或局部化、允许态射以及所保持的操作；缺少
其中任一项，陈述都不足以用于后续证明。

## 练习

**练习 22.1.** 用三次泛性质证明命题 22.1。

**练习 22.2.** 计算 `T(\mathbb C)` 的同伦型，并核对例子 22.3。

**练习 22.3.** 解释为什么 Isaksen 的目标是 pro-spaces 而非 ordinary spaces。

**练习 22.4.** 用 `\operatorname{Spec}\mathbb C` 比较 real-points 与
`C_2`-equivariant Betti realization。

**练习 22.5.** 写出定理 22.7 的全部基假设与局部化元素。

**练习 22.6.** 证明命题 22.8，并指出单靠伴随唯一性还缺少什么条件。

**练习 22.7.** 证明命题 22.9。
