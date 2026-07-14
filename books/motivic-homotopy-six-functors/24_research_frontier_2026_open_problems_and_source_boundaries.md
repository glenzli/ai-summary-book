# 第二十四章：研究边界、比较问题与开放方向

前二十三章出现了若干名字相近而输入不同的理论：概形、商栈、一般代数栈、对数概形、
perfect 概形和复解析栈上都可以尝试建立 motivic homotopy；有限对应、framed
correspondence、Gysin 映射与 norm 又都可被称为 transfer。把它们并列成名单并不能
产生统一理论。真正需要比较的是：源几何是什么，采用何种下降和区间，反演哪个悬挂
坐标，六操作允许哪些态射，以及比较函子保存哪些相干结构。

本章把这些问题整理成可逐项检验的数学形式。近期结果只按
[研究边界核查](FRONTIER_SOURCE_AUDIT_2026_07_08.md) 中确认的版本陈述；其中
perfect motivic homotopy、pullback formalism 与 complex analytic stacks 的结果
仍按预印本状态处理。它们可以提供条件命题和研究方向，但不反向改变第一至第十八章
所采用的基础外部输入。

## 24.1 一个 motivic 理论需要哪些数据

**定义 24.1（几何输入五元组）.** 一个 motivic 构造的几何输入记为

$$
\mathfrak G=(\mathcal C,\tau,I,T,\mathcal E),
$$

其中：

1. `\mathcal C` 是带有限纤维积的几何对象范畴；
2. `\tau` 是 `\mathcal C` 或其光滑对象上的 Grothendieck 拓扑；
3. `I` 是要求同伦不变的区间对象；
4. `T` 是在 pointed 理论中反演的悬挂坐标；
5. `\mathcal E` 是允许 exceptional operations `f_!\dashv f^!` 的态射类，并对
   复合和 base change 封闭。

例如，前八章的默认输入取有限型 `B`-概形为 `\mathcal C`，在光滑站点上取
Nisnevich 拓扑，令 `I=\mathbb A^1`、`T=\mathbb A^1/(\mathbb A^1-0)`，并令
`\mathcal E` 为 separated finite-type 态射。对数、stacky 或 analytic 理论改变的
不止第一项，其他四项也可能随之改变。

**定义 24.2（基本比较数据）.** 设 `\mathfrak G`、`\mathfrak G'` 是两个几何输入。
从前者到后者的基本比较数据包括：

1. 保持所用有限纤维积的几何函子 `u:\mathcal C\to\mathcal C'`；
2. `u` 把 `\tau`-覆盖送到 `\tau'`-局部等价的证明；
3. `u(I)` 在目标 `I'`-局部理论中可缩的证明；
4. `u(T)` 在目标稳定范畴中张量可逆的证明。

**命题 24.3.** 给定定义 24.2 的数据，一个保持相应余极限的预层级函子若先通过
`\tau`-层化，再通过 `I`-局部化和 `T`-稳定化，则诱导稳定 motivic 范畴之间的函子；
缺少第 2、3、4 项中的任一项，都不能仅由局部化泛性质得到该因子化。

**证明.** 层化是关于覆盖筛映射的反射局部化。第 2 项保证函子把这些映射送到目标
等价，故由层化泛性质因子化。`I`-局部化反演 `X\times I\to X`；第 3 项保证其像
为等价，于是第二次因子化。最后，`T`-稳定化是对称幺半地反演 `T`；第 4 项正是
对象反演泛性质的假设。反过来，若某类被反演的映射在目标中不是等价，或 `T` 的像
不可逆，则任何声称通过相应局部化的函子与泛性质矛盾。`\square`

这个命题解释了为什么“取复点”“做 perfectization”或“忘掉 log structure”本身还
不是 realization 定理：必须验证它们经过每一层局部化。

## 24.2 六操作比较比范畴比较更强

**定义 24.4（六操作比较）.** 设 `\mathcal D`、`\mathcal D'` 是同一几何范畴或由
`u:\mathcal C\to\mathcal C'` 联系的两个六操作形式主义。一族函子

$$
R_X:\mathcal D(X)\longrightarrow\mathcal D'(uX)
$$

称为六操作比较，若它带有与复合相干的自然等价

$$
R_Xf^*\simeq (uf)^*R_Y,
\qquad
R_Yf_*\simeq (uf)_*R_X,
$$

并在 `f\in\mathcal E` 时同样与 `f_!`、`f^!` 相容，同时保持张量、internal Hom、
base-change mate、projection formula 和 open--closed localization。

**命题 24.5.** 一族纤维范畴等价 `R_X:\mathcal D(X)\simeq\mathcal D'(uX)` 不足以
给出定义 24.4 的六操作比较。

**证明.** 单个纤维上的范畴等价没有指定 `R_Xf^*` 与 `(uf)^*R_Y` 之间的自然变换。
即使补上 pullback 相容，右伴随相容还需由选定 adjunction 的 mate 构造并验证可逆；
`f_!` 相容又涉及 exceptional 态射类、compactification 或 gluing。最后，逐个自然
等价还必须满足复合和 Cartesian-square pasting。所有这些数据都不包含在裸纤维等价
中。`\square`

**例 24.6（复 Betti realization）.** 对复概形取复点时，
`\mathbb A^1(\mathbb C)=\mathbb C` 可缩，且 Tate 球的复点具有拓扑球的稳定同伦型，
所以定义 24.2 的局部化条件可以成立。但与 `f_!` 的相容还要比较代数几何的紧支撑
推前和拓扑紧支撑推前；这不是“复点保持纤维积”的直接推论。第二十二章因此把
Betti functor 的存在与六操作相容分成不同外部输入。

## 24.3 栈、perfect 与 analytic 三类扩展

前三类前沿扩展可用定义 24.1 的五个坐标逐项比较。

| 扩展 | 几何对象 | 主要新增困难 | 已引用结论的地位 |
| --- | --- | --- | --- |
| algebraic stacks | quotient、scalloped 或指定 Artin stacks | lisse 下降、stabilizer、representability、stacky purity | 第十九、二十章的外部输入 |
| perfect schemes | 正特征 perfect 概形 | Frobenius/perfection、universal homeomorphism、`p` 的可逆性 | 2025 预印本研究边界 |
| complex analytic stacks | 复解析栈 | analytic localization、紧支撑、与代数 analytification 的比较 | 2025--2026 预印本研究边界 |

**研究边界 24.7（PF-21.13；perfect 比较）.** Dahlhausen--Hekking--Wolters 构造的
perfect motivic homotopy theory带有 coefficient-system 六操作，并将其与 universal
homeomorphism localization 及 `\mathbf{SH}[1/p]` 比较。这里 `p` 已可逆是理论结构的
一部分；它不能给出 ordinary integral `\mathbf{SH}` 中的无条件等价。

**命题 24.8.** 设 `L:\mathcal C\to\mathcal C[1/p]` 是反演 `p` 的局部化函子。
若一个态射 `a` 满足 `L(a)` 为等价，则只能推出 `\operatorname{cofib}(a)` 被 `L` 杀掉，
不能推出 `a` 在 `\mathcal C` 中为等价。

**证明.** 正合函子把 cofiber sequence
`A\xrightarrow{a}B\to\operatorname{cofib}(a)` 送到 cofiber sequence。
`L(a)` 为等价当且仅当 `L(\operatorname{cofib}(a))\simeq0`。局部化函子一般有非零
核；例如 `p`-primary torsion 对象在反演 `p` 后为零。因此原 cofiber 未必为零。
`\square`

**研究边界 24.9（RB-24.1、RB-24.2；pullback 与 analytic 比较）.** Magen 的 pullback-formalism
判据把六操作形式主义及其 morphism 的六操作相容性组织在同一框架中，并用于
algebraic/complex analytic stacks；配套的 analytic localization 定理为该构造提供
open--closed gluing。按本书采用的资料快照，这些结果以 2025、2026 预印本版本作为
研究输入，不替代 Hoyois、Ayoub、Cisinski--Deglise 等基础 package。

## 24.4 四种 transfer 的类型不能合并

**定义 24.10（transfer 类型表）.** 对一个几何态射或 correspondence，本文出现的
四类传递具有下列最小输入和输出。

| 结构 | 几何输入 | 额外数据 | 典型输出 |
| --- | --- | --- | --- |
| finite correspondence | finite-over-source cycles | 交积与 proper pushforward | 加性反变作用 |
| framed transfer | finite syntomic map | `L_f` 的稳定 framing | infinite-loop/Gysin 型作用 |
| fundamental class | smoothable lci map | virtual tangent 与 purity transformation | twisted Gysin map |
| norm | finite etale map | symmetric monoidal span coherence | 乘法传递 |

**命题 24.11.** 这四类结构中任意一类的存在，都不形式推出其余三类。

**证明.** 它们的定义域不同。一般 finite correspondence 不由一条 finite syntomic
态射给出；smoothable lci 态射未必 finite，更未必 etale；finite etale 态射虽具有
零 cotangent complex，却仍需独立的对称幺半 norm coherence。输出类型也不同：finite
correspondence 给出加性作用，fundamental class 带 virtual Thom twist，norm 则把分裂
覆盖上的多分量送到张量积。定义域或目标结构的任一差异都排除纯形式蕴含。`\square`

统一 transfer calculus 因而不能只是把四种箭头画在同一张图上；它必须给出比较变换，
并证明复合、base change、distributivity 与 orientation change 的相干。

## 24.5 Slice 与 realization 的交换问题

Slice tower 提供 motivic 谱的权重过滤，而 realization 把 motivic 对象送到另一稳定
范畴。二者的相互作用是可计算性的核心之一，但“realization 保持余极限”不足以保证
它逐层保持 slice。

**命题 24.12（条件性的 slice 比较）.** 设
`R:\mathbf{SH}(S)\to\mathcal C` 为正合且保持小余极限的函子。若对每个 `q` 存在
`\mathcal C` 的反射或余反射子范畴 `\mathcal C^{eff}(q)`，并且：

1. `R` 把 `\mathbf{SH}(S)^{eff}(q)` 送入 `\mathcal C^{eff}(q)`；
2. `R` 与 effective cover 的右伴随相容，即有
   `R i_qr_q\simeq i'_qr'_qR`；

则对所有 `E` 有自然等价

$$
R(f_qE)\simeq f'_qR(E),
\qquad
R(s_qE)\simeq s'_qR(E).
$$

**证明.** 第一式就是假设 2。由第十三章定义，存在 cofiber sequence

$$
f_{q+1}E\longrightarrow f_qE\longrightarrow s_qE.
$$

对它作用正合函子 `R`，再用第一式识别前两项，所得第三项是
`\operatorname{cofib}(f'_{q+1}R(E)\to f'_qR(E))=s'_qR(E)`。`\square`

**注 24.13.** 假设 2 是实质条件。它要求 realization 不仅保持生成子，还与定义
effective cover 的右伴随交换；这通常涉及紧致性、权重或完成条件。即使逐层比较成立，
谱序列的收敛仍需比较 tower 的极限与 completion，不能从命题 24.12 自动推出。

## 24.6 可精确陈述的开放问题

下面的问题只陈述未知或未在本书输入中解决的比较，不作为后续证明的前提。

**问题 24.14（stacky 六操作的最大自然定义域）.** 找出一类对有限纤维积、开闭分解
和所需 compactification 稳定的 algebraic stacks，使 `\mathbf{SH}(-)` 同时满足
lisse descent、六操作、purity 与 compact generation；并证明不同 atlas/lisse
构造与 genuine stacky 构造之间的结构化等价。

**问题 24.15（统一 transfer calculus）.** 构造一个明确的 higher category 或
span/correspondence category，使 finite、framed、Gysin 与 norm structures 由不同
子范畴或装饰得到，并在重叠定义域上恢复已知比较与分配律。

**问题 24.16（一般基上的 slice 收敛）.** 对 arithmetic、equivariant 或 stacky
基，给出可检验的对象条件，保证 slice tower 完备且相应谱序列强收敛；同时确定
球谱、`KGL`、`MGL` 等基本谱的 slices 及隐藏扩张。

**问题 24.17（realization 的检测范围）.** 对每一种 Betti、etale、real etale 或
analytic realization，确定其在何种 cellular、constructible、完成或系数局部化
子范畴上保守，并描述 kernel 中最小的几何生成族。

**问题 24.18（二次型与退化几何）.** 在 derived 或 log intersection theory 中，
构造与第十六章 fundamental classes 相容的 quadratic local degrees，并证明它们在
适当 proper pushforward、excess intersection 与 field trace 下的函子性。

## 24.7 理论收束于比较图，而非主题清单

本书的基础链从 Nisnevich 层和 `\mathbb A^1`-局部化出发，经 Tate 稳定化进入
`\mathbf{SH}`，再由六操作、purity 与代表谱产生 cohomology、Gysin、transfer 和
trace。高级扩展只有在定义 24.2 的局部化条件与定义 24.4 的六操作相干同时验证后，
才能接到这条链上。这个判据把“存在一个同名理论”与“得到可迭代使用的比较定理”
分开，也说明开放问题应当补哪一条箭头、哪一个 mate 或哪一种收敛，而不是只增加
新的主题名称。

## 练习

**练习 24.1.** 对普通概形上的 `\mathbf{SH}` 写出定义 24.1 的五个分量，并说明
其中哪一项控制 `f^!` 的定义域。

**练习 24.2.** 用局部化泛性质证明命题 24.3 中第 3 项的必要性。

**练习 24.3.** 给出一族纤维范畴等价，但尚未给出 pullback 相容数据的抽象例子，
并解释它为何不是六操作比较。

**练习 24.4.** 对 split degree-`d` finite etale cover，分别写出 additive
pushforward 与 norm 的公式，并据此验证命题 24.11 的一个特例。

**练习 24.5.** 证明命题 24.8，并给出导出范畴中一个非零 `p`-primary torsion
对象在反演 `p` 后消失的例子。

**练习 24.6.** 逐行检查命题 24.12 的证明，指出只假设 `R` 保持余极限时缺少哪一步。

**练习 24.7.** 从问题 24.14--24.18 中任选一个，把它改写成“已知输入、所求对象、
需要验证的相干图、预期输出”四栏。

**练习 24.8.** 比较 perfectization 与 complex points 两种几何函子，分别检查
定义 24.2 的四项数据可能在哪一步失败或需要外部定理。
