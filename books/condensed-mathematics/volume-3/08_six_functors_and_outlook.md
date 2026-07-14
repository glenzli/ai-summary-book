# 第八章：六函子关系与开放问题

开嵌入 $j:U\hookrightarrow X$ 已经迫使我们区分两种推前：$j_*$ 允许截面在边界附近
继续存在，$j_!$ 则只保留相对 $X$ 具有适当支撑的截面。proper 映射上二者相合，Serre
对偶中的 trace 才能同时写成普通全局截面与紧支撑推前。六函子形式的内容不是列出六个
符号，而是规定两组伴随、张量闭结构、base change、projection formula 和支撑条件如何
相容。

本章以 $\mathbb C^\times\hookrightarrow\mathbb C$ 的 stalk 计算展示
$f_!\ne f_*$，再从伴随唯一性证明复合与对偶的形式关系，并把前章的 Serre trace 和
GRR 放回同一函子框架。由此自然出现这些相容式在复解析、凝聚与谱化范围内的存在性、
base change、对偶对象和特征类比较问题。

## 8.1 六种运算的类型

对适当的解析或凝聚环化空间映射 $f:X\to Y$，设有稳定闭对称幺半范畴
$\mathcal D(X)$、$\mathcal D(Y)$。六种运算的类型为

$$
f^*: \mathcal D(Y)\to\mathcal D(X),
\qquad
f_*: \mathcal D(X)\to\mathcal D(Y),
$$

$$
f_!: \mathcal D(X)\to\mathcal D(Y),
\qquad
f^!: \mathcal D(Y)\to\mathcal D(X),
$$

以及每个 $\mathcal D(X)$ 内部的

$$
-\otimes_X-,
\qquad
R\mathcal Hom_X(-,-).
$$

它们满足伴随

$$
f^*\dashv f_*,
\qquad
f_!\dashv f^!,
\qquad
A\otimes_X-\dashv R\mathcal Hom_X(A,-).
$$

第一组伴随由逆像和普通推前组织，第二组由支撑与对偶组织。对称幺半结构还应给出
projection formula

$$
f_!(F\otimes_X f^*G)
\simeq
f_!F\otimes_YG.
$$

第二卷构造了特定仿射有限型范围内的 $f_!$ 输入；第五章证明了接受上述伴随和投影公式
后得到的内部 Hom 等价。一般复解析映射上这些函子的存在与相容不作为书内定理宣称。

## 8.2 Worked example：开嵌入处的两个推前

令

$$
j:\mathbb C^\times\hookrightarrow\mathbb C
$$

为开嵌入，$\underline{\mathbb C}_U$ 为 $U=\mathbb C^\times$ 上的常值 sheaf。
计算原点 stalk。

对 extension by zero，

$$
(j_!\underline{\mathbb C}_U)_0=0,
$$

因为 $j_!$ 在补集 $\{0\}$ 上的 stalk 按定义为零。对普通推前，

$$
(j_*\underline{\mathbb C}_U)_0
=
\varinjlim_{0\in V}
\Gamma(V\cap\mathbb C^\times,\underline{\mathbb C}_U).
$$

取足够小的连通圆盘 $V$，穿孔圆盘 $V\setminus\{0\}$ 仍连通，故其常值 sheaf 全局
截面为 $\mathbb C$。限制映射保持常数，于是

$$
(j_*\underline{\mathbb C}_U)_0\cong\mathbb C.
$$

因此自然变换

$$
j_!\underline{\mathbb C}_U
\longrightarrow
j_*\underline{\mathbb C}_U
$$

不是同构。输入是开嵌入与常值 sheaf，步骤是分别计算边界 stalk，输出是 $0$ 与
$\mathbb C$。差异恰好位于缺失的边界点；若 $f$ proper，就没有这种由逃向无穷远产生
的差异。

在经典 sheaf 六函子中，开嵌入还满足 $j^!\simeq j^*$。若
$i:\{0\}\hookrightarrow\mathbb C$，recollement 三角

$$
j_!j^*F\longrightarrow F\longrightarrow i_*i^*F\longrightarrow
$$

把开部分与闭边界重新拼成 $F$。把这条三角提升到 analytic/liquid 模时，必须验证
extension by zero 保持相应局部对象；这不是普通 sheaf 公式自动保证的。

## 8.3 Proper 映射与对偶

六函子形式带有比较变换

$$
f_!\longrightarrow f_*.
$$

对 proper $f$，预期或在相应理论中证明它为同构。把这个同构代入第五章命题 5.7
的伴随与 projection-formula 计算，便得到 Grothendieck duality 的 Hom 形式。

**命题 8.1.** 假设 $f$ proper、$f_!\simeq f_*$，并且 projection formula 成立。则

$$
f_*R\mathcal Hom_X(F,f^!G)
\simeq
R\mathcal Hom_Y(f_*F,G).
$$

**证明.** 第五章命题 5.7 已从两组伴随和 projection formula 证明

$$
f_*R\mathcal Hom_X(F,f^!G)
\simeq
R\mathcal Hom_Y(f_!F,G).
$$

用 proper 比较同构 $f_!F\simeq f_*F$ 代入即得。证毕。

当 $Y=*$、$X$ 为紧复流形且
$f^!\mathbb C\simeq\omega_X[n]$ 时，取 $G=\mathbb C$ 就得到第五章的 derived
Serre duality。$f^!\mathbb C$ 与 dualizing complex 的识别仍是深层几何输入；命题
8.1 只承担函子形式的推导。

## 8.4 复合、base change 与 mates

若 $X\xrightarrow{f}Y\xrightarrow{g}Z$，支撑推前应有 coherent composition

$$
(g\circ f)_!\simeq g_!f_!.
$$

**命题 8.2.** 若上式成立，且三组右伴随存在，则

$$
(g\circ f)^!\simeq f^!g^!.
$$

**证明.** 对 $A\in\mathcal D(X)$、$B\in\mathcal D(Z)$，

$$
\begin{aligned}
\operatorname{Map}(g_!f_!A,B)
&\simeq\operatorname{Map}(f_!A,g^!B)\\
&\simeq\operatorname{Map}(A,f^!g^!B).
\end{aligned}
$$

所以 $f^!g^!$ 是 $g_!f_!\simeq(g\circ f)_!$ 的右伴随。右伴随在自然同构意义下
唯一，得到结论。证毕。

对 Cartesian 方块

$$
\begin{matrix}
X'&\xrightarrow{g'}&X\\
\downarrow f'&&\downarrow f\\
Y'&\xrightarrow{g}&Y
\end{matrix}
$$

base-change 变换的紧支撑形式为

$$
g^*f_!\longrightarrow f'_!g'^*.
$$

它何时为同构取决于所用六函子理论。对普通构造层的 $f_!$，适当的局部紧性与支撑
条件往往已包含在形式体系中；对 coherent/analytic 派生拉回，还可能需要有限 Tor
维数、Tor 独立性或相应的解析横截性。若 $f$ proper 且 $f_!\simeq f_*$，该公式化为
proper base change。不能把某一系数理论的充分条件无条件搬到另一系数理论。对
Riemann--Roch 而言，还要让 Chern character 把 $f_!$ 或 $Rf_*$ 的推前送到
cohomological pushforward，并由 Todd class 修正；第七章的 GRR 输入正表达这种相容。

base-change 变换与其对偶版本不是两条无关公式。给定伴随后，对一个自然变换取 mate
可在 $f_!$ 公式和 $f^!$ 公式之间转换；同构的 mate 仍为同构。这个纯形式事实说明，为何
证明一侧的 base change 往往同时得到对偶侧公式，但它不判断原始交换变换在何种几何
假设下可逆。

## 8.5 三个前章定理在六函子中的位置

相干有限性断言 proper 结构映射把相干对象送到点上的 perfect 对象。Serre 对偶把

$$
R\mathcal Hom_Y(f_!F,G)
$$

由命题 8.1 识别为 $X$ 上的 dualizing Hom。GAGA 断言代数与解析实现不仅等价，而且
与 proper $f_*$、$R\Gamma$ 和 trace 相容。GRR 则比较 $f_*$ 在 $K$-理论与
cohomology 中的两个实现。

这些结论共享六函子语法，却不由语法彼此推出：有限性需要椭圆或 Grauert 输入，GAGA
需要代数化，Serre perfectness 需要 dualizing theory，GRR 需要特征类与推前几何。
六函子形式的作用是让输入的类型和它们之间必须交换的图清楚可检验。

## 8.6 开放问题的数学形状

以下每一问都指定了待构造的函子或待检验的自然变换。

1. **非 proper 解析推前。** 对哪些复解析映射 $f$，analytic/liquid 模范畴中的
   $f_!$ 存在并保持可展示性？哪些支撑条件保证它把 coherent 或 perfect 对象送到可控
   子范畴？
2. **解析 base change。** 在非代数、非 quasi-compact 或带边界的解析空间上，何种
   几何条件使
   $g^*f_!\to f'_!g'^*$ 可逆？失败时的 cofiber 是否能由边界或 wave-front 数据描述？
3. **奇异空间的对偶对象。** 对非约化或奇异凝聚解析空间，$f^!\mathbf1$ 与经典
   dualizing complex 的比较需要哪些有限 Tor 维数和相干性假设？
4. **Trace 与特征类。** categorical trace 到 topological/analytic Chern character
   的比较能否在同一 analytic 范畴中构造，并直接证明与 composition、base change 和
   Todd correction 相容？
5. **谱化与 pyknotic 接口。** 把系数从凝聚阿贝尔群提升到凝聚谱后，哪些
   constructible/coherent 子范畴仍支持六函子？$t$-structure 下的阿贝尔结果能在多大
   范围内由谱级公式恢复？

每个问题都要求先确定范畴、函子类型与可逆性假设；仅有符号相似不足以把经典 sheaf
六函子搬到 condensed/analytic 环境。

## 8.7 由边界 stalk 到整体形式

开嵌入例子在一个 stalk 上看见 $j_!$ 与 $j_*$ 的差，proper 比较则消除这项边界差异。
两组伴随和 projection formula 随后把 trace、dualizing object 与内部 Hom 连成
Serre 对偶；base change 和复合相容再把单个映射的公式组织成几何。至此，前三章的
函数空间、第四章的有限性、第五章的对偶、第六章的比较和第七章的特征数都落在同一
数学框架中，同时各自依赖的深层输入仍清楚分离。

## 练习

**练习 8.1.** 对 $j:\mathbb C^\times\hookrightarrow\mathbb C$，计算
$(j_!\underline{\mathbb C})_x$ 与 $(j_*\underline{\mathbb C})_x$ 在 $x\ne0$ 时的
stalk，并写出自然变换在所有 stalk 上的形状。

**练习 8.2.** 只用伴随同构证明命题 8.2，并检查所得同构关于 $A,B$ 自然。

**练习 8.3.** 在 Cartesian 方块中假设 $f$ proper。把
$g^*f_!\simeq f'_!g'^*$ 逐步改写为普通 proper base change 公式，并指出使用
$f_!\simeq f_*$ 的两个位置。
