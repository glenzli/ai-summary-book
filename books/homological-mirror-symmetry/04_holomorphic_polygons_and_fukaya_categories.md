# 第四章：holomorphic polygon、$A_\infty$ 结构与 Fukaya category

Floer 微分只使用两条 Lagrangian 边界条件；要把若干 Floer cochains 复合，圆盘边界必须依次落在 $L_0,\ldots,L_d$ 上。零维多边形模空间给出 $\mu^d$，一维模空间的断裂端恰好排列成 Stasheff 恒等式中的各项。几何计数与代数恒等式在这里第一次完整相接。本章固定 compact exact Liouville 模型，以第三章的 Floer cochains 为态射，并按第一章和附录 B 的 suspension 约定组织高阶运算；Fredholm 正则性、紧致化、取向与 gluing 采用附录 E 的外部输入，一般 non-exact 或 obstructed 情形留到第五章。

## 4.1 带边界标记点的圆盘

**定义 4.1.** 对 $d\ge1$，记 $\mathcal R^{d+1}$ 为带有 $d+1$ 个按逆时针顺序排列的边界 punctures
$$
\zeta_0,\zeta_1,\ldots,\zeta_d
$$
的圆盘复结构模空间，其中 $\zeta_0$ 作为输出点，$\zeta_1,\ldots,\zeta_d$ 作为输入点。每两个相邻 punctures 之间的边界弧记为 $I_0,\ldots,I_d$。

**事实 4.2.** 对稳定圆盘情形 $d\ge2$，$\mathcal R^{d+1}$ 的实维数为 $d-2$。当 $d=1$ 时对应 strip，需单独作为带两个 strip-like ends 的模型处理；当 $d=2$ 时，它对应三角形，模空间为零维。

**解释 4.3.** 维数 $d-2$ 是圆盘边界上 $d+1$ 个有序点的参数数目减去 $\operatorname{PSL}(2,\mathbb R)$ 的三维自同构群。严格处理需要稳定曲线紧化；本章只使用其边界分解形式。

## 4.2 Polygon 模空间

**约定 4.3A.** 本章固定约定 E.1A 的 Liouville completion
$\widehat M$、compact exact brane 小集合 $\mathscr L$、系数域 $k$、Maslov
cover、background class 与 relative Pin data，并固定一套定义 E.1B 的
coherent Floer data。符号 $\mathcal F^c_{\mathrm{ex}}(\widehat M)$ 只指这一
compact exact 模型。

**定义 4.4.** 对有序对象对 $(L_{i-1},L_i)$，Hamiltonian chord 是路径
$$
x_i:[0,1]\to\widehat M,\qquad
\dot x_i=X_{H_{i-1,i,t}}(x_i),\quad
x_i(0)\in L_{i-1},\quad x_i(1)\in L_i.
$$
Pair data 取到所有 chords 非退化；紧 branes 与 compactly supported data
使 chord 集合有限。给定输入 chords
$$
x_i\in\mathcal X(L_{i-1},L_i),\qquad 1\le i\le d,
$$
和输出 chord $x_0\in\mathcal X(L_0,L_d)$，一个 perturbed holomorphic
polygon 是映射
$$
u:S\to\widehat M,\qquad S\in\mathcal R^{d+1},
$$
满足：

1. 方程 $(du-X_{K_S})^{0,1}_{J_S}=0$；
2. $u(I_i)\subset L_i$；
3. 在固定 strip-like ends 上渐近于 $x_i$。

相应模空间记为
$$
\mathcal M(x_0;x_d,\ldots,x_1).
$$
若解 regular，其维数为
$$
|x_0|-\sum_{i=1}^d|x_i|+d-2.
$$
当 pair Hamiltonian 为零且相邻 Lagrangians 横截时，chords 可识别为原来
的交点模型；自 morphisms 与一般 coherent composition 仍必须使用上述
pair/universal data。

**定义 4.5.** 对 rigid solution
$u\in\mathcal M^0(x_0;x_d,\ldots,x_1)$，relative Pin data 与
determinant-line gluing 给出 orientation-line map $c_u$，边界上的局部系统
parallel transport 给出 $\operatorname{PT}_u$；二者的定义域和值域见
(E.3)--(E.5)。记
$$
\mathfrak m_u=\operatorname{PT}_u\otimes c_u.
$$
这是从输入 chord summands 到输出 chord summand 的次数 $2-d$ 线性映射。
只有所有局部系统平凡秩一且 orientation lines 已选基时，才记其标量为
$n(x_0;x_d,\ldots,x_1)\in k$。

**警告 4.6.** Regularity、energy bound、no-escape、compactness、orientation
和 gluing 不是形式代数事实。本章后续计数只引用外部输入定理 E.6 的
Seidel compact exact 口径；FOOO 的一般 filtered/virtual 口径属于第五章，
不能用来默默扩大本章定理的定义域。

## 4.3 高阶复合

**定义 4.7.** 对每对 branes 定义
$$
CF^\ast(\mathbb L_0,\mathbb L_1)
=\bigoplus_{x\in\mathcal X(L_0,L_1)}
\operatorname{Hom}_k((E_{L_0})_{x(0)},(E_{L_1})_{x(1)})\otimes o_x.
$$
在定理 E.6 的 regular compact 模型中，对 homogeneous chord summands
定义
$$
\mu^d(\xi_d,\ldots,\xi_1)
=\sum_{x_0}\ \sum_{u\in\mathcal M^0(x_0;x_d,\ldots,x_1)}
\mathfrak m_u(\xi_d,\ldots,\xi_1).
\tag{4.1}
$$
零维 compactness 与 chord finiteness 使和为有限和。由维数公式，(4.1)
给出次数 $2-d$ 的映射
$$
\mu^d:
CF^\ast(\mathbb L_{d-1},\mathbb L_d)\otimes\cdots\otimes
CF^\ast(\mathbb L_0,\mathbb L_1)
\to
CF^\ast(\mathbb L_0,\mathbb L_d)[2-d].
$$

**例 4.8.** 当 $d=1$ 时，$\mu^1$ 是 Hamiltonian-chord Floer 微分；在
$H_{01}=0$ 且两对象横截时，它退化为定义 3.13 的交点模型。一般 pair
data 与交点模型只在 continuation 意义下比较。$d=2$ 时，$\mu^2$ 由
holomorphic triangles 计数，并在 Floer cohomology 上诱导乘法。

**反例 4.8A（scalar count 不足以处理局部系统）.** 若某个 $E_{L_i}$
秩大于 $1$，两条不同边界路径可给出不同 holonomy endomorphisms，即使
底层 rigid polygons 的 signed cardinality 相同，所得 $\mu^d$ 也可不同。
因此只记录 $n(x_0;x_d,\ldots,x_1)$ 会丢失运算本身。

**外部输入定理 4.9（compact exact Fukaya $A_\infty$ 方程）.** 在约定
4.3A 下，采用外部输入定理 E.6 构造的运算 $\{\mu^d\}_{d\ge1}$ 良定义、
次数为 $2-d$；其对应 suspended Taylor components $b_d$ 对每个
$d\ge1$ 满足精确恒等式 (B.3)。因此它们给出一个非弯曲
$A_\infty$ 结构。

**证明路线（外部输入）.** 良定义性、有限性和次数来自 E.6(1)--(3)。
推论 E.7 已把 compactified one-dimensional moduli spaces 的全部 broken
boundary、determinant-line sign 与 gluing multiplicity 逐项识别为 (B.3)。
Exactness 的内部作用仅是命题 E.4 所证明的 bubble 排除；其余分析责任
仍属于外部输入 E.6。

**推论 4.10.** 在定理 4.9 的假设下，exact Lagrangian branes 与 Floer
cochains 构成一个非弯曲 $A_\infty$ category；这里暂不声称链级严格含
单位。

**证明.** 对象取 branes，morphism spaces 取 $CF^\ast$，高阶复合取定义 4.7。$A_\infty$ 方程由定理 4.9 给出。证毕。

## 4.4 Exact Fukaya category

**定义 4.11.** 在约定 4.3A 下，compact exact Fukaya category
$$
\mathcal F^c_{\mathrm{ex}}(\widehat M;\mathscr L,k)
$$
是如下 $\mathcal U$-小 $A_\infty$ category：

1. 对象是 $\mathscr L$ 中含 primitive、grading、relative Pin 与局部系统的
   compact exact branes；
2. morphism spaces 是定义 4.7 的 finite Hamiltonian-chord complexes；
3. 高阶复合是 (4.1) 的 orientation-line/local-system maps 之和；
4. coherent perturbation data 与全部分析输入由定理 E.6 提供。

在本书明确固定该 scope 时可简写为 $\mathcal F(M)$；该简写不得用于
monotone、wrapped 或 general obstructed category。

**外部输入定理 4.12（选择独立性）.** 对同一
$(\widehat M,\mathscr L,k)$ 及同一 brane background data，两套满足 E.1B、
E.6 的 coherent perturbation systems 之间存在 continuation
$A_\infty$ functors；它们在 morphism complexes 上为 quasi-isomorphisms，
在 $H^0$ 上本质满，因而给出 quasi-equivalent compact exact Fukaya
categories。若改变对象集合、Maslov cover、background class 或系数域，
该定理不声称所得 categories quasi-equivalent。

**解释 4.13.** 定义 4.11 不是单纯的集合论定义：它压缩了大量分析选择。定理 4.12 保证压缩后的对象可以作为几何不变量。若没有选择独立性，HMS 的 A-side 会依赖非几何辅助数据。

## 4.5 单位与三角闭包

**定义 4.14.** Fukaya category 通常先得到 cohomologically unital $A_\infty$ category：对每个对象 $\mathbb L$，$HF^\ast(\mathbb L,\mathbb L)$ 中有单位元，使 $H^\ast\mathcal F(M)$ 成为含单位范畴。若通过模型替换得到严格单位，则称为 strictly unital Fukaya model。

**外部输入定理 4.14A（compact exact 模型的同调单位）.** 在约定
4.3A 与外部输入定理 E.6 的假设下，若同时选择与 strip-like ends 相容的
unit perturbation data，则 continuation 元给出
$$
[e_{\mathbb L}]\in HF^0(\mathbb L,\mathbb L),
$$
且 $[e_{\mathbb L}]$ 在 $H^\ast\mathcal F^c_{\mathrm{ex}}$ 中是左右单位。
这是 moduli-space 构造的额外外部输入，不由定理 4.9 的无单位
$A_\infty$ identities 自动推出。来源：Seidel，*Fukaya Categories and
Picard--Lefschetz Theory*，Chapters 9--12 的 compact exact 模型。

**外部输入定理 4.15（strictification）.** 设 $k$ 是域，$\mathcal A$ 是
小、非弯曲、cohomologically unital 的 $k$-linear $A_\infty$ category。
则存在小、strictly unital 的 $A_\infty$ category $\mathcal A^{\mathrm{su}}$
以及一个 $A_\infty$ quasi-equivalence
$$
\mathcal A^{\mathrm{su}}\longrightarrow\mathcal A.
$$
该替换依赖选择，并不把原模型中的同调单位自动变成同一底层 graded
category 上的严格单位。来源：Lefevre-Hasegawa，*Sur les
$A$-infini categories*，关于 homological 与 strict unitality 的比较。

**定义 4.16.** Fukaya category 的 split-closed derived category 记为
$$
D^\pi\mathcal F(M):=H^0\operatorname{Perf}(\mathcal F(M)).
$$
这是许多早期 HMS 文献中“derived Fukaya category”的精确版本之一。

**警告 4.17.** 记号 $D^\pi\mathcal F(M)$ 在文献中有变体。有些作者使用 twisted complexes 的 split-closure，有些作者使用 perfect modules。本书在每个定理中说明采用哪一种，但 Morita 口径下这些模型通常表达同一类 idempotent-complete triangulated envelope。

## 4.6 与 wrapped 版本的边界

**定义 4.18.** 若 $M$ 非紧并带 Liouville 结构，wrapped Fukaya category $\mathcal W(M)$ 允许非紧 Lagrangians，并用 Hamiltonian wrapping 产生 morphisms。其 morphism cochains 由 Hamiltonian chords 而非单纯交点生成。

**警告 4.19.** $\mathcal W(M)$ 不是 $\mathcal F(M)$ 的形式小修改。wrapped 版本需要控制无穷远处的 Reeb dynamics、Hamiltonian 增长、sectorial boundary、stops 和 continuation maps。后续章节将使用 Ganatra-Pardon-Shende 的 Liouville sectors 与 descent 作为外部输入。

多边形计数至此给出了一个非弯曲 $A_\infty$ 范畴：零维模空间定义运算，一维紧化的定向边界给出全部关系。Exactness 排除了特定 bubbling，却不代替正则性、取向和 gluing；这些分析责任仍由外部输入承担。HMS 实际比较的往往还不是原始 $\mathcal F(M)$，而是它的 twisted、split-closed 或 perfect/Morita 完成。离开 exact 世界后，首先失效的正是“曲率为零”这一点。

## 练习

**练习 4.1.** 对 $d=1,2,3$，分别说明 $\mathcal R^{d+1}$ 的几何含义。

**练习 4.2.** 将 $d=2$ 的 $\mu^2$ 与 Floer cohomology 上的乘法联系起来，并说明为什么链级乘法只按 $A_\infty$ 意义结合。

**练习 4.3.** 写出一维 polygon 模空间的 broken boundary 如何对应 $A_\infty$ 方程中的一项。

**练习 4.4.** 解释 wrapped Fukaya category 中为什么 morphism 生成元应改为 Hamiltonian chords。
