# 第三章：辛流形、Lagrangian brane 与 exact Floer 口径

## 本章目标

本章建立 HMS 的 A-side 基础对象：辛流形、Lagrangian submanifolds、brane data、exact Floer cochains 和 analytic inputs 的边界。完整 Fukaya category 的构造将在后续章节展开；本章只建立可检查的 exact 入口。

## 依赖前置知识

需要熟悉光滑流形、微分形式、切丛、向量丛、基本同调代数和第一章的 $A_\infty$ 语言。

## 3.1 辛流形与 Lagrangian

**定义 3.1.** 一个辛流形（symplectic manifold）是偶维光滑流形 $M$ 和闭的非退化 $2$-形式 $\omega\in\Omega^2(M)$ 的二元组 $(M,\omega)$。非退化指映射
$$
T_pM\to T_p^\vee M,\qquad v\mapsto \omega_p(v,-)
$$
对每个 $p\in M$ 都是线性同构。

**定义 3.2.** 子流形 $L\subset M$ 称为 Lagrangian，若
$$
\omega|_L=0,\qquad \dim L=\frac12\dim M.
$$

**命题 3.3.** 若 $L\subset(M,\omega)$ 是 Lagrangian，则 $T_pL$ 是辛向量空间 $(T_pM,\omega_p)$ 中的极大 isotropic 子空间。

**证明.** 条件 $\omega|_L=0$ 表示 $\omega_p$ 在 $T_pL$ 上恒为零，所以 $T_pL$ isotropic。任意 $2n$ 维辛向量空间中的 isotropic 子空间维数不超过 $n$：因为 $W\subset W^\omega$，而非退化性给出 $\dim W+\dim W^\omega=2n$，故 $\dim W\le n$。这里 $\dim T_pL=n$，所以 $T_pL$ 是极大 isotropic。证毕。

**定义 3.4.** Exact symplectic manifold 是实光滑流形 $M$ 连同实 $1$-形式 $\lambda\in\Omega^1(M;\mathbb R)$，其外微分
$$
\omega=d\lambda
$$
为辛形式。Lagrangian $L\subset M$ 称为 exact，若存在实值光滑函数 $f_L:L\to\mathbb R$，使得
$$
\lambda|_L=df_L.
$$
函数 $f_L$ 称为 $L$ 的 primitive，并且在 $L$ 的每个连通分支上只确定到加法常数。这里的系数域 $k$ 只用于 Floer cochains 和局部系统，不是辛形式或 primitive 的值域。

**例 3.5.** 余切丛 $T^\ast Q$ 带 tautological $1$-形式 $\lambda_{\mathrm{can}}$，其微分 $d\lambda_{\mathrm{can}}$ 给出标准 exact symplectic structure。零截面 $Q\subset T^\ast Q$ 是 exact Lagrangian，因为 $\lambda_{\mathrm{can}}$ 在零截面上为零。

## 3.2 Brane data

只给出 Lagrangian 子流形不足以定义分次 Floer cochains。固定一个 Maslov
cover（或等价 grading structure）；若 $\operatorname{char}k\ne2$，再固定
background class $b_M\in H^2(M;\mathbb Z/2)$。还需要 primitive、grading、
relative spin/Pin 结构和局部系统等数据。

**定义 3.6.** 固定系数域 $k$。在本书的 exact Calabi-Yau 入口中，一个 Lagrangian brane 是数据
$$
\mathbb L=(L,f_L,\alpha_L,\mathfrak p_L,E_L)
$$
其中：

1. $L\subset M$ 是 exact Lagrangian，$f_L:L\to\mathbb R$ 是已选定的
   primitive，$\lambda|_L=df_L$；
2. $\alpha_L$ 是相对于固定 Calabi-Yau volume form 或 Maslov cover 的 grading；
3. $\mathfrak p_L$ 是用于 determinant-line orientations 的 relative
   $\operatorname{Pin}^{\pm}$ 结构；其与 $b_M$ 的 obstruction 条件见约定
   E.1A。定向情形可使用 relative spin，$\operatorname{char}k=2$ 时可省略
   这一 orientation 数据；
4. $E_L$ 是有限秩 $k$-局部系统。

若 $M$ 非紧，本章只允许两种对象口径：$L$ 紧，或 $(M,\lambda)$ 在无穷远带固定 Liouville/cylindrical structure 且 $L$ 在无穷远为 exact conical。后一情形还要求所选 Hamiltonian 扰动使相关交点集在每个作用量窗口内有限；若使用增长 Hamiltonian 和无限生成元，则进入第六章 wrapped Floer 口径，而不是本章的有限直和模型。

**警告 3.7.** 不同文献对 brane data 的最低要求不同。若 $M$ 不 exact、$L$ 不 exact、Maslov class 不消失，或采用 monotone/Novikov/obstructed 口径，则必须修改定义 3.6。本书后续章节会在进入一般情形前重新声明假设。

## 3.3 横截交与 Floer cochains

**定义 3.8.** 两个 Lagrangians $L_0,L_1\subset M$ 横截相交，若对每个 $p\in L_0\cap L_1$，
$$
T_pL_0+T_pL_1=T_pM.
$$

**引理 3.9.** 若 $L_0,L_1$ 是紧子流形且横截相交，则 $L_0\cap L_1$ 是有限集合。

**证明.** 横截性推出 $L_0\cap L_1$ 是维数
$$
\dim L_0+\dim L_1-\dim M=0
$$
的光滑子流形，因此是离散集合。它又是紧集 $L_0\cap L_1$，因为 $L_0,L_1$ 在 Hausdorff 流形中闭且紧。紧离散空间有限。证毕。

**定义 3.10.** 设 $\mathbb L_0,\mathbb L_1$ 是横截 exact branes，并假设 $L_0\cap L_1$ 有限（例如二者均紧，或在非紧端没有交点）。Floer cochain space 定义为
$$
CF^\ast(\mathbb L_0,\mathbb L_1)
=\bigoplus_{p\in L_0\cap L_1}
\operatorname{Hom}_k((E_{L_0})_p,(E_{L_1})_p)\otimes o_p,
$$
其中 $o_p=o_p^{\mathbb Z}\otimes_{\mathbb Z}k$ 是由相应 Fredholm
determinant line 的两个 orientations 生成的秩一 $k$-module，次数由
grading/Maslov index 给出。Relative Pin data 使这些 lines 的 gluing maps
可相干取向；详见定义 E.5。

**解释 3.11.** 若忽略局部系统和 orientation lines，定义 3.10 只是把相交点生成的向量空间分次化。严格 Floer theory 的难点不在生成元，而在微分、乘法和高阶复合由 holomorphic curves 的计数给出，并且这些计数必须有 compactness、orientation 和 transversality 支撑。

## 3.4 Floer 微分的 exact 版本

**定义 3.12.** 取与 $\omega$ 相容的 almost complex structure $J$。从 $p\in L_0\cap L_1$ 到 $q\in L_0\cap L_1$ 的 Floer strip 是映射
$$
u:\mathbb R\times[0,1]\to M
$$
满足
$$
\partial_su+J(u)\partial_tu=0,\qquad
u(s,0)\in L_0,\quad u(s,1)\in L_1,
$$
以及渐近条件
$$
\lim_{s\to-\infty}u(s,t)=p,\qquad
\lim_{s\to+\infty}u(s,t)=q.
$$

**定义 3.13.** 若相关 moduli spaces 已经完成 regularization、orientation
和 compactification，且每个给定输入只有有限多个刚性 strip 贡献，则对
$\xi\in\operatorname{Hom}_k((E_{L_0})_p,(E_{L_1})_p)\otimes o_p$ 定义
$$
\mu^1(\xi)=
\sum_q\ \sum_{[u]\in\mathcal M^0(q;p)}
(\operatorname{PT}_u\otimes c_u)(\xi),
$$
其中 $\mathcal M^0(q;p)$ 是输入 $p$、输出 $q$ 且除去 $\mathbb R$-平移后
维数为 $0$ 的 Floer strips 模空间；$\operatorname{PT}_u$ 是两条边界上的
局部系统 parallel transport，$c_u:o_p\to o_q$ 是 orientation-line map。
只有局部系统平凡秩一并选定 orientation bases 时，内层和才可写成
$\#\mathcal M^0(q;p)\,q$。

**命题 3.13A（exact strip 的能量恒等式）.** 令
$g_J(v,w)=\omega(v,Jw)$，并假设 $J$ 与 $\omega$ 相容。若 $u$ 是定义
3.12 的有限能量 Floer strip，$\lambda|_{L_i}=df_i$，并且 $u$ 在两端以
$C^1([0,1])$ 收敛到常值 chords $p,q$，则
$$
E(u)\coloneqq\int_{\mathbb R\times[0,1]}|\partial_su|_{g_J}^2\,ds\,dt
=\mathcal A(p)-\mathcal A(q),
$$
其中
$$
\mathcal A(x)=f_1(x)-f_0(x),\qquad x\in L_0\cap L_1.
$$
特别地，固定输入 $p$ 后只有满足 $\mathcal A(q)\le\mathcal A(p)$ 的输出可能出现。

**证明.** 由 Cauchy--Riemann 方程 $\partial_su+J\partial_tu=0$ 得 $\partial_tu=J\partial_su$，所以
$$
|\partial_su|_{g_J}^2
=\omega(\partial_su,J\partial_su)
=\omega(\partial_su,\partial_tu).
$$
积分并使用 $\omega=d\lambda$ 与 Stokes 定理，截断在 $[-S,S]\times[0,1]$ 后得到
$$
\int u^*\omega
=\int_{t=0}u^*\lambda-\int_{t=1}u^*\lambda
+\int_{s=S}u^*\lambda-\int_{s=-S}u^*\lambda.
$$
当 $S\to\infty$ 时，所假设的 $C^1$ 渐近收敛使两条竖直边的积分趋于
零。实际 Floer 解的指数衰减由外部 Fredholm 渐近理论提供；本命题不把
它隐藏在“有限能量”一词中。两条水平边分别位于 $L_0,L_1$，故其贡献为
$$
[f_0(q)-f_0(p)]-[f_1(q)-f_1(p)]
=\mathcal A(p)-\mathcal A(q).
$$
左端即 $E(u)\ge0$，最后的作用量不等式随之得到。证毕。

**外部输入定理 3.14（compact exact 交点模型的 Floer 微分）.** 设
$\widehat M$ 是约定 E.1A 的 Liouville completion，$\mathbb L_0,\mathbb L_1$
是其中横截相交的 compact exact branes。对这对对象取零 Hamiltonian
$H_{01}=0$ 与 contact-type compatible $J_{01}$；于是 Hamiltonian chords
正是交点，且横截性给出 pair datum 的非退化性。再假设外部输入定理 E.6
对这组数据给出的 regularity、energy、no-escape、compactness、
relative-Pin orientation 与 gluing package。则定义 3.13 的和有限、次数为
$+1$，并且
$$
(\mu^1)^2=0.
$$

**证明路线（外部输入）.** 命题 3.13A 控制 strip 能量。若 $v:(D,\partial D)\to(M,L_i)$ 是 $J$-holomorphic disk，则
$$
\int_Dv^*\omega=\int_{\partial D}v^*\lambda
=\int_{\partial D}d(f_i\circ v)=0;
$$
相容性迫使其能量为零，故 $v$ 常值。Exactness 同样排除非平凡 sphere
bubbling。于是定理 E.6 把一维 strip 模空间的紧化边界识别为两层
broken strips；determinant-line gluing sign 使其边界计数正是
$(\mu^1)^2$ 的系数。完整证明仍依赖 E.6 的 Fredholm、no-escape、
compactness、orientation 与 gluing，以上文字只说明该外部输入如何用于
本结论。

一般 compactly supported Hamiltonian pair data 的生成元是 Hamiltonian
chords，而不是定义 3.13 的交点；该版本在定义 4.4--4.7 中给出，并由
定理 4.9 覆盖。两种模型的比较还要使用定理 3.18 的 continuation 外部输入。

**警告 3.15.** 定理 3.14 是外部输入。若没有 exactness 或 monotonicity，disk bubbling 可能产生 obstruction term $\mu^0$，此时 $\mu^1$ 不一定平方为零，必须引入 bounding cochains 或 curved $A_\infty$ 结构。

该定理也不覆盖非紧 conical branes：即使某一对对象碰巧只有有限交点，
构造整个范畴仍需控制无穷远、continuation 与 wrapping；这些内容属于
第六章，而不是把定理 3.14 的“compact”删去即可得到。

## 3.5 Hamiltonian 扰动与不变量性

**定义 3.16.** 时间依赖 Hamiltonian 是光滑函数 $H:[0,1]\times M\to\mathbb R$。本书固定符号约定
$$
\omega(X_{H_t},-) = dH_t.
$$
若 $M$ 紧，或 $H_t$ 紧支撑，或另有完备性假设保证积分曲线在 $t\in[0,1]$ 上存在，则其 Hamiltonian flow 记为 $\varphi_H^t$。

**定义 3.17.** 两个 branes 的 Floer cochains 在非横截时通常通过 Hamiltonian perturbation 定义：
$$
CF^\ast(L_0,L_1):=CF^\ast(\varphi_H^1(L_0),L_1),
$$
其中 $H$ 选择到使交点横截且有限。非紧情形若该条件不能满足，必须改用 wrapped/partially wrapped complex，而不能仍使用定义 3.10 的有限直和。

**外部输入定理 3.18（continuation invariance）.** 合适 exact 假设下，不同 Hamiltonian perturbations 和 almost complex structures 得到的 Floer complexes 在同伦范畴中 quasi-isomorphic。

**解释 3.19.** continuation invariance 是 Fukaya category 成为几何不变量的基础。没有它，Floer complex 只是依赖辅助选择的链复形，不能作为 HMS 的 A-side。

## 3.6 从 Floer cochains 到 Fukaya category

本章只定义了对象和 morphism complexes 的入口。Fukaya category 还需要高阶复合
$$
\mu^d:
CF^\ast(L_{d-1},L_d)\otimes\cdots\otimes CF^\ast(L_0,L_1)
\to CF^\ast(L_0,L_d)[2-d],
$$
这些复合由 holomorphic $(d+1)$-gons 的零维模空间计数定义。其 $A_\infty$ 方程来自一维 polygon 模空间的边界分解。第四章将把这个构造写成定理链，并明确列出 analytic inputs。

## 本章小结

A-side 的最低入口是带 brane data 的 Lagrangian，而不是裸 Lagrangian 子流形。Floer cochains 由横截交点生成，Floer 微分由 holomorphic strips 的计数给出。$\mu^1{}^2=0$、不变量性和高阶 $A_\infty$ 方程都依赖深层分析输入；本书在未建立分析基础时必须把它们标为外部输入。

## 练习

**练习 3.1.** 证明辛向量空间中 isotropic 子空间的维数不超过总维数的一半。

**练习 3.2.** 在 $T^\ast S^1$ 中写出零截面和一个横截 Hamiltonian 扰动的交点，并描述 Floer cochain 的生成元。

**练习 3.3.** 用 exactness 计算 holomorphic strip 的能量，并说明为什么 action 差控制能量。

**练习 3.4.** 解释若存在 disk bubbling，为什么 $\mu^1{}^2=0$ 的边界分解会失败。
