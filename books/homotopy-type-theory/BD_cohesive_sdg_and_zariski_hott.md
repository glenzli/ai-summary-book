# 附录 BD：Cohesive HoTT、合成微分几何与 Zariski 接口

附录 AT 给出 cohesive/modal HoTT 的伴随和 left exact 模态接口。本附录把几何方向进一步写成数学对象：无穷小对象、Kock-Lawvere 公理、切丛、微分、形状模态和 Zariski/合成代数几何接口。它们不是普通 HoTT 或附录 BA 的构造性实分析定理，而是需要额外几何对象语言或模型的结构。

## BD.1 Cohesive 结构

**输入 BD.1（cohesive adjoint string）。** Cohesive HoTT 通常假设一串伴随
$$
\Pi\dashv\mathsf{Disc}\dashv\Gamma\dashv\mathsf{Codisc}
$$
或等价的 shape、discrete、global sections、codiscrete 算子，并带有 exactness 条件。

**定义 BD.2（shape modality）。** $\Pi X$ 表示 $X$ 的形状或同伦型。单位
$$
X\to\mathsf{Disc}(\Pi X)
$$
忘记 cohesive 几何信息，保留同伦形状。

**原则 BD.3（不可由普通 HoTT 推出）。** $\Pi,\mathsf{Disc},\Gamma,\mathsf{Codisc}$ 不是第六章单值性的推论，也不是第八章截断的特例。使用它们必须声明额外形成规则、伴随单位/余单位和 exactness 假设。

## BD.2 无穷小对象与 Kock-Lawvere 公理

**输入 BD.4（线对象）。** 合成微分几何需要一个交换环对象 $\mathbb R_{\mathsf{sdg}}$，它不是附录 AK/AR/BA 的 Cauchy 实数对象。它支持 nilpotent infinitesimals。

**定义 BD.5（一阶无穷小对象）。** 定义
$$
D\coloneqq\sum_{d:\mathbb R_{\mathsf{sdg}}}(d^2=0).
$$
元素 $d:D$ 是一阶无穷小。

**公理 BD.6（Kock-Lawvere，一阶形式）。** 对任意函数 $f:D\to\mathbb R_{\mathsf{sdg}}$，类型
$$
\sum_{a:\mathbb R_{\mathsf{sdg}}}
\sum_{b:\mathbb R_{\mathsf{sdg}}}
\prod_{d:D}\bigl(f(d)=a+b\cdot d\bigr)
$$
可收缩。

**命题 BD.7（导数唯一性，书内证明核）。** 在 BD.6 下，给定 $f:D\to\mathbb R_{\mathsf{sdg}}$，其线性系数 $b$ 唯一。

**证明.** BD.6 断言三元数据 $(a,b,h)$ 的总类型可收缩。可收缩类型任意两点相等；对该路径取第二投影，得到两个候选 $b$ 相等。$\square$

**警告 BD.8（与 Cauchy/Dedekind 实数的区别）。** 附录 AK、AR、AW、BA 的实数服务于构造性分析。它们不自动含非零 nilpotent，也不满足 BD.6。若把 $\mathbb R_C$ 直接替换为 $\mathbb R_{\mathsf{sdg}}$，将改变理论。

## BD.3 切丛与微分

**定义 BD.9（切丛）。** 对 cohesive/SDG 类型 $X$，其切丛定义为指数
$$
T X\coloneqq X^D.
$$
投影 $\pi:T X\to X$ 由在 $0:D$ 处求值得到：
$$
\pi(\gamma)\coloneqq\gamma(0).
$$

**定义 BD.10（点处切向量）。** 点 $x:X$ 处的切向量类型为 fiber
$$
T_xX\coloneqq\sum_{\gamma:D\to X}(\gamma(0)=x).
$$

**定义 BD.11（映射的微分）。** 若 $f:X\to Y$，定义
$$
T f:T X\to T Y,\qquad
T f(\gamma)\coloneqq f\circ\gamma.
$$
它限制为
$$
d f_x:T_xX\to T_{f(x)}Y
$$
其中基点路径由 $\gamma(0)=x$ 通过 $\mathsf{ap}_f$ 得到。

**命题 BD.12（链式法则，书内证明核）。** 对 $f:X\to Y$、$g:Y\to Z$，
$$
T(g\circ f)=Tg\circ Tf.
$$

**证明.** 对 $\gamma:D\to X$，
$$
T(g\circ f)(\gamma)=(g\circ f)\circ\gamma
$$
而
$$
(Tg\circ Tf)(\gamma)=Tg(f\circ\gamma)=g\circ(f\circ\gamma).
$$
由函数复合结合的 judgmental 或函数外延性路径得到相等。$\square$

## BD.4 Microlinearity 与微分形式

**定义 BD.13（microlinear，接口）。** 类型 $X$ 称为 microlinear，若它把指定的有限无穷小极限图送到极限图。形式上，对每个 infinitesimal limit diagram $D_\bullet$，canonical map
$$
X^{\operatorname{colim}D_\bullet}\to\lim_i X^{D_i}
$$
是等价。

**用途.** Microlinearity 使切丛、向量场、微分形式和流的常规计算成立。

**输入 BD.14（de Rham 接口）。** 在合适的 cohesive/SDG 模型中，可定义微分形式、外微分
$$
d:\Omega^k(X)\to\Omega^{k+1}(X)
$$
并证明 $d^2=0$，从而得到 de Rham 上同调。

**边界.** 本书不把 de Rham 复形当作附录 Y 的 EM 型上同调的直接实例。二者之间的比较需要 de Rham theorem 或 cohesive 模型中的额外定理。

## BD.5 Zariski 与合成代数几何接口

**输入 BD.15（局部环对象）。** 合成代数几何通常从一个局部交换环对象 $R$ 开始，并把“仿射空间”“Zariski open”“局部化”等作为对象语言结构。

**定义 BD.16（基本开集接口）。** 对 $f:R$，基本开集 $D(f)$ 可通过局部化或 open modality 表示，使得在 $D(f)$ 上 $f$ 可逆。

**规则 BD.17（Zariski 覆盖）。** 一个族 $D(f_i)$ 覆盖仿射对象，通常需要证明理想生成条件
$$
(f_1,\ldots,f_n)=R
$$
或其类型论版本。覆盖消去原则是 sheaf/gluing 原则，不是普通 $\Sigma$ 或 $\Pi$ 消去。

**事实 BD.18（SAG 使用边界）。** Cherubini-Coquand-Hutzler 的 synthetic algebraic geometry 需要专门的类型论基础。其环对象、Zariski open 和 sheaf 条件不能替换成第八章的集合商代数。

## BD.6 与本书实数和上同调章节的关系

1.  附录 BA 的连续性和紧致性是构造性分析定理；BD 的导数来自 nilpotent infinitesimal 和 Kock-Lawvere 公理。
2.  附录 Y 的上同调由 EM 型表示；BD 的 de Rham 接口需要微分形式和 cohesive 比较定理。
3.  附录 AT 给出 modal/cohesive 算子；BD 给出这些算子服务的几何对象。
4.  合成代数几何结论只有在列出环对象、公理、覆盖、gluing 和模型来源后才能进入内部推导。

## BD.7 几何模型边界

Shape 模态、无穷小对象、Kock--Lawvere 公理、microlinearity 与 Zariski 覆盖属于特定 cohesive/SDG/SAG 语言。具体模型、sheaf 语义和 de Rham 比较定理均未由基础 HoTT 推出；缺少这些输入时，本附录中的几何陈述只能作为条件化接口，不能作用于第十五章的普通集合族或 simplicial 模型。
