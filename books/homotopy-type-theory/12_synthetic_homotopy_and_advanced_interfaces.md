# 第十二章：合成同伦论与高级接口

## 本章目标

本章给出 HoTT 中合成同伦论的教材级地图：高阶同伦群、higher groups、Postnikov towers、cofiber sequences、Eilenberg-Mac Lane 型、上同调、Blakers-Massey、Freudenthal、Hopf fibration、smash product、谱接口和谱序列接口。这里许多内容属于高级外部输入；本章的任务是给出精确定理形态和证明接口，而不是把未展开的高阶相干伪装为基础证明。

## 依赖前置知识

本章依赖前十一章，尤其是 HIT、圆、悬挂、pushout、截断、基本群和单值性。高级定理还依赖连通性、局部系数、cofiber、谱和模型论输入；这些依赖会在相应附录中标出。

## 12.1 高阶同伦群

**定义 12.1.** 对 pointed type $(X,x_0)$，定义
$$
\pi_n(X,x_0)\coloneqq \left\|\Omega^n(X,x_0)\right\|_0
$$
及其群结构。$n=1$ 时群运算来自 loop 复合；$n\ge 2$ 时交换性来自 Eckmann-Hilton。

**定理 12.2（高阶同伦群交换性）.** 若 $n\ge 2$，则 $\pi_n(X,x_0)$ 是交换群。

**证明.** 见附录 AC。二重 loop space 上纵向复合与横向复合有共同单位并满足 interchange law；Eckmann-Hilton 推出二重 loop 复合交换，再对 $\Omega^{n-2}(X,x_0)$ 应用该结论并下降到集合截断。$\square$

## 12.2 Higher groups 与 Postnikov 接口

**定义 12.3（higher group 与 delooping 接口）.** Pointed connected type $B$ 可看作 higher group 的 delooping，其 loop space $\Omega B$ 给出群状对象。普通群 $G$ 的 classifying type $BG$、torsor 和 principal bundle 分类接口见附录 BF。

**定义 12.4（Postnikov tower 接口）.** 对类型 $X$，Postnikov tower 是由截断
$$
\cdots\to \|X\|_{n+1}\to \|X\|_n\to\cdots
$$
组织的近似系统。其 fiber、$k$-invariant、Whitehead theorem 和 obstruction class 的教材接口见附录 BJ。

**定义 12.5（局部系数接口）.** 非单连通类型中的同伦群族和 obstruction class 通常取值于局部系数系统。Abelian group local systems、twisted cohomology 和 twisted $k$-invariants 见附录 BM。

## 12.3 Eilenberg-Mac Lane 型与上同调

**输入 12.6（EM 型塔）.** 对阿贝尔群 $A$ 和自然数 $n$，$K(A,n)$ 是满足
$$
\pi_n(K(A,n))\cong A,\qquad \pi_k(K(A,n))=0\quad(k\ne n)
$$
的指称类型。HoTT 中通常通过 HIT 或谱构造实现。本书采用附录 Y.1 的 EM 型塔输入。

**定义 12.7（上同调）.** 对类型 $X$，定义
$$
H^n(X;A)\coloneqq \left\|X\to K(A,n)\right\|_0.
$$
约化上同调、阿贝尔群结构、反变函子性、悬挂同构和球面计算见附录 Y。

**定理 12.8（球面上同调基本计算）.** 对整数系数，
$$
\widetilde H^k(\mathbb S^n;\mathbb Z)\cong
\begin{cases}
\mathbb Z,& k=n,\\
0,& k\ne n.
\end{cases}
$$

**证明状态.** 见附录 Y.12。证明使用约化上同调悬挂同构、$\mathbb S^{n+1}\simeq\Sigma\mathbb S^n$、维数公理和连通球面的约化 $H^0$ 消失。完整 EM 型塔和 cup product 高阶相干作为高级输入处理。

## 12.4 Blakers-Massey、Freudenthal 与 Hopf fibration

**输入 12.9（Blakers-Massey）.** 给定 pushout 方块，若两条映射分别满足连通性假设，则 gap map 的连通度由两者连通度控制。精确陈述见附录 AL，join connectivity、flattening lemma 和 pushout 路径空间技术分别见附录 AU、AY。

**定理 12.10（Freudenthal 悬挂定理接口）.** 若 $A$ 足够连通，则 unit
$$
A\to\Omega\Sigma A
$$
在相应范围内高连通。

**证明状态.** 见附录 AL.8。由悬挂 pushout 方块和 Blakers-Massey 定理推出。

**输入 12.11（Hopf fibration 接口）.** Hopf fibration 可作为
$$
\mathbb S^1\to\mathbb S^3\to\mathbb S^2
$$
的 fiber sequence。其与 $\pi_3(\mathbb S^2)$ 的关系通过同伦群长正合列给出；见附录 AL、AP。

## 12.5 Smash product、谱与谱序列

**定义 12.12（smash product）.** 对 pointed types $A,B$，smash product
$$
A\wedge B
$$
是 $A\times B$ 对 wedge $A\vee B$ 的商。递归泛性质、球面 smash 和对称幺半结构见附录 AM。

**定义 12.13（谱接口）.** 预谱、Omega 谱、谱映射、稳定等价、稳定范畴和 filtered spectra 的最小接口见附录 AM、AZ。

**定义 12.14（exact couple 谱序列接口）.** 谱序列的代数核心可由 exact couple 生成。附录 AQ 定义 exact couple、derived couple、页 $E_r$、微分 $d_r$ 和条件收敛。

**事实 12.15（核心谱序列模板）.** Serre、Atiyah-Hirzebruch 和 Adams 谱序列的输入格式、$E_2$ 页、微分次数和收敛目标见附录 AV。Steenrod operations、Steenrod algebra、Ext 和 Adams 低维类接口见附录 BN。

**定义 12.16（cofiber 与 Mayer-Vietoris 接口）.** Cofiber、Puppe sequence、cofiber cohomology long exact sequence 和 Mayer-Vietoris long exact sequence 见附录 BK。它们是从 pushout 和 cofiber 进行具体上同调计算的入口。

## 12.6 本章边界

本章不声称已经内部化所有稳定同伦论和谱序列计算。以下内容保留为高级接口或外部输入：

1.  EM 型塔的完整构造和所有高阶相干；
2.  Blakers-Massey 的逐行内部证明；
3.  Hopf fibration 的完整 fiber sequence 相干；
4.  具体局部系数计算和 Postnikov obstruction 计算；
5.  Steenrod operations、Ext 代数和 Adams differentials 的内部构造；
6.  每个具体谱序列的强收敛验证。

## 本章小结

HoTT 的合成同伦论远超圆的基本群。本章把高阶同伦群、上同调、Blakers-Massey、Hopf fibration、smash product、谱、cofiber、Postnikov 和谱序列组织成可审查接口。核心主线仍是前十一章和第十三至十四章；本章高级内容应保留明确状态标签。

## 练习

**练习 12.1.** 用 Eckmann-Hilton 引理解释为什么 $\pi_2(X,x_0)$ 是交换群。

**练习 12.2.** 写出 $H^n(X;A)$ 的反变函子性需要哪些函数复合和截断递归。

**练习 12.3.** 说明 Blakers-Massey 定理在 Freudenthal 悬挂定理中的作用。

**练习 12.4.** 对比 cofiber 长正合列和 Mayer-Vietoris 长正合列所需的输入数据。
