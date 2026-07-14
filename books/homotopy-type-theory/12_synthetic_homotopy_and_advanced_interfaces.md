# 第十二章：从环路到稳定现象

圆的 loop space 已经可以完全计算，但合成同伦论并不等于不断寻找新的 encode--decode 技巧。迭代 loop 会产生高阶同伦群，映射到 Eilenberg--Mac Lane 型会产生上同调，pushout 的路径信息由连通性定理控制，而反复悬挂最终把问题带到谱和稳定范畴。每一次推进都需要新的输入，不能仅凭“类型就是空间”的直觉获得。

本章选择四条彼此衔接的链作为高级主题导读。能由前文规则和指定输入推出的结论会实际推导；Blakers--Massey、EM 型塔与谱序列收敛等大型结果则只以完整假设下的外部接口出现。这样既能看见高级同伦论怎样从 loop、截断和 pushout 生长出来，也能准确知道前十一章尚未证明什么。

## 12.1 迭代 loop 与高阶同伦群

**定义 12.1（迭代 loop）.** 设 $(X,x_0)$ 是带基点类型，其中 $X:\mathcal U_i$。递归定义带基点类型
$$
\Omega^0(X,x_0)\coloneqq(X,x_0),
$$
而递归步先令 $(Y,y_0)\coloneqq\Omega^n(X,x_0)$，再置
$$
\Omega^{n+1}(X,x_0)\coloneqq(y_0=_Yy_0,\mathsf{refl}_{y_0}).
$$
因此每次递归都把当前带基点类型替换为其 loop space，并以反身路径作为新基点；这里不能把底层类型 $Y$ 本身误当成下一阶 loop space。

**定义 12.2（高阶同伦群）.** 对 $n\ge1$，定义底层集合
$$
\pi_n(X,x_0)\coloneqq
\left\|\Omega^n(X,x_0).1\right\|_0.
$$
当 $n=1$ 时，乘法由 loop 复合下降到集合截断；单位与逆分别由反身路径和路径逆下降。这里没有把 $\pi_0$ 称为群：$\|X\|_0$ 一般只有连通分支集合，没有规范群结构。

**定理 12.3（高阶同伦群的代数结构）.** 对每个带基点类型 $(X,x_0)$，$\pi_1(X,x_0)$ 是群；若 $n\ge2$，则 $\pi_n(X,x_0)$ 是阿贝尔群。

**证明（书内证明核）.** $n=1$ 的群律由第二章的路径群胚律逐项下降到集合截断，细节与命题 11.3 相同。若 $n\ge2$，把 $\Omega^nX$ 写成某个类型的二重 loop space。二重 loop 上有两个以反身二阶路径为共同单位的复合，路径代数给出 interchange law；附录 AC 的 Eckmann--Hilton 论证推出两种复合相等且交换。再由集合截断递归下降运算和交换律，得到阿贝尔群。$\square$

**例 12.4（最低两阶的差别）.** 第十一章给出
$$
\pi_1(\mathbb S^1,\mathsf{base})\cong\mathbb Z.
$$
这个群碰巧交换，但原因是具体计算，不是 $\pi_1$ 的一般形式。对 $\pi_2(X,x_0)$，交换性无需计算 $X$：它来自二重 loop 的两种复合。这说明提高一次 loop 不只是增加下标，也会强迫新的代数结构。

## 12.2 EM 型、上同调与一个完整计算

从同伦群转向上同调，需要可表示这些不变量的目标类型。本书不在此构造 Eilenberg--Mac Lane 型，而是先精确说明条件化输入。

**高级输入 12.5（EM 型塔）.** 固定一个 $i$-小阿贝尔群 $A$。假设对每个 $n:\mathbb N$ 给定带基点类型 $K(A,n)$、基点等价
$$
\Omega K(A,n+1)\simeq_* K(A,n),
$$
以及交换 $H$-space 结构。要求 $K(A,0)$ 是底层集合等同于 $A$ 的离散带基点类型；对 $n\ge1$，要求 $K(A,n)$ 是 $(n-1)$-连通的 $n$-type，且
$\pi_n(K(A,n))\cong A$。这里 $\pi_k$ 只在 $k\ge1$ 时使用；零阶信息由 $K(A,0)$ 的离散描述承担。
这些数据合称本章的 EM 型塔输入，详细接口见附录 Y.1。本章所有上同调结论都以该输入为条件；它不是前文 HIT 规则的自动推论。

**定义 12.6（上同调）.** 对 $X:\mathcal U_j$，定义
$$
H^n(X;A)\coloneqq
\left\|X\to K(A,n)\right\|_0.
$$
若 $(X,x_0)$ 带基点，则约化上同调定义为
$$
\widetilde H^n(X;A)
\coloneqq
\left\|X\to_*K(A,n)\right\|_0.
$$
这两个类型位于由 $i,j$ 和 EM 型所在宇宙决定的最大层级中，并且按定义都是集合。对 $n\ge1$，$K(A,n)$ 的交换 $H$-space 运算逐点作用于映射，再下降到集合截断，使它们成为阿贝尔群。

**命题 12.7（反变函子性）.** 若 $f:X\to Y$，则预合成定义群同态
$$
f^*:H^n(Y;A)\to H^n(X;A),
\qquad
[u]\longmapsto[u\circ f],
$$
并满足 $(\mathsf{id}_X)^*=\mathsf{id}$ 与
$(g\circ f)^*=f^*\circ g^*$。

**证明（书内证明，依赖高级输入 12.5）.** 目标是集合，所以可对集合截断代表元消去。两个函子律分别化为函数复合的单位律和结合律；群同态性逐点化为 $K(A,n)$ 的 $H$-space 运算与预合成相容。函数相等使用函数外延性。$\square$

**高级输入 12.8（悬挂与维数）.** 假设带基点悬挂--loop 等价
$$
(\Sigma X\to_*Y)\simeq(X\to_*\Omega Y),
$$
球面等价 $\mathbb S^{m+1}\simeq\Sigma\mathbb S^m$，以及维数输入
$$
\widetilde H^0(\mathbb S^0;A)\cong A,
\qquad
\widetilde H^k(\mathbb S^0;A)=0\quad(k>0),
$$
并假设 $m\ge1$ 时 $\mathbb S^m$ 连通。前两项来自相应 HIT 的 pointed 泛性质与 EM loop 数据；完整证明边界见附录 Y.9-Y.11。

**定理 12.9（约化球面上同调）.** 在高级输入 12.5 与 12.8 下，对 $k,n:\mathbb N$ 有
$$
\widetilde H^k(\mathbb S^n;A)\cong
\begin{cases}
A,&k=n,\\
0,&k\ne n.
\end{cases}
$$

**证明（条件化书内推导）.** 先由悬挂--loop 等价与
$\Omega K(A,r+1)\simeq K(A,r)$ 得到悬挂同构
$$
\widetilde H^{r+1}(\Sigma X;A)
\cong
\widetilde H^r(X;A).
$$
若 $k\ge n$，连续使用该同构 $n$ 次，把目标化为
$\widetilde H^{k-n}(\mathbb S^0;A)$，再用维数输入。若 $k<n$，使用同构 $k$ 次，得到
$\widetilde H^0(\mathbb S^{n-k};A)$；此时 $n-k\ge1$，连通性使约化零次上同调消失。两种情形覆盖全部自然数 $k,n$。$\square$

这个计算展示了高级输入应怎样使用：EM 型塔、悬挂泛性质和维数公理被明确列在定理之前；归纳降维则由本书承担。若没有高级输入 12.8，公式仍可作为期望，但不能称为已经证明的球面计算。

## 12.3 Pushout、连通性与不稳定定理

**定义 12.10（连通类型与连通映射）.** 对 $r\ge-2$，置
$$
\mathsf{isConnected}_r(Z)\coloneqq\mathsf{isContr}(\|Z\|_r).
$$
称 $Z$ 为 $r$-连通，若该类型有项；称映射 $f:A\to B$ 为 $r$-连通，若
$$
\prod_{b:B}
\mathsf{isConnected}_r(\mathsf{fib}_f(b)).
$$
这里负指标按第八章的截断编号理解；精确的自然数编码和边界 convention 固定在附录 AL、AU。

给定 span $B\xleftarrow{f}A\xrightarrow{g}C$ 及其 pushout $P$，存在 gap map
$$
A\longrightarrow B\times_P C.
$$
它比较原交点 $A$ 与在 pushout 中同伦相交后得到的拉回。控制这张映射的 fiber，是从局部粘合数据推断整体路径空间的关键。

**外部输入定理 12.11（Blakers--Massey，采用版）.** 设 $m,n:\mathbb N$。若 $f:A\to B$ 为 $m$-连通且 $g:A\to C$ 为 $n$-连通，则 canonical gap map
$$
A\longrightarrow B\times_{B\sqcup_A C}C
$$
为 $(m+n)$-连通；连通性编号采用定义 12.10 与附录 AL.1-AL.5 的约定。

**来源与边界.** 本书采用 *Homotopy Type Theory: Univalent Foundations of Mathematics*, Theorem 8.10.2；附录 AL 固定转写后的编号，附录 AU、AY 只解释 join connectivity、flattening 与 path-code 包在来源证明中的角色。本章不重证长篇路径族计算，也不把未写出的指标偏移隐含在“足够连通”中。

**外部输入定理 12.12（Freudenthal）.** 若 $n\ge0$ 且 $(X,x_0)$ 是 $n$-连通带基点类型，则悬挂单位
$$
X\longrightarrow\Omega\Sigma X
$$
是 $2n$-连通。

**来源与边界.** 本书采用 HoTT Book, Theorem 8.6.4；附录 AL.8 按本书符号转写该结论。把悬挂写成 pushout 并应用定理 12.11 可以解释 $2n$ 的来源，但 gap map 与悬挂单位的比较及其基点相容仍由外部定理承担。

Hopf fibration
$$
\mathbb S^1\longrightarrow\mathbb S^3\longrightarrow\mathbb S^2
$$
进一步把这种连通性方法接到 fiber sequence 和同伦群长正合列；附录 AP 给出 connecting map 与 exactness 的类型，附录 AL 记录本书采用的 Hopf 输入。本章不把经典空间中的纤维图自动当作 HoTT 内部 fiber sequence。

## 12.4 稳定化为什么需要新的语言

悬挂一次通常只在有限范围内改善同伦群；不断悬挂后，稳定现象更适合由谱组织。对带基点类型 $X,Y$，smash product
$$
X\wedge Y
$$
是 wedge 嵌入 $X\vee Y\to X\times Y$ 的 cofiber。球面满足的 smash 关系把悬挂写成与 $\mathbb S^1$ 的 smash，这提示把一列类型及结构映射
$$
\Sigma E_n\to E_{n+1}
$$
作为单个对象研究。

**高级输入 12.13（谱接口）.** 附录 AZ 固定预谱、$\Omega$-谱、谱映射与稳定等价；附录 AM 给出 smash product 所需的 pointed 泛性质。构造稳定范畴、证明对称幺半相干以及比较不同谱模型都属于外部输入，不能从本章的一列结构映射直接推出。

过滤谱或过滤链复形会产生 exact couple。其代数数据为分次对象 $D,E$ 和次数明确的映射
$$
D\xrightarrow{i}D,
\qquad
D\xrightarrow{j}E,
\qquad
E\xrightarrow{k}D,
$$
满足三角 exactness；微分由 $d=j\circ k$ 产生。附录 AQ 从这些数据构造 derived couples 与各页 $E_r$。但“存在各页”不等于“收敛到目标”：Serre、Atiyah--Hirzebruch 和 Adams 谱序列还分别需要纤维化、胞腔过滤或 Adams resolution，并要核对完备性、穷尽性和极限项。附录 AV、BN 只在这些假设下提供接口。

## 12.5 四条链汇合之处

迭代 loop 解释高阶同伦群为何在二阶以后交换；EM 型把群值不变量表示为映射类型，并让悬挂同构实际算出球面上同调；Blakers--Massey 控制 pushout 产生的新路径；谱则把反复悬挂后的稳定信息和过滤计算组织起来。这四条链共享前十一章的路径、截断与 HIT 基础，却各自增加了不可省略的新输入。后续遇到一个高级结论时，应先辨认它属于哪条链，再检查所需对象是否已经构造、连通度采用何种编号、以及所谓收敛究竟是哪一种极限陈述。

## 练习

**练习 12.1.** 展开 $\Omega^2(X,x_0)$ 的底层类型和基点，写出两种 loop 复合的定义域、值域与共同单位。

**练习 12.2.** 在高级输入 12.5 下，逐项证明命题 12.7 的复合律，并说明集合截断消去为何合法。

**练习 12.3.** 只使用悬挂同构与维数输入，分别计算
$\widetilde H^2(\mathbb S^3;A)$ 和
$\widetilde H^3(\mathbb S^3;A)$，写出每一步降维。

**练习 12.4.** 对 span $B\leftarrow A\to C$ 写出 gap map 的目标拉回类型，并说明其一个 fiber 的项包含哪些数据。

**练习 12.5.** 解释为什么 exact couple 已经给出 $E_r$ 页和微分，却仍不足以断言谱序列强收敛。
