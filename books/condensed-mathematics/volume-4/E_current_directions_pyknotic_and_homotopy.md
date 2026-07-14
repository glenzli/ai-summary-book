# 附录 E：当代方向、pyknotic 对象与凝聚同伦

## E.0 目标

本附录补上凝聚数学当前发展中容易被四卷主线漏掉的方向：pyknotic 对象、凝聚同伦类型、凝聚谱、Galois categories 与 pro-etale/exodromy 接口。本附录的目标不是完整发展这些理论，而是给出严格的定义入口和与前四卷的关系。

## E.1 pyknotic 对象的定义

固定两个 universe，并令 $\mathbf{Comp}$ 为较小层级 compact Hausdorff 空间在较大层级
中的站点，覆盖为有限 jointly surjective 族。这个层级选择是 pyknotic 定义的一部分，
不能在“大范畴”上无条件忽略。令 $\mathcal C$ 为可在其中形成 $\mathcal C$-值
hypersheaf 的 presentable $\infty$-category；集合和阿贝尔群的 1-范畴情形按其离散
$\infty$-范畴理解。

**定义 E.1.1（本附录的 hypercomplete 约定）.** $\mathcal C$ 中的 pyknotic object
是一个满足 hyperdescent 的函子

$$
F:\mathbf{Comp}^{op}\to\mathcal C.
$$

Barwick--Haine 因此把 pyknotic objects 概括为 compacta 站点上的
(hyper)sheaves；具体文献可能先取 sheaf 再另作 hypercompletion。当
$\mathcal C=\mathbf{Set}$ 时，0-截断对象自动 hypercomplete，在固定同一站点和
universe 后这就是凝聚集合。当 $\mathcal C=\mathcal S$ 时称为 pyknotic space/anima；
当 $\mathcal C=\operatorname{Sp}$ 时称为 pyknotic spectrum。若使用“condensed
anima/spectrum”一词，必须同时声明是否已 hypercomplete。

**命题 E.1.2（0-截断比较）.** 若 pyknotic anima $F:\mathbf{CHaus}^{op}\to\mathcal S$ 的每个值都是 0-截断空间，则 $F$ 等价于一个凝聚集合。反过来，每个凝聚集合给出一个 0-截断的 pyknotic anima。

**证明.** 0-截断空间范畴等价于集合范畴。若 $F(S)$ 对每个 $S$ 都 0-截断，则可把 $F$ 视为集合值函子。空间值 sheaf 条件在 0-截断对象上退化为集合值 sheaf 条件，因为 homotopy limit 在 0-截断对象中等于普通 limit。反向地，把集合看成离散空间，即得到 0-截断空间值 sheaf。两构造互逆。证毕。

因此，pyknotic theory 不是与凝聚集合平行的无关对象，而是把凝聚集合提升到同伦值、谱值和高范畴值的版本。

## E.2 sheaf 条件的同伦形式

设 $\mathcal U=\{U_i\to U\}$ 是有限覆盖。对空间值预层 $F$，Čech descent 不是普通
等化子，而是要求自然映射

$$
F(U)\to
\operatorname*{holim}
\left(
\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_UU_j)
\rightrightarrows
\prod_{i,j,k}F(U_i\times_UU_j\times_UU_k)
\cdots
\right)
$$

为等价。右侧是 Čech nerve 的同伦极限。Pyknotic hypersheaf 还要求对任意 hypercover
满足同类 totalization；只检查单个覆盖的 Čech nerve 一般不能替代 hyperdescent。

**命题 E.2.1.** 若 $F$ 是 0-截断空间值预层，则上述同伦 sheaf 条件等价于第一卷的集合值 sheaf 条件。

**证明.** 0-截断空间构成 $\mathcal S$ 的反射全子范畴，等价于 $\mathbf{Set}$，并且有限极限和相关 Čech 极限在该子范畴内计算。由于所有项 0-截断，Čech 同伦极限也是 0-截断，并由普通极限给出。于是条件正是集合值匹配族唯一粘合。证毕。

## E.3 凝聚谱与稳定化

**定义 E.3.1.** 记谱值 sheaf 与超完备谱值 sheaf 范畴分别为

$$
\operatorname{CondSp}
=
\operatorname{Shv}(\mathbf{Comp},\operatorname{Sp}),
\qquad
\operatorname{CondSp}^{\wedge}
=
\operatorname{Shv}^{\wedge}(\mathbf{Comp},\operatorname{Sp}).
$$

第八章和附录 G 的“超完备凝聚谱”取第二个范畴。第一种约定只要求覆盖下降；第二种还
要求所有 hypercover 下降。

**命题 E.3.2.** $\operatorname{CondSp}$ 与
$\operatorname{CondSp}^{\wedge}$ 都是稳定 $\infty$-范畴。

**证明.** 谱范畴 $\operatorname{Sp}$ 稳定，且 sheaf 条件由极限表达。预层范畴 $\operatorname{Fun}(\mathbf{CHaus}^{op},\operatorname{Sp})$ 逐点稳定。sheaf 全子范畴由满足一族极限条件的对象组成，对有限极限、有限余极限和 suspension/loop 封闭。因此它稳定。证毕。

这给出凝聚同伦论的基本对象。solid/analytic/liquid 理论可看作在凝聚谱或凝聚导出范畴中进一步施加 localization 条件。

## E.4 pyknotic abelian groups 与凝聚阿贝尔群

**命题 E.4.1.** 在固定同一 compacta 站点与 universe 后，0-截断 pyknotic abelian
groups 的 1-范畴与凝聚阿贝尔群范畴等价。

**证明.** 阿贝尔群对象是集合值代数结构。若 $F:\mathbf{CHaus}^{op}\to\mathbf{Ab}$ 满足 sheaf 条件，则忘记到集合得到凝聚集合，并且群运算是 sheaf 中的态射。反过来，凝聚阿贝尔群按定义就是阿贝尔群值 sheaf。若从 anima 值观点出发，离散阿贝尔群对象正是 0-截断的 grouplike $E_\infty$-对象，等价于普通阿贝尔群。证毕。

## E.5 凝聚同伦类型

凝聚同伦类型可以理解为 pyknotic anima。它把一个“同伦类型随紧 Hausdorff 参数连续变化”的对象记录为 sheaf

$$
X:\mathbf{CHaus}^{op}\to\mathcal S.
$$

典型来源：

1. 拓扑空间 $T$ 的连续映射对象
   $$
   S\mapsto\operatorname{Map}_{\operatorname{Top}}(S,T)
   $$
   若把右侧取为合适的映射空间，可得到空间值预层。
2. 凝聚集合 $A$ 作为离散同伦类型。
3. 凝聚阿贝尔群 $M$ 经 Dold-Kan 或 Eilenberg-Mac Lane 构造给出凝聚谱/凝聚空间。

**风险点.** 若 $T$ 是一般拓扑空间，映射空间拓扑和 sheaf 条件需要额外检查。凝聚集合只需要集合 $\operatorname{Cont}(S,T)$；凝聚同伦类型还需要连续映射空间本身的同伦结构。

## E.6 Galois categories 与 exodromy 的接口

凝聚/pyknotic 语言与 Galois categories 的接口来自同一个思想：用合适的测试对象和 sheaf 条件重建几何对象。

在经典 Galois theory 中，连通空间或 scheme 的有限覆盖范畴常由基本群控制。pro-etale theory 和 exodromy theory 则把这一思想推广到更细的站点和更高范畴：

1. pro-etale site 提供足够多的局部投射对象。
2. locally constant sheaves 或 constructible sheaves 可由路径/出口路径数据控制。
3. pyknotic/condensed 对象提供处理非离散拓扑和无限极限的范畴环境。

**外部输入定理 E.6.1（Wolf）.** 若 $X$ 是 coherent scheme，则其 hypercomplete
pro-étale $\infty$-topos 等价于 Galois category $\operatorname{Gal}(X)$ 在 pyknotic
spaces 中的连续表示范畴。

因此 pro-étale/pyknotic 接口不是尚未建立的类比；已知定理的精确中介是
$\operatorname{Gal}(X)$，并带有 coherent 与 hypercomplete 假设。本书当前只证明了
sheaf、站点比较和投射测试对象的基础部分；Wolf 定理与完整 exodromy 仍作为外部输入，
后者需要单独教材。

## E.7 与四卷主线的关系

| 方向 | 本书已有基础 | 仍缺内容 |
| --- | --- | --- |
| pyknotic sets/anima | 站点、sheaf、凝聚集合 | 空间值 descent 与高范畴语言 |
| condensed spectra | 派生范畴、Ext/Tor、localization | 稳定 $\infty$-范畴细节 |
| solid/analytic spectra | solid/analytic localization | 谱值版本的完整张量理论 |
| Galois/exodromy | Wolf 的 pro-étale--pyknotic 比较、投射测试对象 | 一般 exodromy、constructible sheaves 与出口路径 |
| formalization | 站点和 sheaf 形式化路线 | 实际 Lean/mathlib 代码 |

## E.8 本附录小结

凝聚数学的现代方向不是只研究集合值 sheaf。更一般的对象是

$$
\operatorname{Shv}(\mathbf{CHaus},\mathcal C)
$$

其中 $\mathcal C$ 可以是集合、阿贝尔群、空间、谱或稳定范畴。前四卷主要处理集合、阿贝尔群、导出范畴、solid/analytic/liquid 模；pyknotic 和凝聚同伦方向则要求把同一站点语言提升到 $\infty$-范畴值 sheaf。

## 练习

**练习 E.1.** 证明 0-截断 pyknotic anima 等价于凝聚集合。

**练习 E.2.** 写出空间值 sheaf 条件中的 Čech nerve 前四项。

**练习 E.3.** 说明为什么 $\operatorname{Shv}(\mathbf{CHaus},\operatorname{Sp})$ 是稳定范畴。

**练习 E.4.** 比较“凝聚阿贝尔群”和“0-截断 pyknotic abelian group”的定义。
