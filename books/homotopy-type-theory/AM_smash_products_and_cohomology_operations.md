# 附录 AM：Smash Product、对称幺半结构与上同调运算

本附录补上 pointed 类型的 smash product。该结构是稳定同伦论、球面乘法、cup product 和谱的基础；近年已有对称幺半性的严格证明路线，因此不能只把它列为远景。

## AM.1 Pointed 类型

**定义 AM.1（pointed 类型）.** pointed 类型为
$$
X_\ast\coloneqq(X,x_0)
$$
其中 $X:\mathcal U$ 且 $x_0:X$。pointed 映射为
$$
f:X_\ast\to_\ast Y_\ast
\coloneqq
\sum_{f:X\to Y}f(x_0)=y_0.
$$

**定义 AM.2（wedge）.** pointed 类型的 wedge
$$
X\vee Y
$$
定义为 pushout
$$
X\leftarrow\mathbf 1\to Y
$$
其中两条映射选取基点。

**定义 AM.3（smash product）.** smash product
$$
X\wedge Y
$$
定义为 cofiber
$$
X\vee Y\to X\times Y,
$$
即 pushout
$$
\begin{array}{ccc}
X\vee Y&\to&X\times Y\\
\downarrow&&\downarrow\\
\mathbf 1&\to&X\wedge Y.
\end{array}
$$

## AM.2 基本泛性质

**定义 AM.4（双基点消失映射）.** 给定 pointed 类型 $X,Y,Z$，一个双基点消失映射是函数
$$
f:X\times Y\to Z
$$
连同路径
$$
\prod_{x:X}f(x,y_0)=z_0,\qquad
\prod_{y:Y}f(x_0,y)=z_0
$$
以及在 $(x_0,y_0)$ 处的相干。

**命题 AM.5（smash product 递归）.** pointed 映射
$$
X\wedge Y\to_\ast Z
$$
等价于双基点消失映射 $X\times Y\to Z$。

**证明（证明核）.** 由 AM.3 的 pushout 递归：从 $X\wedge Y$ 到 $Z$ 的 pointed 映射等价于从 $X\times Y$ 到 $Z$ 的映射，使得 $X\vee Y$ 的像被识别到 $z_0$。展开 wedge 的 pushout 递归，正得到两条基点消失条件及其交点相干。$\square$

**命题 AM.6（球面 smash）.** 有等价
$$
\mathbb S^m\wedge\mathbb S^n\simeq\mathbb S^{m+n}.
$$

**证明状态.** 证明对 $m,n$ 作归纳，使用 $\mathbb S^{k+1}\simeq\Sigma\mathbb S^k$ 与 smash 对悬挂的相容性
$$
\Sigma X\wedge Y\simeq\Sigma(X\wedge Y).
$$
后者需要 pushout 交换和更高相干。本书把该命题作为稳定方向的外部证明核。

## AM.3 对称幺半结构

**输入 AM.7（smash product 的 1-coherent symmetric monoidal structure）.** pointed 类型宇宙在 $\wedge$ 下带有：

1.  结合等价
    $$
    (X\wedge Y)\wedge Z\simeq X\wedge(Y\wedge Z);
    $$
2.  单位对象 $\mathbb S^0$ 与左右单位等价；
3.  交换等价
    $$
    X\wedge Y\simeq Y\wedge X;
    $$
4.  pentagon、triangle、hexagon 等 1-coherent 相干。

**验证状态.** Ljungström 2024 给出该结构的证明。本书据此把 AM.7 作为可审计外部输入。

**命题 AM.8（迭代 smash 的相干消解）.** 任意由 $n$ 个 pointed 类型通过 smash product 加括号和置换得到的两个表达式，在 AM.7 的相干约束下有规范等价。

**证明状态.** 这是 symmetric monoidal coherence 的实例。传统证明需大量 pentagon/hexagon 检查；2024 工作给出适合 HoTT 的迭代 smash 归纳原则。本书只引用 1-coherent 层，不声称有完整 $\infty$-coherent symmetric monoidal 结构。

## AM.4 Cup product 的 smash 版本

**定义 AM.9（外积）.** 对上同调代表元
$$
u:X\to K(R,p),\qquad v:Y\to K(R,q),
$$
外积为 pointed 映射
$$
X\wedge Y\to K(R,p+q)
$$
由 EM 乘法
$$
K(R,p)\wedge K(R,q)\to K(R,p+q)
$$
和 AM.5 的泛性质定义。

**定义 AM.10（cup product 的对角定义）.** 对 $u,v:X\to K(R,\_)$，
$$
u\smile v
$$
可写为
$$
X\xrightarrow{\Delta}X\times X\to X\wedge X
\xrightarrow{u\wedge v}K(R,p)\wedge K(R,q)
\to K(R,p+q),
$$
其中 $X\times X\to X\wedge X$ 需在 reduced/pointed 口径下使用。

**命题 AM.11（graded commutativity 的来源）.** cup product 的符号
$$
u\smile v=(-1)^{pq}v\smile u
$$
来自 smash product 的交换等价在球面因子
$$
\mathbb S^p\wedge\mathbb S^q\simeq\mathbb S^q\wedge\mathbb S^p
$$
上诱导的 Koszul 符号。

**证明状态.** 完整证明需要 EM 型乘法、球面 smash、对称幺半相干和符号计算。附录 Y 把 graded commutativity 作为 EM 乘法输入；本附录说明该输入的几何来源。

## AM.5 谱的最小接口

**定义 AM.12（预谱）.** 预谱为一列 pointed 类型 $E_n$ 与结构映射
$$
\sigma_n:\Sigma E_n\to_\ast E_{n+1}.
$$

**定义 AM.13（Omega 谱）.** 预谱为 Omega 谱，若伴随映射
$$
E_n\to_\ast\Omega E_{n+1}
$$
均为等价。

**定义 AM.14（谱表示上同调）.** 若 $E$ 为谱，可定义
$$
E^n(X)\coloneqq\|X\to_\ast E_n\|_0
$$
或按 suspension shift 调整编号。

**边界 AM.15.** 要得到稳定同伦范畴，还需要谱之间的映射谱、稳定等价、同伦范畴或 $\infty$-范畴结构。本书当前只写预谱/Omega 谱接口，不声称完成稳定同伦论教材。

## AM.6 本附录的接口

1.  附录 Y 的 cup product 输入 Y.14 可由 AM.7-AM.11 解释其相干来源。
2.  第十二章的谱入口应引用 AM.12-AM.15，而非只给一句纲要。
3.  若后续要写稳定同伦论专章，第一步是把 AM.7 的 1-coherent 结构提升或替换为足以定义谱范畴的相干结构。
