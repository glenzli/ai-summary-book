# 附录 AT：Left Exact 模态、Cohesive HoTT 与使用边界

附录 AJ 给出反射子宇宙、模态、局部化和连通/截断分解。本附录补上 left exact 模态和 cohesive/modal HoTT 的规则级接口。它们在合成几何、形状理论和局部化中核心，但不可作为基础 HoTT 的默认规则。

## AT.1 Left exact modality

**定义 AT.1（finite-limit preserving）.** 模态 $L$ 称为 left exact，若它保持终对象和 pullback。即 canonical map
$$
L\mathbf 1\to\mathbf 1
$$
是等价，并且对任意 pullback square
$$
\begin{array}{ccc}
P&\to&A\\
\downarrow&&\downarrow\\
B&\to&C
\end{array}
$$
其 $L$-像仍为 pullback square。

**等价形式 AT.2（路径保持）.** 在单值基础中，left exactness 可用路径空间保持表达：对 $x,y:A$，canonical map
$$
L(x=y)\to(\eta_A(x)=\eta_A(y))
$$
是等价，并且 $L\mathbf 1\simeq\mathbf 1$。

**命题 AT.3（left exact 模态保持截断层级）.** 若 $L$ left exact，且 $A$ 是 $n$-型，则 $L A$ 是 $n$-型。

**证明（证明核）.** 对 $n$ 归纳。可收缩情形由 $L$ 保持终对象和收缩中心。归纳步中，需证明任意 $u,v:L A$ 的路径空间是 $(n-1)$-型。由反射泛性质和路径保持 AT.2，可把 $u,v$ 按 local elimination 降到 $\eta_A(x),\eta_A(y)$ 情形，再用
$$
(\eta_A(x)=\eta_A(y))\simeq L(x=y)
$$
和归纳假设。$\square$

**命题 AT.4（left exact 模态保持有限极限结构）.** 若一个结构由有限极限条件定义，例如 pullback、等化子、终对象，则 $L$ 将其送到相同结构。

**证明.** 由定义 AT.1。等化子可表示为 pullback 到 diagonal；终对象已包含在定义中。$\square$

## AT.2 Closed modality 与 open modality

**输入 AT.5（subuniverse by proposition）.** 给定命题 $P:\mathsf{Prop}$，可定义 closed modality 和 open modality，分别刻画“在 $P$ 上局部”与“在 $\neg P$ 上局部”的类型。其形式通常通过 reflective subuniverse 或 HIT localization 给出。

**使用边界.** 这些模态依赖命题、universe 和 resizing 口径。正文若使用 open/closed modality，必须说明是否假设 propositional resizing，或把相关 universe 提升显式写出。

## AT.3 Cohesive HoTT 的四个算子

**输入 AT.6（cohesive quadruple）.** Cohesive HoTT 通常含有一串伴随
$$
\Pi\dashv \mathsf{Disc}\dashv \Gamma\dashv \mathsf{Codisc}
$$
或相近记号，表达 shape、discrete、global sections、codiscrete 等结构。

**解释.**

1.  $\Pi$ 抽取形状或同伦型；
2.  $\mathsf{Disc}$ 把普通类型看作离散 cohesive 类型；
3.  $\Gamma$ 取全局点；
4.  $\mathsf{Codisc}$ 给出余离散对象。

**规则 AT.7（不可默认原则）.** Cohesive 算子不是普通 HoTT 的定理。它们是额外对象语言结构或模型结构。若在正文使用，必须声明新的类型形成规则、伴随单位/余单位、计算规则和 exactness 假设。

## AT.4 Modal induction

**定义 AT.8（modal induction principle）.** 对模态 $L$，modal induction 是如下原则：若族
$$
B:L A\to\mathcal U
$$
逐点 local，则从
$$
\prod_{a:A}B(\eta_A(a))
$$
可构造
$$
\prod_{u:L A}B(u).
$$

**命题 AT.9（modal induction 唯一性）.** 若 $B(u)$ 均为 local 且为集合或命题，modal induction 得到的延拓在相应层级上唯一。

**证明.** 两个延拓相等可由函数外延性逐点证明。固定 $u:L A$，因目标 local 且由 $\eta_A$ 的反射泛性质决定，两个值在 $\eta_A(a)$ 上相等即可推出在 $u$ 上相等。命题目标时唯一性由命题性直接给出。$\square$

## AT.5 与合成代数几何的接口

**事实 AT.10（synthetic algebraic geometry 边界）.** 合成代数几何使用特定的 modal/cohesive 或 Zariski-like 类型论结构。其“环对象”“仿射线”“Zariski open”等概念不是第八章集合层代数的直接实例。

**使用规则 AT.11.** 任何合成代数几何结论都必须另列：

1.  基础对象语言；
2.  所用模态或覆盖结构；
3.  环对象和局部环对象的定义；
4.  是否假设 choice、excluded middle、resizing；
5.  与 HoTT Book 普通集合层代数的翻译边界。

## AT.6 模态与模型的分界

附录 AJ 的反射模态不自动 left exact，也不自动组成 cohesive adjunction。Open/closed modality、cohesive 算子和 modal induction 只有在相应规则与模型已经给出时可用；具体几何结论还需单独的环对象、覆盖与微局部公理。
