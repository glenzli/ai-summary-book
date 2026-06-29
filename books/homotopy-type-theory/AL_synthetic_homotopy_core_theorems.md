# 附录 AL：Blakers-Massey、Freudenthal 与 Hopf Fibration

本附录补入合成同伦论的核心定理形态。它不把全部高阶相干压进正文，但每个结果都给出精确类型论陈述、使用的输入和证明骨架，避免把第十二章停留在“研究方向”层面。

## AL.1 连通性约定

**定义 AL.1（类型的 $n$-连通性）.** 类型 $A$ 称为 $n$-连通，若
$$
\mathsf{isContr}(\|A\|_n).
$$
映射 $f:A\to B$ 称为 $n$-连通，若每个 fiber 是 $n$-连通。

**定义 AL.2（方块的 gap map）.** 给定交换方块
$$
\begin{array}{ccc}
A&\xrightarrow{f}&B\\
\downarrow g&&\downarrow h\\
C&\xrightarrow{k}&D,
\end{array}
$$
其 gap map 为
$$
A\to B\times_D C,\qquad
a\mapsto(f(a),g(a),\mathsf{sq}_a)
$$
其中 pullback
$$
B\times_D C\coloneqq
\sum_{b:B}\sum_{c:C}h(b)=k(c).
$$

**定义 AL.3（方块 $n$-cartesian）.** 该方块称为 $n$-cartesian，若 gap map 是 $n$-连通。

**定义 AL.4（方块 $n$-cocartesian）.** 方块称为 $n$-cocartesian，若从 pushout $B\sqcup_A C$ 到 $D$ 的诱导映射是 $n$-连通。

## AL.2 Blakers-Massey 定理

**定理 AL.5（Blakers-Massey，HoTT 形式）.** 若方块是 pushout 方块，且
$$
f:A\to B
$$
为 $m$-连通，
$$
g:A\to C
$$
为 $n$-连通，则该方块是 $(m+n)$-cartesian。

更展开地，canonical gap map
$$
A\to B\times_{B\sqcup_A C}C
$$
是 $(m+n)$-连通。

**证明状态.** 这是合成同伦论核心定理。HoTT 证明使用 flattening lemma、fiberwise join、connectedness of joins 和 pushout 的依赖消去。附录 AU 给出这些依赖的证明核和 Blakers-Massey 证明架构；Wei 2024 的 synthetic homotopy theory 文献给出 Hopf、Blakers-Massey、Freudenthal 的统一路线。本书仍把完整逐项证明作为外部义务。

**证明核 AL.6（证明分解）.** 证明 AL.5 可分为以下可检查子义务：

1.  对 pushout $P\coloneqq B\sqcup_A C$，把 gap map fiber 在 $(b,c,p)$ 处的 fiber 化为某个 join 型；
2.  证明若 $X$ 为 $m$-连通、$Y$ 为 $n$-连通，则 join $X\ast Y$ 为 $(m+n+2)$-连通，具体编号依赖本书连通性 convention；
3.  用 flattening lemma 把 pushout 上的依赖族拉回到原方块数据；
4.  用截断归纳关闭 connectedness 证明；
5.  检查编号平移，确保 AL.5 的 $(m+n)$ 与文献使用的 $m+n+2$ convention 对齐。

**边界.** 本书采用 AL.5 的定理陈述，不把子义务 1-4 全部重写为数十页路径代数。完整展开应固定 pushout/join 证明口径并逐项给出相干计算。

## AL.3 Freudenthal 悬挂定理

**定义 AL.7（悬挂单位映射）.** 对 pointed 类型 $(X,x_0)$，定义悬挂单位
$$
\eta_X:X\to\Omega\Sigma X
$$
为
$$
\eta_X(x)\coloneqq \mathsf{merid}(x)\cdot\mathsf{merid}(x_0)^{-1}.
$$

**定理 AL.8（Freudenthal，HoTT 形式）.** 若 $X$ 是 $n$-连通 pointed 类型，且 $n\ge0$，则
$$
\eta_X:X\to\Omega\Sigma X
$$
是 $(2n+1)$-连通。

**证明（由 Blakers-Massey 的证明核）.** 考虑悬挂作为 pushout
$$
\mathbf 1\leftarrow X\to\mathbf 1.
$$
两条映射 $X\to\mathbf 1$ 的 fiber 等价于 $X$，故均为 $n$-连通。对该 pushout 方块应用 AL.5，得到相应 gap map 的连通性。把 gap map 展开并识别为悬挂单位 $\eta_X$ 的连通性陈述；编号由两次 loop/suspension 的 convention 调整给出 $(2n+1)$。$\square$

**推论 AL.9（稳定范围内同伦群同构）.** 在 AL.8 条件下，悬挂诱导
$$
\pi_k(X)\to\pi_{k+1}(\Sigma X)
$$
在 $k\le 2n$ 范围内为同构，并在 $k=2n+1$ 为满射。

**证明状态.** 由 $r$-连通映射对低阶同伦群的影响推出。证明需要一般命题：若 $f$ 为 $r$-连通，则 $\pi_k(f)$ 在 $k<r$ 为同构、$k=r$ 为满射。该命题本书未完全展开，作为同伦群连通性标准引理登记。

## AL.4 Hopf fibration

**输入 AL.10（Hopf fibration 构造）.** 存在 fibration
$$
\mathbb S^1\to\mathbb S^3\to\mathbb S^2
$$
其总空间、底空间和 fiber 按 HoTT 的球面 HIT 定义，并满足长 fiber sequence 的标准性质。

**更具体的 HoTT 路线.** 可把 Hopf fibration 构造为与 $\mathbb S^1$ 在 $\mathbb S^3$ 上的作用相关的总空间，或通过 join/suspension 识别：
$$
\mathbb S^3\simeq \mathbb S^1\ast\mathbb S^1,\qquad
\mathbb S^2\simeq \Sigma\mathbb S^1.
$$

**定理 AL.11（$\pi_3(\mathbb S^2)$ 的生成元入口）.** Hopf fibration 给出
$$
\pi_3(\mathbb S^2)\to\pi_2(\mathbb S^1)
$$
相关的长正合序列边界信息，并导向经典同构
$$
\pi_3(\mathbb S^2)\cong\mathbb Z.
$$

**证明状态.** 同伦群长正合列的证明核见附录 AP。完整 $\pi_3(\mathbb S^2)$ 证明仍需要 Hopf fibration 的 fiber sequence 相干和球面低阶同伦群计算。本书只把 Hopf fibration 本身作为高级定理输入。

## AL.5 球面同伦群的计算边界

**事实 AL.12（低阶球面计算）.** 合成同伦论可在 HoTT 中计算若干低阶球面同伦群；典型方向包括 Brunerie number 和合成上同调计算。

**规则 AL.13（数值计算引用规则）.** 对任意具体数值，例如某个球面同伦群的生成元关系或 Brunerie-type number，本书必须记录：

1.  具体定理陈述；
2.  所用库、commit 或论文版本；
3.  该数值是否已经由对象语言证明给出；
4.  是否依赖不可执行公理、postulate 或实验模块。

## AL.6 本附录的接口

1.  第十二章可以引用 AL.5 和 AL.8 作为合成同伦论核心定理，而非只写“谱序列入口”。
2.  附录 AJ 的连通/截断分解为 AL.1 的连通性口径提供基础。
3.  附录 AM 的 smash product 和上同调运算提供 Hopf、cup product、稳定方向的另一条接口。
