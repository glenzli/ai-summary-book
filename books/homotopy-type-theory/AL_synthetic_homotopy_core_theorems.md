# 附录 AL：Blakers-Massey、Freudenthal 与 Hopf Fibration

本附录固定合成同伦论中两个大型连通性定理的采用版本，并区分定理结论与来源证明机制。Blakers--Massey 和 Freudenthal 均作为精确外部输入；join、flattening 与 path-code 的说明帮助读者理解假设如何进入，但不代替来源中的完整高阶相干证明。

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

**外部输入定理 AL.5（Blakers--Massey，HoTT 形式）.** 设 $m,n:\mathbb N$。若方块是 pushout 方块，且
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

**精确来源与边界.** 这是 The Univalent Foundations Program,
*Homotopy Type Theory: Univalent Foundations of Mathematics*, Theorem 8.10.2
按定义 AL.1-AL.3 的转写。来源证明使用 flattening lemma、fiberwise join、join 连通性和 pushout 依赖消去；本书采用定理结论，不声称附录 AU、AY 已内部重构该证明。

**来源机制 AL.6（证明分解）.** 来源证明可按下列数学步骤理解：

1.  对 pushout $P\coloneqq B\sqcup_A C$ 和 $(b,c,p):B\times_PC$ 构造依赖 code 与比较映射，使所需连通性归约到相关 fibers 的 join；这一步不是把一般 pushout 路径类型直接等同于一次 join；
2.  证明若 $X$ 为 $m$-连通、$Y$ 为 $n$-连通，则 join $X\ast Y$ 为 $(m+n+2)$-连通，具体编号依赖本书连通性 convention；
3.  用 flattening lemma 把 pushout 上的依赖族拉回到原方块数据；
4.  用截断归纳关闭 connectedness 证明；
5.  检查编号平移，确保 AL.5 的 $(m+n)$ 与文献使用的 $m+n+2$ convention 对齐。

这些步骤只解释 Theorem 8.10.2 的结构。若要把 AL.5 升级为书内定理，必须固定 pushout/join 的具体语法并给出每一步相干计算；目前任何后续调用都以 AL.5 的外部输入身份为准。

## AL.3 Freudenthal 悬挂定理

**定义 AL.7（悬挂单位映射）.** 对 pointed 类型 $(X,x_0)$，定义悬挂单位
$$
\eta_X:X\to\Omega\Sigma X
$$
为
$$
\eta_X(x)\coloneqq \mathsf{merid}(x)\cdot\mathsf{merid}(x_0)^{-1}.
$$

**外部输入定理 AL.8（Freudenthal，HoTT 形式）.** 若 $X$ 是 $n$-连通 pointed 类型，且 $n\ge0$，则
$$
\eta_X:X\to\Omega\Sigma X
$$
是 $2n$-连通。

**来源与关系.** 本书采用 HoTT Book, Theorem 8.6.4。该结论也可由 Blakers--Massey 路线解释：考虑悬挂作为 pushout
$$
\mathbf 1\leftarrow X\to\mathbf 1.
$$
两条映射 $X\to\mathbf 1$ 的 fiber 等价于 $X$，故均为 $n$-连通。对该 pushout 方块应用 AL.5，并把 gap map 与悬挂单位比较，得到 $2n$-连通性。完整的比较与相干由来源定理承担。

**条件化推论 AL.9（稳定范围内同伦群同构）.** 另假设标准的连通映射作用定理：若 $f$ 为 $r$-连通，则 $\pi_k(f)$ 在 $k\le r$ 为同构，并在 $k=r+1$ 为满射。在此输入下，AL.8 的悬挂诱导
$$
\pi_k(X)\to\pi_{k+1}(\Sigma X)
$$
在 $k\le 2n$ 范围内为同构，并在 $k=2n+1$ 为满射。

**推导.** 对 $\eta_X$ 应用所列标准输入，并代入 $r=2n$。悬挂诱导与 $\eta_X$ 在 loop 后的映射相同，故得到所述范围。$\square$

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
