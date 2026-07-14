# 第三十八章：Topological Hochschild homology、cyclotomic trace 与 $TC$

Algebraic $K$-theory 对环和稳定范畴高度敏感，往往难以直接计算。Topological Hochschild homology $THH$ 把 Hochschild trace 提升为带圆作用的谱，cyclotomic Frobenius 又把不同固定点信息组织成 $TC$；cyclotomic trace

$$
K\to TC
$$

把 $K$-理论问题转化为圆作用、Tate construction 与 Frobenius 结构的谱论问题。本章从小稳定幂等完备 $\infty$-范畴的 trace 出发，解释 $S^1$-action 的来源，区分 $TC^-$、$TP$ 与 $TC$，并说明 localizing invariance 和纤维序列在计算中的位置。

本章使用稳定 $\infty$-范畴、Morita theory、谱、circle actions、localizing invariants 与非交换 motives。Cyclotomic spectra 的模型和 Nikolaus--Scholze 公式作为外部输入；固定点、homotopy fixed points 与 Tate construction 不会用同一符号混写，并明确区分 localizing invariant $THH$ 与一般不保持滤过余极限的 $TC$。

## 38.1 $THH$ 作为谱值 trace

**定义 38.1.** 对小稳定幂等完备 $\infty$-范畴 $C$，topological Hochschild homology $THH(C)$ 是 $C$ 的谱值 Hochschild trace。若 $C=\operatorname{Perf}(R)$，也写作 $THH(R)$。

**例子 38.2.** 对 $E_1$-ring spectrum $R$，$THH(R)$ 可由 cyclic bar construction 计算：

$$
[n]\mapsto R^{\otimes(n+1)}.
$$

其几何实现给出 $THH(R)$。

**命题 38.3.** $THH$ 对 Morita equivalence 不变。

**证明.** $THH(C)$ 是稳定 Morita $(\infty,2)$-范畴中恒等 bimodule 的 trace。Morita equivalence 识别对象的双模 endomorphism theory 和恒等 bimodule，因此识别 trace。对 $R$ 与 $S$，若 $\operatorname{Perf}(R)\simeq\operatorname{Perf}(S)$，则 $THH(R)\simeq THH(S)$。$\square$

**外部输入定理 38.4.** $THH$ 是 localizing invariant：exact sequence

$$
A\to B\to C
$$

给出谱纤维序列

$$
THH(A)\to THH(B)\to THH(C).
$$

## 38.2 圆作用与 cyclotomic structure

**定义 38.5.** 设 $\mathbb T=S^1$。带圆作用的谱是函子

$$
B\mathbb T\to\mathbf{Sp}.
$$

由 cyclic bar construction 的循环对称性，$THH(C)$ 自然带 $\mathbb T$-作用。

**定义 38.6.** Cyclotomic spectrum 是带 $\mathbb T$-作用的谱 $X$，并对每个素数 $p$ 配有 Frobenius 型结构映射

$$
\varphi_p:X\to X^{tC_p}
$$

满足相干条件。这里 $X^{tC_p}$ 是 Tate construction。

**外部输入定理 38.7.** 对每个小稳定幂等完备 $\infty$-范畴 $C$，$THH(C)$ 自然提升为 cyclotomic spectrum；该提升对 exact functors 自然。

**命题 38.8.** 忘却 cyclotomic 结构后，$THH(C)$ 仍保留圆作用。

**证明.** Cyclotomic spectrum 的数据包含带 $\mathbb T$-作用的底层谱以及额外 Frobenius 映射。忘却额外 Frobenius 数据即得到带圆作用谱。$\square$

## 38.3 $TC$ 的定义形式

**定义 38.9.** 对带圆作用的谱 $X$，记

$$
TC^-(X)=X^{h\mathbb T},\qquad TP(X)=X^{t\mathbb T}.
$$

若 $X$ 是 bounded-below cyclotomic spectrum，则 Nikolaus--Scholze 公式在 $p$-完成后给出

$$
TC(X)^\wedge_p\simeq\operatorname{fib}\left(
TC^-(X)^\wedge_p\xrightarrow{\operatorname{can}-\varphi_p}
TP(X)^\wedge_p
\right)
$$

其中 $\operatorname{can}$ 是 homotopy fixed points 到 Tate construction 的典范映射，$\varphi_p$ 由 cyclotomic Frobenius 及 Tate-orbit 识别诱导。无 bounded-below 假设时应使用 cyclotomic spectra 范畴中的原始 equalizer 定义，不能直接套用这一简式。

**定义 38.10.** 对小稳定 $\infty$-范畴 $C$，定义

$$
TC(C)=TC(THH(C)).
$$

**命题 38.11.** 若 $THH(C)\simeq THH(D)$ 作为 cyclotomic spectra，则 $TC(C)\simeq TC(D)$。

**证明.** $TC$ 是 cyclotomic spectra 上的函子。等价对象在任意 $\infty$-范畴值函子下送为等价对象，故结论成立。$\square$

## 38.4 Cyclotomic trace

**外部输入定理 38.12（cyclotomic trace）.** 存在自然变换

$$
\operatorname{tr}_{cycl}:K(C)\to TC(C)
$$

从 algebraic $K$-theory 到 topological cyclic homology，称为 cyclotomic trace。它对 exact functors 自然，并与 Morita equivalence 相容。

**定义 38.13.** Dennis trace 是自然变换

$$
K(C)\to THH(C),
$$

cyclotomic trace 则利用 $THH$ 的 cyclotomic 结构构造。二者相容，但一般不存在自然映射 $THH(C)\to TC(C)$ 使 Dennis trace 按字面因子化为 cyclotomic trace；“refinement”指额外结构，而不是该方向上的函子分解。

**命题 38.14.** 若 $C\simeq D$ Morita equivalent，则 cyclotomic trace 的方块

$$
\begin{array}{c}
K(C)\to TC(C)\\
\downarrow\quad\downarrow\\
K(D)\to TC(D)
\end{array}
$$

交换且竖箭头为等价。

**证明.** $K$ 与 $TC$ 都是 Morita invariant，cyclotomic trace 对 exact/Morita functors 自然。因此 Morita equivalence 诱导两侧等价，自然性给出交换方块。$\square$

## 38.5 Dundas-Goodwillie-McCarthy 定理

**外部输入定理 38.15（Dundas-Goodwillie-McCarthy）.** 对合适的 connective ring spectra 映射 $A\to B$，若其在 $\pi_0$ 上为带 nilpotent kernel 的满射，则相对 $K$-理论与相对 $TC$ 在 $p$-完成后等价：

$$
\operatorname{fib}(K(A)\to K(B))^\wedge_p
\simeq
\operatorname{fib}(TC(A)\to TC(B))^\wedge_p.
$$

**命题 38.16.** 在定理 38.15 的假设下，若能计算 $TC(A)\to TC(B)$ 的纤维，则可计算相对 $K$-理论的 $p$-完成。

**证明.** 定理直接给出两个纤维谱的 $p$-完成等价。谱等价保留所有同伦群，因此右侧的计算给出左侧相对 $K$-群的 $p$-完成。$\square$

## 38.6 Trace methods 的范畴论意义

**命题 38.17.** Trace methods 先以 localizing invariant $THH$ 提取 Morita 不变量，再以 cyclotomic fixed-point constructions 形成 $TC$；第二步通常不保持滤过余极限。

**证明.** $THH$ 由定理 38.4 是 localizing invariant。$TC$ 由 cyclotomic $THH$ 取 homotopy fixed points、Tate construction 与 Frobenius 的 equalizer/fiber 得到；这些极限型构造一般不保持滤过余极限，所以 $TC$ 一般不是 Blumberg--Gepner--Tabuada 意义下的 localizing invariant。Cyclotomic trace $K\to TC$ 仍然是自然的 Morita 不变量变换，但这不把 $TC$ 变成 localizing invariant。$\square$

**注 38.18.** 非交换 motives 直接控制的是满足相应可加性或局部化公理的不变量，例如非连通 $K$-理论与 $THH$。$TC$ 仍是稳定范畴的 Morita 不变量，但因上述余极限问题，不能不加限定地称为 localizing motive 上的函子。

## 38.7 相对 trace 与形式后果

**定义 38.19.** 对 exact functor $F:C\to D$，定义相对 $K$-理论和相对 $TC$ 为纤维

$$
K(C,D)=\operatorname{fib}(K(C)\to K(D)),
\qquad
TC(C,D)=\operatorname{fib}(TC(C)\to TC(D)).
$$

**命题 38.20.** Cyclotomic trace 自然诱导相对 trace

$$
K(C,D)\to TC(C,D).
$$

**证明.** 自然变换 $K\to TC$ 给出交换方块

$$
\begin{array}{c}
K(C)\to K(D)\\
\downarrow\quad\downarrow\\
TC(C)\to TC(D).
\end{array}
$$

稳定 $\infty$-范畴中，交换方块诱导纤维之间的自然映射。因此得到

$$
\operatorname{fib}(K(C)\to K(D))\to
\operatorname{fib}(TC(C)\to TC(D)).
$$

这就是相对 cyclotomic trace。$\square$

**命题 38.21.** 若 $A\to B\to C$ 是 small stable idempotent-complete $\infty$-categories 的 exact sequence 且 $THH(A)\simeq0$，则

$$
THH(B)\simeq THH(C).
$$

若 $THH(C)\simeq0$，则 $THH(A)\simeq THH(B)$。

**证明.** 由外部输入定理 38.4，有纤维序列

$$
THH(A)\to THH(B)\to THH(C).
$$

在稳定范畴中，纤维为零的态射是等价，所以 $THH(A)\simeq0$ 蕴含 $THH(B)\to THH(C)$ 为等价。若 $THH(C)\simeq0$，则 $THH(B)\to0$ 的纤维为 $THH(B)$，而该纤维等价于 $THH(A)$，故 $THH(A)\simeq THH(B)$。$\square$

**命题 38.22.** 若 $C\to D$ 与 $C'\to D'$ 是 Morita 等价的 exact functors，即有交换到同伦的方块并且 $C\simeq C'$、$D\simeq D'$ 均为 Morita equivalences，则相对 $K$-理论和相对 $TC$ 分别等价。

**证明.** $K$ 和 $TC$ 都对 Morita equivalence 不变。于是得到纤维序列之间的竖向等价方块

$$
K(C)\to K(D),\qquad K(C')\to K(D')
$$

以及对应的 $TC$ 方块。稳定范畴中，两个可比较态射的源和靶均为等价时，其纤维也等价。因此相对 $K$ 与相对 $TC$ 均被识别。$\square$

## 38.8 从 Hochschild trace 到 cyclotomic trace

$THH$ 是谱值 Hochschild trace，天然带圆作用并提升为 cyclotomic spectrum。$TC$ 从 cyclotomic structure 中提取算术信息。Cyclotomic trace $K\to TC$ 把难计算的代数 $K$-理论连接到更可计算的固定点与 Tate 构造。Dundas-Goodwillie-McCarthy 定理说明在 nilpotent 相对情形中，这种近似是 $p$-完成等价。

## 练习

**练习 38.1.** 定义 $THH(C)$ 的 trace 口径。

**练习 38.2.** 对 $E_1$-ring $R$ 写出 cyclic bar construction。

**练习 38.3.** 证明 $THH$ 对 Morita equivalence 不变。

**练习 38.4.** 说明 $THH$ 为什么有圆作用。

**练习 38.5.** 定义 cyclotomic spectrum。

**练习 38.6.** 陈述 $THH$ 的 cyclotomic refinement。

**练习 38.7.** 写出 Nikolaus-Scholze 形式的 $TC$ 公式。

**练习 38.8.** 证明 cyclotomic 等价诱导 $TC$ 等价。

**练习 38.9.** 陈述 cyclotomic trace。

**练习 38.10.** 说明 Dennis trace 与 cyclotomic trace 的关系。

**练习 38.11.** 陈述 Dundas-Goodwillie-McCarthy 定理。

**练习 38.12.** 解释 trace methods 的范畴论意义。

**练习 38.13.** 定义相对 $K$-理论和相对 $TC$。

**练习 38.14.** 证明 cyclotomic trace 诱导相对 trace $K(C,D)\to TC(C,D)$。

**练习 38.15.** 证明 exact sequence 中 $THH(A)=0$ 时 $THH(B)\simeq THH(C)$。
