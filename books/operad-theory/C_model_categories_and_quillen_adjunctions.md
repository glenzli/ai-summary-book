# 附录 C：模型范畴与 Quillen adjunction 复习

本附录整理第十四章和第十九章使用的模型范畴语言。它不是模型范畴教材，只固定本书所需的定义、符号和若干基础命题。

## C.1 Lifting properties

**定义 C.1.** 设 $\mathcal C$ 是范畴，$i:A\to B$ 与 $p:X\to Y$ 是态射。称 $i$ has left lifting property with respect to $p$，记作
$$
i\pitchfork p,
$$
若任意交换方块
$$
\begin{array}{ccc}
A & \longrightarrow & X\\
{\scriptstyle i}\downarrow & & \downarrow{\scriptstyle p}\\
B & \longrightarrow & Y
\end{array}
$$
存在 lift $B\to X$ 使两个三角形交换。

对一类态射 $S$，定义
$$
{}^\pitchfork S=\{i:i\pitchfork s\text{ for all }s\in S\},
\qquad
S^\pitchfork=\{p:s\pitchfork p\text{ for all }s\in S\}.
$$

**定义 C.2.** Weak factorization system 是一对态射类 $(\mathcal L,\mathcal R)$，满足：

1. 每个态射可分解为 $r\circ l$，其中 $l\in\mathcal L$，$r\in\mathcal R$；
2. $\mathcal L={}^\pitchfork\mathcal R$；
3. $\mathcal R=\mathcal L^\pitchfork$。

**命题 C.3.** 在 weak factorization system 中，$\mathcal L$ 与 $\mathcal R$ 都对 retract 封闭。

**证明.** 设 $f$ 是 $\mathcal L$ 中态射的 retract。若 $r\in\mathcal R$，则任意 $f$ 对 $r$ 的 lifting problem 可作为对应 $\mathcal L$-态射对 $r$ 的 lifting problem 的 retract。后者有 lift；取 retract 得到前者的 lift。因此 $f\in{}^\pitchfork\mathcal R=\mathcal L$。$\mathcal R$ 的证明对偶。$\square$

## C.2 Model categories

**定义 C.4.** 模型范畴是完备且余完备的范畴 $\mathcal M$，连同三类态射
$$
\mathsf W,\qquad \mathsf{Cof},\qquad \mathsf{Fib},
$$
称为 weak equivalences、cofibrations、fibrations，满足：

1. $\mathsf W$ 满足 two-out-of-three；
2. 三类态射对 retract 封闭；
3. $(\mathsf{Cof},\mathsf{Fib}\cap\mathsf W)$ 是 weak factorization system；
4. $(\mathsf{Cof}\cap\mathsf W,\mathsf{Fib})$ 是 weak factorization system。

**定义 C.5.** 对象 $X$ 称为 cofibrant，若初对象到 $X$ 的态射是 cofibration。对象 $X$ 称为 fibrant，若 $X$ 到终对象的态射是 fibration。

**定义 C.6.** Cofibrant replacement 是 weak equivalence
$$
QX\to X
$$
其中 $QX$ cofibrant。Fibrant replacement 是 weak equivalence
$$
X\to RX
$$
其中 $RX$ fibrant。

**命题 C.7.** 每个对象都有 cofibrant replacement 和 fibrant replacement。

**证明.** 将初对象到 $X$ 的态射分解为 cofibration 后接 trivial fibration：
$$
\varnothing\to QX\to X.
$$
则 $QX$ cofibrant，且 $QX\to X$ 是 weak equivalence。将 $X$ 到终对象的态射分解为 trivial cofibration 后接 fibration：
$$
X\to RX\to *.
$$
则 $RX$ fibrant，且 $X\to RX$ 是 weak equivalence。$\square$

## C.3 Homotopy category

**定义 C.8.** 模型范畴 $\mathcal M$ 的 homotopy category 记作
$$
\operatorname{Ho}(\mathcal M)=\mathcal M[\mathsf W^{-1}],
$$
即把 weak equivalences 形式反演得到的范畴。

**外部输入定理 C.9.** Homotopy category 可由 cofibrant-fibrant objects and homotopy classes of maps 计算：
$$
\operatorname{Ho}(\mathcal M)(X,Y)
\cong
\pi_0\operatorname{Map}_{\mathcal M}(QX,RY)
$$
在 simplicial model category 中成立；一般模型范畴可用 cylinder/path objects 给出同伦类版本。

**说明 C.10.** $\operatorname{Ho}(\mathcal M)$ 只保留 mapping spaces 的 $\pi_0$。第十九章的 $\mathcal M_\infty$ 保留全部高阶同伦信息。

## C.4 Quillen adjunctions

**定义 C.11.** 设 $\mathcal M,\mathcal N$ 是模型范畴。伴随
$$
F:\mathcal M\rightleftarrows\mathcal N:G
$$
称为 Quillen adjunction，若 $F$ 保 cofibrations 和 trivial cofibrations；等价地，$G$ 保 fibrations 和 trivial fibrations。

**命题 C.12.** 上述两个条件等价。

**证明.** 由伴随，$F(i)\pitchfork p$ 当且仅当 $i\pitchfork G(p)$。若 $F$ 保 cofibrations 和 trivial cofibrations，则对 $\mathcal N$ 中 fibration $p$，要证明 $G(p)$ 是 fibration，即所有 trivial cofibrations $i$ 都满足 $i\pitchfork G(p)$。这等价于 $F(i)\pitchfork p$；而 $F(i)$ 是 trivial cofibration，故有提升。Trivial fibration 情形同理。反向证明对偶。$\square$

**定义 C.13.** Quillen adjunction 的 total left derived functor 和 total right derived functor 记作
$$
\mathbf L F:\operatorname{Ho}(\mathcal M)\rightleftarrows\operatorname{Ho}(\mathcal N):\mathbf R G,
$$
其中
$$
\mathbf L F(X)=F(QX),\qquad
\mathbf R G(Y)=G(RY).
$$

**命题 C.14.** 若 $F\dashv G$ 是 Quillen adjunction，则 $\mathbf L F\dashv\mathbf R G$ 是 homotopy categories 上的伴随。

**证明.** 对 $X\in\mathcal M$、$Y\in\mathcal N$，取 $QX$ cofibrant、$RY$ fibrant。Quillen adjunction 给出普通 Hom 伴随
$$
\mathcal N(FQX,RY)\cong\mathcal M(QX,GRY).
$$
Passing to homotopy classes，并使用 $QX$ cofibrant、$RY$ fibrant，得到
$$
\operatorname{Ho}(\mathcal N)(\mathbf L F X,Y)
\cong
\operatorname{Ho}(\mathcal M)(X,\mathbf R G Y).
$$
自然性来自原伴随的自然性。$\square$

## C.5 Quillen equivalence

**定义 C.15.** Quillen adjunction
$$
F:\mathcal M\rightleftarrows\mathcal N:G
$$
称为 Quillen equivalence，若对每个 cofibrant $X\in\mathcal M$ 和 fibrant $Y\in\mathcal N$，态射
$$
F(X)\to Y
$$
是 weak equivalence 当且仅当其 adjoint
$$
X\to G(Y)
$$
是 weak equivalence。

**命题 C.16.** 若 $F\dashv G$ 是 Quillen equivalence，则导出伴随
$$
\mathbf L F:\operatorname{Ho}(\mathcal M)\rightleftarrows\operatorname{Ho}(\mathcal N):\mathbf R G
$$
是范畴等价。

**证明边界.** 证明要检查导出 unit 和 counit 是同构。对 cofibrant $X$，derived unit 由 $X\to GRF(X)$ 表示；对 fibrant $Y$，derived counit 由 $FQG(Y)\to Y$ 表示。Quillen equivalence 的判别条件说明这些态射是 weak equivalences，故在 homotopy category 中成为同构。完整证明还需处理 replacement 的自然性，本书作为标准模型范畴事实使用。$\square$

## C.6 Monoidal model categories

**定义 C.17.** Symmetric monoidal model category 是 symmetric monoidal closed category
$$
(\mathcal M,\otimes,\mathbb 1)
$$
连同模型结构，满足：

1. pushout-product axiom；
2. unit axiom。

Pushout-product axiom 说：若 $i:A\to B$ 与 $j:C\to D$ 是 cofibrations，则
$$
i\square j:(B\otimes C)\coprod_{A\otimes C}(A\otimes D)\to B\otimes D
$$
是 cofibration；若 $i$ 或 $j$ trivial，则 $i\square j$ trivial。

**定义 C.18.** Monoid axiom 说：由所有
$$
f\otimes X
$$
生成的 weakly saturated class 中态射都是 weak equivalences，其中 $f$ 遍历 trivial cofibrations，$X$ 遍历 $\mathcal M$ 的对象。

**说明 C.19.** Monoid axiom 和其 operadic 加强版本用于把模型结构转移到 monoids、operads 和 operad algebras。第十四章中所有 admissibility 定理都依赖这类条件。

## C.7 本附录小结

模型范畴提供三层信息：weak equivalences 定义同伦理论，cofibrations/fibrations 提供计算工具，Quillen adjunctions 描述模型之间的结构保持函子。Quillen equivalence 在 homotopy category 层面给出等价，在第十九章中进一步提升为 underlying infinity-categories 的等价。
