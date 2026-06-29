# 附录 BM：局部系数、扭曲上同调与 Postnikov 系数系统

本附录补齐 Postnikov tower、Mayer-Vietoris 和谱序列共同需要的局部系数语言。常系数上同调不足以表达非单连通空间的 obstruction class；在 HoTT 中局部系数可自然表示为类型族或 Abelian group-valued local system。

## BM.1 Local system

**定义 BM.1（Abelian group local system）。** 对类型 $X$，一个阿贝尔群局部系数系统是族
$$
L:X\to\mathsf{AbGroup}
$$
其中 transport 沿路径
$$
p:x=y
$$
给出群同构
$$
\mathsf{transport}^L(p):L(x)\cong L(y).
$$

**命题 BM.2（transport 给出 $\infty$-群胚作用，书内证明核）。** 对路径 $p:x=y$、$q:y=z$，
$$
\mathsf{transport}^L(p\cdot q)
=
\mathsf{transport}^L(q)\circ\mathsf{transport}^L(p)
$$
作为群同构相等。

**证明.** 对 $q$ 和 $p$ 依次路径归纳，归约到 $\mathsf{refl}$ 情形。此时 transport 与复合均为恒等同构。$\square$

**定义 BM.3（constant local system）。** 给定阿贝尔群 $A$，常系数系统为
$$
\underline A(x)\coloneqq A.
$$
其 transport 为恒等同构。

## BM.2 Local systems over delooping

**定义 BM.4（$G$-module）。** 对群 $G$，$G$-module 是阿贝尔群 $M$ 加群同态
$$
G\to\mathsf{Aut}_{\mathsf{Ab}}(M).
$$

**命题 BM.5（$BG$ 上局部系统等价于 $G$-module，证明架构）。** 若 $BG$ 是 $G$ 的 delooping，则 $BG$ 上的阿贝尔群局部系统等价于 $G$-module。

**证明架构.** 从 $L:BG\to\mathsf{AbGroup}$ 取基点 fiber $M=L(\ast)$。每个 loop
$$
\Omega BG\simeq G
$$
经 transport 给出 $M$ 的自同构；BM.2 给出乘法相容。反向从 $G$-module 构造 associated family over $BG$，使用 $BG$ 的消去原则或 principal $G$-bundle 分类。两方向互逆依赖 delooping 的 encode-decode。

## BM.3 Twisted Eilenberg-Mac Lane fibration

**定义 BM.6（twisted EM fibration）。** 给定局部系统 $L:X\to\mathsf{AbGroup}$ 和 $n\ge1$，定义 over $X$ 的 fibration
$$
K(L,n)\to X
$$
其 fiber over $x:X$ 为
$$
K(L(x),n).
$$
沿路径的 transport 由 $L$ 的群同构诱导 EM 型之间的等价。

**定义 BM.7（局部系数上同调）。** 定义
$$
H^n(X;L)\coloneqq
\left\|\prod_{x:X}K(L(x),n)\right\|_0
$$
更准确地说，是 twisted EM fibration 的 section type 的 $0$-截断。

**边界 BM.8.** 若 $K(L,n)$ 的 total fibration 只作为外部输入给出，则 BM.7 是接口定义。完整书内定义需要 EM 型对群同构的 functoriality 和 universe-level fibration glueing。

## BM.4 Pullback

**定义 BM.9（pullback local system）。** 对 $f:Y\to X$ 和 $L:X\to\mathsf{AbGroup}$，
$$
f^\ast L(y)\coloneqq L(f(y)).
$$

**命题 BM.10（上同调反变性，证明核）。** 预合成给出映射
$$
f^\ast:H^n(X;L)\to H^n(Y;f^\ast L).
$$

**证明.** section $s:\prod_{x:X}K(L(x),n)$ 送到
$$
y\mapsto s(f(y)).
$$
再取集合截断。群结构保持由 EM 型点态群结构和函数外延性给出。$\square$

## BM.5 Local coefficient long exact sequence

**定义 BM.11（short exact sequence of local systems）。** $0\to L'\to L\to L''\to0$ 是逐点短正合，并且所有映射与 transport 群同构相容。

**定理 BM.12（局部系数长正合列，证明架构）。** 短正合局部系统诱导长正合列
$$
\cdots\to H^n(X;L')\to H^n(X;L)\to H^n(X;L'')
\xrightarrow{\delta}H^{n+1}(X;L')\to\cdots .
$$

**证明架构.** 逐点短正合给出 EM fibration 的 fiber sequence
$$
K(L',n)\to K(L,n)\to K(L'',n).
$$
对 section type 应用 dependent mapping object 保持 fiber sequence，再取 $\pi_0$ 和 loop-space 识别。需要 EM 型的 delooping exactness。

## BM.6 Twisted Mayer-Vietoris

**定理 BM.13（Mayer-Vietoris with local coefficients，证明架构）。** 对 pushout cover
$$
X=U\cup_A V
$$
和局部系统 $L:X\to\mathsf{AbGroup}$，存在长正合列
$$
\cdots\to H^n(X;L)\to
H^n(U;L|_U)\oplus H^n(V;L|_V)
\to H^n(A;L|_A)
\xrightarrow{\delta}H^{n+1}(X;L)\to\cdots .
$$

**证明架构.** 与附录 BK 的常系数证明相同，但 mapping-out 对象换成 twisted EM fibration 的 section space。pushout 的 section 是两个限制 section 加上在 $A$ 上的相容路径。

## BM.7 Postnikov coefficients

**定义 BM.14（homotopy local system）。** 对 connected pointed type $X$，第 $n$ 同伦群族
$$
x\mapsto \pi_n(X,x)
$$
是 $X$ 上的局部系统。路径 transport 由改变基点同构给出。

**命题 BM.15（$\pi_1$ 作用）。** 对 $n\ge2$，$\pi_1(X,x_0)$ 作用在 $\pi_n(X,x_0)$ 上。

**证明.** loop $p:x_0=x_0$ 通过基点 transport 给出
$$
\Omega^n(X,x_0)\to\Omega^n(X,x_0).
$$
该映射保持 loop 复合，下降到集合截断得群自同构。BM.2 给出作用律。$\square$

**定义 BM.16（twisted $k$-invariant）。** Postnikov tower 的第 $n$ 个 obstruction class 位于
$$
H^{n+2}(\|X\|_n;\pi_{n+1}(X))
$$
其中 $\pi_{n+1}(X)$ 按 BM.14 视为 $\|X\|_n$ 上的局部系统。

## BM.8 本附录关闭的缺口

本附录把局部系数系统、$G$-module、twisted EM fibration、局部系数上同调、twisted Mayer-Vietoris 和 Postnikov $k$-invariant 的系数系统接入全书。剩余义务是 EM fibration 的完整 HIT/谱构造、局部系数长正合列的逐行 proof term，以及与具体谱序列的计算接口。
