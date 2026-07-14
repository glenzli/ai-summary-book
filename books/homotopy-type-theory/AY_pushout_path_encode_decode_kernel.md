# 附录 AY：Pushout path-code 的内部接口与外部边界

一般 pushout 的路径空间并不只记录一次穿过粘合边的选择；路径可以在左右两侧移动并反复穿过粘合，因此会出现任意长度的 zigzag 及其相干。若尚未构造这些数据，就不能把一个 span-fiber 宣称为完整的 path code。本附录先精确定义一个 path-code 包必须提供什么，再证明该包一旦给出便足以得到 encode--decode 等价；随后用悬挂二点类型说明“单步 code”为何不够。Blakers--Massey 所需的连通性结论最终作为精确外部输入使用。

固定 span
$$
B\xleftarrow{f}A\xrightarrow{g}C
$$
及其 pushout $P\coloneqq B\sqcup_A C$，构造子记为
$\mathsf{inl}$、$\mathsf{inr}$ 与
$$
\mathsf{glue}_a:\mathsf{inl}(f(a))=\mathsf{inr}(g(a)).
$$

## AY.1 抽象 path-code 包

**定义 AY.1（基于 $b_0$ 的 path-code 包）.** 固定 $b_0:B$。一个 path-code 包由以下数据组成：

1. 类型族 $\mathsf{code}:P\to\mathcal U$；
2. 基码 $c_0:\mathsf{code}(\mathsf{inl}(b_0))$；
3. 函数
   $$
   \mathsf{decode}_x:\mathsf{code}(x)\to
   (\mathsf{inl}(b_0)=x)
   $$
   对每个 $x:P$ 给出；
4. 两族路径
   $$
   \delta_{x,p}:
   \mathsf{decode}_x
   \bigl(\mathsf{transport}^{\mathsf{code}}(p,c_0)\bigr)=p
   $$
   其中 $p:\mathsf{inl}(b_0)=x$，以及
   $$
   \epsilon_{x,z}:
   \mathsf{transport}^{\mathsf{code}}
   (\mathsf{decode}_x(z),c_0)=z
   $$
   其中 $z:\mathsf{code}(x)$。

这一定义没有假设 code 由哪一种 HIT、zigzag 或 join 构造；具体构造必须另外给出上述四项，而不能用“取适当闭包”代替。

**定义 AY.2（encode）.** 对一个 path-code 包，置
$$
\mathsf{encode}_x(p)
\coloneqq
\mathsf{transport}^{\mathsf{code}}(p,c_0).
$$

**命题 AY.3（抽象 encode--decode）.** 每个 path-code 包都逐点给出等价
$$
(\mathsf{inl}(b_0)=x)\simeq\mathsf{code}(x).
$$

**证明（书内证明）.** 取 $\mathsf{encode}_x$ 为正向函数、
$\mathsf{decode}_x$ 为逆向函数。定义 AY.1 的 $\delta$ 正是
$\mathsf{decode}_x\circ\mathsf{encode}_x\sim\mathsf{id}$，而
$\epsilon$ 正是
$\mathsf{encode}_x\circ\mathsf{decode}_x\sim\mathsf{id}$。
因此 $\mathsf{decode}_x$ 是 $\mathsf{encode}_x$ 的准逆；由推论 G.7，
$\mathsf{encode}_x$ 是等价。$\square$

## AY.2 在 pushout 上形成 code 族

若要由 pushout 消去原则构造 $\mathsf{code}:P\to\mathcal U$，必须先给出族
$$
L:B\to\mathcal U,
\qquad
R:C\to\mathcal U
$$
以及对每个 $a:A$ 的等价
$$
e_a:L(f(a))\simeq R(g(a)).
$$
单值性把 $e_a$ 变成宇宙路径 $\mathsf{ua}(e_a)$，pushout 的依赖消去据此给出 $\mathsf{code}$，满足
$$
\mathsf{code}(\mathsf{inl}(b))\equiv L(b),
\qquad
\mathsf{code}(\mathsf{inr}(c))\equiv R(c),
$$
而沿 $\mathsf{glue}_a$ 的 transport 命题地等于 $e_a$ 的底层函数。于是困难不在于写出 `encode`，而在于同时构造 $L$、$R$、所有 $e_a$、`decode` 及定义 AY.1 的两族相干路径。

宇宙层级也属于数据的一部分：若 $A,B,C$ 不在同一宇宙，以上族和 pushout 消去必须先作相应 universe lift。本附录不假设 impredicative universe 来消除这种提升。

## AY.3 单次穿越不是完整 code

对 $b:B$、$c:C$，确有一个自然的“单次穿越”类型
$$
\mathsf{OneStep}(b,c)
\coloneqq
\sum_{a:A}(f(a)=b)\times(g(a)=c),
$$
以及函数
$$
\mathsf{step}_{b,c}:\mathsf{OneStep}(b,c)\to
(\mathsf{inl}(b)=\mathsf{inr}(c)),
$$
其在 $(a,p,q)$ 上的值为
$$
\mathsf{ap}_{\mathsf{inl}}(p)^{-1}
\cdot\mathsf{glue}_a
\cdot\mathsf{ap}_{\mathsf{inr}}(q).
$$
这个函数通常不是等价。

**例 AY.4（悬挂二点类型）.** 取 $B=C=\mathbf 1$、$A=\mathbf 2$，两张映射均为唯一函数。所得 pushout 是 $\Sigma\mathbf 2$，并由附录 AD 等价于 $\mathbb S^1$。此时
$$
\mathsf{OneStep}(\star,\star)\simeq\mathbf 2.
$$
选择一条 meridian 后，北极到南极的路径类型等价于北极的 loop space；附录 N 的计算把后者等价到 $\mathbb Z$。因此完整路径类型包含由任意多次往返产生的整数族，而单步类型只有两个生成元。这个例子排除了把
$\sum_a(f(a)=b)\times(g(a)=c)$ 直接定义成一般 pushout 路径空间的做法。

## AY.4 Blakers--Massey 的采用边界

**外部输入定理 AY.5（Blakers--Massey）.** 设 $m,n:\mathbb N$。若 $f:A\to B$ 为 $m$-连通、$g:A\to C$ 为 $n$-连通，则 canonical gap map
$$
A\longrightarrow B\times_{B\sqcup_A C}C
$$
为 $(m+n)$-连通，其中连通性采用附录 AL.1 的截断编号。

**精确来源.** The Univalent Foundations Program,
*Homotopy Type Theory: Univalent Foundations of Mathematics*,
Theorem 8.10.2。附录 AL.5 是该定理按本书符号的转写。

来源证明使用 flattening、join 的连通性和精心选择的依赖 code 数据。命题 AY.3 解释一旦完整 code 包存在，encode--decode 的最后一步为何形式上直接；它并没有构造来源证明所需的具体包。因而本书后续可以调用 AY.5 的连通性结论，却不能从本附录获得一般 pushout 路径空间与某个未定义 zigzag 或 join-code 的内部等价。
