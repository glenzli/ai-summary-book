# 附录 AU：Join Connectivity、Flattening Lemma 与 Blakers-Massey 证明接口

附录 AL 把 Blakers--Massey 定理列为精确外部输入。Fiberwise join、join connectivity 和 flattening lemma 解释来源证明为何能把局部 fiber 信息转成连通度估计，但它们并不单独给出一般 pushout 路径空间的完整描述。本附录区分可在书内验证的 pushout 构造、另行采用的 join 连通性定理，以及最终仍由外部来源承担的 Blakers--Massey 结论。

## AU.1 Join

**定义 AU.1（join）.** 类型 $A,B$ 的 join 定义为 pushout
$$
A\leftarrow A\times B\to B,
$$
其中两条映射为投影。记作
$$
A\ast B.
$$

**规则 AU.2（join 构造子）.** $A\ast B$ 有点构造子
$$
\mathsf{inl}:A\to A\ast B,\qquad
\mathsf{inr}:B\to A\ast B
$$
和路径构造子
$$
\mathsf{glue}_{a,b}:\mathsf{inl}(a)=\mathsf{inr}(b).
$$

**命题 AU.3（join 的对称性）.** 有等价
$$
A\ast B\simeq B\ast A.
$$

**证明.** 由 pushout 的等价不变性附录 AI，交换 span 的左右端点。也可直接用 pushout 递归定义双向函数，并由依赖消去证明互逆。$\square$

## AU.2 连通性估计

**定义 AU.4（$n$-连通类型）.** 类型 $A$ 是 $n$-连通，若
$$
\mathsf{isContr}(\|A\|_n).
$$

**外部输入定理 AU.5（join connectivity）.** 设 $m,n:\mathbb N$。若 $A$ 是 $m$-连通，$B$ 是 $n$-连通，则
$$
A\ast B
$$
是 $(m+n+2)$-连通。

**来源与边界.** Egbert Rijke, *The join construction*, arXiv:1701.07538, Theorem 6.9；取共同余域为 $\mathbf 1$ 得到上述类型版。该结果依赖单值宇宙及文中列出的 pushout/truncation 输入。本书采用其连通度结论，不把一句“映射到截断目标可收缩”当作证明。

**低维检查 AU.6.** 若 $A$ 和 $B$ 均 merely inhabited，即 $(-1)$-连通，则 $A\ast B$ 是 $0$-连通。直观上，任意 $\mathsf{inl}(a)$ 与 $\mathsf{inr}(b)$ 由 glue 相连，而任意两点可经左右代表和 glue 连接；形式证明需对命题截断代表元消去，目标为集合截断中的连通性命题。

## AU.3 Fiberwise join

**定义 AU.7（fiberwise join）.** 给定同底映射
$$
f:A\to X,\qquad g:B\to X,
$$
其 fiberwise join
$$
A\ast_X B\to X
$$
定义为每个 $x:X$ 上 fiber 的 join：
$$
\mathsf{fib}_{A\ast_XB}(x)\simeq \mathsf{fib}_f(x)\ast\mathsf{fib}_g(x).
$$
可用总空间 pushout 构造：
$$
A\leftarrow A\times_XB\to B
$$
的 pushout，并投影到 $X$。

**命题 AU.8（fiberwise join 的 fiber 计算）.** 对每个 $x:X$，
$$
\mathsf{fib}_{A\ast_XB}(x)
\simeq
\mathsf{fib}_f(x)\ast\mathsf{fib}_g(x).
$$

**证明（证明核）.** 展开 $A\ast_XB$ 为 pushout。取 fiber 等价于把 pullback 沿 pushout 分配；这正是 flattening lemma 的一个实例。具体地，fiber over $x$ 的点来自 $a:A$ 加路径 $f(a)=x$，或 $b:B$ 加路径 $g(b)=x$，路径构造子来自 $(a,b,p:f(a)=g(b))$ 与到 $x$ 的相容路径。整理后得到两个 fiber 的 join。$\square$

## AU.4 Flattening lemma

**定理 AU.9（flattening lemma for pushout）.** 给定 pushout
$$
P\coloneqq B\sqcup_A C
$$
和依赖族
$$
E:P\to\mathcal U.
$$
设其在 $B,C,A$ 上的拉回分别为
$$
E_B:B\to\mathcal U,\qquad
E_C:C\to\mathcal U,\qquad
E_A:A\to\mathcal U
$$
并带有沿 glue 的相容等价。则总空间
$$
\sum_{p:P}E(p)
$$
等价于 total spaces 的 pushout
$$
\left(\sum_{b:B}E_B(b)\right)
\sqcup_{\sum_{a:A}E_A(a)}
\left(\sum_{c:C}E_C(c)\right).
$$

**证明（证明核）.** 从右到左用 pushout 递归：左 total 点 $(b,u)$ 送到 $(\mathsf{inl}(b),u)$，右 total 点送到 $(\mathsf{inr}(c),u)$，glue 相容由 $E$ 沿 pushout glue 的 transport 给出。从左到右对 $p:P$ 作 pushout 依赖消去：若 $p=\mathsf{inl}(b)$，送到左 total；若 $p=\mathsf{inr}(c)$，送到右 total；路径构造子情形由依赖消去的计算规则和相容等价处理。两个复合为恒等由 pushout 依赖消去和 $\Sigma$-路径等价证明。$\square$

## AU.5 Blakers--Massey 中的作用

**外部输入定理 AU.10（Blakers--Massey）.** 设 $m,n:\mathbb N$。对 pushout 方块
$$
\begin{array}{ccc}
A&\to&B\\
\downarrow&&\downarrow\\
C&\to&P
\end{array}
$$
若 $A\to B$ 为 $m$-连通，$A\to C$ 为 $n$-连通，则 gap map
$$
A\to B\times_PC
$$
是 $(m+n)$-连通。

**精确来源.** The Univalent Foundations Program,
*Homotopy Type Theory: Univalent Foundations of Mathematics*, Theorem 8.10.2；这与 AL.5、AY.5 是同一个采用版本。

**边界 AU.11.** 来源证明使用 flattening 与 join 连通性，但不能把固定
$(b,c,p):B\times_PC$ 的 gap fiber 直接宣布为两个普通 fibers 的一次 join。若要内部重构 AU.10，还必须给出：

1.  pushout 路径空间的 encode-decode；
2.  flattening lemma 的 universe 层级；
3.  join connectivity 的编号 convention；
4.  gap fiber 与所构造 path-code fiber 的具体比较及全部 transport 相容。

附录 AY.4 的悬挂二点类型例子说明，单步 span-fiber 一般不等于完整 pushout 路径空间。因此 AU.8-AU.9 可以解释来源机制，却不能代替 AU.10 的外部定理身份。
