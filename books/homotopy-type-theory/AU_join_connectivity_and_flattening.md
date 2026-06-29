# 附录 AU：Join Connectivity、Flattening Lemma 与 Blakers-Massey 证明接口

附录 AL 把 Blakers-Massey 定理列为高级输入，并登记其依赖：fiberwise join、join connectivity 和 flattening lemma。本附录补出这些依赖的证明核，使 Blakers-Massey 的剩余边界更精确。

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

**定理 AU.5（join connectivity）.** 若 $A$ 是 $m$-连通，$B$ 是 $n$-连通，则
$$
A\ast B
$$
是 $(m+n+2)$-连通。

**证明状态 / 证明核.** 标准 HoTT 证明分为四步：

1.  用 join 的 pushout 递归把映射
    $$
    A\ast B\to X
    $$
    化为函数 $A\to X$、$B\to X$ 与相干同伦。
2.  对目标 $X$ 取 $(m+n+2)$-截断目标，使用 $A$ 和 $B$ 的连通性把函数空间降到常值数据。
3.  关键相干由 $A\times B$ 的连通性提供；若 $A$ 为 $m$-连通且 $B$ 为 $n$-连通，则 $A\times B$ 至少为 $\min(m,n)$-连通，但 join 证明实际使用的是路径空间和 suspension-like 增连通机制。
4.  对截断目标的映射空间证明可收缩，从而由截断泛性质得到 $\|A\ast B\|_{m+n+2}$ 可收缩。

完整 proof term 较长，通常借助 encode-decode 或 wedge connectivity 的引理库。本书把 AU.5 作为 Blakers-Massey 的标准外部证明核；使用时必须保持编号 convention。

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

## AU.5 Blakers-Massey 的证明架构

**定理 AU.10（Blakers-Massey 架构）.** 对 pushout 方块
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
的 fiber 可由两个相关 fiber 的 join 控制，从而由 AU.5 推出 $(m+n)$-连通性。

**证明核.** 固定 $(b,c,p):B\times_PC$。gap fiber 的点是 $a:A$ 连同路径 $a$ 映到 $b,c$ 且与 $p$ 相容。由 flattening lemma，把 pushout 路径 $p:\mathsf{inl}(b)=\mathsf{inr}(c)$ 的 fiber 表示为两侧输入 fiber 的 join。输入映射的连通性给这些 fiber 的 $m$、$n$ 连通性；AU.5 给 join 的连通性；编号经由 path fiber 与 gap fiber 的一次 loop/截断平移，得到 $(m+n)$-连通。$\square$

**剩余边界 AU.11.** AU.10 仍是证明架构而非完整逐项证明。完整展开需固定：

1.  pushout 路径空间的 encode-decode；
2.  flattening lemma 的 universe 层级；
3.  join connectivity 的编号 convention；
4.  gap fiber 与 path-code fiber 的具体等价方向。
