# 附录 BK：Cofiber、Puppe 序列与 Mayer-Vietoris

本附录补入 cofiber sequence 和 Mayer-Vietoris 这条上同调计算主线。附录 AP 处理 fiber sequence 的同伦群长正合列；本附录处理 cofiber、悬挂、Puppe 序列和由 pushout 方块得到的上同调长正合列。

## BK.1 Cofiber

**定义 BK.1（cofiber）。** 对 pointed map 或普通映射 $f:A\to B$，定义 cofiber 为 pushout
$$
\mathsf{cofib}(f)\coloneqq
\mathsf{pushout}\bigl(A\xrightarrow{f}B,\ A\to\mathbf 1\bigr).
$$
记规范映射为
$$
i:B\to\mathsf{cofib}(f)
$$
并记 collapsed point 为 $\ast:\mathsf{cofib}(f)$。

**命题 BK.2（cofiber 递归泛性质，书内证明核）。** 给定类型 $X$，定义函数
$$
\mathsf{cofib}(f)\to X
$$
等价于给出函数 $g:B\to X$、点 $x_0:X$，以及同伦
$$
\prod_{a:A}(g(f(a))=x_0).
$$

**证明.** 这是 pushout 非依赖递归原则的直接实例。左腿为 $f:A\to B$，右腿为常值映射 $A\to\mathbf 1$；在目标 $X$ 中的 cocone 数据正是 $g$、$x_0$ 和连接路径族。$\square$

## BK.2 Suspension as cofiber

**命题 BK.3（悬挂是到单位映射的 cofiber）。** 对类型 $A$，
$$
\Sigma A\simeq\mathsf{cofib}(A\to\mathbf 1).
$$

**证明.** $\mathsf{cofib}(A\to\mathbf 1)$ 是 pushout
$$
\mathbf 1\leftarrow A\to\mathbf 1,
$$
这正是悬挂的 HIT 构造：两个端点 north/south 和对每个 $a:A$ 的 meridian。由 pushout 与 suspension 的递归原则互相构造映射，并用 HIT 依赖消去证明互逆。$\square$

## BK.3 Cofiber sequence

**定义 BK.4（cofiber sequence）。** 对 pointed map $f:A\to_\ast B$，有序列
$$
A\xrightarrow{f}B\to\mathsf{cofib}(f)\to\Sigma A\to\Sigma B\to\Sigma\mathsf{cofib}(f)\to\cdots
$$
其中 $\mathsf{cofib}(f)\to\Sigma A$ 由 pushout 的边界映射给出。

**命题 BK.5（边界映射构造，证明架构）。** 存在自然 pointed map
$$
\partial:\mathsf{cofib}(f)\to\Sigma A
$$
使得 $B\to\mathsf{cofib}(f)\to\Sigma A$ pointed-homotopic to constant。

**证明架构.** 用 BK.2 的递归泛性质定义 $\partial$。在 $B$ 上取 north；collapsed point 取 south；对每个 $a:A$，连接路径由 meridian $a$ 给出。相容性由 cofiber 的 pushout 路径构造子给出。

## BK.4 Puppe sequence

**定理 BK.6（Puppe sequence，证明架构 / 外部输入）。** 对 pointed map $f:A\to B$，迭代 cofiber 和 suspension 得到 Puppe 序列
$$
A\to B\to C_f\to\Sigma A\to\Sigma B\to\Sigma C_f\to\Sigma^2A\to\cdots
$$
并且相邻两箭头的复合为零；在映射到任意 spectrum 或 EM 型后诱导长正合列。

**证明架构.** 每一步由 cofiber 的 pushout 泛性质和 suspension-cofiber 等价构造。正合性由 mapping out of cofiber 把 nullhomotopy 转为 fiber 条件，再由 EM 型或谱的 loop-suspension 结构把相邻 fiber 识别为 loop。

## BK.5 Reduced cohomology exactness

**定义 BK.7（cofiber 上同调 connecting map）。** 对阿贝尔群 $G$ 和 cofiber sequence
$$
A\to B\to C_f\to\Sigma A,
$$
定义 connecting homomorphism
$$
\delta:\widetilde H^n(A;G)\to\widetilde H^{n+1}(C_f;G)
$$
为沿 suspension isomorphism 和 Puppe 边界映射的复合。

**定理 BK.8（cofiber 长正合列，证明架构）。** 有长正合列
$$
\cdots\to\widetilde H^n(C_f;G)\to\widetilde H^n(B;G)\to
\widetilde H^n(A;G)\xrightarrow{\delta}
\widetilde H^{n+1}(C_f;G)\to\cdots .
$$

**证明架构.** 将 cofiber sequence 映入 EM 型 $K(G,n)$。由 cofiber 泛性质，映射空间形成 fiber sequence；对 $\pi_0$ 和 loop space 应用附录 AP 的长正合列。悬挂同构见附录 Y。

## BK.6 Mayer-Vietoris

**定义 BK.9（pushout cover）。** 给定 pushout 方块
$$
\begin{array}{ccc}
A&\to&U\\
\downarrow&&\downarrow\\
V&\to&X
\end{array}
$$
称其为 cover datum。这里 $X$ 是 $U$ 和 $V$ 沿 $A$ glue 得到的 homotopy pushout。

**定理 BK.10（Mayer-Vietoris 长正合列，证明架构）。** 对阿贝尔群 $G$，有长正合列
$$
\cdots\to H^n(X;G)\to H^n(U;G)\oplus H^n(V;G)\to H^n(A;G)
\xrightarrow{\delta} H^{n+1}(X;G)\to\cdots .
$$

**证明架构.** 由 pushout 的 mapping-out 泛性质，映射空间
$$
X\to K(G,n)
$$
是
$$
(U\to K(G,n))\times_{(A\to K(G,n))}(V\to K(G,n))
$$
的 pullback。把该 pullback fiber sequence 的 long exact homotopy sequence 下降到 $\pi_0$，并用 $H^n$ 的群结构识别差分映射。

**边界 BK.11（局部系数）。** 若系数为 local system，则 Mayer-Vietoris 中 $G$ 必须沿 $X$ 的 fundamental groupoid 或 higher local system 变化。普通常系数版本不能直接替代局部系数版本。

## BK.7 Excision

**事实 BK.12（excision，外部输入 / 证明架构）。** 在 HoTT 中，excision 可由 Blakers-Massey、pushout 路径空间和 connectedness 条件表达。具体形式通常说某个 pushout 方块的 gap map 高连通。

**连接.** 附录 AL、AU、AY 给出 Blakers-Massey、flattening 和 pushout path code；本附录的 Mayer-Vietoris 是 cohomological shadow。

## BK.8 从 cofiber 到正合列的边界

Cofiber 与 suspension 可由 pushout 在书内形成；Puppe 序列的全体自然性、局部系数 Mayer--Vietoris 以及与谱序列的比较还需要额外外部输入。常系数公式不能替代局部系统，mapping-out 的 fiber 描述也不能在未验证连通性或 EM 型输入时自动产生长正合列。
