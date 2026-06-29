# 附录 BH：集合层代数层级、商结构与局部化

本附录补齐 HoTT 中普通代数的教材接口。第八章说明集合商，附录 I/J/AG 说明结构等同性原则；本附录把这些工具组织成 semigroup、monoid、group、ring、module、ideal、quotient 和 localization 的层级。所有载体默认是集合。

## BH.1 代数结构的通用形式

**定义 BH.1（代数结构）。** 设 $\mathsf{Set}_{\mathcal U}$ 为集合宇宙。一个代数结构可写为
$$
\mathsf{Str}:\mathsf{Set}_{\mathcal U}\to\mathcal U
$$
其中 $\mathsf{Str}(A)$ 包含有限 arity 运算和命题性公理。

**原则 BH.2（结构等同性）。** 若 $\mathsf{Str}$ 的运算 transport 计算由附录 AG 控制，且公理分量为命题，则
$$
(A,s)=(B,t)
$$
等价于载体等价 $e:A\simeq B$ 加上运算保持条件。

**证明状态.** 一元代数签名见附录 J；有限 arity 运算 transport 和命题性公理代数 SIP 见附录 AG。

## BH.2 基础代数对象

**定义 BH.3（semigroup）。** Semigroup 是集合 $A$、二元运算
$$
\mu:A\to A\to A
$$
和结合律
$$
\mu(\mu(x,y),z)=\mu(x,\mu(y,z)).
$$

**定义 BH.4（monoid）。** Monoid 是 semigroup 加单位元 $e:A$，满足左右单位律。

**定义 BH.5（group）。** Group 是 monoid 加逆元函数 $(-)^{-1}:A\to A$，满足左右逆律。

**定义 BH.6（commutative ring）。** Commutative ring 是集合 $R$，带加法交换群、乘法交换幺半群、分配律和 $0$ 吸收律。

**定义 BH.7（module）。** 若 $R$ 是交换环，$R$-module 是阿贝尔群 $M$ 加标量乘法
$$
R\to M\to M
$$
满足单位、结合和分配律。

**命题 BH.8（结构类型的集合性，证明核）。** 若载体 $A$ 是集合，运算类型由有限函数类型构成，公理为命题，则 $\mathsf{Str}(A)$ 的同一结构路径由运算路径决定；若运算所在函数类型是集合，则结构类型是集合。

**证明.** 函数外延性把运算路径化为逐点路径；目标 $A$ 是集合，所以逐点路径类型为命题，函数类型为集合。公理分量为命题，由 $\Sigma$ 类型路径和命题分量删除得到结论。$\square$

## BH.3 同态与子对象

**定义 BH.9（homomorphism）。** 代数对象 $(A,s)$ 到 $(B,t)$ 的同态是函数 $f:A\to B$，满足所有运算保持条件。例如群同态满足
$$
f(xy)=f(x)f(y),\qquad f(e)=e.
$$
在群情形，单位与逆元保持可由乘法保持和群律推出，也可作为冗余命题性公理记录。

**定义 BH.10（subalgebra）。** 子代数可由谓词 $P:A\to\mathsf{Prop}$ 给出，要求 $P$ 对所有运算封闭。载体为
$$
\sum_{a:A}P(a).
$$

**命题 BH.11（子代数载体是集合）。** 若 $A$ 是集合且 $P(a)$ 是命题，则 $\sum_{a:A}P(a)$ 是集合。

**证明.** 由子类型集合性，见附录 F 的子类型外延性和集合性推论。$\square$

## BH.4 正规子群与商群

**定义 BH.12（正规子群）。** 群 $G$ 的正规子群由谓词 $N:G\to\mathsf{Prop}$ 给出，满足：

1.  $N(e)$；
2.  $N(x)\to N(y)\to N(xy)$；
3.  $N(x)\to N(x^{-1})$；
4.  $\prod_{g,x:G}N(x)\to N(gxg^{-1})$。

**定义 BH.13（商关系）。** 定义
$$
x\sim_N y\coloneqq N(x^{-1}y).
$$

**命题 BH.14（商关系是等价关系，证明核）。** $\sim_N$ 是等价关系。

**证明.** 自反性用 $x^{-1}x=e$ 和 $N(e)$。对称性若 $N(x^{-1}y)$，则
$$
y^{-1}x=(x^{-1}y)^{-1}
$$
故由逆封闭得 $N(y^{-1}x)$。传递性若 $N(x^{-1}y)$ 与 $N(y^{-1}z)$，则
$$
x^{-1}z=(x^{-1}y)(y^{-1}z)
$$
由乘法封闭得到。$\square$

**定义 BH.15（商群）。** 商群 $G/N$ 的载体为集合商 $G/{\sim_N}$。乘法由
$$
[x]\cdot[y]\coloneqq[xy]
$$
定义。

**证明义务.** 需证明乘法对 $\sim_N$ well-defined。正规性用于右侧代表元变化；群律由商递归和 $G$ 的群律下降。

## BH.5 理想与商环

**定义 BH.16（ideal）。** 交换环 $R$ 的 ideal 是谓词 $I:R\to\mathsf{Prop}$，满足加法子群条件和吸收律
$$
I(x)\to I(r x).
$$

**定义 BH.17（商环）。** 定义关系
$$
x\sim_I y\coloneqq I(x-y).
$$
商 $R/I$ 的加法和乘法分别由
$$
[x]+[y]=[x+y],\qquad [x]\cdot[y]=[xy]
$$
给出。

**命题 BH.18（商环 well-defined，证明说明）。** 加法 well-defined 由 ideal 的加法子群性质给出；乘法 well-defined 由
$$
xy-x'y'=x(y-y')+(x-x')y'
$$
和 ideal 吸收律给出。

## BH.6 局部化

**定义 BH.19（multiplicative subset）。** 交换环 $R$ 的乘法闭子集 $S:R\to\mathsf{Prop}$ 满足
$$
S(1),\qquad S(s)\to S(t)\to S(st).
$$

**定义 BH.20（fraction relation）。** 在 pairs $(r,s)$ with $S(s)$ 上定义
$$
(r,s)\sim(r',s')
$$
若仅仅存在 $u:S$ 使
$$
u(rs'-r's)=0.
$$
具体公式依赖是否允许 zero divisors；可在整环情形简化为 $rs'=r's$。

**定义 BH.21（localization）。** $S^{-1}R$ 是上述关系的集合商，记元素为 $r/s$。加法与乘法由
$$
\frac r s+\frac {r'}{s'}=\frac{rs'+r's}{ss'},\qquad
\frac r s\cdot\frac {r'}{s'}=\frac{rr'}{ss'}
$$
给出。

**定理 BH.22（局部化泛性质，证明架构）。** 对任意交换环 $A$ 和环同态 $f:R\to A$，若每个 $s:S$ 的 $f(s)$ 可逆，则存在唯一环同态
$$
\bar f:S^{-1}R\to A
$$
使 $\bar f(r/1)=f(r)$。

**证明架构.** 定义 $\bar f(r/s)=f(r)f(s)^{-1}$。well-defined 使用 fraction relation。唯一性由每个 $r/s=(r/1)(s/1)^{-1}$ 和环同态保持逆元推出。

## BH.7 本附录关闭的缺口

本附录把 HoTT 中集合层代数从“结构等同性原则”推进到普通代数学接口：代数层级、同态、子代数、商群、商环、理想和局部化。剩余义务是把每个商结构的 well-defined 证明、泛性质和层级依赖全部展开。
