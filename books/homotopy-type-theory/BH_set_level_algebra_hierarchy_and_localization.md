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

**命题 BH.18（商环 well-defined）。** 加法与乘法在等价类上良定义。

**证明.** 若 $x\sim_I x'$、$y\sim_I y'$，则
$I(x-x')$ 与 $I(y-y')$。由理想的加法子群性质，
$I((x+y)-(x'+y'))$，故加法与代表元选择无关。又
$$
xy-x'y'=x(y-y')+(x-x')y'
$$
的两项分别由理想吸收律属于 $I$，其和仍属于 $I$，故乘法也与代表元
选择无关。零元、单位元、加法逆元以及结合律、交换律和分配律均由
$R$ 中对应等式传到商集；因此这些运算定义交换环 $R/I$。$\square$

## BH.6 局部化

**定义 BH.19（multiplicative subset）。** 交换环 $R$ 的乘法闭子集 $S:R\to\mathsf{Prop}$ 满足
$$
S(1),\qquad S(s)\to S(t)\to S(st).
$$

**定义 BH.20（fraction relation）。** 先定义原始分式类型
$$
\mathsf{Frac}_0(R,S)
\coloneqq
\sum_{r:R}\sum_{s:R}S(s).
$$
对 $x=(r,s,h_s)$ 与 $x'=(r',s',h_{s'})$，定义
$$
x\sim x'
$$
为命题
$$
\left\|
\sum_{u:R}S(u)\times\bigl(u(rs'-r's)=0\bigr)
\right\|.
$$
在整环且 $0\notin S$ 的特殊情形，该关系可简化为 $rs'=r's$；有零因子时不能省略见证 $u$。

**命题 BH.20.1（分式关系是等价关系）.** 关系 $\sim$ 自反、对称且传递。

**证明.** 自反性取 $u=1$；对称性保留同一 $u$ 并取等式的加法逆。传递性目标是命题，所以可依次消去两份命题截断。若
$$
u(rs'-r's)=0,
\qquad
v(r's''-r''s')=0,
$$
则 $w\coloneqq uvs'$ 仍属于 $S$，而交换环计算给出
$$
\begin{aligned}
w(rs''-r''s)
&=vs''\,u(rs'-r's)+us\,v(r's''-r''s')\\
&=0.
\end{aligned}
$$
故 $x\sim x''$。$\square$

**定义 BH.21（localization）。** $S^{-1}R$ 是上述关系的集合商，记元素为 $r/s$。加法与乘法由
$$
\frac r s+\frac {r'}{s'}=\frac{rs'+r's}{ss'},\qquad
\frac r s\cdot\frac {r'}{s'}=\frac{rr'}{ss'}
$$
给出。

**定义 BH.21.1（单位元性质）。** 对交换环 $A$ 与 $a:A$，置
$$
\mathsf{isUnit}_A(a)
\coloneqq
\sum_{b:A}(ab=1)\times(ba=1).
$$
由于 $A$ 是集合且逆元唯一，$\mathsf{isUnit}_A(a)$ 是命题；其项仍包含可用于定义函数的唯一逆元 $b$。

**定理 BH.22（局部化泛性质）。** 对任意交换环 $A$ 和环同态 $f:R\to A$，若给定
$$
\prod_{s:R}S(s)\to\mathsf{isUnit}_A(f(s)),
$$
则存在唯一环同态
$$
\bar f:S^{-1}R\to A
$$
使 $\bar f(r/1)=f(r)$。

**证明（书内证明）.** 在原始分式上定义
$$
h_0(r,s,h_s)\coloneqq f(r)f(s)^{-1},
$$
其中逆元由假设在 $(s,h_s)$ 处给出。要证明它尊重 $\sim$，目标是集合 $A$ 中的等式，因而可对关系中的命题截断消去。对见证
$u$ 应用 $f$ 后，$f(u)$、$f(s)$ 与 $f(s')$ 都可逆；从
$f(u)(f(r)f(s')-f(r')f(s))=0$ 依次消去这些单位，得到
$$
f(r)f(s)^{-1}=f(r')f(s')^{-1}.
$$
集合商递归遂给出 $\bar f$。加法、乘法、零和单位的保持可在原始分式代表上由交换环律直接验证，目标均为命题，故再由商归纳推广。

若 $g:S^{-1}R\to A$ 也延拓 $f$，则对每个原始分式有
$$
g(r/s)=g(r/1)g(s/1)^{-1}=f(r)f(s)^{-1}=\bar f(r/s).
$$
商归纳与函数外延性给出 $g=\bar f$；环同态律的证明分量是命题，因此同态记录也相等。$\square$

## BH.7 商结构的使用边界

商群、商环和局部化都必须先证明运算尊重所取关系，再由集合商消去定义运算。凡本附录只给出公式而未展开 well-defined、泛性质或 universe 依赖的地方，都应读作条件化接口；这些公式本身不提供对代表元作任意选择的权限。
