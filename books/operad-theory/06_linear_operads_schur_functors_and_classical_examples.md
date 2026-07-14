# 第六章：线性 operad、Schur 函子与经典例子

把集合值运算逐项取自由 $R$-模，只完成了“允许线性组合”这一步；代入还要同时处理张量积、诱导模和对称群 coinvariants。以二元运算为例，$m\otimes(v_1\otimes v_2)$ 与重标号后的 $m\sigma\otimes(v_{\sigma(1)}\otimes v_{\sigma(2)})$ 必须代表同一形式值，否则 Schur functor 甚至不能良定义。本章固定交换环 $R$，从模值对称序列重建代入乘积，并由 Schur functor 得到自由线性代数的 monad。结合、交换、Lie 与 Poisson 四类 operad 将展示生成元、对称作用和关系在同一公式中如何协同。

## 6.1 $R$-模值对称序列

**约定 6.1.** 本章固定一个交换环 $R$。所有张量积 $\otimes$ 若不特别说明均为 $\otimes_R$。$\mathbf{Mod}_R$ 表示 $R$-模范畴。

**定义 6.2.** 一个 $R$-模值对称序列是函子
$$
M:\mathbf B_{\mathcal U}\to\mathbf{Mod}_R.
$$
它等价于带左 $R[\Sigma_n]$-模结构的模族 $M(n)=M([n])$，但本书继续以有限集口径作为定义。

**约定 6.2.1（Schur functor 中的右作用）.** 当 arity 公式中出现
$$
M(n)\otimes_{R[\Sigma_n]}V^{\otimes n}
$$
时，$M(n)$ 的函子性左作用按命题 A.9 和约定 A.10 转换为右作用：
$$
m\cdot\sigma=\sigma^{-1}m.
$$
$V^{\otimes n}$ 带左 $\Sigma_n$-作用，按张量因子重排。所有 $\otimes_{R[\Sigma_n]}$ 均按这个右-左作用约定理解。

**定义 6.3.** 设 $M,N$ 是 $R$-模值对称序列。定义代入乘积
$$
(M\circ N)(S)
=
\operatorname*{colim}_{(f:S\to T)\in\operatorname{Fib}(S)}
M(T)\otimes\bigotimes_{t\in T}N(f^{-1}(t)),
$$
其中 colimit 在 $\mathbf{Mod}_R$ 中取，并按目标双射同时作用于 $M(T)$、重排张量因子。函数 $f$ 不要求满射，所以空纤维贡献 $N(\varnothing)$。仅当 $N(\varnothing)=0$ 时，该公式才缩成按非空分块取直和的公式。双射 $S\to S'$ 的作用由定义 1.5 的纤维重标号线性化得到。

**定义 6.4.** 单位对称序列 $I_R$ 定义为
$$
I_R(S)=
\begin{cases}
R, & |S|=1,\\
0, & |S|\ne1.
\end{cases}
$$

**命题 6.5.** $R$-模值对称序列范畴连同 $\circ$ 和 $I_R$ 构成幺半范畴。

**证明.** 与命题 1.8 相同，只是集合的 coproduct、积和群作用商替换为 $R$-模中的直和、张量积和 coinvariants。两个加括号方向都展开为可复合映射 $S\to U\to T$ 上的同一张量装饰，并使用张量积的结合约束
$$
(A\otimes B)\otimes C\cong A\otimes(B\otimes C).
$$
右单位强制 $S\to T$ 为双射，左单位强制 $T$ 为单点；两者均使用 $R\otimes_R M\cong M$，并在 $S=\varnothing$ 时仍成立。五边形和三角形化为同一多层映射数据上的恒等式，再由 $\mathbf{Mod}_R$ 的对称幺半相干性完成。$\square$

**定义 6.6.** 一个 $R$-线性 operad 是幺半范畴
$$
(\operatorname{SymSeq}(\mathbf{Mod}_R),\circ,I_R)
$$
中的幺半对象。

## 6.2 Arity 公式与 coinvariants

**命题 6.7.** 对 $R$-模值对称序列 $M,N$，有自然同构
$$
(M\circ N)(n)
\cong
\bigoplus_{k\ge0}
M(k)\otimes_{R[\Sigma_k]}
\left(
\bigoplus_{\substack{n_1+\cdots+n_k=n\\n_i\ge0}}
\operatorname{Ind}^{\Sigma_n}_{\Sigma_{n_1}\times\cdots\times\Sigma_{n_k}}
\bigl(N(n_1)\otimes\cdots\otimes N(n_k)\bigr)
\right).
$$
令 $H=\Sigma_{n_1}\times\cdots\times\Sigma_{n_k}$，并把张量积看作右 $R[H]$-模；这里
$$
\operatorname{Ind}^{\Sigma_n}_{H}(W)
:=W\otimes_{R[H]}R[\Sigma_n]
$$
是右 $R[\Sigma_n]$-模。括号内的 $\Sigma_k$ 左作用同时置换大小序列、$H$ 的块嵌入和 $N(n_i)$ 因子；coinvariants $-\otimes_{R[\Sigma_k]}-$ 把它与 $M(k)$ 的右作用相平衡。允许 $n_i=0$ 是 arity $0$ 代入所必需的。

**证明.** 在定义 6.3 中把目标有限集选成 $[k]$。函数 $f:[n]\to[k]$ 由有序纤维 $(f^{-1}(1),\ldots,f^{-1}(k))$ 给出，纤维允许为空。令 $n_i=|f^{-1}(i)|$，并选择 $[n_i]\cong f^{-1}(i)$；改变这些选择由 $\Sigma_{n_i}$ 作用，给出显示的诱导模。改变目标坐标 $[k]\cong T$ 由 $\Sigma_k$ 作用，并与 $M(k)$ 的右作用取 balanced tensor product。反向由诱导模中的有序纤维坐标恢复 $f$，故得到自然同构。$\square$

**警告 6.8.** 在线性 operad 中，不能把集合值公式中的 quotient set 直接替换为集合商。正确对象通常是 coinvariants 或 coend。若底环 $R$ 的特征整除某个对称群阶数，coinvariants 与 invariants 一般不同。

## 6.3 Schur functor 与线性代数

**定义 6.9.** 设 $M$ 是 $R$-模值对称序列。其 Schur functor 定义为
$$
S_M:\mathbf{Mod}_R\to\mathbf{Mod}_R,
\qquad
S_M(V)=
\bigoplus_{n\ge0}M(n)\otimes_{R[\Sigma_n]}V^{\otimes n},
$$
其中 $M(n)$ 使用约定 6.2.1 的右作用，$\Sigma_n$ 对 $V^{\otimes n}$ 按张量因子置换左作用。

**命题 6.10.** 存在自然同构
$$
S_{M\circ N}\cong S_M\circ S_N.
$$

**证明.** 展开左侧：
$$
S_{M\circ N}(V)
=
\bigoplus_n (M\circ N)(n)\otimes_{R[\Sigma_n]}V^{\otimes n}.
$$
代入命题 6.7 的公式，并使用诱导-限制伴随的张量形式
$$
\operatorname{Ind}^{\Sigma_n}_{H}(W)\otimes_{R[\Sigma_n]}V^{\otimes n}
\cong
W\otimes_{R[H]}\operatorname{Res}^{\Sigma_n}_{H}V^{\otimes n}.
$$
当 $H=\Sigma_{n_1}\times\cdots\times\Sigma_{n_k}$ 时，限制后的 $V^{\otimes n}$ 识别为
$$
V^{\otimes n_1}\otimes\cdots\otimes V^{\otimes n_k}.
$$
整理直和和 coinvariants 后得到
$$
\bigoplus_k
M(k)\otimes_{R[\Sigma_k]}
\left(
\bigoplus_{n_1}N(n_1)\otimes_{R[\Sigma_{n_1}]}V^{\otimes n_1}
\right)\otimes\cdots\otimes
\left(
\bigoplus_{n_k}N(n_k)\otimes_{R[\Sigma_{n_k}]}V^{\otimes n_k}
\right),
$$
这正是 $S_M(S_N(V))$。所有同构由张量积、直和和诱导模的泛性质给出，因此关于 $V$ 自然。$\square$

**推论 6.11.** 若 $\mathcal P$ 是 $R$-线性 operad，则 $S_{\mathcal P}$ 是 $\mathbf{Mod}_R$ 上的 monad。

**证明.** operad 乘法 $\mathcal P\circ\mathcal P\to\mathcal P$ 经命题 6.10 给出
$$
S_{\mathcal P}S_{\mathcal P}\cong S_{\mathcal P\circ\mathcal P}\to S_{\mathcal P}.
$$
operad 单位 $I_R\to\mathcal P$ 给出 $\operatorname{id}_{\mathbf{Mod}_R}\cong S_{I_R}\to S_{\mathcal P}$。monad 公理由 operad 的结合律和单位律推出。$\square$

**定义 6.12.** 一个 $\mathcal P$-代数是 $R$-模 $V$ 连同 monad 代数结构
$$
S_{\mathcal P}(V)\to V.
$$
等价地，它是 $R$-线性 operad morphism
$$
\mathcal P\to\operatorname{End}_V,
\qquad
\operatorname{End}_V(S)=\operatorname{Hom}_R(V^{\otimes S},V).
$$

## 6.4 线性化与 Ass、Com

**定义 6.13.** 若 $\mathcal O$ 是集合值 operad，定义其自由 $R$-模线性化 $R[\mathcal O]$ 为
$$
R[\mathcal O](S)=R[\mathcal O(S)].
$$
重标号和代入由 $\mathcal O$ 的结构 $R$-线性延拓得到。

**命题 6.14.** $R[\mathcal O]$ 是 $R$-线性 operad。

**证明.** 自由 $R$-模函子把有限乘积上的函数诱导为张量积上的 $R$-线性映射。具体地，对任意有限集映射 $f:S\to T$，集合值代入
$$
\mathcal O(T)\times\prod_{t\in T}\mathcal O(f^{-1}(t))\to\mathcal O(S)
$$
线性延拓为
$$
R[\mathcal O(T)]
\otimes\bigotimes_{t\in T} R[\mathcal O(f^{-1}(t))]
\to R[\mathcal O(S)].
$$
空纤维对应 $\mathcal O(\varnothing)$ 因子。集合值 operad 对可复合有限集映射的单位和结合等式在线性延拓后仍成立。$\square$

**命题 6.15.** $R[\operatorname{Ass}]$-代数等价于含单位的结合 $R$-代数。

**证明.** $R[\operatorname{Ass}]$ 由二元乘法 $m$ 和零元单位 $e$ 生成，满足第四章例 4.15 的结合和单位关系。一个 $R[\operatorname{Ass}]$-代数是在 $R$-模 $A$ 上给出 $R$-双线性乘法
$$
A\otimes A\to A
$$
和元素 $R\to A$，并满足结合律和单位律。这正是含单位结合 $R$-代数。同态条件是 $R$-线性且保持乘法和单位。$\square$

**命题 6.16.** $R[\operatorname{Com}]$-代数等价于含单位交换 $R$-代数。

**证明.** 在命题 6.15 的结构上再加入关系 $(12)\cdot m=m$。在 endomorphism operad 中，$(12)$ 对二元运算的作用交换两个输入，因此该关系等价于
$$
xy=yx.
$$
其余关系给出结合律和单位律。$\square$

## 6.5 Lie operad

**定义 6.17.** 假设 $R$ 是交换环。Lie operad $\operatorname{Lie}_R$ 是由一个二元生成元
$$
[-,-]\in\operatorname{Lie}_R(2)
$$
生成的 $R$-线性 operad，满足以下关系：

1. 反对称关系
   $$
   (12)\cdot[-,-]=-[ -,- ],
   $$
   即交换两个输入使括号变号。
2. Jacobi 关系，在 arity $3$ 中
   $$
   [[x,y],z]+[[y,z],x]+[[z,x],y]=0.
   $$

这里第二条是生成元关系的记号写法；严格地说，它是在自由 $R$-线性 operad 的 arity $3$ 部分中，由三棵二顶点树及其叶标号给出的元素之和。

**命题 6.18.** $\operatorname{Lie}_R$-代数等价于 $R$ 上 Lie 代数。

**证明.** 一个 $\operatorname{Lie}_R$-代数给出 $R$-双线性映射
$$
[-,-]_V:V\otimes V\to V.
$$
反对称关系在 $\operatorname{End}_V(2)$ 中变成
$$
[y,x]_V=-[x,y]_V.
$$
Jacobi 关系变成
$$
[[x,y]_V,z]_V+[[y,z]_V,x]_V+[[z,x]_V,y]_V=0.
$$
这正是 Lie 代数公理。反过来，任意 Lie 代数的括号满足这两类关系，由生成元关系的泛性质给出唯一 operad 代数结构。同态条件正是保持括号的 $R$-线性映射。$\square$

**注 6.19.** 若 $2$ 在 $R$ 中不可逆，反对称 $[y,x]=-[x,y]$ 不推出 $[x,x]=0$。一些作者把 Lie 代数定义中的 alternating 条件 $[x,x]=0$ 作为公理。本书在一般底环语境中会显式说明采用哪一种约定；在含 $\mathbb Q$ 的底环上二者等价。相关最小反例见例 X.6 和说明 X.7。

## 6.6 Poisson operad

**定义 6.20.** Poisson operad $\operatorname{Pois}_R$ 是由交换乘法
$$
m\in\operatorname{Pois}_R(2)
$$
和 Lie 括号
$$
\ell\in\operatorname{Pois}_R(2)
$$
生成的 $R$-线性 operad，关系包括：

1. $m$ 满足交换结合代数关系并有单位；
2. $\ell$ 满足 Lie 反对称和 Jacobi 关系；
3. Leibniz 关系
   $$
   \ell(x,m(y,z))=m(\ell(x,y),z)+m(y,\ell(x,z)).
   $$

**命题 6.21.** $\operatorname{Pois}_R$-代数等价于含单位 Poisson $R$-代数。

**证明.** 由生成元，代数结构是在 $R$-模 $A$ 上给出交换含单位乘法和 Lie 括号。前两类关系分别给出交换代数和 Lie 代数公理。第三类关系在 endomorphism operad 中正是每个 $\ell(x,-)$ 对乘法 $m$ 是导子的条件。反向由任意 Poisson $R$-代数的乘法和括号通过生成元关系的泛性质给出 operad 代数结构。$\square$

## 6.7 经典线性例子尚未覆盖同伦信息

本章只给出经典线性 operad 的入口。Gerstenhaber、BV、$E_n$、Koszul 对偶和 bar-cobar 构造需要链复形、悬挂、符号规则、二次对偶和模型范畴语言。第八章至第十三章会先建立这些工具，再把本章的 $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$ 和 $\operatorname{Pois}$ 纳入同伦代数框架。

## 6.8 Coinvariants 连接运算与线性输入

模值代入与集合值代入的差别集中在一个不能省略的商：对称群同时作用于运算和张量输入，Schur functor 必须取相对张量积或 coinvariants 才能消除重标号。这个机制把线性 operad 变成 $\mathbf{Mod}_R$ 上的 monad，并在 $R[\operatorname{Ass}]$、$R[\operatorname{Com}]$、$\operatorname{Lie}_R$ 与 $\operatorname{Pois}_R$ 中分别恢复结合、交换、Lie 与 Poisson 代数。它也预告了后面的同伦风险：coinvariants 对弱等价的行为依赖底环和等变 cofibrancy，不能只凭逐 arity 计算判断。

## 练习

**练习 6.1.** 写出 $(M\circ N)(0)$ 和 $(M\circ N)(1)$ 的 arity 公式，并指出 $\Sigma_0$ 和 $\Sigma_1$ 的作用。

**练习 6.2.** 证明 $S_{I_R}(V)\cong V$。

**练习 6.3.** 对 $R[\operatorname{Com}]$，直接从 Schur functor 公式推出自由交换 $R$-代数是对称代数
$$
\operatorname{Sym}_R(V)=\bigoplus_{n\ge0}(V^{\otimes n})_{\Sigma_n}.
$$

**练习 6.4.** 在 $\operatorname{End}_V(3)$ 中写出 Jacobi 关系对应的三个树形复合。

**练习 6.5.** 设 $R$ 的特征为 $2$。说明反对称关系 $[y,x]=-[x,y]$ 退化成什么，并解释为什么需要额外小心。
