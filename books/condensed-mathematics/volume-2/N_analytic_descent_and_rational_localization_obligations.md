# 附录 N：Analytic descent 与 rational localization 的证明义务

## N.0 目标

analytic ring 的定义把局部化、测度对象、张量和几何覆盖绑定在一起。本附录不试图重证 Scholze 的 analytic ring 定理，而是把使用 analytic rings 时必须满足的证明义务列成定理级模块，并证明这些义务推出正文使用的 descent 和 gluing 结论。

## N.1 Analytic ring 数据

设 $A$ 是凝聚交换环。一个 analytic structure 由一族测试对象和测度对象

$$
S\mapsto\mathcal M[S]
$$

给出，带有自然态射

$$
A[\underline S]\to\mathcal M[S].
$$

记 cone 为

$$
K_S^\mathcal M
=
\operatorname{cofib}(A[\underline S]\to\mathcal M[S]).
$$

**定义 N.1.** $C\in D(A)$ 称为 $\mathcal M$-analytic，如果

$$
R\operatorname{Hom}_A(K_S^\mathcal M,C)\simeq0
$$

对所有测试 $S$ 成立。

## N.2 证明义务清单

**输入义务 N.2（反射性）.** analytic 对象构成反射局部子范畴

$$
D(A,\mathcal M)\subset D(A),
$$

左伴随记为

$$
L_{(A,\mathcal M)}.
$$

**输入义务 N.3（张量理想性）.** localization 的核为张量理想：

$$
N\in\ker L_{(A,\mathcal M)}
\Rightarrow
N\otimes_A^LX\in\ker L_{(A,\mathcal M)}.
$$

**输入义务 N.4（rational localization）.** 对 Huber pair 或 affinoid datum 的 rational subset $U\subset X$，存在 analytic ring

$$
(A_U,\mathcal M_U)
$$

和 restriction functor

$$
D(A,\mathcal M)\to D(A_U,\mathcal M_U)
$$

与代数 rational localization 相容。

**输入义务 N.5（rational Čech descent）.** 若 $X=\bigcup_iU_i$ 是有限 rational cover，则自然函子

$$
D(A,\mathcal M)
\to
\operatorname{Tot}\bigl(D(A_{U_\bullet},\mathcal M_{U_\bullet})\bigr)
$$

为等价。

## N.3 由义务推出解析张量

**命题 N.6.** 在 N.2-N.3 下，$D(A,\mathcal M)$ 是对称幺半稳定范畴，张量积为

$$
M\otimes^{L,\mathcal M}_AN
=
L_{(A,\mathcal M)}(M\otimes_A^LN).
$$

**证明.** 由附录 K 的幺半 Bousfield localization 判别，核为张量理想使张量积下降到局部范畴。证毕。

**命题 N.7.** 若 $B$ 是 $A$-代数对象，则 analytic localization 后的相对张量积满足

$$
L(M\otimes_B^LN)
\simeq
LM\otimes_{LB}^{L,\mathcal M}LN.
$$

**证明.** 这是附录 K 的 bar construction 证明在 analytic localization 下的代入。证毕。

## N.4 Descent 推出 gluing

**命题 N.8（对象粘合）.** 在 N.5 下，给出 $D(A,\mathcal M)$ 中对象等价于给出：

1. 每个 $U_i$ 上的对象 $M_i$；
2. 每个交叠 $U_{ij}$ 上的等价
   $$
   M_i|_{U_{ij}}\simeq M_j|_{U_{ij}};
   $$
3. 三重交叠上的 cocycle 条件。

**证明.** totalization 的 $0$ 层是各 $U_i$ 上对象，$1$ 层记录交叠相容，$2$ 层记录 cocycle。N.5 断言全局范畴等价于该 cosimplicial diagram 的 totalization。证毕。

**命题 N.9（态射 descent）.** 在 N.5 下，态射空间满足 equalizer 公式

$$
\operatorname{Map}(M,N)
\simeq
\operatorname{Eq}\left(
\prod_i\operatorname{Map}(M_i,N_i)
\rightrightarrows
\prod_{i,j}\operatorname{Map}(M_{ij},N_{ij})
\right)
$$

在二截断相容已足以描述映射空间的情形成立；一般情形由全 cosimplicial totalization 给出。

**证明.** 映射空间由 totalization 中 mapping object 的 limit 计算。写出 cosimplicial limit 的前两级即得 equalizer 公式；高阶相容由完整 totalization 保证。证毕。

## N.5 Huber pair 使用边界

离散 Huber pair $(A,A^+)$ 给出 analytic ring 时，正文中常用以下推论：

1. rational subset 诱导 analytic ring localization；
2. finite rational cover 满足 descent；
3. coherent 或 perfect 对象可由 rational cover 检查；
4. $f_!$、投影公式和对偶构造与 rational descent 相容。

其中 1-2 是 N.4-N.5 的实质输入；3-4 是在稳定范畴、紧生成和投影公式输入下的形式后果。

## 练习

1. 证明 N.6 中 associativity 约束来自普通张量积的 associativity。
2. 写出二开 rational cover 的对象粘合数据。
3. 说明 N.9 中为什么一般要保留高阶 totalization。
4. 解释 analytic ring descent 与普通 sheaf descent 的差别。
