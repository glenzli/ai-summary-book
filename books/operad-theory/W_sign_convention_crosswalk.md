# 附录 W：符号、悬挂与总次数交叉核对表

本附录把附录 E 的约定 E.1--说明 E.25、附录 J 的定义 J.1--警告 J.20、附录 L 的定义 L.1--说明 L.20、附录 P 的命题 P.1--说明 P.9 和附录 S 的定义 S.1--说明 S.13 放在同一套符号转换中。它不引入新定义，而是把文献中的上同调分次、不同 suspension convention 和 Hochschild brace 符号统一转换为本书的同调分次约定。

## W.1 全书默认分次

本书默认使用同调分次：

| 对象 | 约定 |
| --- | --- |
| 链微分 | $d:C_n\to C_{n-1}$，次数 $-1$ |
| 悬挂 | $|sx|=|x|+1$ |
| 悬挂微分 | $d(sx)=-s(dx)$ |
| 去悬挂 | $|s^{-1}x|=|x|-1$ |
| 张量微分 | $d(x\otimes y)=dx\otimes y+(-1)^{|x|}x\otimes dy$ |
| Koszul braiding | $\tau(x\otimes y)=(-1)^{|x||y|}y\otimes x$ |
| Hom differential | $d(f)=d f-(-1)^{|f|}f d$ |

**检查 W.1.** 任一公式若使用 cohomological degree，应先把微分次数从 $+1$ 改为 $-1$，再比较 signs。不能只把上标改成下标。

## W.2 Operadic suspension

对 dg symmetric sequence $M$，
$$
(\Lambda M)(n)=s^{1-n}M(n)\otimes\operatorname{sgn}_n.
$$
因此 arity $n$ 的元素次数改变 $1-n$。

**表 W.2.**

| arity | suspension shift | symmetric group twist |
| --- | --- | --- |
| $0$ | $+1$ | trivial |
| $1$ | $0$ | trivial |
| $2$ | $-1$ | $\operatorname{sgn}_2$ |
| $3$ | $-2$ | $\operatorname{sgn}_3$ |
| $n$ | $1-n$ | $\operatorname{sgn}_n$ |

**说明 W.3.** $A_\infty/L_\infty$ 中最常见的符号差异来自是否把 operadic suspension 吸收到生成元定义中。本文正文避免把 unsuspended 高阶恒等式作为主定义；主定义始终来自 bar-cobar 或 suspended coderivation。

## W.3 Suspended Hochschild cochains

对 dg algebra $A$，设
$$
\widetilde C^p(A,A)=\underline{\operatorname{Hom}}((sA)^{\otimes p},sA).
$$
若
$$
f:A^{\otimes p}\to A
$$
内部次数为 $|f|$，则
$$
\widetilde f=s\circ f\circ(s^{-1})^{\otimes p}
$$
次数为
$$
|\widetilde f|=|f|+1-p.
$$

**表 W.4.**

| cochain | arity | internal degree | suspended degree |
| --- | --- | --- | --- |
| multiplication $\mu$ | $2$ | $0$ | $-1$ |
| differential $d_A$ | $1$ | $-1$ | $-1$ |
| unary chain map $f$ | $1$ | $0$ | $0$ |
| $p$-cochain $f$ | $p$ | $|f|$ | $|f|+1-p$ |

**命题 W.5.** 对内部次数为 $0$ 的 $f\in C^p(A,A)$ 和 $g\in C^q(A,A)$，suspended bracket 的交换符号
$$
(-1)^{|\widetilde f||\widetilde g|}
$$
等于未分次 Gerstenhaber 公式中的
$$
(-1)^{(p-1)(q-1)}.
$$

**证明.** 内部次数为 $0$ 时，
$$
|\widetilde f|=1-p,\qquad |\widetilde g|=1-q.
$$
模 $2$ 下
$$
(1-p)(1-q)\equiv(p-1)(q-1).
$$
故符号相同。$\square$

## W.4 Brace 符号算法

本书采用定义 E.18--定义 E.23 的 suspended brace 算法：

1. 把未悬挂 cochain $f$ 转为 $\widetilde f$；
2. 在 $(sA)^{\otimes p}$ 上按 Koszul rule 做 insertion；
3. 每个插入块 $G$ 穿过其左侧直接进入外层 operation 的 suspended inputs，贡献
   $$
   (-1)^{|G|(\text{这些输入的次数之和})};
   $$
4. 最后去悬挂。

**命题 W.6.** 若所有输入 $a_i$ 内部次数为 $0$，则 suspended brace 算法退化为未分次公式
$$
\epsilon=\sum_j(q_j-1)i_j.
$$

**证明.** 内部次数为 $0$ 时，$sa_i$ 次数为 $1$。若 $g_j$ 内部次数为 $0$ 且 arity 为 $q_j$，则
$$
|\widetilde g_j|=1-q_j.
$$
穿过 $i_j$ 个 suspended inputs 的符号指数为
$$
(1-q_j)i_j\equiv(q_j-1)i_j\pmod2.
$$
对所有插入块求和即得公式。$\square$

## W.5 $A_\infty$ 的 suspended 主定义

本书使用 reduced tensor coalgebra
$$
T^c(sA)=\bigoplus_{n\ge1}(sA)^{\otimes n}.
$$
$A_\infty$-结构是 degree $-1$ coderivation
$$
b:T^c(sA)\to T^c(sA)
$$
满足 $b^2=0$。Taylor 分量
$$
b_n:(sA)^{\otimes n}\to sA
$$
均为次数 $-1$。

未悬挂映射定义为
$$
m_n=s^{-1}\circ b_n\circ s^{\otimes n}.
$$
其次数为
$$
|m_n|=-1-1+n=n-2.
$$

**表 W.7.**

| operation | suspended degree | unsuspended degree |
| --- | --- | --- |
| $b_1$ | $-1$ | $|m_1|=-1$ |
| $b_2$ | $-1$ | $|m_2|=0$ |
| $b_3$ | $-1$ | $|m_3|=1$ |
| $b_n$ | $-1$ | $|m_n|=n-2$ |

**说明 W.8.** 某些文献把额外符号放入 $m_n=s^{-1}b_ns^{\otimes n}$ 的定义。本书不把这种 unsuspended 公式作为基础定义。若引用文献中的 $m_n$ 恒等式，必须先写出其 $m_n$ 与本书 $m_n$ 相差的符号。

## W.6 $A_\infty$ 低阶关系的安全形式

在 suspended convention 中，关系统一写为
$$
\sum_{r+s+t=n}
b_{r+1+t}
(\operatorname{id}^{\otimes r}\otimes b_s\otimes\operatorname{id}^{\otimes t})
=0.
$$

低阶为：

1. $n=1$：
   $$
   b_1b_1=0.
   $$
2. $n=2$：
   $$
   b_1b_2+b_2(b_1\otimes1)+b_2(1\otimes b_1)=0.
   $$
3. $n=3$：
   $$
   \begin{aligned}
   0={}&b_1b_3+b_2(b_2\otimes1)+b_2(1\otimes b_2)\\
   &+b_3(b_1\otimes1\otimes1)+b_3(1\otimes b_1\otimes1)+b_3(1\otimes1\otimes b_1).
   \end{aligned}
   $$

**检查 W.9.** 若一个 unsuspended 公式在 $n=3$ 情形没有同时出现两个二叉树项和三个含 $m_1$ 的 $m_3$ 边界项，则它不是完整的 $A_\infty$ 关系。

## W.7 $L_\infty$ 的 suspended 主定义

本书用 reduced cofree cocommutative coalgebra
$$
S^c(sV)=\bigoplus_{n\ge1}\operatorname{Sym}^n(sV)
$$
上的 degree $-1$ coderivation $q$ 定义 $L_\infty$-结构。Taylor 分量
$$
q_n:\operatorname{Sym}^n(sV)\to sV
$$
次数为 $-1$。

未悬挂括号为
$$
\ell_n=s^{-1}\circ q_n\circ s^{\otimes n},
$$
次数为
$$
|\ell_n|=n-2.
$$

**说明 W.10.** $q_n$ 在 suspended variables 上 graded symmetric。转回 $V$ 后，$\ell_n$ 满足 graded antisymmetric convention；具体符号由 suspension 和 Koszul braiding 共同产生。书写 unsuspended shuffle 公式时，必须从 $q^2=0$ 推出，而不能从记忆中的 cohomological 公式直接搬运。

## W.8 同伦转移符号检查

在 $A_\infty$ 转移公式中：

1. 叶子放置 $i$；
2. 内部边放置 $h$；
3. 内部顶点放置原乘法或原高阶 operation；
4. 根放置 $p$；
5. 对所有相应树求和。

对 dg associative algebra 的低阶部分，形状为
$$
m_2^H=p\mu(i-,i-),
$$
$$
m_3^H=
\sum_{T\in\operatorname{PBT}_3}
\pm\,p\mu(h\mu(i-,i-),i-).
$$

**检查 W.11.** 任何给定的 $m_3^H$ 公式必须说明：

1. 两棵二叉树的相对符号；
2. $h$ 的次数是 $+1$；
3. 微分是同调次数 $-1$；
4. contraction side conditions 是否使用；
5. 该公式是 suspended 展开还是 unsuspended 展开。

## W.9 文献转换表

| 文献公式类型 | 转换动作 |
| --- | --- |
| cohomological $A_\infty$ with $|m_n|=2-n$ | 改为同调分次后得到 $|m_n|=n-2$，并重新计算微分符号 |
| unsuspended brace formula | 先转到 $\widetilde C^\*(A,A)$，用定义 E.18--定义 E.23 的 suspended braces 核对 |
| $L_\infty$ skew brackets in cohomological grading | 转为 $S^c(sV)$ 上 degree $-1$ coderivation 后再展开 |
| operadic suspension $\mathfrak s$ 与 $\Lambda$ 符号不同 | 对照 arity shift $1-n$ 和 $\operatorname{sgn}_n$ twist |
| chain model for $E_2$ | 检查 chains functor 的 monoidal convention 和 differential degree |

## W.10 小结

本书的安全原则是：所有高阶同伦代数符号先在 suspended coalgebra 或 suspended Hochschild cochains 上定义，再转回未悬挂表达。最终教材若要展示 unsuspended 全公式，必须附带本附录的转换路径；否则只能把 suspended 公式作为正式定义，把 unsuspended 公式作为计算说明。
