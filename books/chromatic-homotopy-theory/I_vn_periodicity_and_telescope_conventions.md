# 附录 I：$v_n$-periodicity 与 telescope 约定

## I.1 Type-n 有限谱

**定义 I.1.** 非零有限 $p$-局部谱 $F$ 为 type $n$，若
$$
K(i)_*F=0\quad(0\le i<n),
\qquad K(n)_*F\ne0.
$$
等价地，$n$ 是使 $K(n)_*F\ne0$ 的最小非负整数。finite detection
保证非零有限谱的该最小值存在；零谱 type 为 $\infty$。

**外部输入 I.2.** finite detection 保证非零有限谱有有限 type；
invariant-prime-ideal/thick-subcategory 定理包进一步给出高度单调性：
若 $F$ 为 type $n$，则 $K(m)_*F\ne0$ 对所有 $m\ge n$ 成立。较低
高度的消失已经包含在定义 I.1 中，不再冒充外部推论。

## I.2 $v_n$ self-map

**定义 I.3.** 设 $n\ge1$ 且 $F$ 是 type $n$ 有限谱。映射
$$
v:\Sigma^dF\to F,\qquad d>0,
$$
称为 $v_n$ self-map，若：

1. $K(n)_*v$ 是同构；
2. 对 $m\ne n$，$K(m)_*v$ 是 nilpotent 或在标准 periodicity theorem 表述中满足对应零化条件；
3. $d$ 与 $v_n$ 的周期次数相容；periodicity theorem 可在取幂后把
   $K(n)_*v$ 规范为某个 $v_n$ 幂的乘法，但具体倍数依赖 $F$。

**约定 I.3A（$v_0$）.** 高度零允许 $d=0$，本书取
$p:\mathbb S_{(p)}\to\mathbb S_{(p)}$。它在有理同调上可逆，在正高度
Morava K-theory 上为零。

**外部输入 I.4（periodicity theorem）.** 对 $n\ge1$，每个 type $n$
有限谱存在 $v_n$ self-map。若 $v$ 与 $w$ 是同一有限谱上的两个
$v_n$ self-maps，则存在正整数 $a,b$ 使 $v^a$ 与 $w^b$ 在合适悬挂
后相同；不同 type $n$ 有限谱之间的映射也在分别取幂后与 self-map
交换。定位为 Hopkins--Smith II, Theorem 9 及 Corollaries 3.7、3.8。

**警告 I.5.** Theorem 9 还给出可在取幂后令其他高度作用为零的规范
版本；正文若使用这个加强结论，必须明确引用 Theorem 9 的对应条件，
不能只引用“存在性”。

## I.3 Telescope

**定义 I.6.** 对 $v:\Sigma^dF\to F$，定义 telescope
$$
v^{-1}F=\operatorname*{colim}\left(F\xrightarrow{\tilde v}\Sigma^{-d}F\xrightarrow{\Sigma^{-d}\tilde v}\Sigma^{-2d}F\to\cdots\right),
$$
其中 $\tilde v:F\to\Sigma^{-d}F$ 是 $v$ 的脱悬挂伴随。

**命题 I.7.** 若 $K(n)_*v$ 是同构，则
$$
K(n)_*(v^{-1}F)\cong K(n)_*F[v^{-1}]
$$
在 graded colimit 意义下成立；特别地，若 $K(n)_*F\ne0$，则 $K(n)_*(v^{-1}F)\ne0$。

**证明.** $K(n)_*$ 保持 filtered homotopy colimits，因为 $K(n)\otimes-$ 是左伴随且同伦群与 filtered colimit 相容。于是
$$
K(n)_*(v^{-1}F)\cong\operatorname*{colim}\left(K(n)_*F\xrightarrow{K(n)_*\tilde v}K(n)_{*+d}F\to\cdots\right).
$$
若 $K(n)_*v$ 是同构，此 colimit 是沿可逆映射的周期化，非零性保留。证毕。

**命题 I.8.** 若 $m<n$ 且 $F$ 为 type $n$，则
$$
K(m)_*(v^{-1}F)=0.
$$

**证明.** 由 type 定义，$K(m)_*F=0$。对 defining telescope 逐项取
$K(m)_*$，每一项都为零，filtered colimit 也为零。这里不需要高度
单调性外部输入。证毕。

## I.4 $T(n)$ 的 Bousfield 类

**定义 I.9.** 对 $n\ge1$，$T(n)$ 表示任意 type $n$ 有限谱 $F$ 的
$v_n$ self-map telescope $v^{-1}F$ 的 Bousfield 类代表。另定义
$$
T(0)=\operatorname*{colim}
(\mathbb S_{(p)}\xrightarrow p\mathbb S_{(p)}\xrightarrow p\cdots)
\simeq H\mathbb Q.
$$

**外部输入 I.10.** 不同 type $n$ 有限谱和不同 $v_n$ self-maps 的
telescopes 给出同一个 Bousfield 类。self-map 选择无关由
Hopkins--Smith II, Corollary 3.7 控制；跨有限谱的 Bousfield 类比较还
使用 Theorem 14（class invariance）。因此这不是 periodicity theorem
单独一句话的形式后果。

**警告 I.11.** $T(n)$ 是 telescope Bousfield 类代表，不是 Morava K-theory $K(n)$。在高度至少 $2$ 的一般情形，不能默认 $L_{T(n)}=L_{K(n)}$。

## I.5 Finite localization

**定义 I.12.** $L_n^f$ 是使所有 type $n+1$ 有限谱 acyclic 的 Bousfield localization。等价地，它由厚子范畴 $\mathcal C_{n+1}$ 生成的 acyclics 决定。

**命题 I.13.** 若 $X$ 是 $L_n^f$-local，则对任意 type $n+1$ 有限谱 $F$，有
$$
F\otimes X\simeq0
$$
并非定义的正确方向；正确的是 localization 的 acyclic 类由 type $n+1$ 有限谱生成，local object 与这些 acyclics 正交。

**证明.** Local object 条件是 $F(A,X)\simeq0$ 对所有 acyclic $A$。它不是 $A\otimes X\simeq0$。若 localization 是 smashing 或有额外 tensor ideal 结构，才可把正交条件转换为张量消失。证毕。

## 本附录小结

$v_n$ self-map 和 telescope 是 chromatic periodicity 的核心工具。正式教材必须把 type、self-map、telescope、$T(n)$、$L_n^f$ 和 $K(n)$ 分开。特别是 telescope conjecture 失败后，$T(n)$ 与 $K(n)$ 的混用是严重错误。
