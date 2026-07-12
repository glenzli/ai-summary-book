# 附录 G：低秩例子：$SL_2$、$SL_3$、Springer fibers 和 Schubert singularities

## 本章目标

本附录收集低秩例子，用于检查正文的 shift、fiber、Springer-action convention 和 Schubert closure convention。凡能用矩阵与 flags 完成的论证在此写全；一般 Springer action 仍明确标为外部输入。

## G.1 $SL_2$

**例 G.1.** $SL_2/B\simeq\mathbb P^1$，Schubert cells 为一点和一条 affine line。KL polynomial 只有 $P_{e,s}=1$。

**例 G.2.** $\mathfrak{sl}_2$ nilpotent cone 有零轨道和 regular nilpotent orbit。Springer fiber 在 $0$ 处为 $\mathbb P^1$，在 regular nilpotent 处为一点。

**计算 G.3.** 取
$$
e=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
f=\begin{pmatrix}0&0\\1&0\end{pmatrix},\qquad
h=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$
nilpotent cone 为
$$
\mathcal N=\{x\in\mathfrak{sl}_2\mid \operatorname{tr}(x^2)=0\}.
$$
若 $x=0$，所有 Borel subalgebras 都含 $x$，所以 Springer fiber 是 flag variety $\mathbb P^1$。若 $x=e$，含 $e$ 的 Borel subalgebra 唯一，即上三角 Borel；故 fiber 为一点。

**命题 G.4（Springer sheaf 的逐次 stalk）.** 令
$$
\pi:T^*\mathbb P^1\longrightarrow\mathcal N
$$
为 $SL_2$ Springer resolution，并在第三章的 Betti convention 下置
$$
\operatorname{Spr}=R\pi_\ast E_{T^*\mathbb P^1}[2].
$$
若 $x$ regular nilpotent，则
$$
H^j(i_x^\ast\operatorname{Spr})
=\begin{cases}E,&j=-2,\\0,&j\ne-2;\end{cases}
$$
若 $x=0$，则
$$
H^j(i_0^\ast\operatorname{Spr})
=\begin{cases}E,&j\in\{-2,0\},\\0,&\text{otherwise}.
\end{cases}
$$

**证明.** $\pi$ proper，故 proper base change 给出
$$
i_x^\ast R\pi_\ast E[2]
\simeq R\Gamma(\mathcal B_x,E)[2],
$$
从而
$$
H^j(i_x^\ast\operatorname{Spr})
\simeq H^{j+2}(\mathcal B_x,E).
$$
计算 G.3 已证明 regular fiber 是一点、zero fiber 是 $\mathbb P^1$。一点只有 $H^0=E$；$\mathbb P^1$ 只有 $H^0=E$ 与 $H^2=E$。代入 $j+2=0,2$ 即得所列 degrees。$\square$

**外部输入说明 G.4.1（作用不能由 Euler characteristic 恢复）.** `SPR-1` 在
$H^\ast(\mathcal B_0,E)=H^\ast(\mathbb P^1,E)$ 上构造 $W=S_2$ action。在 coinvariant-algebra convention 中 $H^0$ 是 trivial、$H^2$ 是 sign，所以 ungraded total cohomology 是 regular representation；sign-twisted Springer convention 会交换两者。命题 G.4 的 stalk dimensions 本身不决定这个 action，因此原来的“Euler characteristics 给出 regular representation”不能作为证明。

## G.2 $SL_3$

**例 G.5.** $SL_3$ 的 Weyl group 为 $S_3$，由 simple reflections $s_1,s_2$ 生成。最长元 $w_0=s_1s_2s_1=s_2s_1s_2$，长度为 $3$。

**计算 G.6.** 六个 Schubert cells 的维数为
$$
\ell(e)=0,\quad
\ell(s_1)=\ell(s_2)=1,\quad
\ell(s_1s_2)=\ell(s_2s_1)=2,\quad
\ell(w_0)=3.
$$
闭包关系由 Bruhat order 给出：
$$
e<s_i<s_is_j<w_0\qquad (i\ne j).
$$
两个长度 $1$ 元素互不可比，两个长度 $2$ 元素互不可比。

**例 G.7.** $SL_3$ 的 nilpotent orbits 由 partitions of $3$ 参数化：
$$
(3),\qquad (2,1),\qquad (1,1,1).
$$
对应 regular、subregular 和 zero orbit。Springer fiber 在 zero orbit 上为完整 flag variety；在 regular orbit 上为一点；subregular fiber 是两条 $\mathbb P^1$ 按 Dynkin 图 $A_2$ 相交的曲线。

**命题 G.8（subregular fiber 的直接分类）.** 令 $V=\mathbb C^3$，并取 Jordan type $(2,1)$ 的 nilpotent endomorphism
$$
xe_2=e_1,\qquad xe_1=xe_3=0.
$$
记
$$
I=\operatorname{im}x=\langle e_1\rangle,
\qquad
K=\ker x=\langle e_1,e_3\rangle.
$$
则 Springer fiber
$$
\mathcal B_x
=\{0\subset L\subset P\subset V
\mid \dim L=1,\ \dim P=2,\ xL\subset L,\ xP\subset P\}
$$
是两条 $\mathbb P^1$ 的并
$$
C_1=\{I\subset P\mid I\subset P\subset V,\ \dim P=2\},
\qquad
C_2=\{L\subset K\mid \dim L=1\},
$$
且 $C_1\cap C_2$ 是单点 flag $I\subset K$。特别地，不可约分支交叉图是 $A_2$。

**证明.** 若 line $L$ 满足 $xL\subset L$，则 $x|_L$ 是 nilpotent scalar，故为零，于是 $L\subset K$。若 stable plane $P$ 不等于 $K$，可取 $v\in P$ 的 $e_2$-coefficient 非零；于是 $xv$ 是 $e_1$ 的非零倍数，stable condition 强迫 $I\subset P$。若 $P=K$，也有 $I\subset P$。反过来，任意包含 $I$ 的 plane 都 stable，因为 $x(V)=I$。

现取任意 stable flag $L\subset P$。若 $P=K$，则 $L$ 可为 $K$ 中任意 line，所得 flags 正是 $C_2\simeq\mathbb P(K)\simeq\mathbb P^1$。若 $P\ne K$，两个 planes $P,K$ 的交含 $I$ 且维数为 $1$，故 $P\cap K=I$；又 $L\subset P\cap K$，所以 $L=I$，所得 flags 正是
$C_1\simeq\mathbb P(V/I)\simeq\mathbb P^1$。因此
$\mathcal B_x=C_1\cup C_2$。同时属于两者要求 $L=I$ 且 $P=K$，所以交集只有 $I\subset K$。两条闭不可约曲线互不包含，故它们恰为两个不可约分支，交叉图有两个顶点和一条边，即 $A_2$。$\square$

## G.3 Schubert singularity 检查

**例 G.9.** 在 $SL_2/B$ 和 $SL_3/B$ 中，所有 Schubert varieties 都是光滑的。第一个非平凡 Schubert singularity 出现在更高 rank，例如 type $A_3$ 的某些 Schubert varieties。故 $SL_2$、$SL_3$ 主要用于检验 cell、shift、Springer fiber 和 Weyl group action，而不是检验 KL 多项式的非平凡系数。

**练习 G.10.** 写出六个 $SL_3$ Schubert cells 的维数，并用 Bruhat order 画出闭包关系。

**练习 G.11.** 对 $SL_2$，直接计算 $H^\ast(\mathbb P^1)$ 上的 degree convention，并与第五章 Springer sheaf 的 shift 对齐。

## 本章小结

低秩例子用于校验符号、shift、closure convention、Springer fibers 和 Weyl group action。本附录现在逐次算出了 $SL_2$ Springer sheaf stalk，并从稳定 line/plane 条件直接证明 $SL_3$ subregular fiber 是两条相交的 $\mathbb P^1$；只有 Weyl group action 的构造本身继续作为 `SPR-1` 外部输入。
