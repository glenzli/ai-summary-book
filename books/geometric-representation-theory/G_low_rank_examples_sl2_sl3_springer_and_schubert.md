# 附录 G：低秩例子：$SL_2$、$SL_3$、Springer fibers 和 Schubert singularities

## 本章目标

本附录收集低秩例子，用于检查正文 convention。

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

**命题 G.4.** 在 $SL_2$ 情形，Springer sheaf 的 stalk Euler characteristics 给出 Weyl group $S_2$ 的 regular representation 的维数分布。

**证明.** Springer resolution 在 regular nilpotent orbit 上 fiber 为一点，在零点 fiber 为 $\mathbb P^1$。零点 stalk 的 cohomology 为
$$
H^\ast(\mathbb P^1,\mathbb C)\simeq \mathbb C\oplus\mathbb C[-2],
$$
总维数为 $2=|S_2|$。Springer action 的精确符号取决于 top cohomology normalization；本计算只用于检验维数和 shift。$\square$

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

**命题 G.8.** $SL_3$ 的 subregular Springer fiber 的不可约分支数等于 $A_2$ Dynkin 图的顶点数。

**证明.** 一般 Springer theory 给出 simply-laced 情形下 subregular fiber 的分支交叉图为对应 Dynkin 图。对 $SL_3$，Dynkin 图 $A_2$ 有两个顶点，故 fiber 有两个不可约分支。低秩模型中这两个分支可由稳定 partial flags 的两个选择得到；交点对应同时满足两个 simple-root 条件的 flag。$\square$

## G.3 Schubert singularity 检查

**例 G.9.** 在 $SL_2/B$ 和 $SL_3/B$ 中，所有 Schubert varieties 都是光滑的。第一个非平凡 Schubert singularity 出现在更高 rank，例如 type $A_3$ 的某些 Schubert varieties。故 $SL_2$、$SL_3$ 主要用于检验 cell、shift、Springer fiber 和 Weyl group action，而不是检验 KL 多项式的非平凡系数。

**练习 G.10.** 写出六个 $SL_3$ Schubert cells 的维数，并用 Bruhat order 画出闭包关系。

**练习 G.11.** 对 $SL_2$，直接计算 $H^\ast(\mathbb P^1)$ 上的 degree convention，并与第五章 Springer sheaf 的 shift 对齐。

## 本章小结

低秩例子用于校验符号、shift、closure convention、Springer fibers 和 Weyl group action。它们不替代一般定理，但能暴露 normalization 错误。
