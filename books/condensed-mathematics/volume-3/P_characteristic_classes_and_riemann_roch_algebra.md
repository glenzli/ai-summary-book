# 附录 P：特征类、Chern character 与 Riemann-Roch 的形式代数

## P.0 目标

第七章和附录 K 使用 Chern character、Todd class 和 Riemann-Roch。本附录补齐其中可在书内说明的形式代数：

1. Chern 类的公理化使用方式。
2. splitting principle 如何把向量丛化为线丛根的计算。
3. Chern character 和 Todd class 的公式。
4. 它们对直和、张量积和 $K$-理论的相容性。
5. Riemann-Roch 公式在接受几何输入后的代数结构。

本附录不构造 singular cohomology、de Rham cohomology、cycle class map 或 Chern-Weil theory；这些属于经典几何输入。

## P.1 Chern 类的公理化形式

设 $X$ 是紧复流形或光滑 proper 复代数簇。对复向量丛 $E$，总 Chern 类记为

$$
c(E)=1+c_1(E)+c_2(E)+\cdots+c_r(E),
$$

其中 $r=\operatorname{rk}E$，$c_i(E)\in H^{2i}(X,\mathbb Q)$。

本书使用以下性质作为输入：

1. **自然性。** 对 $f:Y\to X$，
   $$
   c(f^*E)=f^*c(E).
   $$
2. **Whitney 公式。** 对短正合列
   $$
   0\to E'\to E\to E''\to0
   $$
   有
   $$
   c(E)=c(E')c(E'').
   $$
3. **线丛归一化。** 若 $L$ 是线丛，则
   $$
   c(L)=1+c_1(L).
   $$

**输入定理 P.1（splitting principle）.** 对任意秩 $r$ 向量丛 $E$，存在映射 $\pi:Y\to X$，使得 $\pi^*:H^\bullet(X,\mathbb Q)\to H^\bullet(Y,\mathbb Q)$ 单射，并且

$$
\pi^*E
$$

有 filtration，其分级商为线丛 $L_1,\ldots,L_r$。

因此在计算 Chern 类恒等式时，可形式地写

$$
c(E)=\prod_{i=1}^r(1+x_i),
\qquad
x_i=c_1(L_i),
$$

其中 $x_i$ 称为 Chern roots。

## P.2 Chern character

**定义 P.2.** 若 $E$ 的 Chern roots 为 $x_1,\ldots,x_r$，定义

$$
\operatorname{ch}(E)=\sum_{i=1}^r e^{x_i}
=
\sum_{k\ge0}\frac1{k!}\sum_{i=1}^r x_i^k.
$$

这是对称幂级数，因此由 splitting principle 下降为 $X$ 上的 cohomology class。

**命题 P.3（直和可加性）.** 对向量丛 $E,F$，

$$
\operatorname{ch}(E\oplus F)=\operatorname{ch}(E)+\operatorname{ch}(F).
$$

**证明.** 在 splitting space 上，$E$ 的根为 $x_i$，$F$ 的根为 $y_j$，则 $E\oplus F$ 的根为二者并集。因此

$$
\operatorname{ch}(E\oplus F)
=\sum_i e^{x_i}+\sum_j e^{y_j}.
$$

由 splitting principle 的单射性，恒等式下降到 $X$。证毕。

**命题 P.4（张量乘法性）.** 对向量丛 $E,F$，

$$
\operatorname{ch}(E\otimes F)=\operatorname{ch}(E)\operatorname{ch}(F).
$$

**证明.** 在 splitting space 上，$E=\oplus_iL_i$、$F=\oplus_jM_j$，其 Chern roots 分别为 $x_i,y_j$。张量积的线丛分量为 $L_i\otimes M_j$，根为 $x_i+y_j$。于是

$$
\operatorname{ch}(E\otimes F)
=\sum_{i,j}e^{x_i+y_j}
=\left(\sum_i e^{x_i}\right)\left(\sum_j e^{y_j}\right).
$$

证毕。

**推论 P.5.** Chern character 诱导环同态

$$
\operatorname{ch}:K^0(X)\to H^{even}(X,\mathbb Q).
$$

**证明.** $K^0(X)$ 的加法来自直和，乘法来自张量积。命题 P.3 和 P.4 分别给出加法和乘法相容。短正合列在 $K^0$ 中给 $[E]=[E']+[E'']$；Whitney 过滤或 splitting principle 保证这与直和公式一致。证毕。

## P.3 Todd class

**定义 P.6.** 对线丛根 $x$，定义 Todd 因子

$$
\operatorname{td}(x)=\frac{x}{1-e^{-x}}
=
1+\frac{x}{2}+\frac{x^2}{12}-\frac{x^4}{720}+\cdots.
$$

若向量丛 $E$ 的 Chern roots 为 $x_1,\ldots,x_r$，定义

$$
\operatorname{td}(E)=\prod_{i=1}^r\frac{x_i}{1-e^{-x_i}}.
$$

**命题 P.7（Todd class 的直和乘法性）.** 对向量丛 $E,F$，

$$
\operatorname{td}(E\oplus F)=\operatorname{td}(E)\operatorname{td}(F).
$$

**证明.** 在 splitting space 上，$E\oplus F$ 的根是 $E$ 和 $F$ 的根的并集。Todd class 是各根 Todd 因子的乘积，因此等式成立。由 splitting principle 下降。证毕。

**例 P.8（曲线情形）.** 若 $X$ 是复曲线，则只保留到二阶 cohomology：

$$
\operatorname{td}(T_X)=1+\frac12c_1(T_X).
$$

因为更高次数项落在 $H^{\ge4}(X)$，对复曲线为零。

## P.4 Euler characteristic 与 $K$-理论

设 $X$ proper，且 coherent cohomology finite-dimensional。

**定义 P.9.** 对 $E\in K^0(X)$，定义

$$
\chi_X(E)=\sum_i(-1)^i\dim H^i(X,E).
$$

**命题 P.10.** $\chi_X:K^0(X)\to\mathbb Z$ 是群同态。

**证明.** 对短正合列

$$
0\to E'\to E\to E''\to0
$$

有上同调长正合列。有限维长正合列的交错维数和为零，因此

$$
\chi(E)=\chi(E')+\chi(E'').
$$

故 $\chi_X$ 尊重 $K^0$ 的关系。证毕。

## P.5 Riemann-Roch 的形式结构

**输入定理 P.11（HRR）.** 对紧复流形 $X$ 和全纯向量丛 $E$，

$$
\chi_X(E)
=
\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

**命题 P.12（HRR 两边都是 $K$-理论同态）.** 固定 $X$ 后，公式两边都定义从 $K^0(X)$ 到 $\mathbb Q$ 的群同态。

**证明.** 左侧由命题 P.10。右侧中 $\operatorname{ch}$ 是群同态，乘以固定类 $\operatorname{td}(T_X)$ 和取顶次积分都是线性映射。因此右侧是群同态。证毕。

**推论 P.13（验证生成元即可）.** 若 $K^0(X)$ 由向量丛类 $[E_\alpha]$ 生成，并且 HRR 公式对所有 $E_\alpha$ 成立，则它对所有 $K^0(X)$ 中元素成立。

**证明.** 由命题 P.12，左右两边是群同态。群同态在生成元上相同，则处处相同。证毕。

## P.6 $\mathbb P^1$ 计算

令 $H\in H^2(\mathbb P^1,\mathbb Q)$ 满足 $\int_{\mathbb P^1}H=1$。

**命题 P.14.** 对 $\mathcal O(d)$，

$$
\operatorname{ch}(\mathcal O(d))=1+dH.
$$

**证明.** 线丛的 Chern root 为 $c_1(\mathcal O(d))=dH$。因为 $H^2=0$，

$$
e^{dH}=1+dH.
$$

证毕。

**命题 P.15.** 对 $\mathbb P^1$，

$$
\operatorname{td}(T_{\mathbb P^1})=1+H.
$$

**证明.** $T_{\mathbb P^1}$ 是线丛，且 $c_1(T_{\mathbb P^1})=2H$。由例 P.8，

$$
\operatorname{td}(T_{\mathbb P^1})=1+\frac12(2H)=1+H.
$$

证毕。

**推论 P.16.** HRR 右侧对 $\mathcal O(d)$ 等于 $d+1$。

**证明.**

$$
\operatorname{ch}(\mathcal O(d))\operatorname{td}(T_{\mathbb P^1})
=(1+dH)(1+H)=1+(d+1)H.
$$

取顶次积分得到 $d+1$。证毕。

## P.7 与 condensed/analytic trace 的边界

在 condensed/analytic 语言中，Euler characteristic 可理解为 identity endomorphism 的 trace。HRR 进一步断言该 trace 与 characteristic class 积分相等。

本附录只证明特征类表达式的形式代数。以下内容仍是输入：

1. Chern class 或 Chern character 的几何构造。
2. Todd class 与 tangent complex 的相容。
3. trace map 与 integration map 的比较。
4. pushforward 与 characteristic classes 的 Grothendieck-Riemann-Roch 相容性。

## P.8 练习

**练习 P.1.** 若 $L,M$ 是线丛，证明 $\operatorname{ch}(L\otimes M)=\operatorname{ch}(L)\operatorname{ch}(M)$。

**练习 P.2.** 用 Chern roots 证明 $\operatorname{td}(E\oplus F)=\operatorname{td}(E)\operatorname{td}(F)$。

**练习 P.3.** 对复曲线 $X$ 和线丛 $L$，把 HRR 右侧化为

$$
\deg L+1-g.
$$

**练习 P.4.** 说明为什么附录 P 没有证明 HRR 输入定理 P.11。
