# 附录 I：低阶 $A_\infty$、curvature 与 Maurer-Cartan 计算

## I.1 非弯曲低阶方程

**约定 I.1.** 本附录使用正文 suspension convention，但把低阶方程按“结构含义”展开。具体 Koszul 符号由附录 B 的 bar coalgebra 规则确定。

**计算 I.2（一个输入）.** $A_\infty$ 方程在一个输入 $a$ 上给出
$$
\mu^1(\mu^1(a))=0.
$$
因此 $\mu^1$ 是微分。

**计算 I.3（两个输入）.** 对 composable $a_1,a_2$，方程给出
$$
\mu^1\mu^2(a_2,a_1)
=\mu^2(\mu^1a_2,a_1)\pm\mu^2(a_2,\mu^1a_1).
$$
这说明 $\mu^2$ 是链映射，或等价地，$\mu^1$ 对乘法满足带符号 Leibniz 规则。

**计算 I.4（三个输入）.** 对 $a_1,a_2,a_3$，方程给出
$$
\mu^2(\mu^2(a_3,a_2),a_1)\pm
\mu^2(a_3,\mu^2(a_2,a_1))
=
\mu^1\mu^3(a_3,a_2,a_1)\pm
\sum_i\mu^3(\ldots,\mu^1a_i,\ldots).
$$
因此 $\mu^2$ 的 associator 是 $\mu^3$ 的边界。

**推论 I.5.** 在 cohomology category $H^\ast(\mathcal A)$ 中，$\mu^2$ 诱导严格结合的复合。

**证明.** 对 cocycles $a_i$，计算 I.4 右侧为 $\mu^1$-boundary。因此两种结合方式在 cohomology 中相等。证毕。

## I.2 dg category 作为特殊情形

**计算 I.6.** 若 $\mu^d=0$ for $d\ge3$，则计算 I.4 变成
$$
\mu^2(\mu^2(a_3,a_2),a_1)
\pm \mu^2(a_3,\mu^2(a_2,a_1))=0,
$$
即 dg category 的严格结合律在符号约定下成立。

## I.3 Curved 情况

**计算 I.7（零输入）.** Curved $A_\infty$ 方程在零输入上给出
$$
\mu^1(\mu^0)=0
$$
加上可能的单位规范化条件。几何上这表示 boundary of one-dimensional disk moduli 的零维端点相消。

**计算 I.8（一个输入）.** 在一个输入 $x$ 上，低阶方程包含
$$
\mu^1\mu^1(x)+\mu^2(\mu^0,x)\pm\mu^2(x,\mu^0)+\text{higher curvature insertions}=0.
$$
若 curvature 为标量单位 $W\cdot e$，则
$$
(\mu^1)^2(x)=(W_{\mathrm{right}}-W_{\mathrm{left}})\,x
$$
是变形 morphism complex 中常见的形式。

## I.4 Maurer-Cartan 变形

**定义 I.9.** 对 bounding cochain $b$，变形 curvature 为
$$
\mu_b^0=\sum_{d\ge0}\mu^d(b,\ldots,b).
$$

**命题 I.10.** 若 $\mu_b^0=0$，则变形后的一阶运算 $\mu_b^1$ 平方为零。

**证明.** 将 $b$ 插入 curved $A_\infty$ 方程的所有空隙并求和。一个输入的变形方程为
$$
\mu_b^1\mu_b^1(x)+\mu_b^2(\mu_b^0,x)\pm\mu_b^2(x,\mu_b^0)+\cdots=0.
$$
若 $\mu_b^0=0$，所有 curvature 项消失，得到 $\mu_b^1\mu_b^1=0$。证毕。

**命题 I.11.** 若 $\mu_b^0=W(b)e$ 且两个对象的 values $W(b_0)$、$W(b_1)$ 不同，则变形 morphism operator 的平方为非零标量乘恒等。它不是同一 fiber Fukaya category 中的普通 differential；在 curved 或 matrix-factorization 型总范畴中，若该标量可逆，则相应 morphism object contractible。

**证明.** 由计算 I.8，
$$
d^2=(W(b_1)-W(b_0))\operatorname{id}.
$$
若 $c=W(b_1)-W(b_0)$ 可逆，则在标准 $\mathbb Z/2$ 同伦约定且 $2$ 可逆时可取
$$
h=(2c)^{-1}d
$$
给出 contracting homotopy，因为
$$
dh+hd=(2c)^{-1}(d^2+d^2)=\operatorname{id}.
$$
若特征或符号约定不同，需要改用相应 curved-category 的 contractibility convention。证毕。

## 本附录小结

低阶 $A_\infty$ 方程解释了 Fukaya category 的基本现象：$\mu^1$ 是微分，$\mu^2$ 只在 cohomology 上严格结合，curvature 阻碍 differential 平方为零，而 Maurer-Cartan 元正是消除该阻碍的数据。
