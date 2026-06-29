# 附录 S：同伦转移低阶计算与最小模型样例

本附录补充第十三章的定义 13.1--外部输入定理 13.16 和附录 J 的定义 J.1--外部输入定理 J.19。Markl MHT-1--MHT-8 已定位 homotopy transfer 的 operadic existence version；本附录只计算低阶公式，并检查它们如何反映结合律、Jacobi 恒等式和 Massey 型障碍。完整 tree signs、minimal model uniqueness 和 formality obstruction theory 仍作为外部输入。

## S.1 Contraction 数据

设 $(A,d,\mu)$ 是 dg associative algebra，$H$ 是链复形，给定 contraction
$$
H\xrightarrow{i}A\xrightarrow{p}H,\qquad h:A\to A[1]
$$
满足
$$
pi=\operatorname{id}_H,\qquad
ip-\operatorname{id}_A=dh+hd.
$$
本附录还采用 side conditions
$$
ph=0,\qquad hi=0,\qquad h^2=0.
$$

若 $H=H_\*(A)$ 且微分为零，则 $i$ 选取 cycles representative，$p$ 为投影到同调代表，$h$ 为链同伦。

## S.2 转移乘法

**定义 S.1.** 转移到 $H$ 上的二元乘法定义为
$$
m_2^H(x,y)=p\mu(ix,iy).
$$

**命题 S.2.** 若 $x,y$ 是 $H$ 中同调类，则 $m_2^H(x,y)$ 是 $A$ 的乘法诱导的同调乘法。

**证明.** $ix,iy$ 是 cycles。因为 $\mu$ 是链映射，
$$
d\mu(ix,iy)=\mu(dix,iy)+(-1)^{|x|}\mu(ix,diy)=0.
$$
故 $\mu(ix,iy)$ 是 cycle。投影 $p$ 取其同调类，正是同调乘法。若改变 cycles representative，相差 boundary；乘法为链映射保证乘积相差 boundary，故同调类不变。$\square$

## S.3 三元转移运算

采用同调分次。忽略由 suspension 展开产生的全局 convention 差异时，$A_\infty$ 三元运算的典型形状为
$$
m_3^H(x,y,z)
=
p\mu(h\mu(ix,iy),iz)
+
\epsilon\,
p\mu(ix,h\mu(iy,iz)),
$$
其中 $\epsilon$ 由定义 E.18--定义 E.23 和定义 J.5--定义 J.6 的 Koszul sign convention 决定。

**说明 S.3.** 本书最终符号应由 suspended coderivation 统一推出。上式的重点是树形结构：
$$
((xy)z)\quad\text{与}\quad(x(yz))
$$
两棵平面二叉树各出现一次，内部边上放置 $h$，叶上放置 $i$，根上放置 $p$。

## S.4 结合律的同伦边界

**命题 S.4.** $m_2^H$ 在同调上严格结合。

**证明.** 对 $x,y,z\in H$，
$$
m_2^H(m_2^H(x,y),z)
$$
和
$$
m_2^H(x,m_2^H(y,z))
$$
都表示 $A$ 中 cycle
$$
\mu(\mu(ix,iy),iz)
$$
与
$$
\mu(ix,\mu(iy,iz)).
$$
由于 $\mu$ 在 $A$ 中严格结合，这两个 cycle 相等。投影到同调后相等。$\square$

**说明 S.5.** 在最小 $A_\infty$ 模型中，$m_2^H$ 严格结合并不强迫 $m_3^H=0$。$m_3^H$ 记录更高同伦信息；它的同伦类与三重 Massey products 有关，但不等同于一个无选择的单值运算。

## S.5 $m_3$ 与 Massey product 的边界

设 $a,b,c\in H_\*(A)$ 满足
$$
ab=0,\qquad bc=0.
$$
选择 cycles $\alpha,\beta,\gamma$ 表示它们，并选择 chains $u,v$ 使
$$
du=\alpha\beta,\qquad dv=\beta\gamma.
$$
三重 Massey product 的代表形如
$$
\alpha v+\epsilon u\gamma.
$$

**命题 S.6.** 在 contraction 选择使
$$
h(\alpha\beta)=u,\qquad h(\beta\gamma)=v
$$
的情形下，$m_3^H(a,b,c)$ 的代表与上述 Massey product 代表具有同一树形来源。

**证明.** 公式 S.3 的两项分别是
$$
p\mu(h\mu(\alpha,\beta),\gamma)
$$
和
$$
p\mu(\alpha,h\mu(\beta,\gamma)).
$$
代入 $h(\alpha\beta)=u$ 与 $h(\beta\gamma)=v$，得到
$$
p(u\gamma)+\epsilon p(\alpha v).
$$
这与三重 Massey representative 只差 convention 中的符号排序。$\square$

**警告 S.7.** Massey product 通常是含不定性的集合，而 $m_3^H$ 依赖 contraction 选择。正确说法是：最小 $A_\infty$ 结构的三阶运算组织 Massey 型障碍；不能把某个 $m_3^H$ 值无条件等同于唯一 Massey product。

## S.6 形式性的低阶判据

**定义 S.8.** dg algebra $A$ 称为 formal，若它与 $H_\*(A)$ 作为 dg algebra 通过 quasi-isomorphism zigzag 相连，其中 $H_\*(A)$ 微分为零。

**命题 S.9.** 若存在 contraction 使转移到 $H_\*(A)$ 上的最小 $A_\infty$ 结构满足
$$
m_n^H=0\qquad(n\ge3),
$$
则 $A$ 与 $H_\*(A)$ 作为 $A_\infty$-algebras quasi-isomorphic。

**证明.** Homotopy transfer theorem 给出 $A_\infty$ quasi-isomorphism
$$
I_\infty:H_\*(A)\rightsquigarrow A.
$$
若所有 $m_n^H$ for $n\ge3$ 消失，则 $H_\*(A)$ 上的 $A_\infty$ 结构只剩微分 $0$ 和乘法 $m_2^H$，即普通 graded algebra 视为 dg algebra。故 $I_\infty$ 是从该 dg algebra 到 $A$ 的 $A_\infty$ quasi-isomorphism。$\square$

**警告 S.10.** 命题 S.9 得到的是 $A_\infty$-formality。若要得到 dg algebra zigzag formality，需要 rectification 或额外严格化定理；不能省略模型范畴假设。

## S.7 Dg Lie algebra 的转移括号

设 $(\mathfrak g,d,[-,-])$ 是 dg Lie algebra，并有 contraction
$$
H\xrightarrow{i}\mathfrak g\xrightarrow{p}H,\qquad h:\mathfrak g\to\mathfrak g[1].
$$

**定义 S.11.** 转移的二元括号为
$$
\ell_2^H(x,y)=p[ix,iy].
$$

三元括号的树形为
$$
\ell_3^H(x,y,z)
=
\sum_{\sigma\in\operatorname{Sh}(2,1)}
\epsilon(\sigma)\,
p\big[ h[ i x_{\sigma(1)},i x_{\sigma(2)}], i x_{\sigma(3)}\big],
$$
再加上由所选 $L_\infty$ convention 决定的整体符号。

**命题 S.12.** $\ell_2^H$ 在同调上满足 graded Jacobi identity。

**证明.** 对 cycles $ix,iy,iz$，dg Lie algebra 中 Jacobi 恒等式给出
$$
[ix,[iy,iz]]
\pm[iy,[iz,ix]]
\pm[iz,[ix,iy]]=0.
$$
投影 $p$ 到同调后得到 $\ell_2^H$ 的 Jacobi 恒等式。$\square$

**说明 S.13.** $\ell_3^H$ 记录 Jacobi 恒等式在链级代表和同伦选择上的高阶信息。完整 $L_\infty$ 恒等式需要所有 rooted trees 与 shuffle signs；本附录只给低阶形状。

## S.8 与 operad 语言的对应

同伦转移定理可被表述为：给定 $\mathcal P_\infty$-algebra $A$ 与 contraction 到 $H$，可在 $H$ 上构造 $\mathcal P_\infty$-algebra 结构，使 $H\rightsquigarrow A$ 为 $\infty$-quasi-isomorphism。

在 $A_\infty$ 情形：

1. 平面二叉树来自 $\operatorname{Ass}$ 的 Koszul dual cooperad；
2. 内部边放置 $h$；
3. 叶放置 $i$；
4. 根放置 $p$；
5. 顶点放置原乘法 $m_2$。

在 $L_\infty$ 情形：

1. 非平面树来自 Lie/Com Koszul duality；
2. shuffle signs 来自对称群作用和 suspension；
3. Jacobi 的高阶同伦由三元及以上 brackets 表示。

## S.9 使用检查表

在正文使用同伦转移公式时，必须说明：

1. contraction 是否满足 side conditions；
2. 采用 homological 还是 cohomological grading；
3. $m_n$ 或 $\ell_n$ 的次数；
4. signs 是 suspended convention 还是 unsuspended 展开；
5. 转移对象是 dg algebra、$A_\infty$-algebra、dg Lie algebra 还是 $L_\infty$-algebra；
6. formality 结论是 $A_\infty$ 层面还是 strict dg 层面；
7. 是否需要 rectification 定理。

## S.10 小结

低阶同伦转移可以概括为：
$$
m_2^H=p\mu(i-,i-),
$$
$$
m_3^H=\sum_{\text{two binary trees}}p\mu(h\mu(i-,i-),i-)
$$
带符号求和。$L_\infty$ 情形把平面二叉树替换为带 shuffle 反对称化的有根树。完整转移定理保证所有高阶项满足 $A_\infty/L_\infty$ 恒等式；本附录只提供低阶校验和使用边界。
