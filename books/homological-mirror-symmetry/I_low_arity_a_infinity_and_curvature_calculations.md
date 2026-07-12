# 附录 I：低阶 $A_\infty$、curvature 与 Maurer--Cartan 计算

## I.1 非弯曲低阶方程

**约定 I.1.** 对齐次态射 $a_i$ 写 $x_i=sa_i$，并以
$|x_i|=|a_i|-1$ 计算符号。张量顺序、coderivation 延拓和 desuspension
完全采用附录 B 的 (B.1)--(B.3)。

**计算 I.2（一个输入）.** 长度 $1$ 方程是
$$
b_1b_1(x_1)=0.
$$
Desuspension 后即 $\mu^1\mu^1(a_1)=0$，所以 $\mu^1$ 是次数 $+1$
微分。

**计算 I.3（两个输入）.** 对可复合齐次输入，长度 $2$ 方程是
$$
b_1b_2(x_2,x_1)+b_2(b_1x_2,x_1)
+(-1)^{|x_2|}b_2(x_2,b_1x_1)=0.
\tag{I.1}
$$
这就是 suspended 形式的 Leibniz 规则。把每个 $x_i$ 换回 $sa_i$ 并按
附录 B 的 desuspension 约定移动 suspension，得到 $\mu^1$ 与
$\mu^2$ 的通常 Koszul Leibniz 规则；(I.1) 已经唯一确定其中每个符号。

**计算 I.4（三个输入）.** 长度 $3$ 方程是
$$
\begin{aligned}
0={}&b_1b_3(x_3,x_2,x_1)
+b_2(b_2(x_3,x_2),x_1)
+(-1)^{|x_3|}b_2(x_3,b_2(x_2,x_1))\\
&+b_3(b_1x_3,x_2,x_1)
+(-1)^{|x_3|}b_3(x_3,b_1x_2,x_1)\\
&+(-1)^{|x_3|+|x_2|}b_3(x_3,x_2,b_1x_1).
\end{aligned}
\tag{I.2}
$$
因此二元复合的 associator 由 $b_3$ 给出的链同伦控制。

**推论 I.5.** 对非弯曲 $A_\infty$ category $\mathcal A$，$\mu^2$ 在
$H^\ast(\mathcal A)$ 上诱导次数 $0$ 的严格结合复合。

**证明.** 取 $b_1x_i=0$。公式 (I.2) 中最后三项消失，第一项是
$b_1$-boundary，故两个嵌套的 $b_2$ 项在 suspended cohomology 上满足
带 (I.2) 所示符号的结合关系。按固定 desuspension 约定移回
$H^\ast\operatorname{hom}_{\mathcal A}$ 后，这正是次数 $0$ 运算
$\mu^2$ 的普通结合律。若更换 cocycle 代表，(I.1) 表明复合只改变一个
$b_1$-boundary，所以该复合良定义。证毕。

## I.2 dg category 作为特殊情形

**计算 I.6.** 若 $b_d=0$（$d\ge3$），则 (I.2) 只剩
$$
b_2(b_2(x_3,x_2),x_1)
+(-1)^{|x_3|}b_2(x_3,b_2(x_2,x_1))=0.
$$
该 suspended 等式在 desuspension 后就是 dg category 的严格结合律。
同时 (I.1) 给出 dg Leibniz 规则。因此“只保留 $\mu^1,\mu^2$”必须与
附录 B 的 suspension 符号一起使用，不能把上式误读为无符号等式。

## I.3 Curved 情况

**计算 I.7（零输入）.** 对每个对象 $X$，curved 零输入方程恰为
$$
b_1(b_0(X))=0.
\tag{I.3}
$$
它只说明 curvature 对一阶运算闭；单位规范化是另加的严格含单位条件，
不是 (I.3) 的额外求和项。

**计算 I.8（一个输入）.** 对
$x\in s\operatorname{hom}(X_0,X_1)$，方程恰为
$$
b_1b_1(x)+b_2(b_0(X_1),x)
+(-1)^{|x|}b_2(x,b_0(X_0))=0.
\tag{I.4}
$$
一个外部输入时不存在更高 arity 的 curvature insertion。若
$\mu^0_{X_i}=W_i e_{X_i}$，则由 (B.7a)
$b_0(X_i)=-s(W_i e_{X_i})$。按定义 1.8 的 strict-unit laws
desuspend (I.4)，得到
$$
(\mu^1)^2=(W_1-W_0)\operatorname{id}.
\tag{I.5}
$$
所以只有 $W_0=W_1$ 时，$\mu^1$ 才是普通 cochain differential。

## I.4 Maurer--Cartan 变形

**定义 I.9.** 设 $b\in A^1\widehat\otimes\Lambda_{>0}$，写
$\beta=sb$；则 $|\beta|=0$。变形后的 suspended Taylor components 为
$$
b_d^\beta(x_d,\ldots,x_1)
=\sum_{r_0,\ldots,r_d\ge0}
b_{d+r_0+\cdots+r_d}
(\beta^{r_d},x_d,\beta^{r_{d-1}},\ldots,x_1,\beta^{r_0}).
\tag{I.6}
$$
正 valuation 与能量完备性保证该和在每个 valuation 截断下有限。变形
curvature 是
$$
b_0^\beta=\sum_{r\ge0}b_r(\beta^r).
\tag{I.7}
$$
相应 unsuspended curvature 是
$\mu_b^0=-s^{-1}b_0^\beta$。本书把 (I.6)--(I.7) 作为
Maurer--Cartan twisting 的基本公式；若把它展开成 unsuspended
$\mu^r(b,\ldots,b)$，所有符号必须由附录 B 的 graded suspension map
推出，不另写一个无符号公式。

**命题 I.10.** 若 $b_0^\beta=0$，则
$(b_1^\beta)^2=0$，等价地 $(\mu_b^1)^2=0$。

**证明.** 把 (I.6) 代入附录 B 的 curved Stasheff 恒等式。Novikov
完备性允许按 valuation 截断后重排有限和，再取逆极限；所得方程正是
Taylor components $b_d^\beta$ 的 curved 恒等式。对一个输入应用 (I.4)
并使用 $b_0^\beta=0$，得到 $(b_1^\beta)^2=0$。Desuspension 给出结论。
证毕。

**命题 I.11（可逆常数 potential 的准确零化口径）.** 设 $k$ 的特征
不为 $2$，$c\in k^\times$，且 $(V,d)$ 是 $\mathbb Z/2$-分次有限维
$k$-向量空间上的 matrix factorization of the constant $c$，即 $d$ 为奇
映射且 $d^2=c\operatorname{id}_V$。则 $(V,d)$ 在
$H^0\operatorname{MF}(\operatorname{Spec}k,c)$ 中同构于零对象。

**证明.** Matrix-factorization endomorphism complex 的微分为
$$
\delta(f)=d f-(-1)^{|f|}f d.
$$
取奇 endomorphism $h=(2c)^{-1}d$，则
$$
\delta(h)=dh+hd=(2c)^{-1}(d^2+d^2)=\operatorname{id}_V.
$$
所以恒等态射是边界，$(V,d)$ 在同伦范畴中为零。证毕。

**警告 I.12.** 当两个 weak bounding cochains 的 values 相差
$c\ne0$ 时，式 (I.5) 给出的 morphism 数据是 curvature 为 $c$ 的
matrix factorization，而不是一个 $d^2=0$ 的普通 cochain complex。
只有在明确采用上述 matrix-factorization/curved 总模型且 $2c$ 可逆时，
命题 I.11 才给出零化；不同 characteristic 或不同 curved-category
约定必须另行处理。

## 本附录小结

低阶 $A_\infty$ 方程现在都有可复算的 suspended 符号。Curvature 的
一个输入方程只有三项；Maurer--Cartan 元把变形 curvature 消为零后才
得到普通 Floer differential。不同 potential values 的 morphisms 必须按
curved 或 matrix-factorization 对象解释，不能直接称为 cochain complex。
