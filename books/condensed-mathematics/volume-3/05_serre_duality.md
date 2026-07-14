# 第五章：从积分配对到 Serre 对偶

有限维性使对偶向量空间可控，却没有说明应该把 $H^i(X,E)$ 与哪个群配对。对复维数
$n$ 的紧复流形，$(0,i)$ 型 $E$-值形式与
$(0,n-i)$ 型 $E^\vee\otimes\omega_X$-值形式楔积后成为顶次 $(n,n)$-形式，可以积分。
要让这个公式定义上同调配对，必须验证 $\bar\partial$-边界积分为零并处理符号；要让
配对成为完美对偶，则需要 Hodge/Serre 的深层非退化性。两部分的证明责任不同。

本章先在链级完成符号与良定义性，再把 Serre perfectness 保留为精确外部输入，并完整
推出 derived duality、有限 resolution 下的 Ext 形式以及
$f_!\dashv f^!$ 的内部 Hom 公式。$\mathbb P^1$ 上的留数计算给出一张显式配对矩阵，
使抽象 trace 可以逐项核验。

## 5.1 Dolbeault 链级配对

设 $X$ 是无边界紧复流形，$\dim_\mathbb C X=n$，$E$ 是全纯向量丛，且

$$
\omega_X=\Omega_X^n
$$

为 canonical bundle。置

$$
C^i=\Gamma(X,\mathcal A_X^{0,i}(E)),
\qquad
D^{n-i}
=
\Gamma(X,\mathcal A_X^{0,n-i}(E^\vee\otimes\omega_X)).
$$

evaluation 与楔积给出

$$
\langle\alpha,\beta\rangle
=
\int_X\operatorname{ev}(\alpha\wedge\beta).
$$

**命题 5.1（链级相容）.** 若 $\alpha\in C^i$、$\beta\in D^{n-i-1}$，则

$$
\langle\bar\partial\alpha,\beta\rangle
=
(-1)^{i+1}
\langle\alpha,\bar\partial\beta\rangle.
$$

**证明.** Leibniz 规则给出

$$
\bar\partial\operatorname{ev}(\alpha\wedge\beta)
=
\operatorname{ev}(\bar\partial\alpha\wedge\beta)
+
(-1)^i
\operatorname{ev}(\alpha\wedge\bar\partial\beta).
$$

左侧是一个 $\bar\partial$-exact 顶次项。把它看作 de Rham 微分的相应分量后，Stokes
定理和 $\partial X=\varnothing$ 给

$$
\int_X\bar\partial
\operatorname{ev}(\alpha\wedge\beta)=0.
$$

移项即得公式。证毕。

**推论 5.2.** 积分诱导良定义配对

$$
H^i(X,E)
\times
H^{n-i}(X,E^\vee\otimes\omega_X)
\longrightarrow\mathbb C.
$$

**证明.** 第三章把两群识别为 Dolbeault cohomology。若
$\alpha$ 改成 $\alpha+\bar\partial a$ 且 $\beta$ 为 cocycle，则命题 5.1 给

$$
\langle\bar\partial a,\beta\rangle
=
\pm\langle a,\bar\partial\beta\rangle=0.
$$

若 $\beta$ 改成 $\beta+\bar\partial b$ 且 $\alpha$ 为 cocycle，再用命题 5.1 把
$\bar\partial$ 从 $b$ 移到 $\alpha$，便有

$$
\langle\alpha,\bar\partial b\rangle
=\pm\langle\bar\partial\alpha,b\rangle=0.
$$

故配对只依赖两个上同调类。证毕。

## 5.2 Perfectness 与 derived duality

对有限维有界复形 $D^\bullet$，定义线性对偶复形

$$
(D^\vee)^k=\operatorname{Hom}_\mathbb C(D^{-k},\mathbb C),
\qquad
d_{D^\vee}(\varphi)=(-1)^{k+1}\varphi\circ d_D.
$$

命题 5.1 等价于一个复形态射

$$
\Phi:C^\bullet\longrightarrow(D^\bullet)^\vee[-n],
\qquad
\Phi(\alpha)(\beta)=\langle\alpha,\beta\rangle.
$$

**外部输入定理 5.3（向量丛 Serre perfectness）.** 上述上同调配对对每个 $i$ 都是
完美配对。其经典证明可由 Hodge star 与椭圆正则性完成；凝聚/解析版本由
Clausen--Scholze 相干对偶输入给出。附录 AA 展开 Hodge 证明接口，附录 J 登记链级到
导出的完整形式。

**定理 5.4.** 接受输入定理 5.3 后，有自然等价

$$
R\Gamma(X,E)
\simeq
R\operatorname{Hom}_\mathbb C
\bigl(
R\Gamma(X,E^\vee\otimes\omega_X),\mathbb C
\bigr)[-n].
$$

**证明.** 第三章定理 3.8 把两侧括号内的导出全局截面分别表示为
$C^\bullet$ 和 $D^\bullet$。第四章有限性说明其 cohomology 有限维；可取有限维
perfect 代表。态射 $\Phi$ 在第 $i$ 个 cohomology 上正是

$$
H^i(X,E)
\longrightarrow
H^{n-i}(X,E^\vee\otimes\omega_X)^\vee.
$$

输入定理 5.3 说该映射对所有 $i$ 都是同构，所以 $\Phi$ 是 quasi-isomorphism。证毕。

深层输入只负责“配对非退化”；从链级配对到 derived 等价的每一步已在命题 5.1、推论
5.2 与定理 5.4 中完成。

## 5.3 从向量丛到相干层的 Ext 形式

先考虑有全局有限局部自由 resolution 的相干层
$\mathcal F\simeq E^\bullet$。因为有限秩局部自由层对第一变量的
$\mathcal Hom$-导出无须再解析，

$$
R\mathcal Hom_X(\mathcal F,\omega_X)
\simeq
\mathcal Hom_X(E^\bullet,\omega_X).
$$

**定理 5.5（有限 resolution 化约）.** 假设输入定理 5.3 对 resolution 中的向量丛
成立，则

$$
H^i(X,\mathcal F)^\vee
\cong
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X).
$$

**证明.** 定理 5.4 对单项向量丛复形成立。令 $\mathcal T$ 为所有使相应 derived
duality 成立的有界向量丛复形。$R\Gamma$、$R\mathcal Hom$ 和有限维线性对偶都把
distinguished triangle 送到 distinguished triangle，故 $\mathcal T$ 对 shift、有限
直和和 cone 封闭。任一有界 $E^\bullet$ 可由 stupid truncation 从单项复形经有限次
cone 构造，所以

$$
R\Gamma(X,E^\bullet)
\simeq
R\operatorname{Hom}_\mathbb C
\left(
R\Gamma(X,\mathcal Hom(E^\bullet,\omega_X)),
\mathbb C
\right)[-n].
$$

以 $E^\bullet\simeq\mathcal F$ 和上面的派生 Hom 模型代入，取第 $i$ 个
cohomology，得到

$$
H^i(X,\mathcal F)
\cong
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X)^\vee.
$$

第四章有限性保证两群有限维，再取双对偶即得所述同构。证毕。

一般相干层未必有全局有限向量丛 resolution，奇异空间上也应把 $\omega_X[n]$ 替换为
dualizing complex。

**外部输入定理 5.6（Grothendieck--Serre duality）.** 对 proper 复解析空间 $X$，存在
$\omega_X^\bullet\in D^b_{\operatorname{coh}}(X)$，使

$$
R\operatorname{Hom}_\mathbb C(R\Gamma(X,\mathcal F),\mathbb C)
\simeq
R\Gamma\bigl(
X,R\mathcal Hom_X(\mathcal F,\omega_X^\bullet)
\bigr).
$$

若 $X$ 是光滑 $n$ 维复流形，则
$\omega_X^\bullet\simeq\omega_X[n]$。结合第四章的 Grauert 有限性，取 cohomology
便给出任意相干 $\mathcal F$ 的 Ext--Serre 完美配对。附录 AD 保留奇异空间与
dualizing complex 的精确版本。

## 5.4 Trace 是伴随的 counit

令 $f:X\to *$。在闭对称幺半稳定范畴中，假设有

$$
f^*\dashv f_*,
\qquad
f_!\dashv f^!,
$$

以及投影公式

$$
f_!(f^*A\otimes F)\simeq A\otimes f_!F.
$$

trace 定义为第二个伴随的 counit

$$
\operatorname{Tr}_f:
f_!f^!\mathbf1\longrightarrow\mathbf1.
$$

**命题 5.7（六函子 Hom 公式）.** 对 $F\in D(X)$、$B\in D(*)$，有自然等价

$$
f_*R\mathcal Hom_X(F,f^!B)
\simeq
R\mathcal Hom_*(f_!F,B).
$$

**证明.** 对任意 $A\in D(*)$，连续使用两个伴随、闭结构与投影公式：

$$
\begin{aligned}
\operatorname{Map}\bigl(
A,f_*R\mathcal Hom(F,f^!B)
\bigr)
&\simeq
\operatorname{Map}\bigl(
f^*A,R\mathcal Hom(F,f^!B)
\bigr)\\
&\simeq
\operatorname{Map}(f^*A\otimes F,f^!B)\\
&\simeq
\operatorname{Map}(f_!(f^*A\otimes F),B)\\
&\simeq
\operatorname{Map}(A\otimes f_!F,B)\\
&\simeq
\operatorname{Map}\bigl(
A,R\mathcal Hom(f_!F,B)
\bigr).
\end{aligned}
$$

等价关于 $A$ 自然，由 Yoneda 引理得到结论。证毕。

当 $X$ proper 时，比较 $f_!\simeq f_*$，并取
$B=\mathbf1$、$f^!\mathbf1\simeq\omega_X[n]$，命题 5.7 正好给出输入定理 5.6 的
形式。这里 $f^!\mathbf1$ 的几何识别是深层对偶输入；从伴随和投影公式推出 Hom 等价
则已完整证明。

## 5.5 Worked example：$\mathbb P^1$ 上的留数矩阵

取 $d\ge0$。第四章给出

$$
H^0(\mathbb P^1,\mathcal O(d))
=
\bigoplus_{a=0}^d\mathbb C z^a
$$

以及

$$
H^1(\mathbb P^1,\mathcal O(-d-2))
=
\bigoplus_{a=0}^d\mathbb C z^{-a-1}.
$$

因 $\omega_{\mathbb P^1}\simeq\mathcal O(-2)$，乘法后使用 trace

$$
\operatorname{Tr}[g(z)\,dz]
=
\operatorname{Res}_{z=0}g(z)\,dz
$$

得到配对。对基向量，

$$
\operatorname{Res}_{z=0}
z^a z^{-b-1}\,dz
=
\delta_{ab}.
$$

所以配对矩阵为单位矩阵，非退化性不需另引 Hodge 理论。输入是整数 $d$ 与两组 Čech
基，步骤是相乘并取 $z^{-1}$ 系数，输出是显式 perfect pairing。若换成非 proper 曲线，
Stokes 边界项和紧支条件会出现，此时应使用 $f_!$ 而非直接把 $f_*$ 当作紧支撑推前。
高维射影空间的单项式配对见附录 T。

## 5.6 对偶之后的比较问题

Serre 对偶把一个解析对象与其 dualizing transform 联系起来，却仍在同一解析空间内。
第六章改变问题：给定 proper 代数簇 $X$，解析化 $X^{an}$ 是否产生同一个相干范畴与
同一个 $R\Gamma$？答案是 GAGA；其深层等价保留为输入，但 exact 等价到 derived
等价、上同调比较到 Euler characteristic 比较的推导将在正文完成。

## 练习

**练习 5.1.** 逐次检查命题 5.1 的符号，说明 $\alpha$ 的总次数为何产生
$(-1)^i$。

**练习 5.2.** 对 $d=2$ 写出例 5.5 的两组基和完整 $3\times3$ 配对矩阵。

**练习 5.3.** 在命题 5.7 的证明中标出每一行分别使用
$f^*\dashv f_*$、tensor--Hom 伴随、$f_!\dashv f^!$ 或投影公式。
