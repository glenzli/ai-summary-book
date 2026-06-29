# 第二十五章：富 Profunctor、Equipment 与 Beck-Chevalley 条件

## 本章目标

本章把第二十四章的 profunctor 推广到富范畴，并引入 equipment：一种同时记录垂直函子、水平 profunctor 和二重胞腔的双范畴结构。Equipment 是 indexed categories、base change、Beck-Chevalley 条件、双模复合和高阶 correspondence 的共同形式。

## 依赖前置知识

需要富范畴、coend、双范畴、profunctor、Cartesian fibration 和 presentable $\infty$-category 的基本语言。

## 25.1 富 profunctor

**定义 25.1.** 设 $\mathcal V$ 为余完备闭对称幺半范畴，且张量积分别保持余极限。若 $\mathcal A,\mathcal B$ 是小 $\mathcal V$-富范畴，则从 $\mathcal A$ 到 $\mathcal B$ 的 $\mathcal V$-profunctor 是 $\mathcal V$-富函子

$$
M:\mathcal A^{op}\otimes\mathcal B\to\mathcal V.
$$

记作

$$
M:\mathcal A\nrightarrow\mathcal B.
$$

**例子 25.2.** 当 $\mathcal V=\mathbf{Set}$ 时，定义 25.1 恢复普通 profunctor。当 $\mathcal V=\mathbf{Ab}$ 时，得到加性 profunctor；当 $\mathcal V=\mathbf{Sp}$ 时，得到谱值 profunctor。

**定义 25.3.** 富恒等 profunctor 为 Hom 对象

$$
\operatorname{id}_{\mathcal A}(a,a')=\mathcal A(a,a').
$$

## 25.2 富 coend 复合

**定义 25.4.** 若

$$
M:\mathcal A\nrightarrow\mathcal B,\qquad
N:\mathcal B\nrightarrow\mathcal C,
$$

则复合 $N\circ M:\mathcal A\nrightarrow\mathcal C$ 定义为富 coend

$$
(N\circ M)(a,c)=
\int^{b\in\mathcal B}M(a,b)\otimes N(b,c).
$$

**外部输入定理 25.5.** 在上述假设下，小 $\mathcal V$-富范畴、$\mathcal V$-profunctors 与富自然变换构成双范畴 $\mathbf{Prof}_{\mathcal V}$。单位律由富 co-Yoneda 给出，结合律由富 coend 的 Fubini 定理给出。

**命题 25.6.** 富恒等 profunctor 对定义 25.4 的复合起单位作用。

**证明.** 左单位在 $(a,b)$ 处为

$$
\int^{a'\in\mathcal A}\mathcal A(a,a')\otimes M(a',b).
$$

由富 co-Yoneda，该对象自然同构于 $M(a,b)$。右单位同理：

$$
\int^{b'\in\mathcal B}M(a,b')\otimes\mathcal B(b',b)\cong M(a,b).
$$

这些同构与富自然变换相容，因此给出单位律。$\square$

## 25.3 Equipment、companion 与 conjoint

**定义 25.7.** 一个 equipment 是双范畴 $\mathbb E$，其中有一类垂直态射和一类水平态射，并且每个垂直态射 $f:A\to B$ 都有 companion $f_*:A\nrightarrow B$ 与 conjoint $f^*:B\nrightarrow A$，满足相应单位/余单位二重胞腔和三角恒等式。

在 $\mathbf{Prof}$ 中，普通函子 $F:\mathcal C\to\mathcal D$ 的 companion 和 conjoint 正是第二十四章的

$$
F_*(c,d)=\mathcal D(Fc,d),\qquad
F^*(d,c)=\mathcal D(d,Fc).
$$

**命题 25.8.** 在 $\mathbf{Prof}_{\mathcal V}$ 中，富函子 $F:\mathcal A\to\mathcal B$ 有 companion 和 conjoint：

$$
F_*(a,b)=\mathcal B(Fa,b),\qquad
F^*(b,a)=\mathcal B(b,Fa).
$$

**证明.** 与命题 24.7 相同。单位由富函子在 Hom 对象上的结构态射

$$
\mathcal A(a,a')\to\mathcal B(Fa,Fa')
$$

给出。余单位由 $\mathcal B$ 的富复合

$$
\mathcal B(b,Fa)\otimes\mathcal B(Fa,b')\to\mathcal B(b,b')
$$

经 coend 的泛性质诱导。三角恒等式化为富范畴的单位律和结合律。$\square$

## 25.4 二重胞腔与 square

**定义 25.9.** 在 equipment 中，一个二重胞腔形如

$$
\begin{matrix}
A&\xRightarrow{M}&B\\
f\downarrow&&\downarrow g\\
A'&\xRightarrow{N}&B'
\end{matrix}
$$

它可理解为从 $M$ 到 $g^*Nf_*$ 或从 $g_*M$ 到 $Nf_*$ 的比较态射，具体方向依赖所采用的 companion/conjoint 约定。

在 $\mathbf{Prof}$ 中，这样的二重胞腔就是自然变换

$$
M(a,b)\to N(fa,gb).
$$

**例子 25.10.** 若有普通自然方块

$$
\begin{matrix}
\mathcal A&\xrightarrow{F}&\mathcal B\\
u\downarrow&&\downarrow v\\
\mathcal A'&\xrightarrow{G}&\mathcal B',
\end{matrix}
$$

以及自然变换 $\alpha:vF\Rightarrow Gu$，则 $\alpha$ 诱导 profunctor 二重胞腔

$$
F_*\to u^*G_*v_*.
$$

它在对象上由态射

$$
\mathcal B(Fa,b)\to\mathcal B'(Gu(a),v(b))
$$

给出，具体为先用 $v$ 作用，再与 $\alpha_a$ 比较。

## 25.5 Exact squares 与 Beck-Chevalley 条件

**定义 25.11.** 在 equipment 中，一个方块称为 exact，若由 companion/conjoint 得到的典范 mate 是同构。对 ordinary indexed categories，这类同构通常称为 Beck-Chevalley isomorphism。

**例子 25.12.** 设有拉回方块

$$
\begin{matrix}
X'&\xrightarrow{g'}&X\\
f'\downarrow&&\downarrow f\\
Y'&\xrightarrow{g}&Y.
\end{matrix}
$$

在集合、sheaf 或合适 topos 的情形中，拉回函子与依赖和/推前之间常有 Beck-Chevalley 比较

$$
g^*f_*\to f'_*g'^*.
$$

当该比较为同构时，称 base change 对该方块成立。

**命题 25.13.** 在 $\mathbf{Set}$ 的 slice 范畴中，若上方方块是集合的拉回方块，则 pullback 与 dependent sum 满足 Beck-Chevalley 同构。

**证明.** 对映射 $p:E\to X$，沿 $f:X\to Y$ 的 dependent sum 是复合 $E\to X\to Y$。先沿 $f$ 求和再沿 $g:Y'\to Y$ 拉回，得到集合

$$
Y'\times_YE.
$$

另一方面，先沿 $g':X'\to X$ 拉回得到

$$
X'\times_XE,
$$

再沿 $f':X'\to Y'$ 求和，即视为 $Y'$ 上对象。由于 $X'\cong Y'\times_YX$，有自然同构

$$
X'\times_XE\cong Y'\times_YE.
$$

该同构与到 $Y'$ 的结构映射相容，因此给出 Beck-Chevalley 同构。$\square$

## 25.6 Indexed categories 与 Cartesian fibration

**定义 25.14.** Indexed category 是伪函子

$$
\mathcal F:B^{op}\to\mathbf{Cat}.
$$

它把 $b\in B$ 送到纤维范畴 $\mathcal F(b)$，把箭头 $\alpha:b\to c$ 送到重索引函子

$$
\alpha^*:\mathcal F(c)\to\mathcal F(b).
$$

**命题 25.15.** 普通 Grothendieck fibration $p:E\to B$ 与 indexed category $B^{op}\to\mathbf{Cat}$ 等价到伪自然等价。

**证明.** 从 indexed category 到 fibration 是第十九章的 ordinary Grothendieck construction。反向地，给定 fibration $p:E\to B$，每个 $b$ 的纤维为 $E_b$。对 $\alpha:b\to c$ 和 $y\in E_c$，选择 Cartesian lift $\alpha^*y\to y$，得到重索引函子 $\alpha^*:E_c\to E_b$。Cartesian lift 的复合唯一性给出 $(\beta\alpha)^*\cong\alpha^*\beta^*$ 的伪函子相干。两种构造互逆到等价。$\square$

**注 25.16.** 第十九章的 Cartesian fibration 与 straightening 是命题 25.15 的 $\infty$-范畴版本。Equipment 语言进一步允许在每个纤维之间使用 profunctors 作为水平态射，而不只使用函子。

## 25.7 高阶 equipment 与 Morita

**外部输入定理 25.17.** 存在多种模型把 equipment、double category 或 framed bicategory 提升到 $(\infty,2)$-范畴或更高结构。小 $\infty$-范畴、functors、correspondences 和二重胞腔可组织成高阶 equipment；稳定 presentable $\infty$-categories、bimodules 和相对张量积给出 Morita 型高阶 equipment。

这些模型用于精确定义高阶 Beck-Chevalley 条件、base change、six functor formalisms 和 extended TFT 中的可对偶性。

## 25.8 本章小结

富 profunctor 把第二十四章的广义态射推广到富环境。Equipment 同时记录垂直函子和水平 profunctor，使 companion、conjoint、base change 与 Beck-Chevalley 条件成为统一语言。它是从 ordinary indexed categories 过渡到 Cartesian fibrations、correspondences 和 Morita $(\infty,2)$-范畴的关键桥梁。

## 练习

**练习 25.1.** 写出 $\mathcal V$-profunctor 的定义。

**练习 25.2.** 在 $\mathcal V=\mathbf{Ab}$ 时解释 $\mathcal V$-profunctor 的含义。

**练习 25.3.** 写出富 profunctor 的复合 coend 公式。

**练习 25.4.** 用富 co-Yoneda 证明恒等富 profunctor 的右单位律。

**练习 25.5.** 定义 companion 与 conjoint。

**练习 25.6.** 对富函子 $F$ 写出 $F_*$ 与 $F^*$。

**练习 25.7.** 在 $\mathbf{Prof}$ 中说明二重胞腔为何是自然变换。

**练习 25.8.** 解释 Beck-Chevalley 条件的直观含义。

**练习 25.9.** 在集合拉回方块中验证命题 25.13 的同构。

**练习 25.10.** 从 indexed category 构造普通 Grothendieck fibration 的对象和态射。

**练习 25.11.** 从普通 Grothendieck fibration 构造重索引函子 $\alpha^*$。

**练习 25.12.** 比较 indexed category 与 Cartesian fibration。

**练习 25.13.** 说明 equipment 比 ordinary bicategory 多记录了什么。

**练习 25.14.** 解释 Morita 理论为什么自然需要 equipment 或 $(\infty,2)$-equipment 语言。
