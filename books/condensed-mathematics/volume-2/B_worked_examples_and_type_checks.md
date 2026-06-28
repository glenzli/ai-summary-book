# 附录 B：例子与类型检查

## B.0 目标

第二卷的对象很抽象。本附录集中做几个类型检查和例子，确保公式不是只在符号层面成立。

## B.1 有限离散测试对象

设 $S=\{1,\ldots,n\}$ 是有限离散集合。则

$$
\mathbb Z^\square[S]\cong\mathbb Z[\underline S]\cong\underline{\mathbb Z}^{\oplus n}.
$$

因此

$$
\mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T]
$$

退化为普通自由阿贝尔群公式

$$
\mathbb Z^n\otimes\mathbb Z^m\cong\mathbb Z^{nm}.
$$

## B.2 无限 profinite 对象

令 $S=\mathbb Z_p$。它是 profinite 空间。第一卷附录 F 给出

$$
C(S,\mathbb Z)
$$

为自由阿贝尔群，因此存在集合 $I$ 使

$$
\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}.
$$

这里 $I$ 依赖基的选择，不是典范对象。典范的是 $\mathbb Z^\square[S]$ 本身。

## B.3 解析化的类型检查

设 $(A,\mathcal M)$ 是解析环。公式

$$
C\otimes_{(A,\mathcal M)}^LD
=
L_{(A,\mathcal M)}(C\otimes_A^LD)
$$

中：

1. $C,D\in D(A,\mathcal M)$，也可看成 $D(A)$ 中对象。
2. $C\otimes_A^LD$ 先在 $D(A)$ 中形成。
3. 结果不一定解析，因此应用 $L_{(A,\mathcal M)}$。
4. 最终结果回到 $D(A,\mathcal M)$。

这说明解析张量积不是普通张量积的重命名，而是普通张量积后的 localization。

## B.4 solid 环单位

solid 张量的单位是

$$
\mathbb Z^\square,
$$

不是任意选择的 $\underline{\mathbb Z}$。在许多情形下二者容易混淆，因为 $\underline{\mathbb Z}$ 的 solidification 就是 $\mathbb Z^\square$。定义 solid 环时，单位态射应写为

$$
\mathbb Z^\square\to R.
$$

## B.5 $p$-liquid 判别式

若 $V$ 是 $p$-liquid 实向量空间，则对极不连通 $S$ 有

$$
\operatorname{Hom}_{\mathbb R}(\mathcal M_{<p}[S],V)
\cong
V(S).
$$

这个公式说明从 $S$ 到 $V$ 的截面已经自动对 $\mathcal M_{<p}[S]$ 中允许的测度可积分。

## B.6 Rational localization 的类型检查

给定 rational subset

$$
U=U\left(\frac{g_1,\ldots,g_n}{f}\right)
\subset\operatorname{Spa}(A,A^+),
$$

局部化产生 Huber pair

$$
(B,B^+)
$$

和解析环映射

$$
(A,A^+)^\square\to(B,B^+)^\square.
$$

对应的限制函子方向是

$$
D((A,A^+)^\square)\to D((B,B^+)^\square).
$$

也就是从全空间的模限制到开子空间，而不是反方向。

## B.7 投影公式的类型检查

设

$$
f:\operatorname{Spec}A\to\operatorname{Spec}\mathbb Z.
$$

若

$$
M\in D(\mathbb Z^\square),
\qquad
N\in D(A^\square),
$$

则

$$
f^*M=M\otimes_{\mathbb Z^\square}^LA^\square
\in D(A^\square).
$$

所以

$$
f^*M\otimes_{A^\square}^LN
\in D(A^\square),
$$

可以作用 $f_!$。右侧

$$
M\otimes_{\mathbb Z^\square}^Lf_!N
$$

也在 $D(\mathbb Z^\square)$ 中。因此投影公式两边类型一致。

## B.8 第二卷中不能误读的地方

1. solidification 不是 $p$-进完备化，但在 $p$-进例子中会相互作用。
2. liquid 不是 Banach 空间范畴的改名。
3. analytic ring 不是普通凝聚环加一个形容词，而是包含测度理论 $\mathcal M$ 的结构。
4. $f_!$ 不是普通 forgetful functor，而是带紧支撑和边界控制的推前。
5. 复几何定理在第二卷只是目标语言，证明属于第三卷。

## 练习

**练习 B.1.** 对有限集合 $S,T$，直接写出 $\mathbb Z[S]\otimes\mathbb Z[T]\cong\mathbb Z[S\times T]$ 的基对应。

**练习 B.2.** 解释为什么 $\prod_I\mathbb Z\otimes\prod_J\mathbb Z$ 在普通阿贝尔群中不应直接化简为 $\prod_{I\times J}\mathbb Z$。

**练习 B.3.** 对解析化公式做一次类型检查。

**练习 B.4.** 写出 $f_!$ 投影公式两边所在的范畴。
