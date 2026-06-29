# 附录 C：Stein、Cech 与相干分解

## C.0 目标

第三卷主体多次使用“局部到整体”的证明路线。本附录补充经典复几何中的三种基本工具：Stein 开集、Cech 复形和相干层的局部分解，并说明它们如何进入 condensed/analytic 表述。

## C.1 Stein 开集

**定义 C.1.** 复解析空间 $U$ 称为 Stein，如果它满足全纯凸性、全纯函数分离点，并有足够多全纯函数给出局部坐标。对本书而言，最重要的是 Stein 空间上的相干层上同调消失。

**输入定理 C.2（Cartan 定理 B）.** 若 $U$ 是 Stein 空间，$\mathcal F$ 是相干解析层，则

$$
H^i(U,\mathcal F)=0,\qquad i>0.
$$

**输入定理 C.3（Cartan 定理 A）.** 若 $U$ 是 Stein 空间，$\mathcal F$ 是相干解析层，则 $\mathcal F$ 由全局截面生成。

**使用说明.** 第三卷不重证 Cartan A/B；它们是复几何中的基础输入。condensed/analytic 方法需要把这些定理产生的函数空间放入 liquid/analytic 范畴。

## C.2 Cech 复形

设 $\mathfrak U=\{U_i\}_{i\in I}$ 是 $X$ 的开覆盖。记

$$
U_{i_0\cdots i_p}=U_{i_0}\cap\cdots\cap U_{i_p}.
$$

对 sheaf $\mathcal F$，Cech $p$-cochains 为

$$
C^p(\mathfrak U,\mathcal F)=
\prod_{i_0<\cdots<i_p}\mathcal F(U_{i_0\cdots i_p}).
$$

微分定义为

$$
(dc)_{i_0\cdots i_{p+1}}
=
\sum_{k=0}^{p+1}(-1)^k
c_{i_0\cdots\widehat{i_k}\cdots i_{p+1}}
|_{U_{i_0\cdots i_{p+1}}}.
$$

**命题 C.4.** $d^2=0$。

**证明.** 对 $c\in C^p$，$(d^2c)_{i_0\cdots i_{p+2}}$ 是双和

$$
\sum_{a<b}(-1)^{a+b}
c_{i_0\cdots\widehat{i_a}\cdots\widehat{i_b}\cdots i_{p+2}}

+
\sum_{b<a}(-1)^{a+b-1}
c_{i_0\cdots\widehat{i_b}\cdots\widehat{i_a}\cdots i_{p+2}}.
$$

同一个删去两个指标的项出现两次，符号相反，因此相消。证毕。

## C.3 Acyclic 覆盖

**定义 C.5.** 开覆盖 $\mathfrak U$ 对 $\mathcal F$ 称为 acyclic，如果所有有限交 $U_{i_0\cdots i_p}$ 上都有

$$
H^q(U_{i_0\cdots i_p},\mathcal F)=0,\qquad q>0.
$$

**命题 C.6.** 若 $\mathfrak U$ 对 $\mathcal F$ acyclic，则 Cech 复形计算 sheaf cohomology：

$$
H^n(C^\bullet(\mathfrak U,\mathcal F))
\cong
H^n(X,\mathcal F).
$$

**证明.** 附录 I 定理 I.2 给出有限覆盖情形的 Cech-to-derived spectral sequence

$$
E_1^{p,q}=
\prod_{i_0<\cdots<i_p}H^q(U_{i_0\cdots i_p},\mathcal F)
\Rightarrow
H^{p+q}(X,\mathcal F).
$$

acyclic 条件使 $q>0$ 行消失。$q=0$ 行正是 Cech 复形 $C^\bullet(\mathfrak U,\mathcal F)$，而 $d_1$ 是 Cech 微分。于是

$$
E_2^{p,0}=H^p(C^\bullet(\mathfrak U,\mathcal F)),
\qquad
E_2^{p,q}=0\ (q>0).
$$

第一象限中没有非零高阶微分能离开或进入唯一非零行，故 $E_2=E_\infty$。总次数 $n$ 的过滤只有一个非零分级片，因此

$$
H^n(C^\bullet(\mathfrak U,\mathcal F))
\cong
H^n(X,\mathcal F).
$$

证毕。

## C.4 Stein 覆盖的作用

**输入定理 C.8（Stein Leray 覆盖）.** 对紧复流形 $X$ 和相干解析层 $\mathcal F$，可取有限开覆盖 $\mathfrak U=\{U_i\}$，使每个有限交

$$
U_{i_0\cdots i_p}
$$

是 Stein，或者至少对 $\mathcal F$ 是 acyclic：

$$
H^q(U_{i_0\cdots i_p},\mathcal F)=0,\qquad q>0.
$$

在该输入下，Cartan B 给出有限交上的高上同调消失，因此 Cech 复形可用于计算 $R\Gamma(X,\mathcal F)$。

condensed/analytic 版本中，每个

$$
\mathcal F(U_{i_0\cdots i_p})
$$

不仅是向量空间，还带有自然拓扑或 liquid 结构。

**警告 C.9.** Cech 复形能计算 $R\Gamma(X,\mathcal F)$，但不自动证明 $H^i(X,\mathcal F)$ 有限维。即使 $U$ 是 Stein，$\mathcal O(U)$ 也可能是无限维复向量空间；单位圆盘就是基本例子。有限性需要 Grauert、Fredholm-Hodge 或 Clausen-Scholze 的有限性输入；形式传播机制见附录 M。

## C.5 相干层的局部分解

相干层局部有有限表示

$$
\mathcal O_U^m\to\mathcal O_U^n\to\mathcal F|_U\to0.
$$

因此很多证明可先对 $\mathcal O_U$ 或有限自由层处理，再通过 exact sequence 推广到一般相干层。

**命题 C.10.** 若某性质 $P(\mathcal F)$ 对短正合列满足 two-out-of-three，并且对 $\mathcal O_U$ 成立，则在局部有限分解允许的范围内，$P$ 可推广到相干层。

**证明.** 使用有限表示给出的 exact sequence。若 $P$ 对 $\mathcal O_U^m$ 和 $\mathcal O_U^n$ 成立，则由 two-out-of-three 推得对 cokernel $\mathcal F$ 成立。对更长分解用归纳。证毕。

## C.6 教材级证明边界

本附录完整写出了 Cech 复形和 acyclic 覆盖的代数部分。Cartan A/B、Stein 覆盖存在性和复解析空间的深层局部理论作为经典复几何输入使用。

## 练习

**练习 C.1.** 直接验证 Cech 微分满足 $d^2=0$。

**练习 C.2.** 说明为什么 Stein 覆盖对相干层是 acyclic 覆盖。

**练习 C.3.** 用两项自由分解说明如何把一个对自由层成立的有限性命题推广到相干层。
