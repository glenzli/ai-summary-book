# 第十八章：一圈匹配的显式环积分例子

## 本章目标

本章给出一个完全书内的 one-loop matching 例子。例子不用标准模型的群论细节，而用一个重实标量场产生的泛函行列式展示：如何从 UV 中的重自由度得到低能局域算符，以及环积分如何决定 Wilson 系数。

## 依赖前置知识

需要第二章的路径积分匹配、第三章的维数正规化、第四章的 EOM 冗余和第十五章的 RGE 语言。

## 18.1 欧氏设置

令 $\phi$ 为轻实标量，$S$ 为质量 $M$ 的重实标量。考虑欧氏作用量中关于 $S$ 的二次部分
$$
S_E[\phi,S]
=S_E^{\rm light}[\phi]
+{1\over 2}\int d^d x\,
S\left(-\partial^2+M^2+U(\phi)\right)S ,
$$
其中 $U(\phi)$ 是由轻场构成的局域函数，质量维数为二。

**定义 18.1（重场一圈阈值作用量）.** 积掉 $S$ 后的重场阈值贡献定义为
$$
\Delta S_E[\phi]
={1\over 2}{\rm Tr}\log\left(-\partial^2+M^2+U\right)
-{1\over 2}{\rm Tr}\log\left(-\partial^2+M^2\right).
$$
第二项只去掉与轻场无关的真空能。

**备注 18.2.** 这一设置等价于 covariant derivative expansion 的最简单无规范场版本。若 $S$ 带规范荷，$\partial$ 必须换成协变导数，且迹还包含规范和内部指标。

## 18.2 行列式展开

记
$$
G_M=(-\partial^2+M^2)^{-1}.
$$
则
$$
\Delta S_E
={1\over 2}{\rm Tr}\log(1+G_MU)
={1\over 2}{\rm Tr}(G_MU)
-{1\over 4}{\rm Tr}(G_MUG_MU)+O(U^3).
$$
一阶项给出 tadpole 型局域项。二阶项决定 $U^2$ 和导数修正。

**命题 18.3（二点核）.** 在维数正规化和 $\overline{\rm MS}$ 记号下，
$$
I(p^2)
\coloneqq
\mu^{4-d}\int {d^d k\over (2\pi)^d}
{1\over (k^2+M^2)((k+p)^2+M^2)}
$$
满足低动量展开
$$
I(p^2)
={1\over 16\pi^2}
\left(
{1\over \bar\epsilon}
-\log {M^2\over \mu^2}
\right)
-{p^2\over 96\pi^2M^2}
+O\left({p^4\over M^4}\right),
$$
其中 $1/\bar\epsilon=1/\epsilon-\gamma_E+\log 4\pi$。

**证明.** 用 Feynman 参数
$$
{1\over AB}=\int_0^1 dx\,{1\over [xA+(1-x)B]^2}
$$
并平移 $q=k+(1-x)p$，得
$$
I(p^2)
=\int_0^1 dx\,\mu^{4-d}
\int {d^d q\over(2\pi)^d}
{1\over [q^2+M^2+x(1-x)p^2]^2}.
$$
标准维数正规化积分给出
$$
\mu^{4-d}
\int {d^d q\over(2\pi)^d}
{1\over(q^2+\Delta)^2}
={1\over16\pi^2}
\left(
{1\over\bar\epsilon}
-\log{\Delta\over\mu^2}
\right)+O(\epsilon).
$$
令 $\Delta=M^2+x(1-x)p^2$，在 $p^2\ll M^2$ 下展开
$$
\log{\Delta\over\mu^2}
=\log{M^2\over\mu^2}
+{x(1-x)p^2\over M^2}
+O(p^4/M^4).
$$
最后使用
$$
\int_0^1 x(1-x)\,dx={1\over6}.
$$
代回即得所述公式。$\square$

**推论 18.4（局域 Wilson 系数）.** 二阶阈值作用量为
$$
\Delta S_E^{(2)}
=-{1\over4}\int {d^d p\over(2\pi)^d}U(p)U(-p)
\left[
{1\over16\pi^2}
\left({1\over\bar\epsilon}-\log{M^2\over\mu^2}\right)
-{p^2\over96\pi^2M^2}
+O\!\left({p^4\over16\pi^2M^4}\right)
\right].
$$
因此对应的局域欧氏拉氏量包含
$$
\Delta{\cal L}_E^{(2)}
=-{1\over64\pi^2}
\left(
{1\over\bar\epsilon}
-\log{M^2\over\mu^2}
\right)U^2
+{1\over384\pi^2M^2}
(\partial_\mu U)(\partial_\mu U)
+O\!\left({(\partial^2U)^2\over16\pi^2M^4}\right).
$$
这里余项仍为 quadratic in $U$；在动量空间它始终由 $U(p)U(-p)O(p^4/(16\pi^2M^4))$ 相乘。行列式展开的 cubic sector 不属于 $\Delta{\cal L}_E^{(2)}$。其零导数领先项另为
$$
\Delta{\cal L}_E^{(3)}
={U^3\over192\pi^2M^2}
+O\!\left({U^2\partial^2U\over16\pi^2M^4}\right),
$$
其中导数余项只表示经分部积分等价的 cubic-in-$U$ 局域结构。

**证明.** 将命题 18.3 代入二阶作用量。Fourier 变换恒等式
$$
\int {d^d p\over(2\pi)^d}p^2U(p)U(-p)
=\int d^dx\,(\partial_\mu U)(\partial_\mu U)
$$
给出导数项系数，$p^4$ 余项则在分部积分后可写为 $(\partial^2U)^2$ 型。cubic sector 来自 $\frac16{\rm Tr}(G_MU)^3$；其零外动量积分为
$$
\int {d^4k\over(2\pi)^4}{1\over(k^2+M^2)^3}
={1\over32\pi^2M^2},
$$
故得到所列 $U^3$ 系数。$\square$

## 18.3 一个低能算符

取
$$
U(\phi)=a\phi^2,
$$
其中 $a$ 为无量纲耦合。推论 18.4 给出
$$
\Delta{\cal L}_E^{(2)}
\supset
{a^2\over384\pi^2M^2}
\left[\partial_\mu(\phi^2)\right]^2.
$$
该项为维数六局域算符，因为 $[\phi]=1$ 且
$$
\left[\partial_\mu(\phi^2)\right]^2=6.
$$
它的 Wilson 系数按预期带有环因子 $1/(16\pi^2)$ 和重尺度抑制 $1/M^2$。

**警告 18.5（方案依赖）.** $U^2$ 项含有反项和重整化方案依赖；导数展开中的局域算符也可被场重定义和 EOM 在不同基之间移动。物理可观测量不依赖于这种基选择。

## 18.4 与 SMEFT 的关系

**原则 18.6.** SMEFT one-loop matching 的结构与本章相同，但实际计算必须额外记录：

1.  重场的 Lorentz 表示和自旋统计；
2.  $SU(3)_c\times SU(2)_L\times U(1)_Y$ 表示；
3.  与轻场的所有规范不变耦合；
4.  规范固定和 ghost 贡献；
5.  匹配尺度与重整化方案；
6.  目标算符基和 EOM 约化规则。

**结论 18.7.** 一圈匹配不是“把树级系数加上环因子”。它是先计算非局域低能振幅或泛函行列式，再按 $p^2/M^2$ 展开为局域算符，并最后投影到选定算符基。

## 本章小结

本章完成了一个可复核的一圈环积分推导。它展示了 EFT 的核心机制：重自由度在低能下留下局域算符，其系数由匹配积分、方案、尺度和基选择共同决定。

## 练习

**练习 18.1.** 重新推导命题 18.3 中的 $p^2$ 系数。

**练习 18.2.** 对 $U=a\phi^2+b\phi^4/M^2$，列出到 $1/M^2$ 为止可能出现的局域项。

**练习 18.3.** 说明为什么 $(\partial_\mu\phi^2)^2$ 可通过分部积分写成含 $\phi^3\Box\phi$ 的形式，并解释这与 EOM 基选择的关系。
