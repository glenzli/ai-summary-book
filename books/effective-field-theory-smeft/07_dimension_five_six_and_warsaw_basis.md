# 第七章：维数五、维数六与 Warsaw basis

## 本章目标

本章介绍 SMEFT 中最常用的维数六 Warsaw basis 的组织方式，并说明本书采用的范围。

## 依赖前置知识

需要第六章的 SMEFT 定义和第四章的算符基概念。

## 7.1 维数六分类原则

**定义 7.1（baryon-number conserving sector）.** 若所有算符均满足 $\Delta B=0$，则称其属于 baryon-number conserving SMEFT 扇区。

**外部输入 7.2（Warsaw basis 计数）.** 在 baryon number 守恒、不展开 flavor 指标，并从每个非自伴 dagger pair 中只计一个代表（Hermitian conjugate 不另计）的结构口径下，Warsaw basis 的维数六独立算符数为
$$
15+19+25=59,
$$
分别对应纯玻色算符、双费米子算符和四费米子算符。若放开 baryon number 守恒，还出现额外四费米子算符。

**来源.** Grzadkowski、Iskrzynski、Misiak、Rosiek 的 Warsaw basis 原始论文。

## 7.2 结构分块

**定义 7.3（纯玻色类）.** 纯玻色维数六算符只含 $H$、规范场强和协变导数。典型结构包括
$$
X^3,\qquad H^6,\qquad H^4D^2,\qquad X^2H^2,
$$
其中 $X$ 表示 $G_{\mu\nu}$、$W_{\mu\nu}$ 或 $B_{\mu\nu}$。

**定义 7.4（双费米子类）.** 双费米子维数六算符含一个费米子双线性，典型结构包括
$$
\psi^2H^3,\qquad
\psi^2XH,\qquad
\psi^2H^2D.
$$

**定义 7.5（四费米子类）.** 四费米子维数六算符含四个费米子场，典型结构为
$$
(\bar\psi\Gamma\psi)(\bar\psi\Gamma\psi),
$$
并需按 Lorentz、颜色、弱同位旋和 flavor 指标分类。

## 7.3 构造维数六算符的步骤

以 $X^2H^2$ 为例。场强维数为 $2$，Higgs 维数为 $1$，故
$$
[X^2H^2]=2+2+1+1=6.
$$
规范不变性要求规范指标全部收缩。对 $SU(2)_L$ 和 $U(1)_Y$，可形成
$$
H^\dagger H\,W_{\mu\nu}^IW^{I\mu\nu},\qquad
H^\dagger H\,B_{\mu\nu}B^{\mu\nu},
$$
以及混合结构
$$
H^\dagger\tau^I H\,W_{\mu\nu}^IB^{\mu\nu}.
$$
若把一个场强换为 dual field strength，还得到 CP-odd 候选。

**原则 7.6（算符构造检查表）.** 每个候选算符必须检查：

1.  Lorentz 指标是否完全收缩；
2.  $SU(3)_c$、$SU(2)_L$、$U(1)_Y$ 是否为 singlet；
3.  质量维数是否为目标维数；
4.  是否与总导数、Bianchi 恒等式或 EOM 冗余；
5.  是否与其他四费米子结构 Fierz 相关；
6.  Hermitian conjugate 是否独立。

## 7.4 flavor 问题

**警告 7.7（flavor 展开）.** 59 不是实际全 flavor Wilson 参数的数目。引入三代 flavor 后，系数带有 flavor 张量指标，并受 Hermiticity、交换对称性和 flavor 假设影响。完整计数属于第二十章的主题。

**定义 7.8（最小 flavor 假设）.** 常见简化包括 flavor universal、minimal flavor violation、只开第三代、或只开 CP-even 系数。这些是假设，不是 SMEFT 定义的一部分。

## 7.5 Baryon number violation

若不要求 $\Delta B=0$，维数六允许若干四费米子结构导致质子衰变型过程。它们与主线 flavor-conserving collider SMEFT 的实验边界非常不同。

**原则 7.9.** 明确写出“Warsaw basis 59 个算符”时，必须同时说明这是 baryon-number conserving、未展开 flavor，且每个非自伴 dagger pair 只计一个代表的结构计数；Hermitian 拉氏量仍须恢复共轭项。

## 本章小结

Warsaw basis 是 SMEFT 维数六最常用的坐标系统。本书用其组织主线，但始终区分算符结构、flavor 展开和拟合假设。

## 练习

**练习 7.1.** 用质量维数检查 $X^2H^2$、$\psi^2H^3$、$\psi^2XH$ 均为维数六。

**练习 7.2.** 解释为什么 flavor universal 假设会减少 Wilson 参数，但不是由规范对称性推出。

**练习 7.3.** 用超荷检查 ${\cal O}_{eB}=(\bar\ell\sigma^{\mu\nu}e)HB_{\mu\nu}$ 的规范不变性。
