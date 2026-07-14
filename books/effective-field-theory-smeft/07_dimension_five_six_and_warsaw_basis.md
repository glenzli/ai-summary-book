# 第七章：维数五、维数六与 Warsaw basis

维数五的 Weinberg 算符只有一个结构类型，到了维数六却立刻出现玻色、双费米子和四费米子三大类。若直接背诵“59 个算符”，很容易混淆结构数、Hermitian 共轭项和展开 flavor 后的参数数，也看不出为什么某个候选项会被 EOM 或 Fierz 恒等式删除。Warsaw basis 的价值在于为 baryon-number conserving 的维数六商空间选择一套稳定坐标，使 UV 匹配、RGE 和实验响应可以在同一语言中交流。这里不重复完整目录，而以 $X^2H^2$ 为可操作样例，从质量维数、Lorentz 收缩、规范 singlet、CP 对偶场强到冗余约化逐步构造候选项，再说明 flavor 与重子数假设如何改变后续的 Wilson 空间而不改变 SMEFT 的定义。

## 7.1 维数六分类原则

**定义 7.1（baryon-number conserving sector）.** 若所有算符均满足 $\Delta B=0$，则称其属于 baryon-number conserving SMEFT 扇区。

**外部输入 7.2（Warsaw basis 计数）.** 在 baryon number 守恒、不展开 flavor 指标，并从每个非自伴 dagger pair 中只计一个代表（Hermitian conjugate 不另计）的结构口径下，Warsaw basis 的维数六独立算符数为
$$
15+19+25=59,
$$
分别对应纯玻色算符、双费米子算符和四费米子算符。若放开 baryon number 守恒，还出现额外四费米子算符。

**来源.** Grzadkowski、Iskrzynski、Misiak、Rosiek 的 Warsaw basis 原始论文。

这个总数只说明商空间的维数，还没有给出构造坐标的方法。要从一个 UV 振幅识别具体列，先按场内容把候选项分块，再在每一块内完成规范收缩与冗余约化。

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

**原则 7.6（算符构造条件）.** 每个候选算符必须满足：

1.  Lorentz 指标是否完全收缩；
2.  $SU(3)_c$、$SU(2)_L$、$U(1)_Y$ 是否为 singlet；
3.  质量维数是否为目标维数；
4.  是否与总导数、Bianchi 恒等式或 EOM 冗余；
5.  是否与其他四费米子结构 Fierz 相关；
6.  Hermitian conjugate 是否独立。

## 7.4 flavor 问题

**警告 7.7（flavor 展开）.** 59 不是实际全 flavor Wilson 参数的数目。引入三代 flavor 后，系数带有 flavor 张量指标，并受 Hermiticity、交换对称性和 flavor 假设影响。完整计数属于第二十章的主题。

**定义 7.8（最小 flavor 假设）.** 常见简化包括 flavor universal、minimal flavor violation、只开第三代、或只开 CP-even 系数。这些是假设，不是 SMEFT 定义的一部分。

Flavor 假设改变的是每个结构所携带的系数张量，而不是上面得到的规范不变场结构。类似地，是否保留 baryon-number violating sector 是对 Wilson 空间的选择；一旦放开它，实验接口会从 collider 与电弱数据延伸到质子衰变等完全不同的能区。

## 7.5 Baryon number violation

若不要求 $\Delta B=0$，维数六允许若干四费米子结构导致质子衰变型过程。它们与主线 flavor-conserving collider SMEFT 的实验边界非常不同。

**原则 7.9.** 明确写出“Warsaw basis 59 个算符”时，必须同时说明这是 baryon-number conserving、未展开 flavor，且每个非自伴 dagger pair 只计一个代表的结构计数；Hermitian 拉氏量仍须恢复共轭项。

## 7.6 “59”所计数的对象

$15+19+25=59$ 计的是 baryon-number conserving、未展开 flavor 且每个非自伴 dagger pair 只取一个代表的维数六结构。它不是 Hermitian 拉氏量中书写项的总数，更不是三代 Wilson 实参数数。$X^2H^2$ 的构造说明，Warsaw 坐标来自规范 singlet 候选再对 IBP、EOM、Bianchi 与 Fierz 关系取商；flavor universal、CP-even 或第三代专属等条件则是在这个空间上另取子空间。

## 练习

**练习 7.1.** 用质量维数检查 $X^2H^2$、$\psi^2H^3$、$\psi^2XH$ 均为维数六。

**练习 7.2.** 解释为什么 flavor universal 假设会减少 Wilson 参数，但不是由规范对称性推出。

**练习 7.3.** 用超荷检查 ${\cal O}_{eB}=(\bar\ell\sigma^{\mu\nu}e)HB_{\mu\nu}$ 的规范不变性。
