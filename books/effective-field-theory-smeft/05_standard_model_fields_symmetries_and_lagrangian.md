# 第五章：标准模型的场、对称性与拉氏量

## 本章目标

本章固定 SMEFT 的低能领先项：标准模型场内容、规范群和重整化拉氏量。

## 依赖前置知识

需要规范场论、手征费米子和 Higgs 机制的基本语言。

## 5.1 规范群与场

**定义 5.1（标准模型规范群）.** 本书取
$$
G_{\mathrm{SM}}
=
SU(3)_c\times SU(2)_L\times U(1)_Y.
$$

**定义 5.2（每代费米子表示）.** 每一代标准模型费米子在 $G_{\mathrm{SM}}$ 下的表示为
$$
q:(3,2)_{1/6},\qquad
u:(3,1)_{2/3},\qquad
d:(3,1)_{-1/3},
$$
$$
\ell:(1,2)_{-1/2},\qquad
e:(1,1)_{-1}.
$$
Higgs 场为
$$
H:(1,2)_{1/2}.
$$

**约定 5.3.** 本书把右手场写作四分量 Dirac 记号中的手征投影或等价 Weyl 记号；在算符分类中只记录其规范表示和手征性。

## 5.2 协变导数与场强

对表示 $(R_3,R_2)_Y$ 中的场 $\Psi$，协变导数取为
$$
D_\mu\Psi
=\left[
\partial_\mu
-ig_sG_\mu^AT^A_{R_3}
-igW_\mu^I t^I_{R_2}
-ig'YB_\mu
\right]\Psi.
$$
场强定义为
$$
G_{\mu\nu}^A=\partial_\mu G_\nu^A-\partial_\nu G_\mu^A
+g_sf^{ABC}G_\mu^BG_\nu^C,
$$
$$
W_{\mu\nu}^I=\partial_\mu W_\nu^I-\partial_\nu W_\mu^I
+g\epsilon^{IJK}W_\mu^JW_\nu^K,
\qquad
B_{\mu\nu}=\partial_\mu B_\nu-\partial_\nu B_\mu.
$$
对任意场，
$$
[D_\mu,D_\nu]\Psi
=-i\left(g_sG_{\mu\nu}^AT^A+gW_{\mu\nu}^It^I+g'YB_{\mu\nu}\right)\Psi.
$$
这个恒等式是构造含场强算符的基本来源。

## 5.3 标准模型拉氏量

**定义 5.4（SM 拉氏量）.** 标准模型拉氏量写为
$$
\mathcal L_{\mathrm{SM}}
=
-\frac14G_{\mu\nu}^A G^{A\mu\nu}
-\frac14W_{\mu\nu}^I W^{I\mu\nu}
-\frac14B_{\mu\nu}B^{\mu\nu}
+\sum_\psi i\bar\psi\slashed D\psi
+(D_\mu H)^\dagger D^\mu H
-V(H)
-\mathcal L_Y,
$$
其中
$$
V(H)=-m^2H^\dagger H+\lambda(H^\dagger H)^2,
$$
而 Yukawa 项为
$$
\mathcal L_Y
=
\bar q Y_u \widetilde H u
+\bar q Y_d H d
+\bar \ell Y_e H e
+\mathrm{h.c.}
$$

**说明 5.5.** 上式的符号随 Weyl/Dirac 记号会有转置和共轭差异。本书后续只使用规范不变结构，不依赖某一套分量记号。

## 5.4 电弱破缺

取
$$
H={1\over\sqrt2}\binom{0}{v+h},\qquad
v^2={m^2\over\lambda}.
$$
树级质量关系为
$$
m_W^2={g^2v^2\over4},\qquad
m_Z^2={(g^2+g'^2)v^2\over4},\qquad
m_h^2=2\lambda v^2.
$$
定义
$$
s_W={g'\over\sqrt{g^2+g'^2}},\qquad
c_W={g\over\sqrt{g^2+g'^2}},
$$
则
$$
A_\mu=s_WW_\mu^3+c_WB_\mu,\qquad
Z_\mu=c_WW_\mu^3-s_WB_\mu.
$$
SMEFT 高维算符会修正这些关系；因此后续必须区分拉氏量参数和实验输入参数。

## 5.5 意外对称性

**命题 5.6（重整化 SM 的 accidental baryon and lepton number）.** 在只允许维数不超过四且规范不变的局域算符时，重整化标准模型自动守恒 baryon number 和 lepton number，忽略非微扰 anomaly 效应。

**推导说明.** 枚举维数不超过四的规范不变费米子双线性、Yukawa 项和 Higgs 势。允许项均可赋予一致的 $B$ 和 $L$ 荷。完整 anomaly 讨论属于外部标准模型输入。$\square$

**例 5.7（为什么 Weinberg 算符是第一处 lepton number violation）.** 维数四内没有规范不变的 $\ell\ell$ Majorana mass 项，因为 $\ell\ell$ 的超荷为 $-1$。乘上两个 Higgs 后超荷变为零，并可收缩 $SU(2)_L$ 指标，得到维数五结构。

## 本章小结

SMEFT 的低能领先项是 $\mathcal L_{\mathrm{SM}}$。后续高维算符必须保持 $G_{\mathrm{SM}}$ 规范不变，并按场维数和 flavor 结构分类。

## 练习

**练习 5.1.** 检查 $\bar q Y_u\widetilde H u$ 的超荷为零。

**练习 5.2.** 说明为什么 Majorana neutrino mass 不能作为维数四 SM 项出现。

**练习 5.3.** 用 $D_\mu H$ 展开推导 $m_W^2=g^2v^2/4$。
