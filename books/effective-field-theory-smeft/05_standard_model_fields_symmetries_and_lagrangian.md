# 第五章：标准模型的场、对称性与拉氏量

SMEFT 中的“所有高维算符”并不是任意场的组合；它们必须建立在一套固定的低能粒子谱和规范表示上。一个超荷写错的偶极算符不会因质量维数正确而被允许，一个遗漏的轻场也不能靠调整 Wilson 系数补救。因此在进入 Warsaw basis 之前，需要把标准模型的手征费米子、Higgs 双重态和三类规范场放进 $SU(3)_c\times SU(2)_L\times U(1)_Y$ 表示表，并用同一协变导数约定重建重整化拉氏量。电弱破缺随后把 $g,g',v$ 变成 $m_W,m_Z$ 等可测组合，也预示高维算符会同时改动场归一化、质量关系与顶点。本章末的 Weinberg 算符例子则显示，规范量子数如何把轻子数破坏推迟到维数五。

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

## 5.6 SMEFT 的低能起点

$\mathcal L_{\mathrm{SM}}$ 固定了 SMEFT 的传播自由度、规范表示和领先 EOM，也因此固定了后续算符商的类型。电弱破缺把未破缺相参数映到质量本征态，却会被高维项重新修正；从这一刻起，拉氏量参数与实验输入必须分开。重整化拉氏量中偶然保存的 $B$ 与 $L$ 也不是任意高维项的对称性：Weinberg 结构在维数五首先破坏轻子数，预告了 SMEFT 维数展开中的第一个新物理层级。

## 练习

**练习 5.1.** 检查 $\bar q Y_u\widetilde H u$ 的超荷为零。

**练习 5.2.** 说明为什么 Majorana neutrino mass 不能作为维数四 SM 项出现。

**练习 5.3.** 用 $D_\mu H$ 展开推导 $m_W^2=g^2v^2/4$。
