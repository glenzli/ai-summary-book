# 附录 F：基变换细表

## F.1 破缺相旋转

本附录使用
$$
A_\mu=s_WW_\mu^3+c_WB_\mu,\qquad
Z_\mu=c_WW_\mu^3-s_WB_\mu,
$$
故
$$
W_\mu^3=c_WZ_\mu+s_WA_\mu,\qquad
B_\mu=c_WA_\mu-s_WZ_\mu.
$$
场强在双线性阶满足同样旋转。

## F.2 $X^2H^2$ 到 Higgs-gauge contact

取 $H=(0,(v+h)/\sqrt2)^T$。只列含一个 Higgs 的 neutral contact。

| Warsaw 算符 | 破缺相含 $h$ 的项 | 物理接口 |
| --- | --- | --- |
| ${\cal O}_{HG}$ | $v h\,G_{\mu\nu}^AG^{A\mu\nu}$ | $hgg$ |
| ${\cal O}_{H\widetilde G}$ | $v h\,\widetilde G_{\mu\nu}^AG^{A\mu\nu}$ | CP-odd $hgg$ |
| ${\cal O}_{HB}$ | $v h(c_WA_{\mu\nu}-s_WZ_{\mu\nu})^2$ | $h\gamma\gamma,hZ\gamma,hZZ$ |
| ${\cal O}_{HW}$ | $v h[2W^+_{\mu\nu}W^{-\mu\nu}+(s_WA_{\mu\nu}+c_WZ_{\mu\nu})^2]$ | $hWW,h\gamma\gamma,hZ\gamma,hZZ$ |
| ${\cal O}_{HWB}$ | $-v h(s_WA_{\mu\nu}+c_WZ_{\mu\nu})(c_WA^{\mu\nu}-s_WZ^{\mu\nu})$ | mixed neutral contacts |

因此 neutral photon contact 的 symbolic 组合为
$$
c_{h\gamma\gamma}
\propto
c_W^2C_{HB}+s_W^2C_{HW}-s_Wc_WC_{HWB},
$$
而 $hZ\gamma$ contact 的 symbolic 组合为
$$
c_{hZ\gamma}
\propto
2s_Wc_W(C_{HW}-C_{HB})
-(c_W^2-s_W^2)C_{HWB}.
$$
比例系数依 Higgs-basis 规范化约定而定。

## F.3 Yukawa-like 算符

以 down-type 为例：
$$
{\cal O}_{dH}^{pr}
=(H^\dagger H)(\bar q_p d_rH).
$$
破缺后
$$
{\cal O}_{dH}^{pr}
=
{(v+h)^3\over2\sqrt2}\bar d_{Lp}d_{Rr}.
$$
因此它同时修正质量项和 Higgs Yukawa：
$$
{(v+h)^3\over2\sqrt2}
={v^3\over2\sqrt2}
 +{3v^2h\over2\sqrt2}
 +{3vh^2\over2\sqrt2}
 +{h^3\over2\sqrt2}.
$$
质量修正与单 Higgs Yukawa 修正的相对系数不同，这就是 SMEFT 中 mass diagonalization 与 Higgs coupling shift 不能分开的原因。

## F.4 Neutral dipole

对 charged lepton dipole，
$$
{\cal O}_{eB}^{pr}=(\bar\ell_p\sigma^{\mu\nu}e_r)HB_{\mu\nu},
$$
$$
{\cal O}_{eW}^{pr}=(\bar\ell_p\sigma^{\mu\nu}e_r)\tau^IH W_{\mu\nu}^I.
$$
取 Higgs 下分量后，neutral 部分为
$$
{v+h\over\sqrt2}\bar e_{Lp}\sigma^{\mu\nu}e_{Rr}
\left[
C_{eB}^{pr}B_{\mu\nu}
-C_{eW}^{pr}W_{\mu\nu}^3
\right].
$$
于是 photon 和 $Z$ dipole 组合为
$$
C_{e\gamma}^{pr}=c_WC_{eB}^{pr}-s_WC_{eW}^{pr},
$$
$$
C_{eZ}^{pr}=-s_WC_{eB}^{pr}-c_WC_{eW}^{pr}.
$$
这些组合进入 $\ell_i\to\ell_j\gamma$、$(g-2)_\ell$ 和 EDM。

## F.5 Current 算符到顶点修正

| Warsaw 算符族 | 破缺相效果 | 典型观测量 |
| --- | --- | --- |
| ${\cal O}_{H\ell}^{(1)}$ | $Z\ell\ell$ left-handed vertex shift | LEP/SLC, LFU |
| ${\cal O}_{H\ell}^{(3)}$ | $W\ell\nu$ 与 $Z\ell\ell$ shift | $G_F$, beta decay |
| ${\cal O}_{He}$ | right-handed charged lepton $Z$ shift | asymmetries |
| ${\cal O}_{Hq}^{(1,3)}$ | quark $Z/W$ vertex shift | EWPO, top, flavor |
| ${\cal O}_{Hu},{\cal O}_{Hd}$ | right-handed quark $Z$ shift | Z-pole, LHC tails |
| ${\cal O}_{Hud}$ | right-handed charged current | beta decay, top |

## F.6 使用边界

本附录给出破缺相和常用接口组合，但不是完整 Higgs basis 定义。完整基变换还需要：

1.  kinetic normalization；
2.  输入参数重定义；
3.  mass diagonalization；
4.  EOM 选择；
5.  flavor basis；
6.  目标基的规范化约定。

