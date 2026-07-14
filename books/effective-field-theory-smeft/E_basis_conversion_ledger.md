# 附录 E：基变换与接口台账

## E.1 基变换原则

算符基是 Wilson 空间的坐标系。两个基之间的变换必须保留 on-shell 可观测量，但可改变：

1.  Wilson 系数坐标；
2.  输入参数位移；
3.  场归一化；
4.  冗余算符的分配；
5.  pseudo-observable 的定义。

**规则 E.1.** 基变换表必须写明源基、目标基、规范化、EOM 使用、输入方案和截断阶数。

## E.2 Warsaw 到破缺相的三个例子

取 unitary gauge 中
$$
H={1\over\sqrt2}\binom{0}{v+h}.
$$

**例 E.2（${\cal O}_{HG}$）.**
$$
{\cal O}_{HG}=H^\dagger H\,G_{\mu\nu}^A G^{A\mu\nu}
={1\over2}(v+h)^2G_{\mu\nu}^A G^{A\mu\nu}.
$$
因此
$$
{C_{HG}\over\Lambda_{\rm ref}^2}{\cal O}_{HG}
\supset
{C_{HG}v\over\Lambda_{\rm ref}^2}hG_{\mu\nu}^AG^{A\mu\nu}
 +{C_{HG}\over2\Lambda_{\rm ref}^2}h^2G_{\mu\nu}^AG^{A\mu\nu}.
$$
这给出 Higgs basis 中的 $hgg$ 和 $hhgg$ contact。

**例 E.3（${\cal O}_{HWB}$）.** 因为
$$
H^\dagger\tau^3H=-{(v+h)^2\over2},
$$
所以
$$
{C_{HWB}\over\Lambda_{\rm ref}^2}{\cal O}_{HWB}
\supset
-{C_{HWB}v^2\over2\Lambda_{\rm ref}^2}W_{\mu\nu}^3B^{\mu\nu}
-{C_{HWB}v\over\Lambda_{\rm ref}^2}hW_{\mu\nu}^3B^{\mu\nu}.
$$
第一项产生中性规范场 kinetic mixing；第二项产生 Higgs-neutral-gauge contact。转到 Higgs basis 前必须先做 kinetic diagonalization。

**例 E.4（${\cal O}_{HD}$）.** 在中性背景下
$$
H^\dagger D_\mu H
\supset {i(v+h)^2\over4}(gW_\mu^3-g'B_\mu),
$$
故
$$
{\cal O}_{HD}
\supset
{(v+h)^4\over16}(gW_\mu^3-g'B_\mu)^2.
$$
该算符修正中性规范玻色子质量矩阵，并进入 $T$ 参数或 Higgs basis 的 neutral-current vertex 重定义。

## E.3 结构级转换台账

| Warsaw 结构 | 常见目标接口 | 必须处理 |
| --- | --- | --- |
| $X^3$ | anomalous triple gauge coupling | normalization、CP convention |
| $X^2H^2$ | Higgs basis contact $hVV$、kinetic terms | field redefinition、input scheme |
| $H^6,H^4D^2$ | Higgs self-coupling、wavefunction、oblique corrections | vev shift、Higgs normalization |
| $\psi^2H^3$ | fermion mass/Yukawa shifts | mass basis、Yukawa diagonalization |
| $\psi^2XH$ | dipole moments、radiative decays | chirality convention、running |
| $\psi^2H^2D$ | gauge-fermion vertex shifts | input scheme、flavor basis |
| four-fermion | contact interactions、LEFT/WET | flavor order、Fierz convention |
| Weinberg operator | neutrino mass matrix | Majorana convention |

## E.4 SMEFT 到 LEFT 的接口

低于电弱尺度时，$W,Z,h,t$ 不再是动力学自由度。匹配形式为
$$
{\cal L}_{\rm SMEFT}(\mu_{\rm EW})
\longrightarrow
{\cal L}_{\rm LEFT}(\mu_{\rm EW})
$$
并随后用低能 RGE 运行。

**例 E.5（charged-current 接口）.** SMEFT 中的
$$
{\cal O}_{H\ell}^{(3)},\quad
{\cal O}_{Hq}^{(3)},\quad
{\cal O}_{\ell q}^{(3)}
$$
会共同进入低能 charged-current 四费米子系数。若只保留其中一个而不说明 flavor 假设，就不能把低能 beta decay 或 meson decay 约束唯一解释为某个高尺度 Wilson 系数。

## E.5 收口状态

本附录给出基变换的内部接口和三个显式破缺相例子。完整 Warsaw-to-Higgs-basis、Warsaw-to-SILH 和 SMEFT-to-LEFT 全表仍属于高级附表工程；正式使用时必须引用具体转换文献或工具版本。
