# 第十三章：Warsaw basis 的维数六算符表

实际匹配得到一个颜色、弱同位旋和 flavor 张量后，必须能在一套明确的独立算符中找到它的坐标。Warsaw basis 提供的正是这种索引系统，但“59”只有在 baryon number 守恒、不展开 flavor，且每个非自伴 dagger pair 只计一个代表时才成立。这里按 $X^3$、$H^6$、$H^4D^2$、$X^2H^2$、双费米子和四费米子依次列出维数六结构，并始终显示规范指标与 flavor 指标的收缩方式。读表时应把它与前面的商空间定义合用：目录中的对象是 IBP、领先 EOM、Bianchi 与四维 Fierz 约化后的代表，不是任意局域单项式；用于 Hermitian 拉氏量时，非自伴项还必须连同共轭系数和 dagger 一起恢复。

## 13.1 记号

**约定 13.1.** 本章用 $X$ 表示规范场强，$H$ 表示 Higgs 双重态，$\psi$ 表示标准模型费米子。Flavor 指标记作 $p,r,s,t$。Pauli 矩阵为 $\tau^I$，颜色生成元为 $T^A$。

**约定 13.2（双向协变导数）.** 定义
$$
H^\dagger i\overleftrightarrow D_\mu H
\coloneqq
iH^\dagger D_\mu H-i(D_\mu H)^\dagger H,
$$
$$
H^\dagger i\overleftrightarrow D_\mu^I H
\coloneqq
iH^\dagger\tau^I D_\mu H-i(D_\mu H)^\dagger\tau^I H.
$$

**约定 13.3（Hermitian conjugates）.** 本章列表对非 Hermitian 算符只列一个 chirality/flavor 方向；拉氏量中应加入 Hermitian conjugate，Wilson 系数与其共轭按 flavor 指标相应配对。Hermitian 算符的 Wilson 系数矩阵另满足自身的 Hermiticity 条件。

在这些约定下，目录先从没有 flavor 指标的纯玻色 sector 开始。它最直接地显示 CP-even 场强平方与含 dual field strength 的 CP-odd 候选如何成对出现。

## 13.2 纯玻色算符：15 个

**列表 13.3（$X^3$ 类）.**
$$
\mathcal O_G=f^{ABC}G_\mu^{A\nu}G_\nu^{B\rho}G_\rho^{C\mu},
\qquad
\mathcal O_{\widetilde G}=f^{ABC}\widetilde G_\mu^{A\nu}G_\nu^{B\rho}G_\rho^{C\mu},
$$
$$
\mathcal O_W=\epsilon^{IJK}W_\mu^{I\nu}W_\nu^{J\rho}W_\rho^{K\mu},
\qquad
\mathcal O_{\widetilde W}=\epsilon^{IJK}\widetilde W_\mu^{I\nu}W_\nu^{J\rho}W_\rho^{K\mu}.
$$

**列表 13.4（$H^6$ 与 $H^4D^2$ 类）.**
$$
\mathcal O_H=(H^\dagger H)^3,
$$
$$
\mathcal O_{H\Box}=(H^\dagger H)\Box(H^\dagger H),
\qquad
\mathcal O_{HD}=(H^\dagger D_\mu H)^\ast(H^\dagger D^\mu H).
$$

**列表 13.5（$X^2H^2$ 类）.**
$$
\mathcal O_{HG}=H^\dagger H\,G_{\mu\nu}^A G^{A\mu\nu},
\qquad
\mathcal O_{H\widetilde G}=H^\dagger H\,\widetilde G_{\mu\nu}^A G^{A\mu\nu},
$$
$$
\mathcal O_{HW}=H^\dagger H\,W_{\mu\nu}^I W^{I\mu\nu},
\qquad
\mathcal O_{H\widetilde W}=H^\dagger H\,\widetilde W_{\mu\nu}^I W^{I\mu\nu},
$$
$$
\mathcal O_{HB}=H^\dagger H\,B_{\mu\nu}B^{\mu\nu},
\qquad
\mathcal O_{H\widetilde B}=H^\dagger H\,\widetilde B_{\mu\nu}B^{\mu\nu},
$$
$$
\mathcal O_{HWB}=H^\dagger\tau^I H\,W_{\mu\nu}^I B^{\mu\nu},
\qquad
\mathcal O_{H\widetilde WB}=H^\dagger\tau^I H\,\widetilde W_{\mu\nu}^I B^{\mu\nu}.
$$

这十五个纯玻色结构在 dagger 下自伴。加入一个费米子双线性后，chirality 与 flavor 方向开始重要：Yukawa-like 和 dipole 算符通常不是自伴项，current 类则多在交换 flavor 指标后回到自身。

## 13.3 双费米子算符：19 个结构

**列表 13.6（$\psi^2H^3$ 类）.**
$$
\mathcal O_{eH}^{pr}=(H^\dagger H)(\bar\ell_p e_r H),
$$
$$
\mathcal O_{uH}^{pr}=(H^\dagger H)(\bar q_p u_r\widetilde H),
\qquad
\mathcal O_{dH}^{pr}=(H^\dagger H)(\bar q_p d_r H).
$$

**列表 13.7（$\psi^2XH$ dipole 类）.**
$$
\mathcal O_{eB}^{pr}=(\bar\ell_p\sigma^{\mu\nu}e_r)H B_{\mu\nu},
\qquad
\mathcal O_{eW}^{pr}=(\bar\ell_p\sigma^{\mu\nu}e_r)\tau^I H W_{\mu\nu}^I,
$$
$$
\mathcal O_{uG}^{pr}=(\bar q_p\sigma^{\mu\nu}T^A u_r)\widetilde H G_{\mu\nu}^A,
\quad
\mathcal O_{uW}^{pr}=(\bar q_p\sigma^{\mu\nu}u_r)\tau^I\widetilde H W_{\mu\nu}^I,
\quad
\mathcal O_{uB}^{pr}=(\bar q_p\sigma^{\mu\nu}u_r)\widetilde H B_{\mu\nu},
$$
$$
\mathcal O_{dG}^{pr}=(\bar q_p\sigma^{\mu\nu}T^A d_r)H G_{\mu\nu}^A,
\quad
\mathcal O_{dW}^{pr}=(\bar q_p\sigma^{\mu\nu}d_r)\tau^I H W_{\mu\nu}^I,
\quad
\mathcal O_{dB}^{pr}=(\bar q_p\sigma^{\mu\nu}d_r)H B_{\mu\nu}.
$$

**列表 13.8（$\psi^2H^2D$ current 类）.**
$$
\mathcal O_{H\ell}^{(1)pr}
=(H^\dagger i\overleftrightarrow D_\mu H)(\bar\ell_p\gamma^\mu\ell_r),
\qquad
\mathcal O_{H\ell}^{(3)pr}
=(H^\dagger i\overleftrightarrow D_\mu^I H)(\bar\ell_p\tau^I\gamma^\mu\ell_r),
$$
$$
\mathcal O_{He}^{pr}
=(H^\dagger i\overleftrightarrow D_\mu H)(\bar e_p\gamma^\mu e_r),
$$
$$
\mathcal O_{Hq}^{(1)pr}
=(H^\dagger i\overleftrightarrow D_\mu H)(\bar q_p\gamma^\mu q_r),
\qquad
\mathcal O_{Hq}^{(3)pr}
=(H^\dagger i\overleftrightarrow D_\mu^I H)(\bar q_p\tau^I\gamma^\mu q_r),
$$
$$
\mathcal O_{Hu}^{pr}
=(H^\dagger i\overleftrightarrow D_\mu H)(\bar u_p\gamma^\mu u_r),
\qquad
\mathcal O_{Hd}^{pr}
=(H^\dagger i\overleftrightarrow D_\mu H)(\bar d_p\gamma^\mu d_r),
$$
$$
\mathcal O_{Hud}^{pr}
=i(\widetilde H^\dagger D_\mu H)(\bar u_p\gamma^\mu d_r).
$$

四费米子 sector 还要区分同种流、异种流和 scalar/tensor contractions。同种流交换会在 flavor 展开后产生额外对称关系，而四维 Fierz 关系已用于选择下面的代表；圈级计算若在 $d_{\rm DR}$ 维进行，仍须按第四章保留 evanescent 补空间。

## 13.4 四费米子算符：25 个结构

**列表 13.9（$(\bar L L)(\bar L L)$）.**
$$
\mathcal O_{\ell\ell}^{prst}=(\bar\ell_p\gamma_\mu\ell_r)(\bar\ell_s\gamma^\mu\ell_t),
$$
$$
\mathcal O_{qq}^{(1)prst}=(\bar q_p\gamma_\mu q_r)(\bar q_s\gamma^\mu q_t),
\qquad
\mathcal O_{qq}^{(3)prst}=(\bar q_p\gamma_\mu\tau^I q_r)(\bar q_s\gamma^\mu\tau^I q_t),
$$
$$
\mathcal O_{\ell q}^{(1)prst}=(\bar\ell_p\gamma_\mu\ell_r)(\bar q_s\gamma^\mu q_t),
\qquad
\mathcal O_{\ell q}^{(3)prst}=(\bar\ell_p\gamma_\mu\tau^I\ell_r)(\bar q_s\gamma^\mu\tau^I q_t).
$$

**列表 13.10（$(\bar R R)(\bar R R)$）.**
$$
\mathcal O_{ee}^{prst}=(\bar e_p\gamma_\mu e_r)(\bar e_s\gamma^\mu e_t),
$$
$$
\mathcal O_{uu}^{prst}=(\bar u_p\gamma_\mu u_r)(\bar u_s\gamma^\mu u_t),
\qquad
\mathcal O_{dd}^{prst}=(\bar d_p\gamma_\mu d_r)(\bar d_s\gamma^\mu d_t),
$$
$$
\mathcal O_{eu}^{prst}=(\bar e_p\gamma_\mu e_r)(\bar u_s\gamma^\mu u_t),
\quad
\mathcal O_{ed}^{prst}=(\bar e_p\gamma_\mu e_r)(\bar d_s\gamma^\mu d_t),
$$
$$
\mathcal O_{ud}^{(1)prst}=(\bar u_p\gamma_\mu u_r)(\bar d_s\gamma^\mu d_t),
\quad
\mathcal O_{ud}^{(8)prst}=(\bar u_p\gamma_\mu T^A u_r)(\bar d_s\gamma^\mu T^A d_t).
$$

**列表 13.11（$(\bar L L)(\bar R R)$）.**
$$
\mathcal O_{\ell e}^{prst}=(\bar\ell_p\gamma_\mu\ell_r)(\bar e_s\gamma^\mu e_t),
\quad
\mathcal O_{\ell u}^{prst}=(\bar\ell_p\gamma_\mu\ell_r)(\bar u_s\gamma^\mu u_t),
\quad
\mathcal O_{\ell d}^{prst}=(\bar\ell_p\gamma_\mu\ell_r)(\bar d_s\gamma^\mu d_t),
$$
$$
\mathcal O_{qe}^{prst}=(\bar q_p\gamma_\mu q_r)(\bar e_s\gamma^\mu e_t),
$$
$$
\mathcal O_{qu}^{(1)prst}=(\bar q_p\gamma_\mu q_r)(\bar u_s\gamma^\mu u_t),
\quad
\mathcal O_{qu}^{(8)prst}=(\bar q_p\gamma_\mu T^A q_r)(\bar u_s\gamma^\mu T^A u_t),
$$
$$
\mathcal O_{qd}^{(1)prst}=(\bar q_p\gamma_\mu q_r)(\bar d_s\gamma^\mu d_t),
\quad
\mathcal O_{qd}^{(8)prst}=(\bar q_p\gamma_\mu T^A q_r)(\bar d_s\gamma^\mu T^A d_t).
$$

**列表 13.12（scalar/tensor 四费米子类）.**
$$
\mathcal O_{\ell edq}^{prst}=(\bar\ell_p^j e_r)(\bar d_s q_{tj}),
$$
$$
\mathcal O_{\ell equ}^{(1)prst}=(\bar\ell_p^j e_r)\epsilon_{jk}(\bar q_s^k u_t),
\qquad
\mathcal O_{\ell equ}^{(3)prst}=(\bar\ell_p^j\sigma_{\mu\nu} e_r)\epsilon_{jk}(\bar q_s^k\sigma^{\mu\nu} u_t),
$$
$$
\mathcal O_{quqd}^{(1)prst}=(\bar q_p^j u_r)\epsilon_{jk}(\bar q_s^k d_t),
\qquad
\mathcal O_{quqd}^{(8)prst}=(\bar q_p^jT^A u_r)\epsilon_{jk}(\bar q_s^kT^A d_t).
$$

## 13.5 Baryon number violating operators

**外部输入 13.13.** 若放开 baryon number 守恒，Warsaw basis 在维数六还包含四个 baryon-number violating 结构，常记为
$$
\mathcal O_{duq\ell},\quad
\mathcal O_{qque},\quad
\mathcal O_{qqq\ell},\quad
\mathcal O_{duue}.
$$
本书主线不使用它们；质子衰变和 GUT 匹配属于后续高级章节。

## 13.6 从目录到 Hermitian 拉氏量

把 15 个纯玻色、19 个双费米子和 25 个四费米子结构相加得到 59。这个数字采用守恒 baryon number、未展开 flavor 且非自伴 dagger pair 只计一个代表的口径。构造 Hermitian 拉氏量时，约定 13.3 要求恢复所有共轭项；进入拟合时还要展开 flavor 指标，并施加相应 Hermiticity、交换与 CP 条件。因此目录提供的是算符商空间的坐标标签，而不是现成的实参数向量。

## 练习

**练习 13.1.** 检查 $\mathcal O_{uG}$ 的规范量子数为 singlet。

**练习 13.2.** 说明 $\mathcal O_{H\widetilde G}$ 为什么是 CP-odd 候选。

**练习 13.3.** 数出 13.3-13.12 中的结构数，验证 $15+19+25=59$。
