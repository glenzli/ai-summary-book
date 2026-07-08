# 第十章：树级匹配例子

## 本章目标

本章给出几个可手算的树级匹配例子，展示 UV 模型如何产生 SMEFT 算符。所有例子只用于训练匹配方法，不代表模型偏好。

## 依赖前置知识

需要第二章的匹配、第五章的标准模型场内容和第七章的 SMEFT 算符结构。

## 10.1 重实 singlet 标量

考虑重实标量 $S$，取
$$
\mathcal L
\supset
\frac12(\partial S)^2-\frac12M^2S^2
-aS(H^\dagger H)
-\frac{\kappa}{2}S^2(H^\dagger H).
$$

**命题 10.1（树级诱导 Higgs 势修正）.** 在 $E\ll M$ 时，$S$ 的树级交换诱导
$$
\Delta\mathcal L_{\mathrm{EFT}}
\supset
\frac{a^2}{2M^2}(H^\dagger H)^2
-\frac{a^2}{2M^4}(H^\dagger H)\Box(H^\dagger H)+\cdots.
$$

**证明（书内推导）.** 领先阶忽略 $\kappa$，重场方程为
$$
(\Box+M^2)S=-aH^\dagger H.
$$
形式求解并展开：
$$
S=-\frac{a}{M^2+\Box}(H^\dagger H)
=-\frac{a}{M^2}\left(1-\frac{\Box}{M^2}+\cdots\right)(H^\dagger H).
$$
代回 $S$ 的二次部分得到所列项。$\square$

**说明 10.2.** 第一项修正 SM Higgs quartic，第二项可通过分部积分和场重定义映射到 Warsaw basis 中的 Higgs-derivative 类算符组合。

## 10.2 重 $Z'$ 向量

设重向量 $X_\mu$ 与 SM 流 $J^\mu$ 耦合：
$$
\mathcal L
\supset
\frac12M_X^2X_\mu X^\mu+g_X X_\mu J^\mu.
$$

**命题 10.3（树级流-流算符）.** 积掉 $X_\mu$ 在领先阶产生
$$
\Delta\mathcal L_{\mathrm{EFT}}
=
-\frac{g_X^2}{2M_X^2}J_\mu J^\mu+\cdots.
$$

**证明说明.** 代数方程给出 $X_\mu=-g_XJ_\mu/M_X^2+\cdots$。代回质量项和耦合项，合并得 $-g_X^2J^2/(2M_X^2)$。若 $J^\mu$ 是费米子流，则得到四费米子算符；若含 Higgs 流，则得到 $\psi^2H^2D$ 或 $H^4D^2$ 型算符。$\square$

**例 10.4（两种流的投影）.** 若
$$
J^\mu=\bar q\gamma^\mu q,
$$
则
$$
J_\mu J^\mu
=(\bar q\gamma_\mu q)(\bar q\gamma^\mu q),
$$
投影到 Warsaw basis 中的 $O_{qq}^{(1)}$、$O_{qq}^{(3)}$ 及其 flavor 置换组合，取决于 $X_\mu$ 的颜色和弱同位旋表示。若中间态携带颜色八重态流，需先用
$$
T^A_{ij}T^A_{kl}
=\frac12\left(\delta_{il}\delta_{kj}-\frac1{N_c}\delta_{ij}\delta_{kl}\right)
$$
把颜色八重态收缩投影回 Warsaw basis；Warsaw 的 $LLqq$ 表中不另列独立的 $O_{qq}^{(8)}$。

若
$$
J^\mu=H^\dagger i\overleftrightarrow D^\mu H,
$$
则
$$
J_\mu J^\mu
=(H^\dagger i\overleftrightarrow D_\mu H)
(H^\dagger i\overleftrightarrow D^\mu H),
$$
它属于 $H^4D^2$ 类，并可与 Warsaw basis 的 ${\cal O}_{HD}$、${\cal O}_{H\Box}$ 组合互相转换。

## 10.3 重向量诱导 semileptonic 算符

取
$$
J^\mu
= (\lambda_\ell)_{pr}\bar\ell_p\gamma^\mu\ell_r
+(\lambda_q)_{st}\bar q_s\gamma^\mu q_t,
$$
其中 $\lambda_\ell,\lambda_q$ 为 Hermitian flavor 矩阵，使实重向量耦合到 Hermitian current。若只写单个 off-diagonal flavor 分量，应同时包含 Hermitian conjugate。
由命题 10.3 得
$$
\Delta{\cal L}_{\rm EFT}
=-{g_X^2\over2M_X^2}
\left[
\begin{aligned}
&(\lambda_\ell)_{pr}(\lambda_\ell)_{uv}
(\bar\ell_p\gamma_\mu\ell_r)(\bar\ell_u\gamma^\mu\ell_v)\\
&+2(\lambda_\ell)_{pr}(\lambda_q)_{st}
(\bar\ell_p\gamma_\mu\ell_r)(\bar q_s\gamma^\mu q_t)\\
&+(\lambda_q)_{st}(\lambda_q)_{uv}
(\bar q_s\gamma_\mu q_t)(\bar q_u\gamma^\mu q_v)
\end{aligned}
\right].
$$
交叉项对应 Warsaw 结构
$$
{\cal O}_{\ell q}^{(1)prst}
=(\bar\ell_p\gamma_\mu\ell_r)(\bar q_s\gamma^\mu q_t).
$$
因此
$$
C_{\ell q}^{(1)prst}(\Lambda)
=-{g_X^2\over M_X^2}
(\lambda_\ell)_{pr}(\lambda_q)_{st}
$$
在本章的归一化下成立。若把 SMEFT 写成 $C_i{\cal O}_i/\Lambda^2$ 且取 $\Lambda=M_X$，并取单位 flavor coupling，则无量纲 $C_{\ell q}^{(1)}=-g_X^2$。

**警告 10.5.** 这里假设 $X_\mu$ 是规范一致 UV 理论中的重向量，且只保留代数质量项。完整规范理论还需处理 Goldstone、ghost、kinetic mixing 和规范固定。

## 本章小结

树级匹配的共同步骤是：写出重场方程，按 $1/M$ 展开，代回得到局域算符，再用第四章的冗余关系投影到选定基。

## 练习

**练习 10.1.** 对重 $Z'$ 若取 $J^\mu=\bar q\gamma^\mu q$，写出产生的四夸克算符结构。

**练习 10.2.** 说明为什么重 singlet 标量的 $(H^\dagger H)^2$ 项不是维数六算符。

**练习 10.3.** 对第 10.3 节的 semileptonic 例子，解释为什么交叉项前有因子 $2$。
