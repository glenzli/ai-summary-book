# 第十章：树级匹配例子

给定一个 UV 拉氏量，树级匹配并不是看到重场质量 $M$ 就写下 $1/M^2$。重场的自旋、规范表示和源决定生成哪类算符，逆核展开决定导数与背景场修正的相对符号，最后还要把结果投影到指定的 Warsaw 坐标。重实 singlet 标量提供最完整的例子：完成平方得到非局域核 $(M^2+\Box+\kappa H^\dagger H)^{-1}$，其低能与小背景展开同时产生 Higgs quartic 位移、$\mathcal O_{H\Box}$ 和 $\mathcal O_H$。随后重向量的流-流交换把同一方法带到四费米子及 Higgs-current 算符，并用颜色恒等式和 flavor 指标说明“产生一个流平方”距离得到可用 Wilson 系数还差一次基投影。

## 10.1 重实 singlet 标量

考虑重实标量 $S$，并令 $X\equiv H^\dagger H$。本例只保留下列含 $S$ 的项：
$$
\mathcal L
\supset
\frac12(\partial S)^2-\frac12M^2S^2
-aSX
-\frac{\kappa}{2}S^2X.
$$

**命题 10.1（树级诱导 Higgs 局域算符）.** 若在所考察的低能场构型和运动学域内同时满足
$$
\frac{|\Box|}{M^2}\ll1,
\qquad
\frac{|\kappa X|}{M^2}\ll1,
$$
则把 $S$ 在树级积掉并保留到显式 $M^{-4}$ 给出
$$
\Delta\mathcal L_{\mathrm{EFT}}
\supset
\frac{a^2}{2M^2}X^2
-\frac{a^2}{2M^4}X\Box X
-\frac{a^2\kappa}{2M^4}X^3.
$$

**证明（书内推导）.** 丢弃边界项后，含 $S$ 的拉氏量可写成
$$
\mathcal L_S
=-\frac12S K S-aSX,
\qquad
K=M^2+\Box+\kappa X.
$$
这里 $X$ 在 $K$ 中表示乘法算符。重场方程为 $KS=-aX$。完成平方得到
$$
\mathcal L_S
=-\frac12\left(S+aK^{-1}X\right)
K\left(S+aK^{-1}X\right)
+\frac{a^2}{2}XK^{-1}X,
$$
所以树级非局域贡献为
$$
\Delta\mathcal L_{\mathrm{EFT}}^{\mathrm{tree}}
=\frac{a^2}{2}X(M^2+\Box+\kappa X)^{-1}X.
$$
在上述两个小参数条件下，逆核的 Neumann 展开为
$$
K^{-1}
=\frac1{M^2}
-\frac{\Box+\kappa X}{M^4}
+\frac{(\Box+\kappa X)^2}{M^6}
+\cdots.
$$
代入后，$M^{-4}$ 阶的两个贡献分别是 $-a^2X\Box X/(2M^4)$ 与 $-a^2\kappa X^3/(2M^4)$，符号都来自逆核展开的一阶负号；首个未列项为 $a^2X(\Box+\kappa X)^2X/(2M^6)$。$\square$

**号差核验.** 对常背景取 $\Box X=0$，经典解为 $S_{\rm cl}=-aX/(M^2+\kappa X)$。直接代回给出
$$
\Delta\mathcal L_{\rm EFT}^{\rm tree}
=\frac{a^2X^2}{2(M^2+\kappa X)}
=\frac{a^2}{2M^2}X^2
-\frac{a^2\kappa}{2M^4}X^3
+O(M^{-6}),
$$
独立确认了 $\kappa X^3$ 项的负号。

**说明 10.2（算符投影与展开域）.** $X^2$ 的场单项式维数为四，只修正 SM Higgs quartic。按第十三章约定，$X\Box X={\cal O}_{H\Box}$ 且 $X^3={\cal O}_H$，故本例在匹配尺度给出
$$
c_{H\Box}=-\frac{a^2}{2M^4},
\qquad
c_H=-\frac{a^2\kappa}{2M^4},
$$
其中 $\Delta\mathcal L_{\rm EFT}\supset c_i\mathcal O_i$。分部积分还给出 $X\Box X\simeq-(\partial_\mu X)(\partial^\mu X)$，等号只在作用量中模边界项成立。$|\Box|/M^2\ll1$ 是 $Q^2/M^2\ll1$ 的算符简写，而 $|\kappa X|/M^2\ll1$ 是独立的背景场展开条件；仅有 $Q\ll M$ 不能无条件删除 $\kappa$ 项。

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
c_{\ell q}^{(1)prst}(\mu_{\rm match})
=-{g_X^2\over M_X^2}
(\lambda_\ell)_{pr}(\lambda_q)_{st}
$$
在树级匹配尺度 $\mu_{\rm match}\simeq M_X$ 成立；这里 $\Delta{\cal L}_{\rm EFT}\supset c_i{\cal O}_i$，故 $[c_{\ell q}^{(1)}]=-2$。等价地，若按本书约定写成
$$
\Delta{\cal L}_{\rm EFT}
\supset {C_{\ell q}^{(1)prst}\over\Lambda_{\rm ref}^2}
{\cal O}_{\ell q}^{(1)prst},
$$
则无量纲坐标为
$$
C_{\ell q}^{(1)prst}(\mu_{\rm match})
=-{g_X^2\Lambda_{\rm ref}^2\over M_X^2}
(\lambda_\ell)_{pr}(\lambda_q)_{st}.
$$
取 $\Lambda_{\rm ref}=M_X$ 和单位 flavor coupling 时，才简化为 $C_{\ell q}^{(1)}=-g_X^2$。

**警告 10.5.** 这里假设 $X_\mu$ 是规范一致 UV 理论中的重向量，且只保留代数质量项。完整规范理论还需处理 Goldstone、ghost、kinetic mixing 和规范固定。

## 10.4 从逆核到 Warsaw 系数

三个例子的共同对象是重场逆核，而不是孤立的 $1/M^2$ 因子。对 singlet 标量，$|\Box|/M^2$ 与 $|\kappa H^\dagger H|/M^2$ 分别控制动量和背景展开；对重向量，流的规范与 flavor 类型决定生成的四费米子或 Higgs-current 组合。代回经典解只得到局域候选项，颜色恒等式、IBP 和 EOM 投影才把它们变成 Warsaw 系数。若再选 $\Lambda_{\rm ref}=M$，这只是单尺度模型中的坐标便利，物理系数仍由耦合和匹配关系决定。

## 练习

**练习 10.1.** 对重 $Z'$ 若取 $J^\mu=\bar q\gamma^\mu q$，写出产生的四夸克算符结构。

**练习 10.2.** 令 $X=H^\dagger H$，从 $a^2X(M^2+\Box+\kappa X)^{-1}X/2$ 展开到 $M^{-4}$。说明 $X^2$ 为什么不是维数六算符，识别两个维数六项，并写出展开所需的两个独立条件。

**练习 10.3.** 对第 10.3 节的 semileptonic 例子，解释为什么交叉项前有因子 $2$。
