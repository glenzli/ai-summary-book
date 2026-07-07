# 第二十三章：响应矩阵 worked examples

## 本章目标

本章把第八章和第二十二章的抽象响应矩阵写成可直接计算的 worked examples。这里不使用具体实验数据，而是给出从 Wilson 参数到观测量位移的矩阵形式；真实拟合只需把实验中心值和协方差接上。

## 依赖前置知识

需要第八章的 Fisher 矩阵、第二十一章的输入方案、第二十二章的 observables-to-operators 图谱。

## 23.1 响应矩阵定义

令归一化观测量为
$$
\Delta_a={O_a-O_a^{\rm SM}\over O_a^{\rm SM}}.
$$
在线性 SMEFT 截断下
$$
\Delta_a=M_{ai}\theta_i,
$$
其中 $\theta_i$ 是选定 Wilson 参数坐标。若保留维数六平方项，则
$$
\Delta_a=M_{ai}\theta_i+Q_{aij}\theta_i\theta_j.
$$

**原则 23.1.** 响应矩阵必须附带参数坐标定义。不同基、不同输入方案或不同 flavor 口径会改变 $M_{ai}$ 的列。

## 23.2 电弱精密：$m_W$ 响应行

第二十一章给出
$$
{\delta m_W^2\over m_W^2}
={s^2\epsilon_\alpha-c^2\epsilon_Z-s^2\epsilon_G\over c^2-s^2}
+\epsilon_W.
$$
取参数向量
$$
\theta=(\epsilon_\alpha,\epsilon_G,\epsilon_Z,\epsilon_W)^T.
$$
则 $m_W^2$ 的响应矩阵只有一行：
$$
M_{m_W^2}
=
\begin{pmatrix}
{s^2\over c^2-s^2}&
-{s^2\over c^2-s^2}&
-{c^2\over c^2-s^2}&
1
\end{pmatrix}.
$$
若使用 $m_W$ 而非 $m_W^2$，线性阶有
$$
{\delta m_W\over m_W}
={1\over2}{\delta m_W^2\over m_W^2},
$$
因此响应行整体乘以 $1/2$。

## 23.3 Higgs：两参数退化

考虑只保留 Higgs-gluon contact 与 top Yukawa shift 的简化参数
$$
\theta=(c_g,\delta y_t)^T,
$$
其中 $c_g$ 表示归一化后的 ${\cal O}_{HG}$ 贡献，$\delta y_t$ 表示 top Yukawa 的相对位移。写
$$
{\delta\sigma(gg\to h)\over\sigma_{\rm SM}}
=2(r_g c_g+r_t\delta y_t),
$$
$$
{\delta\Gamma(h\to gg)\over\Gamma_{\rm SM}}
=2(r_g c_g+r_t\delta y_t),
$$
其中 $r_g,r_t$ 是由振幅归一化和 loop 函数决定的响应系数。对信号强度
$$
\mu_{gg\to h\to ZZ}
=1+\delta_{\rm prod}+\delta_{ZZ}-\delta_{\rm tot},
$$
若暂时只考虑生产端修正，则
$$
\Delta_\mu=
\begin{pmatrix}
2r_g&2r_t
\end{pmatrix}
\binom{c_g}{\delta y_t}.
$$
这个一行矩阵有一个零方向：
$$
r_g c_g+r_t\delta y_t=0.
$$
因此单个 Higgs signal strength 不能分离 contact 与 top Yukawa shift。

## 23.4 高质量 dilepton：两 bin 响应

设两个 invariant-mass bin 的代表 partonic 能量为 $s_1,s_2$，只考虑一个 semileptonic Wilson 参数
$$
\theta={C_{\ell q}\over\Lambda^2}.
$$
线性干涉给
$$
\Delta_b=M_b\theta,\qquad
M_b=\kappa_b s_b,
$$
其中 $\kappa_b$ 包含 PDF、chirality 和 SM propagator 权重。两个 bin 的响应矩阵为
$$
M=
\begin{pmatrix}
\kappa_1s_1\\
\kappa_2s_2
\end{pmatrix}.
$$
若保留平方项，
$$
\Delta_b=\kappa_bs_b\theta+\rho_bs_b^2\theta^2.
$$
当 $s_b$ 增大时，线性灵敏度增强，平方项和维数八风险也同时增强。

## 23.5 Flavor：SMEFT 到 LEFT 的响应组合

低能 flavor 观测量常可写为
$$
\Delta_a=N_{a\alpha}C_\alpha^{\rm LEFT}(\mu_{\rm low}).
$$
而 LEFT 系数来自高尺度 SMEFT：
$$
C_\alpha^{\rm LEFT}(\mu_{\rm low})
=R_{\alpha\beta}^{\rm LEFT}(\mu_{\rm low},m_W)
T_{\beta i}(m_W)
R_{ij}^{\rm SMEFT}(m_W,\Lambda)
C_j^{\rm SMEFT}(\Lambda).
$$
因此总响应矩阵为
$$
M_{aj}
=
N_{a\alpha}
R_{\alpha\beta}^{\rm LEFT}
T_{\beta i}
R_{ij}^{\rm SMEFT}.
$$
这条公式解释了为什么 flavor 结果必须报告匹配、运行和低能矩阵元。

## 本章小结

响应矩阵是把教材定义变成拟合对象的关键中间层。它显示 Wilson 系数限制不是单个数字，而是依赖基、尺度、输入方案、flavor 假设和数据协方差的线性代数对象。

## 练习

**练习 23.1.** 对第 23.2 节，把 $m_W^2$ 的响应行改写为 $m_W$ 的响应行。

**练习 23.2.** 对第 23.3 节，找出与 $r_gc_g+r_t\delta y_t$ 正交的 flat direction。

**练习 23.3.** 对第 23.4 节，说明为什么加入第二个 bin 不能约束一个参数模型中的新方向，但能改善同一参数的误差。

