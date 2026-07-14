# 第二十三章：响应矩阵算例

算符图谱只说明“可能影响”，拟合需要的是每个独立 Wilson 坐标对每个观测量的导数。把归一化观测量位移写成 $\Delta_a=M_{ai}\theta_i$ 后，输入方案、破缺相展开、RGE 与低能矩阵元都被组合进同一个 Jacobian；矩阵的秩直接告诉我们哪些 Wilson 组合仍是 flat direction。四个相互衔接的例子展示这种压缩：$m_W$ 给出一条输入参数响应行，Higgs 信号强度产生 contact 与 top Yukawa 的退化，高质量 dilepton 的两个 bin 显示能量增长，flavor 观测则把 SMEFT 运行、阈值匹配、LEFT 运行和矩阵元写成矩阵乘积。每个数值矩阵都只有在基、尺度、flavor、输入方案与截断固定后才有定义。

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
\Delta_a=M_{ai}\theta_i+\mathsf Q_{aij}\theta_i\theta_j.
$$

**原则 23.1.** 响应矩阵必须附带参数坐标定义。不同基、不同输入方案或不同 flavor 口径会改变 $M_{ai}$ 的列。
若参数列采用无量纲 $C_i^{(6)}(\mu)$，矩阵元素显含
$\Lambda_{\rm ref}^{-2}$；若采用有量纲
$c_i^{(6)}(\mu)=C_i^{(6)}(\mu)/\Lambda_{\rm ref}^2$，矩阵元素相应多两个质量
维数。两种坐标不能在同一矩阵中混用。

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
\theta=c_{\ell q}^{(6)}(\mu_{\rm obs})
\coloneqq {C_{\ell q}^{(6)}(\mu_{\rm obs})\over\Lambda_{\rm ref}^2},
\qquad [\theta]=-2.
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
\Delta_b=\kappa_bs_b\theta+q_bs_b^2\theta^2,
$$
其中 $q_b$ 是由接触振幅平方、PDF 与归一化决定的无量纲系数。忽略
$\kappa_b,q_b$ 的过程权重时，插入层级可用
$$
\rho_{\rm ins,b}
\coloneqq |s_b\theta|
=|C_{\ell q}^{(6)}(\mu_{\rm obs})|
{s_b\over\Lambda_{\rm ref}^2}
$$
估计；局域展开则使用
$$
\rho_{\rm loc,b}\coloneqq{\sqrt{s_b}\over M_{\rm gap}}.
$$
增大 $s_b$ 会同时提高线性灵敏度和 $\rho_{\rm ins,b}$，却只有在给定物理
$M_{\rm gap}$ 后才能判断 $\rho_{\rm loc,b}$。二者分别控制 Wilson 插入与遗漏
重奇点，不能由同一个任意参考尺度替代。

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
R_{ij}^{\rm SMEFT}(m_W,\mu_{\rm high})
C_j^{\rm SMEFT}(\mu_{\rm high}).
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

## 23.6 矩阵秩所表达的物理

$m_W$ 行展示输入参数反解，Higgs 行展示秩亏，高质量 dilepton 列展示能量增长，
flavor 链则把两段 RGE、一次阈值匹配和低能矩阵元乘成总响应。矩阵的秩是数据能否
分离 Wilson 方向的坐标不变量，但具体元素仍依赖基、定义尺度、输入方案与 cuts。
在高能例子中，$\rho_{\rm ins}$ 与 $\rho_{\rm loc}$ 还必须附在矩阵旁：前者检验
所保留的插入层级，后者检验局域 EFT 的物理能区。

## 练习

**练习 23.1.** 对第 23.2 节，把 $m_W^2$ 的响应行改写为 $m_W$ 的响应行。

**练习 23.2.** 对第 23.3 节，找出与 $r_gc_g+r_t\delta y_t$ 正交的 flat direction。

**练习 23.3.** 对第 23.4 节，说明为什么加入第二个 bin 不能约束一个参数模型中的新方向，但能改善同一参数的误差。

**练习 23.4.** 对 $s_1=(0.5\,\mathrm{TeV})^2$、
$s_2=(1.0\,\mathrm{TeV})^2$，取
$C_{\ell q}^{(6)}(\mu_{\rm obs})=0.2$、
$\Lambda_{\rm ref}=1\,\mathrm{TeV}$ 和
$M_{\rm gap}=2.5\,\mathrm{TeV}$，分别计算两个 bin 的
$\rho_{\rm ins,b}$ 与 $\rho_{\rm loc,b}$。
