# 第二十二章：可观测量到算符的图谱

从可观测量反查算符时，最危险的捷径是画一条一对一箭头。$h\to\gamma\gamma$ 同时接收规范-Higgs contact、SM loop 耦合位移和总宽度修正，$m_W$ 同时接收输入参数位移与直接质量修正，高质量 dilepton 则把多个 quark flavor 和手征结构卷积进 PDF。更可靠的图谱应给出从一组 Warsaw 算符到一族观测量的多对多映射，并标出共同的退化与尺度接口。以下表格按电弱、Higgs、top、高能散射和低能 flavor 组织这种映射；随后用 dilepton contact interaction 完成一次从算符振幅、逐 bin 响应到有效性条件的推演，明确 Wilson 灵敏度由 $|C|Q^2/\Lambda_{\rm ref}^2$ 描述，而局域解析边界由 $Q/M_{\rm gap}$ 决定。

## 22.1 电弱精密

| 观测量族 | 领先算符族 | 主要退化 | 必报元数据 |
| --- | --- | --- | --- |
| $m_W$ | ${\cal O}_{HD},{\cal O}_{HWB},{\cal O}_{H\ell}^{(3)},{\cal O}_{\ell\ell}$ | 输入方案退化 | 输入集、muon-decay 约定、线性/平方截断 |
| $Zf\bar f$ partial widths | ${\cal O}_{Hf}^{(1,3)}$、${\cal O}_{HD}$、${\cal O}_{HWB}$ | universal vs nonuniversal | flavor 假设、LEP/SLC 协方差 |
| asymmetries | 同上 | left/right coupling 组合 | pseudo-observable 定义 |
| triple gauge coupling | ${\cal O}_W,{\cal O}_{HWB},{\cal O}_{HW},{\cal O}_{HB}$ | diboson 与 Higgs 相关 | anomalous-coupling 规范化 |

电弱表中的“领先算符族”已经包含两类路径：算符可直接改变目标顶点，也可先改变输入量再通过 $g,g',v$ 的反解进入预测。Higgs 通道还多出总宽度这一共享分母，因此必须把产生与衰变放在同一个响应系统中。

## 22.2 Higgs

| 通道 | 领先算符族 | 直接效应 | 共同拟合风险 |
| --- | --- | --- | --- |
| gluon fusion | ${\cal O}_{HG}$、top Yukawa-like ${\cal O}_{uH}$ | $hG_{\mu\nu}G^{\mu\nu}$ contact 与 top loop 改变 | 与 $t\bar th$、Higgs decay 退化 |
| $h\to\gamma\gamma$ | ${\cal O}_{HW},{\cal O}_{HB},{\cal O}_{HWB}$、dipoles | $hF_{\mu\nu}F^{\mu\nu}$ contact | 输入方案和 SM loop 干涉 |
| $h\to ZZ^\ast,WW^\ast$ | ${\cal O}_{HD},{\cal O}_{H\Box},{\cal O}_{HW},{\cal O}_{HWB}$ | vertex、kinetic 和 width 改变 | 与 EWPO 强相关 |
| Yukawa channels | ${\cal O}_{eH},{\cal O}_{uH},{\cal O}_{dH}$ | fermion mass 与 Yukawa 同时重定义 | mass scheme |

Top 与高能散射把响应推向更大的硬不变量。下面表中的能量增长描述 Wilson 插入的灵敏度；是否仍能使用局域 SMEFT，则由独立的物理谱隙条件决定。

## 22.3 Top 与高能散射

| 观测量族 | 领先算符族 | 有效性问题 |
| --- | --- | --- |
| $t\bar t$ inclusive | ${\cal O}_{uG}$、四夸克算符 | PDF、scale、quadratic terms |
| $t\bar t$ differential high-$p_T$ | 四夸克、dipole、current 算符 | 插入随 $|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ 增强；按 $Q/M_{\rm gap}$ 切 bin |
| single top | ${\cal O}_{Hq}^{(3)}$、${\cal O}_{Hud}$、dipoles | charged-current flavor 假设 |
| high-mass dilepton | semileptonic 四费米子 | contact term 随 $|C_{\ell q}^{(6)}(\mu)|s/\Lambda_{\rm ref}^2$ 增长，局域性另查 $\sqrt s/M_{\rm gap}$ |
| VBS/diboson tails | gauge-Higgs、$X^3$、dimension-eight | 维数六平方和维数八竞争 |

低能 flavor 与 EDM 的困难不同：实验能区本身通常远低于电弱尺度，但 Warsaw 系数必须先跨越电弱阈值变成 LEFT 坐标，再与强子、核或原子矩阵元相乘。

## 22.4 Flavor、EDM 与低能

| 方向 | SMEFT 算符族 | 低能接口 | 主要不确定性 |
| --- | --- | --- | --- |
| $b\to s\ell^+\ell^-$ | semileptonic 四费米子、current 算符 | LEFT/WET Wilson 系数 | hadronic matrix elements |
| charged-current anomalies | ${\cal O}_{\ell q}^{(3)}$、${\cal O}_{Hq}^{(3)}$ | beta decay、meson decay、tau decay | CKM 和 flavor 假设 |
| LFV decays | lepton dipoles、four-lepton、semileptonic | low-energy decay rates | flavor off-diagonal bounds |
| EDM | CP-odd bosonic、fermion dipoles、four-fermion | hadronic/nuclear matrix elements | RGE mixing、matrix elements |
| $(g-2)_\ell$ | lepton dipole、semileptonic tensor | QED/QCD running | chirality enhancement |
| $0\nu\beta\beta$ | Weinberg operator、LNV higher-dimensional operators | nuclear EFT | Majorana phases、nuclear matrix elements |

## 22.5 全局拟合矩阵

固定基、$\Lambda_{\rm ref}$、定义尺度、flavor/CP 与输入方案后，令
$\theta_i$ 表示独立实 Wilson 坐标。全局拟合可抽象为
$$
d_a=t_a^{\rm SM}+M_{ai}\theta_i+\mathsf Q_{aij}\theta_i\theta_j+\eta_a,
$$
其中 $d_a$ 是数据，$M_{ai}$ 是线性 SMEFT 响应，$\mathsf Q_{aij}$ 是维数六平方响应，
$\eta_a$ 包含理论和实验误差。若取无量纲
$\theta_i=C_i^{(6)}(\mu)$，$M$ 与 $\mathsf Q$ 分别显含
$\Lambda_{\rm ref}^{-2}$ 与 $\Lambda_{\rm ref}^{-4}$；若取有量纲
$\theta_i=C_i^{(6)}(\mu)/\Lambda_{\rm ref}^2$，这些尺度因子移入坐标。
两种写法给出同一预测，但矩阵元素的单位不同。

**规则 22.1（图谱到拟合）.** 任何一行 observables-to-operators 图谱进入拟合前，必须补齐：

1.  $\Lambda_{\rm ref}$、Wilson 系数定义尺度 $\mu$ 与条件化的 $M_{\rm gap}$；
2.  RGE 是否使用；
3.  输入参数方案；
4.  flavor 和 CP 口径；
5.  covariance matrix；
6.  构造 $Q$ 的方式及基于 $Q/M_{\rm gap}$ 的 EFT validity cut；
7.  $|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ 的逐 bin 插入层级；
8.  是否保留 $\mathsf Q_{aij}$。

## 22.6 算例：high-mass dilepton

考虑 $pp\to\ell^+\ell^-$ 的高质量尾部。在 parton 层面，semileptonic contact operator 给出
$$
{\cal A}_{\rm EFT}\sim
{C_{\ell q}^{(6)}(\mu)\over\Lambda_{\rm ref}^2}s,
$$
而 SM Drell-Yan 振幅由 $\gamma/Z$ 交换给出。线性干涉对截面的贡献尺度为
$$
{\Delta\sigma\over\sigma_{\rm SM}}
\sim
{C_{\ell q}^{(6)}(\mu)s\over\Lambda_{\rm ref}^2}
\times(\hbox{chirality and PDF weights}).
$$
因此高 $m_{\ell\ell}$ bin 的 Wilson 插入灵敏度增强；该数值增长本身不等于
EFT validity 条件，后者取决于该 bin 的硬尺度相对物理谱隙的位置。

正式使用该图谱时必须指定：

1.  lepton flavor：$e,\mu,\tau$ 或组合；
2.  quark flavor 和 chirality；
3.  invariant-mass bin；
4.  PDF 和 QCD order；
5.  条件化的 $M_{\rm gap}$ 与 $Q_{\max}/M_{\rm gap}$；
6.  $\Lambda_{\rm ref}$、$C_{\ell q}^{(6)}(\mu)$ 与逐 bin 的 $\rho_{\rm ins,\ell q}$；
7.  是否保留 quadratic term；
8.  是否与 LEP 或 low-energy 数据联合。

**规则 22.1A（dilepton 的两个无量纲量）.** 取该 partonic channel 的
$Q=\sqrt{\hat s}$。Dimension-six contact insertion 的大小为
$$
\rho_{\rm ins,\ell q}
=|C_{\ell q}^{(6)}(\mu)|\frac{Q^2}{\Lambda_{\rm ref}^2},
$$
而局域展开参数为
$$
\rho_{\rm loc}\coloneqq\frac{Q}{M_{\rm gap}}.
$$
每个保留 bin 应满足 $\rho_{\rm loc}\le\rho_*<1$，其中 $\rho_*$ 是分析事先
选定的最大局域比值。$\rho_{\rm ins,\ell q}$ 在 Wilson 坐标重标度下不变，
$\rho_{\rm loc}$ 则定位最近遗漏的物理 pole/threshold；两者必须分别检查。
只有在明确 UV matching 的模型中先声明单一物理重尺度 $M$、验证
$M_{\rm gap}=M$，再选择 $\Lambda_{\rm ref}=M$ 时，二者才会使用同一个数值尺度，
但仍承担不同的检查职责。

**结论 22.2.** 该例展示了本章表格的用法：表格只给起点，正式分析还必须把算符族投影到外态、flavor、能区和 likelihood。

## 22.7 从图谱到数据映射

可观测量与算符之间是多对多映射：输入位移、共享总宽度、RGE mixing、阈值匹配和矩阵元都会使一行数据响应多个 Wilson 方向。图谱只能确定可能非零的矩阵块，真正的 $M_{ai}$ 与 $\mathsf Q_{aij}$ 还依赖外态、cuts、尺度、输入方案和协方差。Dilepton 例子进一步把 $\rho_{\rm ins}$ 与 $\rho_{\rm loc}$ 分开：前者衡量所选 Wilson 插入层级，后者衡量局域展开离遗漏奇点的距离；高能灵敏度不能代替这两项判断。

## 练习

**练习 22.1.** 说明为什么 $h\to\gamma\gamma$ 不能只约束一个 Wilson 系数。

**练习 22.2.** 对 high-mass dilepton 约束，分别写出逐 bin 的
$\rho_{\rm loc}$ 与 $\rho_{\rm ins,\ell q}$，并说明各自所需的外部输入。

**练习 22.3.** 选一个 flavor 观测量，说明 SMEFT 到 LEFT 匹配为何不可省略。

**练习 22.4.** 对 high-mass dilepton 算例，分别说明灵敏度为何随
$|C_{\ell q}^{(6)}(\mu)|s/\Lambda_{\rm ref}^2$ 增大，以及局域性为何由
$\sqrt s/M_{\rm gap}$ 控制；解释二者一般不能用任意 $\Lambda_{\rm ref}$ 合并。
