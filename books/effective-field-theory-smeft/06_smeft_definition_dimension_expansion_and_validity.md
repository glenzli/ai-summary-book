# 第六章：SMEFT 的定义、维数展开与适用范围

把标准模型拉氏量后面写上 $C_i\mathcal O_i/\Lambda_{\rm ref}^2$ 还不足以定义 SMEFT。必须先说明低能谱中没有额外轻态，Higgs 以 $SU(2)_L$ 双重态线性实现，局域算符已经在 IBP、EOM 与代数恒等式下取商，并把 Hermitian 与非 Hermitian 算符族分别处理。更重要的是，作用量按维数列到六并不等于可观测量只含一次维数六插入：Weinberg 算符的多次插入、输入参数位移和平方项会改变逆尺度阶数。以维数五的唯一结构和维数六的 Warsaw 计数为锚点，本章给出 SMEFT、HEFT、LEFT 的分界，并把局域性 $Q/M_{\rm gap}$、Wilson 插入大小和圈/对数展开拆成彼此独立的有效性条件。

## 6.1 定义

**定义 6.1（SMEFT）.** 标准模型有效场论是满足以下条件的 EFT：

1.  显式低能自由度恰为第五章列出的标准模型场，不含额外轻 sterile neutrino、axion-like field 或其他 BSM 场；
2.  规范群的 Lie-algebra 层面约定为 $G_{\mathrm{SM}}=SU(3)_c\times SU(2)_L\times U(1)_Y$，场表示与第五章一致；
3.  Higgs 场 $H$ 是线性实现的 $SU(2)_L$ 双重态；
4.  拉氏量按第四章的 IBP/EOM/代数商包含全部允许的局域 Lorentz 与 $G_{\mathrm{SM}}$ 不变量；
5.  算符按质量维数、逆参考尺度次数和圈次数展开，并在有限阶截断。

为使拉氏量逐阶 Hermitian，先把固定维数的算符族按 Hermitian conjugation（dagger）分区。令 $\mathfrak H_d$ 表示在 dagger 下闭合的自伴算符族，令 $\mathfrak N_d$ 从每一对互不相同的非自伴算符 $\{\mathcal O,\mathcal O^\dagger\}$ 中只选一个代表。因此一般式应写成
$$
\mathcal L_{\mathrm{SMEFT}}
=
\mathcal L_{\mathrm{SM}}
+
\sum_{d>4}\frac1{\Lambda_{\rm ref}^{d-4}}
\left\{
\sum_{a\in\mathfrak H_d}C_a^{(d)}(\mu)\mathcal O_a^{(d)}(\mu)
+\sum_{b\in\mathfrak N_d}
\left[
C_b^{(d)}(\mu)\mathcal O_b^{(d)}(\mu)
+\bigl[C_b^{(d)}(\mu)\bigr]^*\mathcal O_b^{(d)\dagger}(\mu)
\right]
\right\}.
$$
这里压低了 flavor 多重指标。对逐分量满足 $\mathcal O_a^\dagger=\mathcal O_a$ 的算符，$C_a^{(d)}$ 必须为实数；若 dagger 同时置换 flavor 指标，则第一项表示相应 Wilson 张量满足 Hermiticity 关系。Baryon number、lepton number、CP 或 flavor 对称性是另加的 sector 假设，不属于 SMEFT 定义本身。本书维数六主线另行限制到 baryon-number conserving sector。

**约定 6.1A（SMEFT 尺度数据）.** 一个可计算 SMEFT 问题至少区分
$$
(M_{\rm gap},\Lambda_{\rm ref},\mu_{\rm match},\mu_{\rm obs},Q,v).
$$
$M_{\rm gap}$ 是最近遗漏 BSM 奇点的物理尺度，$\Lambda_{\rm ref}$ 只归一化 Wilson 坐标，$\mu_{\rm match}$ 与 $\mu_{\rm obs}$ 是重整化尺度，$Q$ 来自过程运动学，$v$ 是电弱尺度。若 UV completion 未指定，$M_{\rm gap}$ 通常未知，所有有效性结论都必须明确写成对它和 UV 耦合计数的条件陈述。

## 6.2 维数展开

**外部输入定理 6.2（维数五算符分类，SMEFT-D5）.** 在上述标准模型场内容、线性 $SU(3)_c\times SU(2)_L\times U(1)_Y$ 规范实现、局域 Lorentz 标量以及分部积分和领先 EOM 等价下，维数五只有 Weinberg 算符这一类，连同其 Hermitian conjugate 与 flavor 分量。带 flavor 指标的一个约定为
$$
(\mathcal O_5)_{rs}
=\epsilon_{jk}\epsilon_{mn}
(\ell_r^j)^T \mathsf C\ell_s^m\,H^kH^n,
$$
其中 $\mathsf C$ 是收缩 Lorentz spinor indices 的 charge-conjugation matrix，且 flavor 系数可取 $C_5^{rs}=C_5^{sr}$。下文对所有有序 $r,s$ 求和，并在拉氏量中放置因子 $1/2$，以免对称 flavor 对重复计数。$(\mathcal O_5)_{rs}$ 携带 $\Delta L=2$，不是自伴算符；其 dagger 携带 $\Delta L=-2$，必须与共轭系数同时出现。它在电弱破缺后产生 Majorana neutrino mass。完整唯一性分类使用外部输入 SMEFT-D5；精确来源为附录 B 所列 Warsaw-classification 论文 Sec. 3、Eq. (3.1)。

**量纲与量子数检查.** 质量维数为 $2(3/2)+2=5$。两个 lepton 双重态总 hypercharge 为 $-1$，两个 Higgs 双重态总 hypercharge 为 $+1$，所以 $U(1)_Y$ 中性；两个 $\epsilon$ 张量完成 $SU(2)_L$ singlet contraction，且所有场为 color singlet。Fermi statistics 与 $SU(2)$ contraction 给出 flavor 对称性。以上只说明该算符良定义并具有所需量子数；“不存在第二个独立等价类”的分类结论依赖完整 Lorentz/规范表示枚举，作为外部输入而不是由量纲计数冒充证明。

**定义 6.3（作用量保留到 $p=2$）.** 若作用量只保留逆参考尺度次数 $p\le2$，则
$$
\mathcal L_{\mathrm{SMEFT}}^{[p\le2]}
=
\mathcal L_{\mathrm{SM}}
+
\frac1{2\Lambda_{\rm ref}}\sum_{r,s}
\left[
C_5^{rs}(\mu)(\mathcal O_5)_{rs}(\mu)
+\bigl[C_5^{rs}(\mu)\bigr]^*(\mathcal O_5)_{rs}^\dagger(\mu)
\right]
+
\frac1{\Lambda_{\rm ref}^2}
\left\{
\sum_{a\in\mathfrak H_6}C_a^{(6)}(\mu)\mathcal O_a^{(6)}(\mu)
+\sum_{b\in\mathfrak N_6}
\left[
C_b^{(6)}(\mu)\mathcal O_b^{(6)}(\mu)
+\bigl[C_b^{(6)}(\mu)\bigr]^*\mathcal O_b^{(6)\dagger}(\mu)
\right]
\right\}.
$$
若假设 exact lepton number，可令 $C_5=0$；若另假设 exact baryon number，还须删除 baryon-number violating dimension-six sector。两个假设必须分别声明。

**Weinberg/Warsaw 计数口径.** “维数五只有一个结构类型”是从非自伴对 $\{\mathcal O_5,\mathcal O_5^\dagger\}$ 中只计一个代表，不是从 Hermitian 拉氏量删除 dagger。Warsaw 的 $59$ 采用同一结构计数口径：在 baryon number 守恒且未展开 flavor 时，第十三章的 $15$ 个纯玻色结构、$7$ 个自伴 current 结构和 $20$ 个自伴 current-current 结构属于 $\mathfrak H_6$，共 $42$ 个；$3$ 个 $\psi^2H^3$、$8$ 个 dipole、$\mathcal O_{Hud}$ 和 $5$ 个 scalar/tensor 四费米子结构给出 $\mathfrak N_6$ 的 $17$ 个代表。因此 $42+17=59$，但构造 Hermitian 拉氏量时每个非自伴代表都必须补上带共轭系数的 dagger。带 flavor 时，自伴算符族的 dagger 还会置换指标，故 $59$ 既不是 flavor 展开后的算符分量数，也不是 Wilson 实参数数。

**警告 6.3A（拉氏量截断与可观测量截断）.** 即使作用量只列到维数六，振幅的 $p=2$ 项也可能同时含一次维数六插入和两次维数五插入；振幅的 $p=4$ 项还含两次维数六插入。因而“作用量列到维数六”“线性维数六”和“预测到 $1/\Lambda_{\rm ref}^2$”是不同陈述，必须同时声明 lepton-number 假设、插入次数和圈阶。

## 6.3 SMEFT、HEFT 与 LEFT

| 理论 | 自由度 | 对称性实现 | 典型适用区间 |
| --- | --- | --- | --- |
| SMEFT | SM 场，含 Higgs 双重态 | $SU(2)_L\times U(1)_Y$ 线性实现 | 新物理高于电弱尺度且 decoupling |
| HEFT | Higgs singlet-like 标量与 Goldstone 非线性实现 | 电弱对称性非线性实现 | 强电弱破缺或非双重态 Higgs 情形 |
| LEFT | $W,Z,h,t$ 已积掉的低能场 | $SU(3)_c\times U(1)_{\rm em}$ | $Q,\mu<m_W$ 且无这些重外态 |

**原则 6.4（理论选择）.** 对 $Q\ll m_W$ 且外态不含 $W,Z,h,t$ 的过程，原则上可在完整 SMEFT 中保留这些重场并计算低能展开；但若要得到只含轻场的局域拉氏量、系统重求和 $m_W$ 到 $Q$ 的对数并使用低能算符基，则应在电弱阈值把 SMEFT 匹配到 LEFT。故“使用 SMEFT 系数解释低能数据”必须附带 SMEFT-to-LEFT matching 和 running，不能把两套系数直接同名。若 Higgs 不属于线性双重态或电弱 Goldstone 以非线性方式实现，SMEFT 不是正确主线，应使用 HEFT。

## 6.4 适用性

**原则 6.5（相互独立的有效性检查）.** 对给定外态和 phase-space bin，以 $m_{\rm ext,max}$ 记最大外态质量，并令
$$
Q=\max\!\left(\max_a\sqrt{|I_a|},m_{\rm ext,max},v\ \text{when relevant}\right).
$$
至少分别检查：

1.  **局域性：** 存在固定 $\rho<1$ 使 bin 内 $Q/M_{\rm gap}\le\rho$，且没有被遗漏的 pole、threshold 或额外轻态；
2.  **插入展开：** 对实际参与过程的算符，$\rho_{\rm ins,i}^{(d)}=|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ 连同耦合、helicity 和群论因子满足所声明的层级，并比较相邻非零 $p$ 阶；
3.  **圈与对数展开：** 所选耦合的 loop parameter 足够小；若 $\epsilon_{\rm loop}|\log(\mu_{\rm match}/\mu_{\rm obs})|$ 不小，则使用相应 RGE resummation；
4.  **可观测量阶：** flux、相空间、cuts、输入参数位移和多次插入均按第三章同一 $(p,L)$ 方案展开。

这些是必要条件，不是对未知 UV 理论的充分收敛定理。

**反例 6.5A（$Q<\Lambda_{\rm ref}$ 不是有效性判据）.** 保持有量纲系数
$$
c_i^{(d)}=\frac{C_i^{(d)}}{\Lambda_{\rm ref}^{d-4}}
$$
不变，并把 $\Lambda_{\rm ref}\mapsto a\Lambda_{\rm ref}$、$C_i^{(d)}\mapsto a^{d-4}C_i^{(d)}$。所有振幅不变，但通过选择 $a$ 可让同一个 bin 满足或不满足 $Q<\Lambda_{\rm ref}$。因此运动学有效性必须与物理 $M_{\rm gap}$ 比较；若 $M_{\rm gap}$ 未知，只能报告条件化结果，不能把拟合约定中的 $\Lambda_{\rm ref}$ 宣称为已测得新粒子质量。

**例 6.5B（系数限制通常约束质量与耦合的组合）.** 若某弱耦合 UV 模型在树级给出
$$
c^{(6)}=\frac{g_*^2}{M^2},
$$
则对 $c^{(6)}$ 的限制直接约束 $M/|g_*|$，而不是单独约束 $M$。把它转换为 $M$ 的下界需要额外给定 $g_*$ 范围和 matching 假设；强耦合或 loop-generated 系数会改变这一映射。

**警告 6.6（维数六平方项）.** 若观测量写为
$$
\sigma
=
\sigma_{\mathrm{SM}}
+
\frac{1}{\Lambda_{\rm ref}^2}\sigma_{\mathrm{int}}
+
\frac{1}{\Lambda_{\rm ref}^4}\sigma_{\mathrm{quad}}
+\cdots,
$$
则 $\sigma_{\mathrm{quad}}$ 与 $A_0$ 对完整 $A_4$ 的干涉同属 $p=4$，后者包括一次维数八和允许的多次低维插入。保留或丢弃维数六平方项是带有理论假设的部分截断方案。

## 6.5 可计算性条件

若要从 SMEFT 得到确定的预测，至少需给出：

1.  过程外态、所有独立硬不变量、bin cuts 和 $Q_{\max}$；
2.  EFT 自由度、规范实现与是否需要 SMEFT-to-LEFT matching；
3.  所用算符商空间、基和 evanescent/EOM 投影方案（若为圈级）；
4.  $\Lambda_{\rm ref}$、Wilson 系数定义尺度 $\mu$、$\mu_{\rm match}$ 与条件化的 $M_{\rm gap}$；
5.  flavor、CP、baryon number 与 lepton number 假设；
6.  输入参数方案；
7.  保留的 $(p,L)$ 集合及多次插入规则；
8.  matching/RGE 的方案、阈值顺序和 logarithmic accuracy；
9.  维数、圈阶和数据建模的理论误差估计。

**例 6.7（不完整陈述）.** “限制 $C_{HWB}$”不是完整物理命题。完整说法必须包含：在 baryon-number conserving Warsaw basis、给定 $\Lambda_{\rm ref}$ 与定义尺度 $\mu$、某输入方案、数据集和 $Q_{\max}$、某 flavor/CP 口径及某 $(p,L)$ 截断规则下，限制有量纲组合 $c_{HWB}=C_{HWB}/\Lambda_{\rm ref}^2$ 的某个置信区间；若再给出新物理质量结论，还须声明 $M_{\rm gap}$ 与 UV coupling/matching 假设。

## 6.6 一条完整 SMEFT 陈述

例 6.7 表明，单独写出 $C_{HWB}$ 没有确定一个物理问题。SMEFT 先由标准模型场内容和线性规范实现定义，再由算符基、flavor/CP sector、输入方案与 $(p,L)$ 截断选出具体计算。$M_{\rm gap}$ 控制遗漏奇点造成的局域边界，$\Lambda_{\rm ref}$ 只归一化 Wilson 坐标；插入层级、圈展开和大对数还须分别判断。把这些数据补齐后，系数限制才可在 RGE、换基或 SMEFT-to-LEFT 匹配中无歧义传播。

## 练习

**练习 6.1.** 验证 Weinberg 算符的质量维数为五；说明它为什么不是自伴算符，写出 Hermitian 拉氏量中的系数组合，并解释为何结构计数仍只记一个类型。

**练习 6.2.** 解释为什么含 $Q\gtrsim M_{\rm gap}$ 的 LHC bin 不能用局域截断 SMEFT 无条件解释，以及为何未知 $M_{\rm gap}$ 时该判断只能条件化报告。

**练习 6.3.** 判断下列情形应优先使用 SMEFT、HEFT 还是 LEFT：低能核 beta decay、强耦合电弱破缺、高能 Higgs pair production。

**练习 6.4.** 取同一有量纲系数 $c^{(6)}=1\,\mathrm{TeV}^{-2}$，分别选 $\Lambda_{\rm ref}=1\,\mathrm{TeV}$ 与 $10\,\mathrm{TeV}$，求对应无量纲 $C^{(6)}$，并说明为何两种写法给出相同振幅却使 $Q/\Lambda_{\rm ref}$ 不同。
