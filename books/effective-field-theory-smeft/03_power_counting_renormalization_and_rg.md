# 第三章：幂计数、重整化与重整化群

匹配在某个尺度给出 Wilson 系数，却不会自动告诉我们哪些图应同时保留，也不会保证把结果直接搬到另一个实验尺度仍然可靠。一次维数六插入的树图、标准模型一圈修正和维数六一圈图属于不同的逆尺度与圈阶；若只写一个“高阶”标签，截断误差和对数重求和都会失去口径。这里用双重分次 $(p,L)$ 记录逆参考尺度次数与圈数，在维数正规化和 $\overline{\mathrm{MS}}$ 方案中考察反项如何混合算符。一个两算符运行系统随后把匹配尺度 $\mu_{\rm match}$ 的初值传到观测尺度 $\mu_{\rm obs}$，并显示 Wilson 系数的尺度依赖如何与矩阵元的尺度依赖抵消，留下到所算阶为止与任意重整化尺度无关的物理预测。

## 3.1 幂计数

**定义 3.1（规范幂计数）.** 在四维弱耦合、单重尺度 EFT 中，若算符 $\mathcal O_i^{(d)}$ 的 Wilson 系数写为 $C_i^{(d)}/\Lambda_{\rm ref}^{d-4}$，并且矩阵元中没有额外的软/共线奇异、阈值增强或大多重数，则相对于同外态的维数四参考振幅，其一次插入贡献按
$$
C_i^{(d)}\left(\frac{Q}{\Lambda_{\rm ref}}\right)^{d-4}
$$
估计。这里必须另行记录耦合常数、环因子、群论因子和选择定则；省略它们只定义一个粗粒度的 canonical-dimension counting。局域性本身仍由 $Q/M_{\rm gap}$ 控制，不能由任意选择的 $\Lambda_{\rm ref}$ 判断。

**警告 3.2.** 幂计数不是单纯量纲分析。强耦合、近阈值、手征破缺、loop suppression、flavor 选择和 helicity selection 都可能改变实际重要性排序。

**定义 3.2A（逆尺度次数与圈次数）.** 对一个含高维顶点的连通图，设维数 $d_v>4$ 的顶点出现 $n_v$ 次，定义其逆尺度次数
$$
p\coloneqq\sum_v n_v(d_v-4),
$$
并以 $L$ 记圈数。该图属于 bidegree $(p,L)$，带有 $\Lambda_{\rm ref}^{-p}$ 以及所选微扰计数约定下的 $L$ 圈因子。于是一次维数六插入与两次维数五插入都属于 $p=2$；一次维数八、两次维数六、一次维数五加一次维数七都属于 $p=4$。

**规则 3.2B（一致的双重截断）.** 报告计算阶数时必须列出保留的 $(p,L)$ 集合，并在每个保留 bidegree 中包含使振幅有限所需的图、反项、外线 residue 和输入参数位移。例如：

- $(0,0),(0,1),(2,0)$ 是“SM NLO + EFT LO”；
- 再加入 $(2,1)$ 才是线性 $1/\Lambda_{\rm ref}^2$ 下的 EFT NLO，并需要一次维数六插入的一圈反项；
- 保留 $p=4$ 时，不能把“两次维数六插入”与“一次维数八插入”按逆尺度次数分开。

若理论含多个耦合或强耦合 sector，还须在 $(p,L)$ 之外增加相应耦合/NDA 分次。

**反例 3.2C（“维数六拉氏量”等于 $p=2$ 预测是错的）.** 若 $C_5\ne0$，两个维数五顶点也贡献 $p=2$。反之，即使拉氏量只显式列到维数六，两个维数六插入仍在振幅的 $p=4$ 项出现。作用量中列出的最大算符维数不能单独决定某个可观测量保留了哪些逆尺度阶。

**规则 3.3（振幅与二次可观测量截断）.** 先假设某精确选择定则排除了奇数 $p$，并写
$$
A=A_0+{1\over\Lambda_{\rm ref}^2}A_2
+{1\over\Lambda_{\rm ref}^4}A_4+\cdots.
$$
这里 $A_2$ 包含所有总插入次数 $p=2$ 的贡献，$A_4$ 包含所有 $p=4$ 的贡献；下标不是“只插入一个相同维数算符”的同义词。若奇数 $p$ 未被排除，必须把 $A_1/\Lambda_{\rm ref}$、$A_3/\Lambda_{\rm ref}^3$ 等一并展开。

设某截面或衰变率在固定 flux、phase-space cuts 和测度后由半正定 sesquilinear form $\langle-,-\rangle$ 给出。则
$$
\begin{aligned}
\sigma
&=\langle A,A\rangle\\
&=\sigma_0
+\frac2{\Lambda_{\rm ref}^2}\operatorname{Re}\langle A_0,A_2\rangle\\
&\quad+\frac1{\Lambda_{\rm ref}^4}
\left(\langle A_2,A_2\rangle
+2\operatorname{Re}\langle A_0,A_4\rangle\right)
+O(\Lambda_{\rm ref}^{-6}).
\end{aligned}
$$
在 lepton number 守恒、只考虑一次维数六插入时，$A_2$ 才可简写为 $A_6$。若只加入 $\langle A_2,A_2\rangle/\Lambda_{\rm ref}^4$，就已经部分进入 $p=4$，并遗漏 $A_4$ 中的一次维数八、两次维数六等振幅与 $A_0$ 的干涉。若 cuts、相空间或输入参数映射依赖 EFT 参数，还必须一并展开测度和参数，不能只展开振幅。

**例 3.4（选择定则）.** 若 $A_0$ 与 $A_2$ 因 helicity、color 或 CP 选择定则不干涉，则 $p=2$ 的可观测量修正消失，领先修正可能来自 $\langle A_2,A_2\rangle$ 或 $A_0$ 与 $A_4$ 的干涉。这时只保留“维数六平方项”会选择 $p=4$ 的一部分，而不是给出完整首个非零阶。

## 3.2 重整化

**定义 3.5（系数与算符的重整化约定）.** 在维数正规化 $d_{\rm DR}=4-2\epsilon$ 和 $\overline{\mathrm{MS}}$ 方案中，先取一个在所算 bidegree 闭合的扩大算符列向量 $\mathcal O$，其中可含物理、EOM/BRST-exact 和 evanescent 算符。把所有由 $d_{\rm DR}$ 维归一化产生的 $\mu$ 幂写成对角矩阵
$$
U_\mu^{(\epsilon)}=\operatorname{diag}(\mu^{\kappa_i\epsilon}),
$$
并定义
$$
C_0=A_C C(\mu),
\qquad
A_C\coloneqq U_\mu^{(\epsilon)}Z_C.
$$
指数 $\kappa_i$ 由把四维算符、场与耦合延拓到 $d_{\rm DR}$ 维的约定固定；同一计算中不得改变。由裸作用量与重整化作用量表示同一项，
$$
C_0^T\mathcal O_0=C^T\mathcal O,
$$
必有
$$
\mathcal O_0=A_C^{-T}\mathcal O.
$$
这固定了“算符混合矩阵”和“系数混合矩阵”之间的逆转置，避免只凭指标位置猜测符号或转置。

**定义 3.5A（evanescent operator）.** 若一个 $d_{\rm DR}$ 维局域张量结构在四维代数恒等式下消失，但在 $\epsilon\ne0$ 时独立，则称其为 evanescent operator。它与 $1/\epsilon$ pole 相乘可对四维有限项产生贡献，因此只能在重整化并选定有限投影方案后删除，不能在 loop integrand 中预先置零。

**定义 3.6（Wilson 系数反常维数矩阵）.** 裸系数满足 $dC_0/d\log\mu=0$，故
$$
\frac{dC}{d\log\mu}=\gamma_C C,
\qquad
\gamma_C
=-A_C^{-1}\frac{dA_C}{d\log\mu}.
$$
这里总导数同时作用于 $Z_C$ 所依赖的重整化耦合。相应算符插入满足
$$
\frac{d\mathcal O}{d\log\mu}=-\gamma_C^T\mathcal O,
$$
使 $C^T\mathcal O$ 在所算阶内尺度不变。

**命题 3.7（允许混合的选择定则）.** 假设作用量、regulator 与 subtraction scheme 保持一个无 anomaly 的精确对称群 $G$。则局域反项也必须为 $G$-invariant，因而带不同精确守恒量或属于不能组成 $G$ singlet 的 sector 之间，$Z_C$ 的矩阵元为零。在允许的 block 内，单个算符一般不保持不变，因此单个 Wilson 系数通常不是 RG 不变量。

**证明（条件式书内推导）.** 在所述假设下，Ward/Slavnov--Taylor identities 要求发散局域泛函及其反项保持 $G$。若源算符与候选反项的量子数不能在所有精确守恒量下匹配，其系数只能为零。反之，同一允许 block 中不存在该选择定则障碍，圈图插入 $\mathcal O_j$ 产生 $\mathcal O_i$ 型 UV pole 时必须用非对角反项吸收。一般 gauge-theory 中“可选取保持 BRST identities 的反项”本身是外部重整化输入；本证明只推出其对 mixing blocks 的后果。$\square$

**外部输入 3.7A（局域反项与阶数闭合，EFT-REN）.** 微扰重整化的局域性保证 UV poles 可由对称性允许的局域反项吸收。对 EFT，这不表示“固定 canonical dimension 的物理基”自动闭合：

1.  在线性 $p=2$，一次维数六插入的一圈图既可混合到维数六物理/EOM/evanescent 结构，也可伴随 Higgs 质量等有量纲 SM 参数修正 $d\le4$ 参数的 running；
2.  在 $p=4$，两次维数六插入的发散一般需要维数八反项；
3.  四费米子 sector 在 $d_{\rm DR}$ 维通常还需 evanescent 算符才能闭合。

因此闭合对象是“固定对称性、bidegree、参数 spurions 和 regulator 后的扩大算符空间”，不是只列物理代表的四维表。

**规则 3.7B（线性与非线性 RGE 的边界）.** 在只保留一次高维算符插入的 sector，RGE 对 Wilson 系数是线性的，形如 $dC^{(6)}/d\log\mu=\gamma_{66}C^{(6)}$。到 $p=4$ 时，一般结构为
$$
\frac{dC_i^{(8)}}{d\log\mu}
=(\gamma_{88})_{ij}C_j^{(8)}
+(\gamma_{8\leftarrow66})_{i,jk}C_j^{(6)}C_k^{(6)}+\cdots.
$$
故一个 $59\times59$ 的线性维数六矩阵不能承担双插入或完整 $1/\Lambda_{\rm ref}^4$ 的运行。

## 3.3 匹配与运行

**定义 3.8（匹配-运行工作流）.** 若 UV 模型在物理阈值 $M_{\rm gap}$ 附近选择 $\mu_{\rm match}$ 匹配到 EFT，低能实验在尺度 $\mu_{\rm obs}$ 测量，则标准流程为：

1.  在 $\mu_{\rm match}\simeq M_{\rm gap}$ 处匹配得到 $C_i(\mu_{\rm match})$；
2.  用与 matching 相同基/方案的 RGE 演化到 $\mu_{\rm obs}$；
3.  在 $\mu_{\rm obs}$ 处计算矩阵元或可观测量；
4.  与数据比较。

**外部输入 3.9（SMEFT 维数六一圈 RGE，SMEFT-RGE6）.** 在 baryon-number conserving Warsaw basis、线性一次维数六插入、维数正规化和 $\overline{\mathrm{MS}}$ 口径下，完整一圈反常维数矩阵来自 Jenkins--Manohar--Trott 与 Alonso--Jenkins--Manohar--Trott 的三篇系列计算。本书不重算该矩阵，只使用其结构和部分例子；三篇文献各自承担的算符分区见附录 B。

## 3.4 RGE 的解与尺度抵消

以下只讨论规则 3.7B 的线性 sector。一般地，若 $\gamma_C(\mu_1)$ 与 $\gamma_C(\mu_2)$ 不必对易，RGE 的解为路径有序指数
$$
C(\mu)=\mathcal P\exp\!\left(
\int_{\log\mu_{\rm match}}^{\log\mu}\gamma_C(e^t)\,dt
\right)C(\mu_{\rm match}).
$$
若 $\gamma_C$ 在考虑区间内可视为常数，则矩阵 RGE
$$
{dC\over d\log\mu}=\gamma_C C
$$
的解为
$$
C(\mu)=\exp\left[\gamma_C\log{\mu\over\mu_{\rm match}}\right]
C(\mu_{\rm match}).
$$
在把 $\gamma_C$ 固定为匹配尺度值的 leading-log 近似下，
$$
C_i(\mu)=C_i(\mu_{\rm match})
+(\gamma_C)_{ij}C_j(\mu_{\rm match})
\log{\mu\over\mu_{\rm match}}
+O\!\left(\|\gamma_C\|^2\log^2{\mu\over\mu_{\rm match}},
\beta_g\,\partial_g\gamma_C\log^2{\mu\over\mu_{\rm match}}\right).
$$
当 $\|\gamma_C\log(\mu/\mu_{\rm match})\|\not\ll1$ 时，固定阶 leading log 不能作为受控近似，应使用演化算符并按相应 logarithmic accuracy 计数。

**命题 3.9A（换基下的 RGE 协变性）.** 若 $\mathcal O'=B(\mu)\mathcal O$ 且 $C'=B(\mu)^{-T}C$，则
$$
\gamma_C'
=B^{-T}\gamma_C B^T
+\frac{dB^{-T}}{d\log\mu}B^T.
$$
特别地，对与 $\mu$ 无关的换基，$\gamma_C'=B^{-T}\gamma_C B^T$。

**证明.** 对 $C'=B^{-T}C$ 求总导数，并用 $C=B^TC'$：
$$
\frac{dC'}{d\log\mu}
=\left(
\frac{dB^{-T}}{d\log\mu}B^T
+B^{-T}\gamma_C B^T
\right)C'.
$$
与反常维数定义比较即得。$\square$

**命题 3.10（物理量的 $\mu$ 独立性到所算阶）.** 设某可观测量在一个线性 Wilson sector 中为
$$
O=O_0(g(\mu),m(\mu),\mu)+C_i(\mu)M_i(g(\mu),m(\mu),\mu).
$$
令 $\mathscr D_\mu$ 表示同时作用于显式 $\mu$、SM 耦合和质量的总 RG 导数。若在保留阶有
$$
\mathscr D_\mu O_0=0,
\qquad
\mathscr D_\mu C=\gamma_C C,
\qquad
\mathscr D_\mu M=-\gamma_C^T M,
$$
则 $\mathscr D_\mu O=0$ 到该阶成立；实际截断计算的残余尺度依赖属于更高圈阶或更高逆尺度阶。

**证明.** 直接求导：
$$
\mathscr D_\mu O
=C^T\gamma_C^T M-C^T\gamma_C^T M=0.
$$
其中已使用 $\mathscr D_\mu O_0=0$；若三条 RG 方程只计算到有限阶，等号相应理解为模去遗漏阶。$\square$

**解释 3.11.** Wilson 系数的尺度依赖不是物理效应本身；它必须与矩阵元的尺度依赖合并后才给出尺度无关的预测。

## 3.5 尺度依赖如何消失

双重分次 $(p,L)$ 决定哪些插入和圈图属于同一近似，扩大算符空间则保证这些图的 UV pole 有地方被吸收。在线性高维插入 sector，反常维数矩阵把匹配初值演化到观测尺度；到双插入阶，运行一般出现 Wilson 系数的二次项，不能再由同一个线性矩阵承担。命题 3.10 最终把这套记账变成物理陈述：系数与矩阵元的尺度依赖相消，残余 $\mu$ 变化只能估计尚未计算的更高阶，而不是新的可观测效应。

## 练习

**练习 3.1.** 设 $\mu dC/d\mu=\gamma C$ 且 $\gamma$ 为常数，解出 $C(\mu)$。

**练习 3.2.** 解释为什么“只打开一个 Wilson 系数”的说法一般不稳定于 RG。

**练习 3.3.** 对二维上三角矩阵
$$
\gamma=\begin{pmatrix}\gamma_1&a\\0&\gamma_2\end{pmatrix}
$$
写出 leading-log 解。

**练习 3.4.** 列出 $p=4$ 的所有插入分拆：$8$、$6+6$、$5+7$、$5+5+6$ 和 $5+5+5+5$，并说明令 $C_5=0$ 会删除其中哪些项。

**练习 3.5.** 从 $C'=B^{-T}C$ 推导当 $B$ 与 $\mu$ 无关时的 $\gamma_C'$，并核验 $C^TM$ 的尺度导数不变。
