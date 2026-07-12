# 附录 D：练习解题路线

本附录给出章末练习的解题路线。它不替代正文推导，只提供核对方向。

## 第 0 章

**0.1.** 用 $[\mathcal L]=4$。标量动能 $(\partial\phi)^2$ 给 $[\phi]=1$；Dirac 动能 $\bar\psi i\slashed\partial\psi$ 给 $[\psi]=3/2$；规范动能 $F^2$ 给 $[A_\mu]=1$；Yukawa 项 $\bar\psi H\psi$ 给 $[y]=0$。

**0.2.** 需指定自由度、对称性、尺度和截断；否则“所有可能修正”没有可计算的有限参数集。

## 第 1 章

**1.1.** 使用几何级数展开；$p^4/M^6$ 对应四导数局域算符。

**1.2.** 接近阈值时传播子分母可接近零，低能 Taylor 展开失效。

**1.3.** 有
$$
{1\over M^2-p^2}
={1\over M^2}\left(1+{p^2\over M^2}+{p^4\over M^4}+{p^6\over M^6}+\cdots\right).
$$
位置空间对应 $J^2/M^2$、$J(-\partial^2)J/M^4$、$J(\partial^2)^2J/M^6$、$J(-\partial^2)^3J/M^8$，符号依 Fourier 约定。

## 第 2 章

**2.1.** UV 树图由重标量交换给出；低能展开后领先常数项对应 $\phi^4$ 接触项。

**2.2.** EOM 冗余算符在 on-shell 振幅中可由场重定义消去，因此 off-shell Green 函数匹配能看到更多坐标信息。

**2.3.** 若 $\Delta{\cal L}=g^2\phi^4/(8M^2)$，四个相同外线给顶点因子 $i\,4!\,g^2/(8M^2)=i3g^2/M^2$，对应 $s,t,u$ 三个重场交换道的常数项。

## 第 3 章

**3.1.** 常数 $\gamma$ 时 $C(\mu)=C(\Lambda)(\mu/\Lambda)^\gamma$；若含 $16\pi^2$ 则相应放入指数。

**3.2.** 反常维数矩阵非对角时，运行会诱导其他系数。

**3.3.** Leading-log 下
$$
C_1(\mu)=C_1(\Lambda)+(\gamma_1C_1(\Lambda)+aC_2(\Lambda))\log{\mu\over\Lambda},
$$
$$
C_2(\mu)=C_2(\Lambda)+\gamma_2C_2(\Lambda)\log{\mu\over\Lambda}.
$$

## 第 4 章

**4.1.** 用 $\partial_\mu(\phi^2\partial^\mu\phi^2)=
(\partial_\mu\phi^2)^2+\phi^2\Box\phi^2$。

**4.2.** EOM 冗余算符改变 off-shell 二点或三点 Green 函数，但不改变 on-shell S-matrix。

**4.3.** 一阶变化为
$$
\delta S_0=\epsilon\int d^4x\,\phi^3{\delta S_0\over\delta\phi}.
$$
用 ${\delta S_0/\delta\phi}=-(\Box\phi+m^2\phi+\lambda\phi^3/6)$ 的约定，可产生与 $\phi^3\Box\phi$、$\phi^4$、$\phi^6$ 等价的项。

## 第 5 章

**5.1.** $\bar q$ 超荷为 $-1/6$，$\widetilde H$ 为 $-1/2$，$u$ 为 $2/3$，总和为零。

**5.2.** 左手 lepton 双重态无法在维数四构成规范不变 Majorana mass；需 Higgs 插入形成维数五 Weinberg 算符。

**5.3.** 在 $H=(0,(v+h)/\sqrt2)^T$ 下，$D_\mu H$ 中 charged gauge 部分给出
$$
{g^2v^2\over4}W_\mu^+W^{-\mu},
$$
因此 $m_W^2=g^2v^2/4$。

## 第 6 章

**6.1.** $2[\ell]+2[H]=3+2=5$。$(\mathcal O_5)_{rs}$ 携带 $\Delta L=2$，而 $(\mathcal O_5)_{rs}^\dagger$ 携带 $\Delta L=-2$，所以二者不同。对所有有序 flavor 指标求和时，Hermitian 组合为
$$
{1\over2\Lambda_{\rm ref}}\sum_{r,s}
\left[C_5^{rs}(\mathcal O_5)_{rs}
+(C_5^{rs})^*(\mathcal O_5)_{rs}^\dagger\right],
\qquad C_5^{rs}=C_5^{sr}.
$$
结构分类从这一非自伴对中只选一个代表，故唯一类型仍计为一；h.c. 不另增结构数，但不能从拉氏量中省略。

**6.2.** EFT 展开参数不再小，局域截断不能控制误差。

**6.3.** 低能核 beta decay 使用 LEFT；强耦合电弱破缺优先 HEFT；若 Higgs pair production 能区低于新物理尺度且 Higgs 为线性双重态，使用 SMEFT。

**6.4.** 由 $c^{(6)}=C^{(6)}/\Lambda_{\rm ref}^2$，取 $\Lambda_{\rm ref}=1\,\mathrm{TeV}$ 时 $C^{(6)}=1$，取 $\Lambda_{\rm ref}=10\,\mathrm{TeV}$ 时 $C^{(6)}=100$。两者保持同一个有量纲系数 $c^{(6)}$，所以振幅不变；$Q/\Lambda_{\rm ref}$ 随坐标约定改变，不能单独充当物理有效性判据。

## 第 7 章

**7.1.** $[X^2H^2]=2+2+1+1=6$；$[\psi^2H^3]=3+3=6$；$[\psi^2XH]=3+2+1=6$。

**7.2.** Flavor universality 是 Wilson 张量的额外约束，不由 $G_{\mathrm{SM}}$ 推出。

**7.3.** $\bar\ell$ 超荷为 $+1/2$，$e$ 为 $-1$，$H$ 为 $+1/2$，总超荷为零；$B_{\mu\nu}$ 是规范 singlet，Lorentz 指标由 $\sigma^{\mu\nu}B_{\mu\nu}$ 收缩。

## 第 8 章

**8.1.** 缺少尺度和能区时不能判断 $E/\Lambda$ 展开是否可信。

**8.2.** 对二维 Gaussian likelihood，协方差矩阵的本征向量给出误差椭圆主轴。

**8.3.** $F=M^TM=\begin{pmatrix}2&2\\2&2\end{pmatrix}$。本征向量 $(1,1)$ 的本征值为 $4$，$(1,-1)$ 的本征值为 $0$，后者是 flat direction。

## 第 9 章

**9.1.** 传播子下一项 $q^2/m_W^4$ 对应带两个导数的四费米子修正。

**9.2.** $F_{\mu\nu}$ 维数为 $2$，四个 $F$ 给维数八。

**9.3.** Fermi 理论来自树级 $W$ 交换；Euler-Heisenberg EFT 来自一圈电子盒图在 $E\ll m_e$ 下的局域展开。

## 第 10 章

**10.1.** 得到 $(\bar q\gamma_\mu q)(\bar q\gamma^\mu q)$，还需指定颜色收缩。若来自颜色八重态流，需用 $T^A_{ij}T^A_{kl}=\frac12(\delta_{il}\delta_{kj}-\delta_{ij}\delta_{kl}/N_c)$ 投影回 Warsaw 的 $O_{qq}^{(1,3)}$ 及 flavor 置换组合，而不是引入独立的 $O_{qq}^{(8)}$。

**10.2.** 记 $A=\Box+\kappa X$，则
$$
(M^2+A)^{-1}={1\over M^2}-{A\over M^4}+O(M^{-6}),
$$
从而
$$
\Delta\mathcal L_{\rm EFT}^{\rm tree}
=\frac{a^2}{2M^2}X^2
-\frac{a^2}{2M^4}X\Box X
-\frac{a^2\kappa}{2M^4}X^3
+O(M^{-6}).
$$
$[X]=2$，故 $X^2$ 是维数四的 SM quartic 修正，而 $X\Box X$ 与 $X^3$ 都是维数六。展开同时要求 $|\Box|/M^2\ll1$ 和 $|\kappa X|/M^2\ll1$；前者是低动量条件，后者是独立的背景场条件。

**10.3.** 因为
$$
(J_\ell+J_q)^2=J_\ell^2+2J_\ell J_q+J_q^2.
$$
交叉项同时可从两个顺序取出，因此有因子 $2$。

## 第 11 章

**11.1.** $\mathcal O_{HG}\supset (C_{HG}/\Lambda^2)vhG^2$。

**11.2.** 多个 Wilson 系数可同时修正产生、衰变和总宽度，信号强度只给组合。

**11.3.** 若允许不可见宽度或未观测宽度改变，则 $\delta_{\rm tot}$ 可吸收一部分 $\delta_i+\delta_f$ 的效应；因此同一信号强度可对应不同 Wilson 组合。

## 第 12 章

**12.1.** 直接展开并保留到 $\Lambda^{-4}$，得到 $|A_6|^2+2\mathrm{Re}(A_{\mathrm{SM}}A_8^\ast)$。

**12.2.** 粗略要求 $C s/(16\pi\Lambda^2)\lesssim 1$，具体系数依 partial wave 归一化。

**12.3.** 许多 positivity 论证约束前向振幅中 $s^2$ 项的符号；在四维 EFT 中这类项常由维数八算符给出，而维数六常对应 $s$ 项或受 crossing/IR subtleties 影响。

## 第 13 章

**13.1.** $\bar q u\widetilde H$ 在颜色、弱同位旋和超荷下均可收缩为 singlet，再乘 $G_{\mu\nu}^A T^A$。

**13.2.** Dual field strength 含 $\epsilon_{\mu\nu\rho\sigma}$，对应 CP-odd 结构。

**13.3.** 纯玻色 $4+3+8=15$，双费米子 $3+8+8=19$，四费米子 $5+7+8+5=25$。

## 第 14 章

**14.1.** $3\times3$ Hermitian 矩阵有 $3$ 个实对角元和 $3$ 个复非对角元，共 $9$ 个实参数。

**14.2.** MFV 允许 Yukawa spurion 产生 flavor changing，但其方向和大小受 Yukawa 结构约束。

**14.3.** Full flavor 下 $C_{H\ell}^{(1)}$ 为 Hermitian $3\times3$ 矩阵，含 $9$ 个实参数；diagonal nonuniversal 只保留三个实对角元，含 $3$ 个实参数。

## 第 15 章

**15.1.** 对角矩阵时每个系数独立指数运行。

**15.2.** $C_1\simeq(4/16\pi^2)\ln(m_Z/1\mathrm{TeV})$，数值为负且约几百分点量级。

**15.3.** $b$ 物理尺度低于电弱尺度，$W,Z,h,t$ 不再是 LEFT/WET 的动力学自由度。高尺度 SMEFT 系数必须先在 $m_W$ 附近匹配为低能四费米子、dipole 等系数，再运行到 $m_b$。

## 第 16 章

**16.1.** $\mathcal O_{HB}\supset (C_{HB}/\Lambda^2)vhB_{\mu\nu}B^{\mu\nu}$。

**16.2.** 使用 $|\mathcal A_0+\delta\mathcal A|^2=|\mathcal A_0|^2+2\mathrm{Re}(\mathcal A_0^\ast\delta\mathcal A)+O(\delta\mathcal A^2)$。

**16.3.** 设 $G=(1+\epsilon)G'$，则 $G^2=(1+2\epsilon)G'^2$。取 $\epsilon=C_{HG}v^2/\Lambda^2$，与原系数 $1-2C_{HG}v^2/\Lambda^2$ 相乘，线性阶为 $1+O(\Lambda^{-4})$。

## 第 17 章

**17.1.** 表格至少含基、尺度、flavor、CP、输入方案、截断、平方项、能区、误差。

**17.2.** 常见隐含假设包括只开一个系数、固定 flavor、忽略 RG、线性截断和固定输入方案。

**17.3.** 模板应至少含 EFT、基、Wilson 尺度、flavor、CP、输入方案、截断、数据、协方差、有效性切割、工具版本和输出格式。

## 第 18 章

**18.1.** Feynman 参数化后展开
$$
\log(M^2+x(1-x)p^2)=\log M^2+{x(1-x)p^2\over M^2}+O(p^4/M^4),
$$
再用 $\int_0^1x(1-x)dx=1/6$，得到 $-p^2/(96\pi^2M^2)$。

**18.2.** 到 $1/M^2$ 为止包含 $a^2\phi^4$ 型局域项、$ab\phi^6/M^2$ 型项和 $a^2(\partial_\mu\phi^2)^2/M^2$ 型导数项；具体有限部分依重整化方案。

**18.3.** 分部积分给
$$
(\partial_\mu\phi^2)^2=-\phi^2\Box(\phi^2)
$$
差一个边界项。展开 $\Box(\phi^2)$ 后可写成含 $\phi^3\Box\phi$ 和 $\phi^2(\partial\phi)^2$ 的组合；用低阶 EOM 可在算符基之间移动这些项。

## 第 19 章

**19.1.** 需要报告生产模式、衰变道、输入方案、相关 Wilson 系数、定义尺度、是否保留平方项、SM 高阶修正和实验协方差。

**19.2.** 高 $p_T$ bin 中 partonic energy 可能接近 $\Lambda$，此时 $E/\Lambda$ 展开失控；不报告有效性切割就无法判断 EFT 截断是否可信。

**19.3.** 例如 $b\to s\ell^+\ell^-$：高尺度使用 SMEFT，在电弱尺度匹配到 LEFT/WET，再运行到 $b$ 物理尺度并使用 hadronic matrix elements。

**19.4.** 八项检查为 EFT 类型、基与尺度、数据能区、flavor 组合、单系数或多系数、RGE/匹配、协方差、EFT validity cut。

## 第 20 章

**20.1.** Hermitian 矩阵有 $n_g$ 个实对角元和 $n_g(n_g-1)/2$ 个复非对角元，总实参数数为
$$
n_g+2{n_g(n_g-1)\over2}=n_g^2.
$$

**20.2.** 代入 $n_g=2$：一般二指标复矩阵 $8$，Hermitian 二指标 $4$，CP-conserving Hermitian $3$，diagonal $2$，universal $1$，generic Hermitian 四指标 $16$，同种流交换对称四指标 $10$，一般 chiral 四指标 $32$。

**20.3.** 标准模型规范群只约束 gauge 指标；generation 是重复的同表示副本。Flavor universality 是 Wilson 张量在 generation 空间中与单位矩阵成比例的额外假设。

## 第 21 章

**21.1.** 对
$$
e=gg'(g^2+g'^2)^{-1/2}
$$
取对数微分，得到
$$
{\delta e\over e}=s^2{\delta g\over g}+c^2{\delta g'\over g'},
$$
再乘以 $2$ 得 $\delta\alpha/\alpha$。

**21.2.** 当 $\epsilon_\alpha=\epsilon_W=0$，
$$
{\delta m_W^2\over m_W^2}
={-c^2\epsilon_Z-s^2\epsilon_G\over c^2-s^2}.
$$

**21.3.** 输入方案决定哪些观测量被用来反解 $g,g',v$。换输入方案会把同一 Wilson 组合在“输入位移”和“直接观测量修正”之间重新分配。

## 第 22 章

**22.1.** $h\to\gamma\gamma$ 同时受 $hF_{\mu\nu}F^{\mu\nu}$ contact、SM loop 归一化、top/W coupling shift、输入方案和总宽度影响，因此不是单系数观测量。

**22.2.** 至少报告 bin 能区、partonic energy proxy、是否剔除 $E>\Lambda$ 区域、是否保留维数六平方项、以及维数八误差估计。

**22.3.** Flavor 观测量通常在低于电弱尺度测量；$W,Z,h,t$ 已被积掉。若不做 SMEFT 到 LEFT/WET 匹配，高尺度 Wilson 系数无法与低能 hadronic matrix elements 相连。

**22.4.** Contact 振幅随 $s/\Lambda^2$ 增大，因此高 $s$ 提升灵敏度；同一个增长也使 $s/\Lambda^2$ 展开接近失控，因此有效性风险同步增加。

## 第 23 章

**23.1.** 由
$$
{\delta m_W\over m_W}={1\over2}{\delta m_W^2\over m_W^2}
$$
可知响应行是第 23.2 节 $m_W^2$ 响应行的一半。

**23.2.** 约束方向为 $(r_g,r_t)$。一个 flat direction 可取
$$
(c_g,\delta y_t)=(r_t,-r_g),
$$
因为 $r_gc_g+r_t\delta y_t=0$。

**23.3.** 单参数模型只有一个 Wilson 方向，第二个 bin 只增加同一方向的信息量；若两个 bin 误差独立，Fisher 信息相加，误差减小，但参数空间维数不增加。

## 第 24 章

**24.1.** 若 $s^2=0.23$，则 $c^2=0.77$，$c^2-s^2=0.54$。因此
$$
M_{m_W}={1\over2}
\begin{pmatrix}
0.23/0.54&-0.23/0.54&-0.77/0.54&1
\end{pmatrix}
\simeq
\begin{pmatrix}
0.213&-0.213&-0.713&0.500
\end{pmatrix}.
$$

**24.2.** 两行相同，故秩为 $1$。零方向由 $2a_g+2\delta y_t-\delta_\Gamma=0$ 给出，可取 $(1,-1,0)$ 和 $(1,0,2)$。

**24.3.** 新 Fisher 信息为
$$
F={0.25^2\over0.1^2}+{1^2\over0.5^2}=6.25+4=10.25,
$$
故
$$
\sigma_c=1/\sqrt{10.25}\simeq0.312.
$$
