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

**8.1.** $c_i=C_i/\Lambda_{\rm ref}^2$ 的限制只固定低能有量纲系数组合；例如树级
matching 可能给 $c_i\sim g_*^2/M^2$，不另给 $g_*$ 就不能反推出 $M$。还须报告
条件化的物理谱隙 $M_{\rm gap}$、数据硬尺度 $Q$ 与 $Q/M_{\rm gap}$ cut。
$\Lambda_{\rm ref}$ 只是 Wilson 坐标归一化，不能替代这个物理质量信息。

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

**12.1.** 在题设的固定 sesquilinear form 下，按正文的总逆尺度次数 $p$ 展开：
$$
\begin{aligned}
\langle A,A\rangle_0
&=\langle A_0,A_0\rangle_0
+{2\over\Lambda_{\rm ref}^2}\operatorname{Re}\langle A_0,A_2\rangle_0\\
&\quad+{1\over\Lambda_{\rm ref}^4}
\left[
\langle A_2,A_2\rangle_0
+2\operatorname{Re}\langle A_0,A_4\rangle_0
\right]
+O(\Lambda_{\rm ref}^{-6}).
\end{aligned}
$$
这里下标 $2,4$ 是 $p$-grading，不是单个算符的 canonical dimension。只列高维
顶点插入部分，$p=\sum_v n_v(d_v-4)$ 给出
$$
\begin{aligned}
A_2^{\rm ins}
&=A_2^{[6]}+A_2^{[5,5]},\\
A_4^{\rm ins}
&=A_4^{[8]}+A_4^{[6,6]}+A_4^{[5,7]}
+A_4^{[5,5,6]}+A_4^{[5,5,5,5]}.
\end{aligned}
$$
因此插入部分对 $p=4$ 可观测量系数的贡献完整展开为
$$
\begin{aligned}
\langle A_2^{\rm ins},A_2^{\rm ins}\rangle_0
&=\langle A_2^{[6]},A_2^{[6]}\rangle_0
+2\operatorname{Re}\langle A_2^{[6]},A_2^{[5,5]}\rangle_0
+\langle A_2^{[5,5]},A_2^{[5,5]}\rangle_0,\\
2\operatorname{Re}\langle A_0,A_4^{\rm ins}\rangle_0
&=2\operatorname{Re}\left\langle A_0,
A_4^{[8]}+A_4^{[6,6]}+A_4^{[5,7]}
+A_4^{[5,5,6]}+A_4^{[5,5,5,5]}\right\rangle_0.
\end{aligned}
$$
若 exact lepton number 或其他选择定则禁止 dimension-five/seven vertices，才可删除
相应项；输入参数重展开还可对 $A_2,A_4$ 增加同阶项。所谓 dimension-six square 是
$$
\langle A_2^{[6]},A_2^{[6]}\rangle_0,
$$
它来自一次 dimension-six 插入振幅的平方。两次 dimension-six 插入振幅则是
$A_4^{[6,6]}$，属于振幅本身，并通过
$2\operatorname{Re}\langle A_0,A_4^{[6,6]}\rangle_0$ 与 SM 干涉；二者不是同一个
对象。$\square$

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

**17.1.** 表格至少含 EFT、基、$\Lambda_{\rm ref}$、Wilson 定义尺度 $\mu$、flavor、
CP、输入方案、截断、平方项、数据硬尺度 $Q$、条件化的 $M_{\rm gap}$、协方差与
误差。有效性栏应报告 $Q_{\max}/M_{\rm gap}$；插入层级另报
$|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$，不能把两栏合成
$Q/\Lambda_{\rm ref}$。

**17.2.** 常见隐含假设包括只开一个系数、固定 flavor、忽略 RG、线性截断和固定
输入方案；还应检查图是否把任意 Wilson normalization $\Lambda_{\rm ref}$ 当成物理
阈值。只有在 UV matching 中先声明单一物理重尺度 $M$、验证 $M_{\rm gap}=M$，
才可再选择 $\Lambda_{\rm ref}=M$；这是模型假设加坐标选择，不是一般 validity
判据。

**17.3.** 模板应至少含 EFT、基、$\Lambda_{\rm ref}$、$\mu_{\rm match}$、$\mu_{\rm obs}$、
条件化的 $M_{\rm gap}$、flavor、CP、输入方案、截断、数据、协方差、基于
$Q/M_{\rm gap}$ 的有效性切割、Wilson 插入层级、工具版本和输出格式。

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

**19.2.** 高 $p_T$ dilepton bin 的硬尺度 $Q$ 可能接近最近遗漏物理奇点
$M_{\rm gap}$；此时局域展开不受控。因此必须报告 $Q$ 的构造、条件化的
$M_{\rm gap}$ 与逐 bin 的 $Q/M_{\rm gap}$ cut。Wilson 插入大小
$|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ 是另一项检查。

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

**22.2.** 至少报告 bin 能区、构造 $Q$ 的 partonic proxy、条件化的
$M_{\rm gap}$、逐 bin 的 $Q/M_{\rm gap}$ 及 cut、$\Lambda_{\rm ref}$ 与 Wilson 定义
尺度 $\mu$。另报 $|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ 的插入层级、是否保留
维数六平方项及维数八误差估计；$Q/\Lambda_{\rm ref}$ 单独不是有效性判据。

**22.3.** Flavor 观测量通常在低于电弱尺度测量；$W,Z,h,t$ 已被积掉。若不做 SMEFT 到 LEFT/WET 匹配，高尺度 Wilson 系数无法与低能 hadronic matrix elements 相连。

**22.4.** Contact insertion 随
$|C_{\ell q}^{(6)}(\mu)|s/\Lambda_{\rm ref}^2$ 增大，所以高 $s$ 提升 Wilson
灵敏度；局域展开是否接近边界则由 $\sqrt s/M_{\rm gap}$ 判断。二者是独立检查：
改变 $\Lambda_{\rm ref}$ 并相应重标度 $C_{\ell q}^{(6)}$ 不改变前者，也不改变物理
$M_{\rm gap}$。只有在 UV matching 中先声明单一物理重尺度 $M$、验证
$M_{\rm gap}=M$，再选择 $\Lambda_{\rm ref}=M$ 时，两种尺度才会数值相同；两项检查
仍不能合并。

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
