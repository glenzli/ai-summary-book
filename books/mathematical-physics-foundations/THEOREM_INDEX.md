# 定理索引与证明责任

状态含义固定如下：

- P：相对于序章先修合同，正文给出覆盖全部结论的完整证明。
- S：正文给出标准物理形式推导，并保留 regulator、微扰阶数、定义域或有效能区边界。
- E：正文精确陈述外部输入，证明路线不承担完成证明的责任；来源编号回链到 [SOURCES.md](SOURCES.md)。

| 编号 | 状态 | 精确内容 | 直接依赖或来源 |
|---|---|---|---|
| 0.1 | P | 未登记状态的文字不能作为无条件前提 | P/S/E 引用规则 |
| 0.2 | P | 有限无环章节依赖图保持外部输入状态 | 有限 DAG 拓扑排序 |
| 1.1 | P | \((r,s)\)-张量分量的坐标变换律 | 链式法则、张量基 |
| 1.2 | P | 光滑微分形式上 \(d^2=0\) | 混合偏导交换、交错性 |
| 1.3 | P | Cartan 公式 \(\mathcal L_X=d\iota_X+\iota_Xd\) | 外微分公式、Lie 括号 |
| 1.4 | P | \(C^2\) Lagrangian 的固定端点 Euler--Lagrange 方程 | 分部积分、bump 函数 |
| 2.1 | P | 非退化二形式唯一确定光滑 Hamilton 向量场 | \(\omega^\flat\) 的光滑逆 |
| 2.2 | P | Poisson 括号的双线性、反对称、Leibniz 与 Jacobi 恒等式 | 1.3, \(d\omega=0\) |
| 2.3 | P | Hamilton 流在其存在区间保持辛形式 | 1.3, 2.1 |
| 2.4 | P | Hamilton 群作用下的不变 Hamiltonian 产生 moment-map 守恒量 | moment map 定义 |
| 2.5 | E | 辛流形的局部 Darboux 标准形 | E-2.5 |
| 3.1 | P | \(T_eG\) 与左不变向量场 Lie 代数自然同构 | 左平移、括号自然性 |
| 3.2 | P | 有限维光滑群表示的微分是 Lie 代数表示 | 相关向量场与括号 |
| 3.3 | P | 有限维复不可约表示的 Schur 引理 | 核、像、复特征值 |
| 3.4 | E | 紧群 Peter--Weyl 稠密性与正则表示分解 | E-3.4 |
| 3.5 | E | \(SU(2)\) 有限维复不可约表示的最高权分类 | E-3.5 |
| 4.1 | P | 主丛局部联络形式的规范变换律 | 联络等变性、Maurer--Cartan 形式 |
| 4.2 | P | 曲率在规范变换下共轭协变 | 4.1、Maurer--Cartan 恒等式 |
| 4.3 | P | Bianchi 恒等式 \(D_AF_A=0\) | 分次 Leibniz、Jacobi |
| 4.4 | P | invariant polynomial 的 Chern--Weil 形式闭合 | 4.3、无穷小 Ad-不变性 |
| 4.5 | E | Chern--Weil 类的联络独立性及特征类识别 | E-4.5, E-A.2 |
| 5.1 | P | 稠密定义算符的伴随闭；对称蕴含 \(A\subset A^*\) | 伴随定义、图收敛 |
| 5.2 | P | 可闭算符的序列判据与唯一最小闭扩张 | 图闭包 |
| 5.3 | E | Banach 空间闭图定理 | E-5.3 |
| 5.4 | E | von Neumann deficiency-index 自伴扩张判据 | E-5.4 |
| 5.6 | E | 无界自伴算符的 PVM 谱定理与 Borel 函数演算定义域 | E-5.6 |
| 5.8 | E | 强连续酉群的 Stone 定理 | E-5.8 |
| 5.9 | P | Schwartz 函数 Fourier 导数/乘法恒等式及 \(\mathcal S\)-保持性 | 分部积分、\(L^1\) 估计 |
| 5.10 | E | Schwartz 空间上的 Fourier 反演与拓扑自同构 | E-5.10 |
| 5.11 | E | \(s>k+n/2\) 时 \(H^s(\mathbb R^n)\hookrightarrow C_b^k\) | E-5.11 |
| 6.1 | P | 有界 Hamiltonian 与范数可微有界可观测量的 Heisenberg 方程 | 算符范数乘积法则 |
| 6.2 | E | 有限自由度 Weyl CCR 的 Stone--von Neumann 唯一性 | E-6.2 |
| 6.3 | E | 保转移概率射线双射的 Wigner 定理 | E-6.3 |
| 6.4 | P | 有限维不可约角动量表示的 \(j,m,J^2\) 分类 | 3.3、升降算符 |
| 6.5 | S | WKB 首阶相位满足 Hamilton--Jacobi 方程 | S-6.5；平滑相位区、\(\hbar\) 展开 |
| 7.1 | P | 一阶局部场论的 Euler--Lagrange 方程 | 紧支撑变分、bump 函数 |
| 7.2 | P | 竖直连续作用量对称的 Noether 第一守恒流 | 7.1、全散度边界项 |
| 7.3 | P | 无显含坐标 Lagrangian 的 canonical 能动张量壳上守恒 | 7.1、链式法则 |
| 7.4 | P | spacelike-compact Klein--Gordon 解的辛配对与 Cauchy 面无关 | 散度定理、支撑条件 |
| 7.5 | E | 全局双曲时空 Green-hyperbolic 算子的先进/推迟 Green 算子 | E-7.5 |
| 8.1 | P | 正定实矩阵的有限维 Euclidean Gaussian 源积分 | 有限维谱定理、Tonelli |
| 8.2 | P | 有限维中心 Gaussian 的完整 Wick 配对公式 | 8.1、矩母函数 |
| 8.3 | S | 固定有限 regulator、固定阶数的微扰/Wick 展开 | 8.2, S-8.3；Taylor 展开 |
| 8.4 | E | massive Euclidean \(\phi^4_4\) 的 BPHZ 逐阶重整化 | E-8.4 |
| 8.5 | S | 本书 \(Z_\phi,\gamma_\phi\) 约定下的 Callan--Symanzik 方程 | S-8.5；固定裸参数求导 |
| 8.6 | S | 有质量隙、阈值下解析时的 EFT 局部 \(E/M\) 展开 | S-8.6；明示 power counting 与截断 |
| 8.7 | S | 局部规范切片上的 Faddeev--Popov 形式恒等式 | S-8.7；可逆 FP 算子、无 Gribov 假设 |
| 9.1 | P | 对称 Fock 有限粒子域上的 CCR | 对称张量定义 |
| 9.2 | P | smeared 自由 Klein--Gordon 场的等时 CCR | 9.1、Fourier 反演 |
| 9.3 | E | Wightman \(n\) 点分布的存在与酉唯一重构 | E-9.3 |
| 9.4 | P | Wightman 谱条件排除负能一粒子谱 | 联合谱包含关系 |
| 9.5 | P | 自由 Fock 真空的算符 Wick 配对公式 | 9.1、归纳 |
| 10.1 | P | Yang--Mills 作用量的规范不变性 | 4.2、Ad-不变内积 |
| 10.2 | P | 无边界或紧支撑变分下 \(D_A*F_A=0\) | 4.3、协变分部积分 |
| 10.3 | S | 局部 BRST 变换的分次形式幂零性 | S-10.3；ghost 奇性、分次 Jacobi |
| 10.4 | S | 四维 Fujikawa regulator 下的手征异常密度 | S-10.4；热核 cutoff、表示归一化 |
| 10.5 | E | 闭偶维 spin 流形扭正手征 Dirac 算子的指标公式 | E-10.5 |
| A.1 | P | 有限维复正规算符的标准正交本征基 | 维数归纳、正规性 |
| A.2 | E | de Rham 上同调与实奇异上同调自然同构 | E-A.2 |
| A.3 | E | Fubini--Tonelli、Radon--Nikodym 与 \(C_0\)-Riesz 表示 | E-A.3a--c |
| B.1 | P | 链映射诱导同调函子 | cycle/boundary 良定义性 |
| B.2 | E | 闭定向 Riemann 流形的 Hodge 代表定理 | E-B.2 |
| B.3 | E | 闭流形椭圆算子的 Fredholm 性与指标稳定性 | E-B.3 |
| C.1 | P | \(L^1\) 卷积的良定义性及 Fourier 乘积公式 | E-A.3a、换元 |
