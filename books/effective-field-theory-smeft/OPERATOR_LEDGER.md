# 算符与命题台账

审计日期：2026-07-15

本文件为教材内部交叉引用台账。它不替代正文定义，而记录每类对象在哪里定义、哪里使用、哪些部分依赖外部文献。

## 1. EFT 命题链

| 对象 | 位置 | 书内状态 | 外部依赖 |
| --- | --- | --- | --- |
| 尺度类型与 Wilson 坐标 | 序章、第 1、6、17、22--24 章、`NOTATION.md` | $M_{\rm gap}$、$\Lambda_{\rm ref}$、$\mu_{\rm match}$ 已分型；$\rho_{\rm loc}=Q/M_{\rm gap}$ 与 $\rho_{\rm ins,i}^{(d)}=|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ 已分开；参考尺度重标度由命题 1.5 证明 | 未知 $M_{\rm gap}$ 时，$\rho_{\rm loc}$ 只能条件化 |
| Hermitian SMEFT 拉氏量与 dagger sectors | 第 4、6、7、13、14、20 章、`NOTATION.md` | 自伴族系数约束与非自伴 $C\mathcal O+C^*\mathcal O^\dagger$ 已显式；结构数和 Wilson 实参数数已分型 | 维数五分类与 Warsaw 结构分类分别使用 SMEFT-D5、SMEFT-WARSAW6 |
| EFT 作为低能局域/渐近展开 | 第 1 章 | 重传播子一致余项界、解析顶点局域化、统一渐近定义及 divergent asymptotic 反例已闭合 | 一般多圈 decoupling 使用 EFT-DEC；一般 QFT 级数不声称收敛 |
| Wilson coefficient | 第 2 章 | 定义已闭合 | 无 |
| 树级匹配 | 第 2、10 章 | 三个 worked examples 已闭合；重实 singlet 的完整二次逆核已展开到 $M^{-4}$，同时保留 $X\Box X$ 与 $\kappa X^3$ 并声明 $|\Box|/M^2$、$|\kappa X|/M^2$ 条件 | 无；singlet 系数为书内完成平方推导 |
| 圈级 matching 与 IR 消去 | 第 2、18 章 | 共同 soft 部分消去由命题 2.7B 证明；标量行列式例子已闭合 | 区域分解方法使用 EFT-REGIONS；具体 SMEFT 模型匹配仍依赖专门文献 |
| 幂计数与截断 | 第 1、3、6、8、12 章及附录 D 12.1 | $(p,L)$ bidegree、多次插入与可观测量展开已闭合；$A_2^{[5,5]}$、$A_4^{[6,6]}$ 和允许的 $d=5$ 组合已显式；误差模型与严格余项已区分 | NDA/强耦合数值层级依赖 UV 假设 |
| RGE 与 leading log | 第 3、15 章 | 裸作用量固定逆转置约定、路径有序解、换基协变和尺度抵消已闭合 | 线性维数六完整矩阵使用 SMEFT-RGE6；$p=4$ 非线性 mixing 未外推 |
| EOM、基商与 RG descent | 第 2、4 章 | 一/二阶场重定义、typed quotient、命题 4.9A 不变子空间判据已闭合 | 量子等价使用 EFT-EQ/EFT-EQ-HO；SMEFT quotient 闭合使用 SMEFT-EOM-RG |
| Gauge/BRST 与 evanescent 投影 | 第 2-4 章 | 计算顺序与四维/DR 边界已声明 | 一般 algebraic renormalization 使用 EFT-REN |
| SMEFT 有效性 | 第 6、8、12、17、22--24 章及附录 D | 参考尺度反例、$\rho_{\rm loc}$ 局域性、$\rho_{\rm ins}$ 插入层级与 loop/log 检查已分别定义；第 24 章基准矩阵不再从 $x_b=s_b/\Lambda_{\rm ref}^2$ 推断局域性 | 未知 $M_{\rm gap}$ 与 UV coupling 只能条件化；须先有 $M_{\rm gap}=M$ 的单物理尺度 matching，才可另选 $\Lambda_{\rm ref}=M$ |
| 破缺相输入方案 | 第 16 章 | 线性例子已闭合 | 完整 Feynman 规则为外部输入 |
| Flavor 参数计数 | 第 20 章 | 通用张量计数已闭合 | Warsaw exact 2499 计数的逐项 Fierz 审计为外部输入 |
| 电弱输入方案 | 第 21 章 | $\{\alpha,G_F,m_Z\}$ 线性系统已闭合 | 具体 Wilson-to-epsilon 映射依规范化 |
| Observables-to-operators 图谱 | 第 22 章 | 结构级图谱已闭合；dilepton map 已分开 Wilson 插入大小与 $Q/M_{\rm gap}$ validity cut | 数值响应矩阵依数据集和工具 |
| 响应矩阵与 Fisher 信息 | 第 8、22--24 章 | 基变换协变性、秩亏例子、dilepton 的 $\rho_{\rm loc}/\rho_{\rm ins}$ 分型与 flavor 矩阵链已显式 | 真实 $M_{ai}$、$\mathsf Q_{aij}$、协方差、PDF 和过程权重依具体数据集 |

## 2. Warsaw basis 结构台账

本书采用 baryon number 守恒、未展开 flavor 指标的 Warsaw basis 计数。第十三章列出 $15+19+25=59$ 个结构；每个非自伴 dagger pair 只列一个代表，Hermitian conjugate 不另增结构数，但必须在拉氏量中恢复。

按第十三章列表逐项核对 Hermiticity：

| Warsaw 分区 | 自伴算符族 | 非自伴 pair 的代表 | 合计 |
| --- | ---: | ---: | ---: |
| 纯玻色 | 15 | 0 | 15 |
| 双费米子 | 7 | 12 | 19 |
| 四费米子 | 20 | 5 | 25 |
| **总计** | **42** | **17** | **59** |

这里“自伴算符族”允许 dagger 置换 flavor 指标；对应 Wilson 张量须满足 Hermiticity 关系。$42+17=59$ 是结构数，不是 flavor 展开后的实参数数。

| 扇区 | 数目 | 结构 | 位置 |
| --- | --- | --- | --- |
| $X^3$ | 4 | ${\cal O}_G,{\cal O}_{\widetilde G},{\cal O}_W,{\cal O}_{\widetilde W}$ | 13.2 |
| $H^6$ | 1 | ${\cal O}_H$ | 13.2 |
| $H^4D^2$ | 2 | ${\cal O}_{H\Box},{\cal O}_{HD}$ | 13.2 |
| $X^2H^2$ | 8 | ${\cal O}_{HG},{\cal O}_{H\widetilde G},{\cal O}_{HW},{\cal O}_{H\widetilde W},{\cal O}_{HB},{\cal O}_{H\widetilde B},{\cal O}_{HWB},{\cal O}_{H\widetilde WB}$ | 13.2 |
| $\psi^2H^3$ | 3 | ${\cal O}_{eH},{\cal O}_{uH},{\cal O}_{dH}$ | 13.3 |
| $\psi^2XH$ | 8 | ${\cal O}_{eB},{\cal O}_{eW},{\cal O}_{uG},{\cal O}_{uW},{\cal O}_{uB},{\cal O}_{dG},{\cal O}_{dW},{\cal O}_{dB}$ | 13.3 |
| $\psi^2H^2D$ | 8 | ${\cal O}_{H\ell}^{(1)},{\cal O}_{H\ell}^{(3)},{\cal O}_{He},{\cal O}_{Hq}^{(1)},{\cal O}_{Hq}^{(3)},{\cal O}_{Hu},{\cal O}_{Hd},{\cal O}_{Hud}$ | 13.3 |
| $(\bar LL)(\bar LL)$ | 5 | ${\cal O}_{\ell\ell},{\cal O}_{qq}^{(1)},{\cal O}_{qq}^{(3)},{\cal O}_{\ell q}^{(1)},{\cal O}_{\ell q}^{(3)}$ | 13.4 |
| $(\bar RR)(\bar RR)$ | 7 | ${\cal O}_{ee},{\cal O}_{uu},{\cal O}_{dd},{\cal O}_{eu},{\cal O}_{ed},{\cal O}_{ud}^{(1)},{\cal O}_{ud}^{(8)}$ | 13.4 |
| $(\bar LL)(\bar RR)$ | 8 | ${\cal O}_{\ell e},{\cal O}_{\ell u},{\cal O}_{\ell d},{\cal O}_{qe},{\cal O}_{qu}^{(1)},{\cal O}_{qu}^{(8)},{\cal O}_{qd}^{(1)},{\cal O}_{qd}^{(8)}$ | 13.4 |
| scalar/tensor 四费米子 | 5 | ${\cal O}_{\ell edq},{\cal O}_{\ell equ}^{(1)},{\cal O}_{\ell equ}^{(3)},{\cal O}_{quqd}^{(1)},{\cal O}_{quqd}^{(8)}$ | 13.4 |

计数检查：
$$
4+1+2+8+3+8+8+5+7+8+5=59.
$$

## 3. 研究边界台账

| 对象 | 当前状态 | 原因 |
| --- | --- | --- |
| Baryon-number violating dimension-six operators | 第 13.5 节只列名 | 主线不讨论质子衰变和 GUT 匹配 |
| 完整 flavor 参数计数 | 第 14 章给原则和基本表 | 全量表依赖具体 flavor 假设 |
| 完整 dimension-eight basis | 第 12 章给边界 | 结构规模大，作为高级研究边界 |
| 完整 SMEFT 维数六线性 RGE 矩阵 | 第 3、15 章给接口与最小例子 | 数值矩阵由 SMEFT-RGE6 外部承载 |
| $p=4$ 双插入/维数八 RGE | 第 3 章只给一般非线性类型 | 完整 counterterm basis 与 anomalous tensors 超出第一版 |
| 全局拟合数值复现 | 第 17、19、22 章给元数据标准 | 依赖数据集、协方差、条件化的 $M_{\rm gap}$ 和工具版本 |
| Basis conversion 全表 | 附录 E 给接口和例子 | 完整逐项表依目标基规范化 |

## 4. 使用规则

引用一个 Wilson 系数时，正文必须同时给出：

1.  算符基；
2.  flavor 指标；
3.  CP 假设；
4.  定义尺度；
5.  $\Lambda_{\rm ref}$、条件化的 $M_{\rm gap}$，以及可观测量所用的 $\rho_{\rm loc}$ 与 $\rho_{\rm ins}$；
6.  保留的 $(p,L)$、多次插入和 evanescent/EOM projection；
7.  与可观测量相连时的输入参数方案及 $Q_{\max}$。

缺少这些信息时，该表达式只能作为形式公式，不能作为可复核物理结果。
