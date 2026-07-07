# 算符与命题台账

审计日期：2026-07-08

本文件为教材内部交叉引用台账。它不替代正文定义，而记录每类对象在哪里定义、哪里使用、哪些部分依赖外部文献。

## 1. EFT 命题链

| 对象 | 位置 | 书内状态 | 外部依赖 |
| --- | --- | --- | --- |
| EFT 作为低能局域展开 | 第 1 章 | 定义和推导已闭合 | Weinberg、Georgi、Burgess 作为历史和教材来源 |
| Wilson coefficient | 第 2 章 | 定义已闭合 | 无 |
| 树级匹配 | 第 2、10 章 | 三个 worked examples 已闭合 | 无 |
| 一圈匹配 | 第 18 章 | 标量行列式例子已闭合 | SMEFT 具体模型匹配仍依赖专门文献 |
| 幂计数 | 第 3 章 | 规则已闭合 | NDA 数值约定依赖文献口径 |
| RGE 与 leading log | 第 3、15 章 | 最小矩阵例子已闭合 | 完整 SMEFT 反常维数矩阵为外部输入 |
| EOM 与基变换 | 第 4 章 | 原理已闭合 | 逐项基转换未展开 |
| 破缺相输入方案 | 第 16 章 | 线性例子已闭合 | 完整 Feynman 规则为外部输入 |
| Flavor 参数计数 | 第 20 章 | 通用张量计数已闭合 | Warsaw exact 2499 计数的逐项 Fierz 审计为外部输入 |
| 电弱输入方案 | 第 21 章 | $\{\alpha,G_F,m_Z\}$ 线性系统已闭合 | 具体 Wilson-to-epsilon 映射依规范化 |
| Observables-to-operators 图谱 | 第 22 章 | 结构级图谱已闭合 | 数值响应矩阵依数据集和工具 |

## 2. Warsaw basis 结构台账

本书采用 baryon number 守恒、未展开 flavor 指标的 Warsaw basis 计数。第十三章列出 $15+19+25=59$ 个结构。

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
| 完整 SMEFT RGE 矩阵 | 第 15 章给最小例子 | 专门文献和工具链更适合承载 |
| 全局拟合数值复现 | 第 17、19 章给元数据标准 | 依赖数据集、协方差和工具版本 |
| Basis conversion 全表 | 附录 E 给接口和例子 | 完整逐项表依目标基规范化 |

## 4. 使用规则

引用一个 Wilson 系数时，正文必须同时给出：

1.  算符基；
2.  flavor 指标；
3.  CP 假设；
4.  定义尺度；
5.  截断阶数；
6.  与可观测量相连时的输入参数方案。

缺少这些信息时，该表达式只能作为形式公式，不能作为可复核物理结果。
