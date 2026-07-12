# 资料源

本文件记录《有效场论与标准模型有效场论》当前版本使用的主要来源。在线版本访问日期：2026-07-11；另行复核者在条目中单列日期。粗体 source key 供正文与附录 B 稳定引用；序号只用于阅读，不作为 locator。

## EFT 基础

1.  **EFT-BURGESS.** C. P. Burgess, *Introduction to Effective Field Theory*, https://arxiv.org/abs/hep-th/0701053.
    用途：EFT 的尺度分离、局域作用量、幂计数和匹配的教材来源。
2.  S. Weinberg, *Phenomenological Lagrangians*, Physica A 96, 327-340 (1979).
    用途：对称性允许的最一般低能拉氏量思想。
3.  H. Georgi, *Effective Field Theory*, Annual Review of Nuclear and Particle Science 43, 209-252 (1993).
    用途：EFT 的物理组织原则和幂计数口径。
4.  J. F. Donoghue, *General Relativity as an Effective Field Theory: The Leading Quantum Corrections*, https://arxiv.org/abs/gr-qc/9405057.
    用途：引力 EFT 例子。
5.  J. Gasser and H. Leutwyler, *Chiral Perturbation Theory to One Loop*, Annals Phys. 158, 142-210 (1984).
    用途：手征微扰论作为 Goldstone EFT 的标准来源。
6.  **EFT-AC.** T. Appelquist and J. Carazzone, *Infrared Singularities and Massive Fields*, Phys. Rev. D 11, 2856-2861 (1975), https://doi.org/10.1103/PhysRevD.11.2856.
    用途：第一章外部输入 1.6 的 decoupling theorem；正文不把其结论扩张到耦合随重质量增长或强耦合情形。
7.  **EFT-KOS.** S. Kamefuchi, L. O'Raifeartaigh, A. Salam, *Change of Variables and Equivalence Theorems in Quantum Field Theories*, Nucl. Phys. 28, 529-549 (1961), https://doi.org/10.1016/0029-5582(61)90056-6.
    用途：第二章外部输入 2.7A 的 on-shell equivalence theorem 原始来源；原文以 point transformations 和 canonical formalism 为主。
8.  **EFT-ARZT.** C. Arzt, *Reduced Effective Lagrangians*, Phys. Lett. B 342, 189-195 (1995), https://arxiv.org/abs/hep-ph/9304230, https://doi.org/10.1016/0370-2693(94)01419-D.
    用途：EFT 中使用经典 EOM 约化冗余项以及包含量子圈时的适用边界。
9.  **EFT-EQ-HO.** J. C. Criado and M. Perez-Victoria, *Field redefinitions in effective theories at higher orders*, JHEP 03 (2019) 038, https://arxiv.org/abs/1811.09413v2, https://doi.org/10.1007/JHEP03(2019)038.
    用途：第四章命题 4.5A/警告 4.5B 的高阶场重定义、EOM 非等价、matching 与 gauge covariance 边界；主要 locator 为 Secs. 3--5 及 Apps. B、D。
10. **EFT-REGIONS.** M. Beneke and V. A. Smirnov, *Asymptotic expansion of Feynman integrals near threshold*, Nucl. Phys. B 522, 321-344 (1998), https://arxiv.org/abs/hep-ph/9711391, https://doi.org/10.1016/S0550-3213(98)00138-2.
    用途：第二章外部输入方法 2.7C 的 expansion by regions；Sec. 2 给出区域步骤，Sec. 3 形式化 threshold expansion。其范围不是任意非微扰 QFT 的收敛定理。
11. **EFT-BRST.** O. Piguet and S. P. Sorella, *Algebraic Renormalization: Perturbative Renormalization, Symmetries and Anomalies*, Lecture Notes in Physics Monographs 28 (1995), https://doi.org/10.1007/978-3-540-49192-7.
    用途：第三、四章关于 Ward/Slavnov--Taylor identities、Yang--Mills 重整化和 BRST cohomology 的外部边界；主要 locator 为 pp. 21-79。

## SMEFT 核心

12. **SMEFT-W79.** S. Weinberg, *Baryon- and Lepton-Nonconserving Processes*, Phys. Rev. Lett. 43, 1566-1570 (1979).
    用途：标准模型场内容下维数五 lepton-number-violating 算符的原始来源。正文把非自伴 Weinberg pair 计作一个结构类型，同时在 Hermitian 拉氏量中显式恢复带共轭系数的 dagger。
13. W. Buchmuller and D. Wyler, *Effective Lagrangian Analysis of New Interactions and Flavor Conservation*, Nucl. Phys. B268 (1986).
    用途：早期标准模型维数六算符分类。
14. **SMEFT-WARSAW.** B. Grzadkowski, M. Iskrzynski, M. Misiak, J. Rosiek, *Dimension-Six Terms in the Standard Model Lagrangian*, JHEP 10 (2010) 085, https://arxiv.org/abs/1008.4884v3. 本次复核：2026-07-12。
    用途：Sec. 3、Eq. (3.1) 给出唯一维数五类型；Sec. 3、Tables 2-3 给出 Warsaw basis；Secs. 5-7 给出零、二、四费米子 sector 的独立性分析。摘要所述 $15+19+25=59$ 假设 baryon number 守恒、不展开 flavor，并且不把非自伴算符的 Hermitian conjugate 另计为新结构；构造 Hermitian 拉氏量时仍须恢复 h.c.，故该结构数不能直接解释为 Wilson 实参数数。
15. **SMEFT-JMT-I.** E. E. Jenkins, A. V. Manohar, M. Trott, *Renormalization Group Evolution of the Standard Model Dimension Six Operators I: Formalism and lambda Dependence*, JHEP 10 (2013) 087, https://arxiv.org/abs/1308.2627v4.
    用途：维数六一圈 RGE 的 formalism 与 Higgs self-coupling 部分；Sec. 3、Eqs. (3.8)-(3.11) 用于 EOM 子空间与 quotient 闭合，正文不重算矩阵。
16. **SMEFT-JMT-II.** E. E. Jenkins, A. V. Manohar, M. Trott, *Renormalization Group Evolution of the Standard Model Dimension Six Operators II: Yukawa Dependence*, JHEP 01 (2014) 035, https://arxiv.org/abs/1310.4838v3.
    用途：维数六一圈 anomalous-dimension matrix 的 Yukawa 部分和 flavor mixing。
17. **SMEFT-JMT-III.** R. Alonso, E. E. Jenkins, A. V. Manohar, M. Trott, *Renormalization Group Evolution of the Standard Model Dimension Six Operators III: Gauge Coupling Dependence and Phenomenology*, JHEP 04 (2014) 159, https://arxiv.org/abs/1312.2014v5.
    用途：维数六一圈 anomalous-dimension matrix 的 gauge-coupling 部分；与前两篇合并才构成正文外部输入 SMEFT-RGE6。另用于维数六插入修正 $d\le4$ SM 参数 running 的边界。
18. **SMEFT-WORKFLOW.** G. Isidori, F. Wilsch, D. Wyler, *The Standard Model Effective Field Theory at Work*, https://arxiv.org/abs/2303.16922.
    用途：现代 SMEFT 工作流、匹配、运行、实验解释和适用性边界。
19. **SMEFT-ATLAS.** J. Aebischer, A. J. Buras, J. Kumar, *SMEFT ATLAS: The Landscape Beyond the Standard Model*, https://arxiv.org/abs/2507.05926.
    用途：现代 SMEFT 应用方向、flavor、EDM、$g-2$、$Z$-pole、Higgs、高能散射和 operator/RGE 相关性地图。
20. **SMEFT-D8.** C. W. Murphy, *Dimension-8 Operators in the Standard Model Effective Field Theory*, https://arxiv.org/abs/2005.00059.
    用途：dimension-eight 作为研究边界和高阶截断风险的来源。
21. A. Adams, N. Arkani-Hamed, S. Dubovsky, A. Nicolis, R. Rattazzi, *Causality, Analyticity and an IR Obstruction to UV Completion*, https://arxiv.org/abs/hep-th/0602178.
    用途：positivity bounds 的理论背景。
22. **SMEFT-SNOWMASS.** Snowmass Theory Frontier, EFT/SMEFT 相关报告，尤其 https://arxiv.org/abs/2210.03199。
    用途：EFT/SMEFT 作为现代高能物理共同语言的状态说明。

## 标准模型与 QFT 背景

23. M. D. Schwartz, *Quantum Field Theory and the Standard Model*.
    用途：标准模型场内容、规范固定、Feynman 规则和重整化背景。
24. M. E. Peskin and D. V. Schroeder, *An Introduction to Quantum Field Theory*.
    用途：路径积分、散射振幅和重整化基础。
25. S. Weinberg, *The Quantum Theory of Fields*, Vol. I-II.
    用途：场论一般结构、对称性、有效拉氏量思想。

## 工具与高级边界

26. **SMEFT-BROKEN.** A. Dedes et al., *Feynman Rules for the Standard Model Effective Field Theory in $R_\xi$-gauges*, https://arxiv.org/abs/1704.03888.
    用途：SMEFT 破缺相、规范固定和 Feynman 规则边界。
27. A. Celis et al., *DsixTools: The Standard Model Effective Field Theory Toolkit*, https://arxiv.org/abs/1704.04504.
    用途：工具链、RGE 和 LEFT 匹配的外部实现来源。
28. T. Giani, G. Magni, J. Rojo, *SMEFiT*, https://arxiv.org/abs/2302.06660.
    用途：全局拟合作为外部分析工具的来源。
