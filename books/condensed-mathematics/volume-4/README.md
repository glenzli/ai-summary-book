# 凝聚数学讲义：第四卷

作者：Dr. Stochastic Parrot  
状态：计算与稳定接口教材稿
副题：形式化、计算与例子

本卷接续 [第一卷](../volume-1/)、[第二卷](../volume-2/) 与 [第三卷](../volume-3/)。前三卷建立凝聚基础、solid/analytic/liquid 结构和复几何应用；第四卷把这些定义落实为带输入、步骤、输出与失败条件的计算，并在最后进入凝聚谱的稳定接口。

## 建议阅读顺序

1. [序章：把定义变成可复核计算](00_preface_and_scope.md)
2. [第一章：匹配族的机器可检规格](01_formalized_condensed_mathematics.md)
3. [第二章：站点、覆盖与 sheaf 条件的计算](02_site_and_sheaf_computations.md)
4. [第三章：Ext 与 Tor 的投射分解计算](03_ext_tor_computation_templates.md)
5. [第四章：solid 张量积例子](04_solid_tensor_examples.md)
6. [第五章：analytic ring 的 Dirac--测度计算](05_analytic_ring_examples.md)
7. [第六章：liquid 化中的连续性与正合性](06_liquid_functional_analysis_examples.md)
8. [第七章：两个站点上的投射局部对象](07_pro_etale_and_condensed.md)
9. [第八章：凝聚谱中的六种运算与开放问题](08_catalogue_and_open_problems.md)
10. [附录 A：形式化蓝图](A_formalization_blueprints.md)
11. [附录 B：练习解答与计算样板](B_worked_solutions_and_computations.md)
12. [附录 C：solid、analytic、liquid 的类型检查](C_solid_analytic_liquid_type_checks.md)
13. [附录 D：pro-etale 与凝聚数学的比较细节](D_pro_etale_comparison_details.md)
14. [附录 E：当代方向、pyknotic 对象与凝聚同伦](E_current_directions_pyknotic_and_homotopy.md)
15. [附录 F：凝聚基础的形式化证明义务](F_formal_proof_obligations_for_condensed_basics.md)
16. [附录 G：凝聚谱、pyknotic 接口与同伦方向](G_condensed_spectra_and_pyknotic_interfaces.md)

## 当前范围

正文沿一条连续计算线展开：有限覆盖给出 sheaf 等化子，投射分解给出 Ext/Tor，profinite 有限商给出 solid 测度外积，Dirac cone 给出 analytic localization，连续截面给出 liquid 复形的正合性，最后这些对象级计算升级为 mapping 与 tensor spectra。

第二章完成有限覆盖、可表 sheaf 和基子站点比较；第三至第五章完整展开循环对象、Cantor 测度、单点紧化测度与普通换底失败；第六章计算 $C^\infty([0,1])$ 的逆极限及微分积分复形；第七章在有限 étale 分支上比较两个站点的下降形状；第八章计算循环凝聚谱，并把六函子相容性表述为生成 cone 的保持问题。

附录保留参考形态：A、F 记录形式化证明义务，B 给补充解答，C、D 负责类型与站点边界，E、G 汇集 pyknotic 和谱值接口。理解正文所需的计算与证明机制已经写回数字章节；附录 G 与第八章互相交叉引用，不再承担唯一主线。

## 资料

资料源见 [SOURCES.md](SOURCES.md)，符号约定见 [NOTATION.md](NOTATION.md)，审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，分卷答案见 [SOLUTIONS.md](SOLUTIONS.md)。
