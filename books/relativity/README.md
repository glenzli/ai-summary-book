# 相对论讲义：从 Minkowski 时空到 Einstein 方程

作者：Dr. Stochastic Parrot

状态：正式教材范围的一版闭合草稿

本书是一部中文相对论教材，目标是在尽量严格的数学表述下写完狭义相对论和广义相对论的核心主线，并把正式教材中应出现的旋转黑洞、宇宙学扰动、后 Newton 近似和数值相对论入口纳入同一套符号系统。它不是历史随笔，也不是只给结论的物理科普；正文会固定符号、给出定义、推导基本公式、标明外部输入定理，并配套习题与答案。

## 阅读路径

### 第零部分：预备

0. [第零章 数学预备、单位与证明约定](00_mathematical_preliminaries.md)

### 第一部分：狭义相对论与场

1. [第一章 Minkowski 时空与 Lorentz 几何](01_minkowski_spacetime.md)
2. [第二章 Lorentz 变换、时钟与尺](02_lorentz_transformations.md)
3. [第三章 相对论力学与应力能量张量](03_relativistic_mechanics.md)
4. [第四章 电磁场的协变形式](04_covariant_electromagnetism.md)
5. [第五章 作用量、Noether 定理与场论入口](05_variational_principles.md)

### 第二部分：广义相对论

6. [第六章 流形、张量、联络与曲率](06_differential_geometry.md)
7. [第七章 测地线、等效原理与局部惯性系](07_geodesics_equivalence.md)
8. [第八章 Einstein 方程](08_einstein_equations.md)
9. [第九章 Schwarzschild 几何与经典检验](09_schwarzschild_geometry.md)
10. [第十章 黑洞、视界与因果结构](10_black_holes_causal_structure.md)
11. [第十一章 FLRW 宇宙学](11_flrw_cosmology.md)
12. [第十二章 线性化引力与引力波](12_linearized_gravity_waves.md)
13. [第十三章 初值问题、能量与整体结构](13_initial_value_energy_global.md)

### 第三部分：高级入口与观测近似

14. [第十四章 应用、近似与边界](14_applications_and_limits.md)
15. [第十五章 Kerr 几何、旋转黑洞与高级黑洞入口](15_kerr_geometry.md)
16. [第十六章 宇宙学扰动、结构形成与规范问题](16_cosmological_perturbations.md)
17. [第十七章 后 Newton 近似、参数化检验与数值相对论](17_post_newtonian_numerical_relativity.md)

### 附录与工具

- [APPENDIX_A_CALCULATION_MANUAL.md](APPENDIX_A_CALCULATION_MANUAL.md)：计算手册。
- [APPENDIX_B_WORKED_EXAMPLES.md](APPENDIX_B_WORKED_EXAMPLES.md)：详细例题。
- [NOTATION.md](NOTATION.md)：符号、单位和指标约定。
- [SOURCES.md](SOURCES.md)：资料源和外部输入边界。
- [THEOREM_INDEX.md](THEOREM_INDEX.md)：定义、命题和外部输入定理索引。
- [SOLUTIONS.md](SOLUTIONS.md)：章末习题答案。
- [MATH_REVIEW.md](MATH_REVIEW.md)：数学与物理审查清单。
- [GLOSSARY.md](GLOSSARY.md)：常用术语表。
- [COMPLETION_STATUS.md](COMPLETION_STATUS.md)：完成度、闭合边界与后续扩展说明。
- [CLOSURE_REVIEW.md](CLOSURE_REVIEW.md)：教材内容收口审查。

## 严格性口径

本书采用“核心主线自证明，深结果输入化”的标准：

- 狭义相对论中 Lorentz 几何、四矢量、四动量、电磁场张量、能动张量和作用量推导在书内完成。
- 广义相对论中流形张量、Levi-Civita 联络、曲率、测地线方程、Einstein 方程的形式性质、牛顿极限、Schwarzschild 轨道近似、Friedmann 方程、线性化引力波、Kerr 基本结构、宇宙扰动增长和弱场后 Newton 入口在书内推导或说明。
- Cauchy 问题适定性、正质量定理、奇点定理、Kerr 唯一性、面积定理、精确后 Newton 展开、EOB 构造和数值相对论稳定性等深定理作为外部输入定理，不在本书内部完整证明。

## 前置知识

读者应熟悉多元微积分、线性代数、常微分方程和基础经典力学。第六章会补充微分几何的最小工具，但本书不会替代一门完整的微分几何课程。

## 教材目标

读完本书后，读者应能：

1. 用 Minkowski 几何重新表达狭义相对论。
2. 熟练使用四矢量、张量指标和自然单位。
3. 从变分原理推导自由粒子、测地线和基本场方程。
4. 理解 Einstein 方程的几何含义、守恒律和牛顿极限。
5. 计算 Schwarzschild、FLRW 和线性化引力中的基本可观测量。
6. 读懂 Kerr、宇宙扰动、后 Newton 近似和数值相对论的基本公式入口。
7. 区分书内证明结论、标准输入公式和外部深定理。
