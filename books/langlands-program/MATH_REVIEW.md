# 数学审查记录

本文档记录《Langlands 纲领》草稿的审查清单、当前风险和后续补证任务。

## 全书审查清单

- [ ] 每章是否列出本章目标和依赖前置知识。
- [ ] 每个新定义是否包含完整数据和公理。
- [ ] 每个非平凡命题是否给出证明、证明草图或“外部输入定理”标记。
- [ ] 每个外部输入定理是否能在 `SOURCES.md` 中追溯。
- [ ] 是否避免把 Langlands 对应写成无条件的一一对应；是否说明已知情形、猜想情形和归一化。
- [ ] 是否区分局部对象和整体对象。
- [ ] 是否区分 classical modular forms、adelic automorphic forms 和 automorphic representations。
- [ ] 是否区分 complex representations、l-adic representations、mod p representations。
- [ ] 是否说明 Frobenius 归一化。
- [ ] 是否说明 Haar 测度和 Fourier 变换归一化。

## 当前风险

- 第一章中 `\mathbb A_K/K` 紧性、Poisson summation 和 self-duality 暂作为外部输入；附录 B 已补 Haar 测度、商测度、restricted product 积分和卷积基础，附录 F 已补 Fourier/Pontryagin/Poisson 的接口与若干基本计算。完整 Fourier inversion、Plancherel 和 Poisson summation 仍作为外部输入。
- 第二章中 Tate thesis 的局部函数方程和整体函数方程暂作为外部输入；后续应拆成若干引理证明。
- 第三章中局部和全局类域论作为外部输入；后续需要在代数数论附录中补 Artin 映射的存在、唯一性、norm subgroup 定理和局部-整体相容性。
- 第四章中 Haar 测度存在唯一性、Satake 同构和 Harish-Chandra 理论作为外部输入；后续应补紧群平均、Iwasawa 分解和球 Hecke 代数计算。
- 第五章中 `GL(n)` 局部 Langlands 和一般 reductive 群的 L-packet 陈述作为外部输入或猜想；后续应补 `GL(1)` 完整证明、`GL(2)` 例子和非分歧 Satake 参数计算。
- 第六章中模形式空间有限维性、Hecke 算子良定义性、Euler 乘积、newform 函数方程和 Deligne Galois 表示作为外部输入；附录 H 已补 Hecke 双陪集代表、Fourier 系数计算、Petersson 内积和 adelic Hecke 比较。模曲线线丛和 Atkin-Lehner-Li 完整理论仍需扩写。
- 第七章中 strong approximation、经典-adelic 对应、cuspidal spectrum 张量积分解、newform 生成自守表示和好素数局部 Langlands 相容作为外部输入；附录 H 已补好素数球 Hecke 代数与经典 $T_p$ 的比较，strong approximation 和 newform 生成定理仍作为外部输入。
- 第八章中最小模型、Neron-Ogg-Shafarevich、Tate 算法、导子公式、Frobenius 多项式和模性定理作为外部输入；后续应补 Neron 模型、Kodaira 类型和坏约化局部因子计算。
- 第九章中 Fontaine-Mazur 作为猜想背景；稳定格、Deligne 表示、椭圆曲线模性、模性提升和 `$R=T$` 原理作为外部输入；后续应补 Galois deformation theory、Selmer 群和 Taylor-Wiles patching 的精确定理版本。
- 第十章中局部-整体相容性、Serre 模性定理、Ribet 降层和 Frey 曲线局部导子计算作为外部输入；后续应补精确定理陈述、局部类型分类和 conductor 计算。
- 第十一章中 reductive group 结构定理、根资料分类、pinning 的 Galois 作用、unramified Satake parameter 作为外部输入；附录 G 已补 `GL(n)`、`SL(n)`、`PGL(n)`、restriction of scalars torus 和若干 L 同态计算。完整 algebraic groups 证明附录仍需扩写。
- 第十二章中 `GL(n)` LLC、tori 的 LLC、Archimedean LLC、enhanced LLC、endoscopic character identities 和 classical groups 的已知情形作为外部输入或猜想；后续应补 component group、rigid inner forms、Kottwitz 符号和 tempered/discrete 参数的精确定义。
- 第十三章中自守表示的 restricted tensor product 分解、离散谱理论、初始收敛估计、Godement-Jacquet、Rankin-Selberg、Langlands-Shahidi 和一般 L 函数解析性质作为外部输入或猜想；附录 I 已补 Godement-Jacquet、Rankin-Selberg、局部函数方程和 converse theorem 的积分接口。Spectral decomposition、Eisenstein series 常数项和 Langlands-Shahidi 局部系数仍需扩写。
- 第十四章中 Bernstein-Zelevinsky 分类、`GL(n)` LLC、Rankin-Selberg 理论、强重数一、converse theorem、Lafforgue 函数域定理和数域 regular algebraic Galois 表示构造作为外部输入；附录 I 已补 `GL(n)` 标准 L 函数和 Rankin-Selberg L 函数的积分来源。精确归一化、局部因子 convention 和数域开放情形边界仍需扩写。
- 第十五章中一般函子性、强转移、endoscopic transfer、任意 symmetric/exterior/tensor power lifts 仍属猜想或部分已知；solvable base change、automorphic induction、若干低阶 lifts、Arthur-Mok 型转移作为外部输入。
- 第十六章中 Arthur trace formula、稳定 trace formula、transfer factor、fundamental lemma、twisted trace formula 作为外部输入；后续应补 invariant trace formula 的精确项和测度归一化。
- 第十七章中 Arthur packets、multiplicity formula、classical groups 标准转移和 L 函数判别符号作为外部输入；后续应补局部 A-packet 和 global Arthur parameter 的精确定义。
- 第十八至二十二章中代数栈、affine Grassmannian、几何 Satake、Hecke eigensheaves、categorical geometric Langlands、sheaf-function dictionary、shtukas 和 Ngô 支持定理作为外部输入；后续应补 derived stack 和 sheaf theory 口径。
- 附录 A-D 和 F 已加入若干关键证明：数域乘积公式、ray class 的 idele 描述、Artin 导子基本性质、卷积结合律、商测度、smooth dual、中心特征、有限长度可容许性、$X_0(2)$ 的 genus 计算、非 Archimedean Fourier 基本计算和 adele Poisson summation 的推导。它们仍不是完整专著级证明替代；类域论、Haar 存在唯一性、Fourier inversion、模曲线代数化和 newform 理论仍作为外部输入。
- 费马应用章当前证明的是“由模性定理、Ribet 降层和 Frey 曲线性质推出费马大定理”的严格逻辑链；不是 Wiles-Taylor-Wiles 证明本身。
- `S_2(\Gamma_0(2))=0` 使用模曲线 genus 公式；附录 D 已补 $\mu=3$、$c=2$、$e_2=1$、$e_3=0$ 的计算和权 $2$ cusp forms 与微分形式的局部说明。一般 $X_0(N)$ 维数公式的完整 Riemann-Hurwitz 证明仍作为外部输入。

## 本轮严格性审查记录

- 已修正第二章中全局类域论接口：有限像 Galois 表示只对应有限阶 Hecke 特征；一般非有限阶 Hecke quasi-character 应放在 Weil 侧。
- 已修正第三章小结，明确局部一般连续特征对应 Weil 参数，而不是普通 profinite Galois 表示。
- 已把第四章的基础群论口径从含混的 tdlc 表述收紧为 locally profinite group，并注明 Van Dantzig 定理的角色。
- 已把第五章 `GL(n)` 局部 Langlands 陈述改为“同构类之间的双射”。
- 已在第六章 Deligne 接口中显式声明 Frobenius 归一化风险。
- 已在第七章区分算术归一化和 unitary automorphic normalization，避免把 $L(f,s)$ 与标准自守 L 函数的不同 convention 混用。
- 已把第六、七、八章中模形式和椭圆曲线 Galois 表示的 Frobenius 写法改为算术 Frobenius，并明确与局部 Langlands 几何 Frobenius convention 的换算风险。
- 已在第九章区分 residual modularity、lift modularity、elliptic curve modularity 和 general Fontaine-Mazur expectation，避免把模性提升定理写成无条件黑箱。
- 已在第十章把 Ribet 降层拆为局部-整体相容、残余导子和 level 删除三个层次，避免把“降层”写成单步口号。
- 已补写第十一章，把一般 Langlands 的结构入口从 `GL(n)` 扩展到 connected reductive groups、root datum、dual group、L group、L homomorphism 和 unramified Satake parameter。
- 已在第十一章区分 Galois 型 L 群、局部 Weil 型 L 群和数域情形中仍属纲领性的全局 Langlands 群，避免把全局 L 群写成无条件已构造对象。
- 已补写第十二章，把局部 Langlands 从“参数对应表示”的粗略说法收紧为 coarse packet、enhanced parameter、component group、inner form 和 endoscopic compatibility 的分层表述。
- 已在第十二章区分定理性已知情形、一般猜想和后续章节只作为接口使用的外部输入，避免把一般 reductive 群 LLC 写成已完全证明的一一对应。
- 已补写第十三章，把全局自守表示、尖点条件、restricted tensor product、非分歧 Satake 参数、部分 Euler 乘积和完全 L 函数接成统一接口。
- 已在第十三章区分形式 Euler 乘积、局部因子定义、已知解析定理和一般 Langlands 解析猜想，避免把任意 $r:{}^LG\to\operatorname{GL}(V)$ 的解析延拓写成无条件结论。
- 已补写第十四章，把 `GL(n)` 的局部 LLC、Langlands 分类、全局标准和 Rankin-Selberg L 函数、强重数一、converse theorem、函数域 Lafforgue 定理和数域 Galois 表示接口集中整理。
- 已在第十四章区分函数域 `GL(n)` 全局定理、数域 regular algebraic 已知构造和数域完整全局 Langlands 猜想，避免把数域一般情形写成既成定理。
- 已补写第十五章，把函子性分为弱转移、强转移、L 函数相容性、`GL(N)` 目标 converse theorem、base change、automorphic induction、低阶 functorial lifts、endoscopy 和 Galois 表示侧复合。
- 已在第十五章区分一般 functoriality 猜想与已知特殊情形，避免把 symmetric powers、exterior powers、tensor products 或 endoscopic transfer 写成全体已知定理。
- 已补写第十六章，建立 trace formula、稳定轨道积分、endoscopic data、transfer factor、fundamental lemma 和 twisted trace formula 的接口。
- 已补写第十七章，建立 Arthur 参数、A-packet、multiplicity formula、standard transfer 和非 tempered 离散谱的接口。
- 已补写第十八至二十二章，建立几何 Langlands 主线：$\operatorname{Bun}_G$、Hecke 修改、affine Grassmannian、几何 Satake、Hecke eigensheaves、谱侧范畴和函数域桥梁。
- 已补写附录 A-E，集中记录代数数论、Haar 测度、光滑表示、模曲线维数公式和外部输入定理索引。
- 已新增全书定理索引 [THEOREM_INDEX.md](THEOREM_INDEX.md)，把主要结果标记为已证、证明草图、外部输入或猜想。
- 已新增章节依赖图 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，明确 `GL(1)`、费马应用、一般 Langlands 和几何 Langlands 四条阅读路径。
- 已新增核心习题解答 [SOLUTIONS.md](SOLUTIONS.md)，覆盖 restricted product、局部特征、Frobenius 归一化、Hecke 关系、Satake 参数、函子性和费马应用链。
- 已扩充附录 A：加入非 Archimedean 赋值、数域乘积公式证明、ray class groups、idele class group 商描述、norm subgroup theorem 口径和 Artin 导子基本性质。
- 已扩充附录 B：加入卷积结合律、开紧平均投影、商测度和 restricted product 积分公式。
- 已扩充附录 C：加入 Schur 引理、中心特征、smooth dual、有限长度和可容许性稳定性。
- 已扩充附录 D：加入 $X_0(2)$ 的指数、cusp 数、椭圆点数、genus 和权 $2$ 微分形式的计算。
- 已新增附录 F：固定局部紧 Abel 群 Fourier 分析、self-dual measure、Schwartz-Bruhat 空间、adeles 自对偶和 Poisson summation 在 Tate thesis 中的用法。
- 已新增附录 G：补 `GL(n)`、`SL(n)`、`PGL(n)`、classical dual groups、split L 群、restriction of scalars torus 和 determinant/symmetric square L 同态计算。
- 已新增附录 H：补 Hecke 双陪集算子、$\Gamma_0(N)$ 好/坏素数代表、Fourier 系数公式、Petersson 内积和 adelic Hecke algebra 比较。
- 已新增附录 I：补 Godement-Jacquet、Rankin-Selberg、Whittaker 模型、全局 unfolding、converse theorem 和函子性检测的积分接口。

## 后续补证计划

1. 继续扩写 Fourier 分析附录：Fourier inversion、Plancherel、Poisson summation 和 $\mathbb A_K/K$ 紧性的完整证明。
2. 继续扩写代数数论附录：Dedekind 域完整证明、分解群、惯性群、高阶分歧群和类域论 class formation。
3. 扩充第三章：加入 ray class groups、idele class characters 的 conductor 和 Dirichlet 特征比较。
4. 扩充第六章和附录 H：补 Hecke 算子交换关系、diamond operators、Petersson 伴随公式的完整证明和 Atkin-Lehner-Li old/new 分解。
5. 补写椭圆曲线章，定义 Neron 模型、约化类型和导子。
6. 扩充第五章：加入 `GL(2)` 的 principal series、Steinberg 表示和 supercuspidal 参数例子。
7. 扩充第六章：加入 diamond operators、Petersson 内积、newform oldform 分解和 Atkin-Lehner involution。
8. 扩充第七章：补 strong approximation 证明、经典-adelic 对应的良定义性检查和 $K_0(N)$-双陪集与 Hecke 算子的精确比较。
9. 扩充第八章：补 Neron 模型、Tate 算法表、Kodaira 符号和 conductor-discriminant 关系。
10. 扩充第九章：补 deformation functor、Mazur representability、local deformation conditions、Selmer duality 和 Taylor-Wiles patching 的最小版本。
11. 扩充第十章：补 Ribet 降层精确陈述、Serre weight/level recipe、Frey 曲线在 $2$ 和 $q\mid abc$ 处的 Tate algorithm 计算。
12. 扩充第十一章和附录 G：补 pinning 与 diagram automorphism、非分歧群的 hyperspecial subgroup、Satake 同构证明和更多 classical group root datum 表。
13. 扩充第十二章：补 Bernstein-Zelevinsky 分类、Langlands quotient theorem、component group 例子、$\operatorname{SL}_2$ packet 计算和 enhanced LLC 的 rigid inner twist 版本。
14. 扩充第十三章和附录 I：补自守形式空间的精确定义、Eisenstein series、残余谱、Godement-Jacquet 局部计算细节、Rankin-Selberg unfolding 完整模型和 Langlands-Shahidi 局部系数。
15. 扩充第十四章：补 `GL(n)` 局部 LLC 的归一化比较、Zelevinsky multisegments、Rankin-Selberg 局部因子定义、converse theorem 精确假设和 Lafforgue 定理的纯性条件。
16. 扩充第十五章：补 Arthur-Clozel base change 精确陈述、automorphic induction 局部公式、isobaric sum、low symmetric power lift 的条件和 endoscopic transfer factor 的接口。
17. 扩充第十六章：补 Arthur trace formula 的几何侧、谱侧、稳定化和 fundamental lemma 的精确定理版本。
18. 扩充第十七章：补 Arthur classification 的 group-by-group 陈述、unitary groups 的 Mok 版本和内形式修正。
19. 扩充几何章节：补 D-modules、perverse sheaves、Beilinson-Drinfeld Grassmannian、factorization 和 singular support。
20. 扩充附录：把 A-E 和新建索引文件继续改写为可独立阅读的证明卷与交叉引用系统。
