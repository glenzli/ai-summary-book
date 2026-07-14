# 定理账本：内部证明、外部输入与研究边界

本文件是《Geometric Representation Theory》的审校账本。它不新增数学结论，而是把当前和计划章节中的结论分成三类：

1. **内部证明**：当前草稿已经给出证明，后续只需压缩、校正符号或补充细节。
2. **外部输入**：正文可使用，但必须在 `SOURCES.md` 和后续附录 D 中定位来源、定理编号、版本和假设。
3. **边界说明**：只用于解释范围、失败模式或研究方向，不得作为证明步骤。

## 0. 使用规则

若正文需要调用某个结论，应按下列顺序检查：

1. 该结论是否在本文件中列为内部证明；
2. 若不是，是否在 `SOURCES.md` 中列为外部输入；
3. 若只出现在边界说明中，则不得用于推出后续命题；
4. 若结论涉及 D-modules 或 perverse sheaves，必须同时检查底域、拓扑、系数和 t-structure 约定；
5. 若结论涉及 geometric Langlands、Coulomb branches 或 symplectic duality，当前只能作为研究边界，除非新增 locator 批次。

## 1. 第一批章节

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 序章 | 严格性标准、对象分层、外部输入使用流程 | 无 | geometric Langlands 和 Coulomb branches 只作边界 |
| 附录 A | quotient stack 对象语言；$[H/K/H]\simeq BK$；Betti local systems 只见 $\pi_0(K)$ | 六函子存在性、proper base change、Verdier duality | sheaves on $BK$ 不等于正维 $K$ 的 algebraic representations |
| 附录 C | equivariant perverse shift；middle-extension heart image；decomposition/semismall 假设表 | BBD t-structure、BBD 6.2.5、semismall intersection-form theorem | 非 canonical splitting、modular failure、proper/projective 边界已分开 |
| 附录 D | theorem/section locator 队列；BBD、semismall、Satake 主链已细化 | 无 | 其余条目仍有页码/版本终校缺口 |
| 附录 E | left/right D-module 转换；de Rham shift convention | Riemann-Hilbert、twisted D-module global sections | $\rho$ shift 待 locator 锁定 |
| 附录 F | 普通与 descent convolution 类型；有 coherence 假设的 associativity/pentagon；单位；$!\ne\ast$ 与 proper 非 t-exact 反例 | 六函子 Beck--Chevalley；各场景的 support-properness | ind-scheme 只允许 finite support；perverse preservation 需 semismall 等额外输入 |
| 第一章 | 稳定子计算；$G$-equivariant vector bundle 与 $B$-表示的对应；$SL_2$ flag variety 的显式坐标 | Borel fixed point、Bruhat decomposition、$G/B$ projective smooth、highest weight theorem | 一般 reductive group 的结构定理不在本书重证 |
| 第二章 | Verma module 泛性质；highest weight morphism 判据；category $\mathcal O$ 的定义和基本封闭性 | PBW theorem、Harish-Chandra isomorphism、BGG reciprocity、Kazhdan-Lusztig character formula | Harish-Chandra modules 只作后续接口 |
| 第三章 | 固定 Betti domain；quotient-stack/descent 类型；forgetful equivariant perversity；光滑 IC 证明；proper Verdier duality 形式推论 | Bernstein--Lunts equivariance、Betti 六函子、BBD perverse/middle extension、BBD decomposition | sheaf models 不混用；arbitrary semisimple input、modular splitting、proper RHL 均被排除 |
| 第四章 | Hecke convolution correspondence 的类型检查；Grothendieck group 上乘法的定义；$SL_2$ 的低阶 Hecke 计算 | IC sheaf 与 Kazhdan-Lusztig basis、KL conjecture、Soergel categorification | 正特征和 modular KL 需另设章节 |
| 第五章 | Springer map well-defined、properness；Springer fibers；Steinberg convolution 结合性 | Springer semismallness/perverse semisimplicity、$W$-action、$H^{BM}_{top}(Z)\simeq\mathbb C[W]$、Springer correspondence | affine Hecke/K-theory 版本后置 |
| 第六章 | Betti equivariant local systems 经 $A_G(x)$ 参数化；Springer 归一化；接受 semismallness 后的 simple-summand 类型；cuspidal datum 的 restriction 判据 | equivariant simple-perverse classification、generalized Springer、normalized restriction、character sheaves | 不使用 algebraic $G_x$-representations 替代 local systems；另一种 Springer action 需张量 sign |
| 第七章 | Weyl algebra 例子；$\mathcal O_X$ characteristic variety；left/right 转换 convention | Bernstein inequality、Riemann-Hilbert、regularity 判据 | irregular D-modules 不进入基础链 |
| 第八章 | $U(\mathfrak g)$ 作用于 global sections；localization 与 global sections 伴随 | Beilinson-Bernstein localization、TDO global sections、KL character formula | Verma/IC 对应依赖 convention |
| 第九章 | line bundle cohomology 的 $G$-作用；translation functor 的 $\mathcal O$ 封闭性；$\pi_s$ proper | Borel-Weil-Bott、translation theory、wall crossing-localization 对应 | 权和 $\rho$ shift 需 locator |
| 第十章 | Harish-Chandra bimodule 基本例子；characteristic cycle of $\mathcal O_X$ | primitive ideals、associated varieties、Joseph theory、microlocal compatibility | real groups 和 primitive spectrum 后置 |
| 第十一章 | Soergel bimodule category monoidal 类型 | Soergel categorification、Soergel conjecture、Elias-Williamson Hodge theory | modular/parity phenomena 后置 |
| 第十二章 | fpqc quotient 与 Betti reduction 边界；torsor descent convolution；support-stage independence；associativity/unit；$GL_1$ 与 $GL_2$ lattice fibers/semismall check | AFFGR representability/orbits；finite-support properness；semismall decomposition | unreduced torus functor 不等于 $\mathbb Z$；一般 t-exactness 留给 Satake 输入 |
| 第十三章 | Tannaka weight grading 仅构造 $T^\vee\to H$；$GL_1$ symmetric equivalence；$GL_2$ tensor-square 核对 | convolution/fusion、fiber functor、neutral Tannaka、geometric Satake、root datum 识别 | classical、integral/modular、mixed-characteristic Satake 分离 |
| 第十四章 | Finite-support Iwahori category；torsor-descent convolution；coherent associativity；affine simple-reflection 局部模型；长度可加标准对象卷积 | affine flag ind-projectivity、finite-support convolution properness、Iwahori orbits、Cartan decomposition、affine KL theorem | affine KL 多版本与 mixed/monodromic coefficients 需分离 |
| 第十五章 | affine Kac-Moody residue cocycle 检查 | Kac-Moody localization、critical level、FLE | factorization categories 只作接口 |
| 第十六章 | Hecke correspondence、eigensheaf 类型定义、$GL_n$ modification 与 $GL_1$ 平移模型 | geometric Satake--Hecke action；任何确定版 global equivalence 另需独立 locator | 局部作用不推出全局等价；不进入基础证明链 |
| 第十七章 | quiver moment map；$T^*\operatorname{Gr}(v,W)$ 识别；$T^*\mathbb P^1\to\mathcal N(\mathfrak{sl}_2)$ 坐标与 fibers | Nakajima representation construction、Hecke correspondences 的 Kac--Moody relations | 一般 slice 同构后置 |
| 第十八章 | KLR induction 结合性；$R(2)$ divided-difference 表示；$R(2)\simeq M_2(E[e_1,e_2])$ | KLR/Rouquier categorification、canonical basis theorem | 一般 $Q_{ij}$ 与 grading convention 由外部版本控制 |
| 第十九章 | conical symplectic resolution 和 quantization 定义；$T^*\mathbb P^1$ incidence resolution；Weyl algebra 正权点模 | BLPW--Losev category $\mathcal O$、具体 symplectic-duality 定理 | symplectic duality 不作全称定理 |
| 第二十章 | torus pure gauge 卷积；权一 matter 分支的消失阶；接受 BFN 后卷积结合性与 loop-rotation specialization | BFN Coulomb branch construction、finite generation、Poisson 性、quantization flatness | symplectic singularity 与 resolution 需逐个 $(G,N)$ 验证 |
| 第二十一章 | Hall multiplication 结合性的滤过计数；Gaussian Hall 公式与二步旗标恒等式 | CoHA、critical CoHA、DT wall crossing | potential、vanishing cycles 与 orientation data 不由 ordinary Hall 自动给出 |
| 第二十二章 | crystal string 条件；$A_1$ Grassmannian fibers 与 incidence graph 给出 $B(n)$ | Lusztig/Kashiwara canonical bases、Nakajima/KLR 几何模型 | canonical/dual canonical convention 需锁定 |
| 第二十三章 | $GL_1$ Hecke coherence；$A$-型曲面二形式；$A_1$ parity；同构 $K_0$ 的 dual-number 反例；nonreduced loop family | ADE 最小分辨率、一般 modular/parity 理论及各确定版前沿等价 | 按 local-to-global、Poisson-to-symplectic、torsion、categorical comparison、derived models 五类障碍组织 |
| 附录 B/G/H/I/J | Coxeter convention；$SL_2$ Springer stalk 与 $SL_3$ stable-flag 完整计算；Soergel 低阶计算；Satake 检查表；前沿流程 | Springer $W$-action、MV packages、Soergel 等外部定理 | Euler characteristic 不决定 Springer action；Tannaka 不自动识别 root datum |
| 正式化文件 | 完备矩阵、模型假设矩阵、内部证明核、低阶计算核、P0 locator 第一批 | 页码级 locator 和最终假设翻译仍需外部核验 | 不新增数学定理 |

## 2. 计划章节的外部输入预登记

| 主题 | 预期外部输入 | 当前状态 |
| --- | --- | --- |
| Springer correspondence | Springer、Borho-MacPherson、Kazhdan-Lusztig Steinberg variety 构造 | 第五、六章已写入骨架，待 locator |
| Beilinson-Bernstein localization | Localization theorem、acyclicity theorem、regular dominant 条件、twisted D-modules | 第八章已写入骨架，待 locator |
| Geometric Satake | Mirkovic--Vilonen 等价、convolution/fusion、fiber/weight functors、Tannakian reconstruction | 第十二、十三章与附录 I 已正文闭合到外部输入边界；MV section/theorem locators 已登记 |
| Soergel bimodules | Soergel categorification theorem、Elias-Williamson Hodge theory | 第十一章已写入骨架，待 locator |
| Quiver varieties | Nakajima construction、Kac-Moody action | 第十七章已写入骨架，待 locator |
| KLR/Rouquier categorification | KLR algebras、cyclotomic quotients、canonical bases | 第十八章已写入骨架，待 locator |
| Symplectic duality | BLPW category $\mathcal O$、Koszul duality、twisting/shuffling | 第十九章已写入骨架，待 locator |
| Coulomb branches | BFN convolution/Borel-Moore homology 构造、quantization | 第二十章已写入骨架，待 locator |
| Geometric Langlands | 各确定版本的 global equivalence、FLE、multiplicity one、IndCoh/factorization formalism | 第十六章给出局部接口，第二十三章拆分 local-to-global 障碍；未定位的时效性结果不进入证明链 |

## 3. 当前外部输入缺口

正文的内部推导与低秩模型已经落实；尚需外部核验的部分集中在：

1. 第一至第二十三章已有连续定义链、证明或外部输入标记、例子与练习；外部引用仍需稳定 theorem locator 与版本翻译。
2. Kazhdan-Lusztig、Springer、Riemann-Hilbert、Beilinson-Bernstein、Borel-Weil-Bott、Soergel/Elias-Williamson、Nakajima、KLR/Rouquier、BLPW、BFN、CoHA 和 canonical bases 仍需精确 theorem locator；BBD decomposition、semismall 与 geometric Satake 主链本轮已推进到 theorem/section level，仍需出版页码终校。
3. 六函子、perverse sheaf、quotient stack、semismall 和 affine-convolution 技术已按 Betti 模型分拆；若新增 $\ell$-adic、mixed Hodge 或 modular 证明链，仍需另建模型专表。
4. quiver varieties、Coulomb branches、symplectic duality 和 geometric Langlands 已有正文入口；作为外部定理使用时仍需 locator 和模型假设翻译。
