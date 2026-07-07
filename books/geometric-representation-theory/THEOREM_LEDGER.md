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
| 附录 A | quotient stack 的对象语言；equivariant sheaf 的 descent 解释；导出范畴符号约定 | 六函子存在性、proper base change、Verdier duality | 不同 sheaf theory 不自动等价 |
| 附录 C | perverse normalization；middle extension 使用格式；decomposition theorem 检查表 | BBD t-structure、middle extension、decomposition theorem | mixed/Hodge/l-adic 版本不可混用 |
| 附录 D | locator 队列 | 无 | 当前不是精确 locator |
| 附录 E | left/right D-module 转换；de Rham shift convention | Riemann-Hilbert、twisted D-module global sections | $\rho$ shift 待 locator 锁定 |
| 附录 F | convolution 通用结合性模板 | 具体场景的 base change/properness | ind-proper 场景需单独处理 |
| 第一章 | 稳定子计算；$G$-equivariant vector bundle 与 $B$-表示的对应；$SL_2$ flag variety 的显式坐标 | Borel fixed point、Bruhat decomposition、$G/B$ projective smooth、highest weight theorem | 一般 reductive group 的结构定理不在本书重证 |
| 第二章 | Verma module 泛性质；highest weight morphism 判据；category $\mathcal O$ 的定义和基本封闭性 | PBW theorem、Harish-Chandra isomorphism、BGG reciprocity、Kazhdan-Lusztig character formula | Harish-Chandra modules 只作后续接口 |
| 第三章 | quotient stack 口径的 equivariant sheaves；perverse support 条件的形式稳定性；同伦商上的局部系统说明 | constructibility 六函子、BBD perverse t-structure、decomposition theorem、Riemann-Hilbert | l-adic、analytic 和 D-module 口径不可直接混用 |
| 第四章 | Hecke convolution correspondence 的类型检查；Grothendieck group 上乘法的定义；$SL_2$ 的低阶 Hecke 计算 | IC sheaf 与 Kazhdan-Lusztig basis、KL conjecture、Soergel categorification | 正特征和 modular KL 需另设章节 |
| 第五章 | Springer map well-defined、properness；Springer fibers；Steinberg convolution 结合性 | Springer $W$-action、$H^{BM}_{top}(Z)\simeq\mathbb C[W]$、Springer correspondence | affine Hecke/K-theory 版本后置 |
| 第六章 | nilpotent orbit 稳定子与 component group；equivariant local system 参数；character sheaf induction 类型检查 | generalized Springer correspondence、character sheaves 与 finite groups of Lie type | cuspidality 待专门定义 |
| 第七章 | Weyl algebra 例子；$\mathcal O_X$ characteristic variety；left/right 转换 convention | Bernstein inequality、Riemann-Hilbert、regularity 判据 | irregular D-modules 不进入基础链 |
| 第八章 | $U(\mathfrak g)$ 作用于 global sections；localization 与 global sections 伴随 | Beilinson-Bernstein localization、TDO global sections、KL character formula | Verma/IC 对应依赖 convention |
| 第九章 | line bundle cohomology 的 $G$-作用；translation functor 的 $\mathcal O$ 封闭性；$\pi_s$ proper | Borel-Weil-Bott、translation theory、wall crossing-localization 对应 | 权和 $\rho$ shift 需 locator |
| 第十章 | Harish-Chandra bimodule 基本例子；characteristic cycle of $\mathcal O_X$ | primitive ideals、associated varieties、Joseph theory、microlocal compatibility | real groups 和 primitive spectrum 后置 |
| 第十一章 | Soergel bimodule category monoidal 类型 | Soergel categorification、Soergel conjecture、Elias-Williamson Hodge theory | modular/parity phenomena 后置 |
| 第十二章 | affine Grassmannian functor、convolution Grassmannian、结合性形式证明 | ind-projectivity、orbit classification、convolution t-exactness | ind-scheme 技术需附录 |
| 第十三章 | 接受 Satake 后 simple objects 参数推论 | geometric Satake、Tannakian reconstruction、root datum 识别 | sheaf-function dictionary 后置 |
| 第十四章 | Iwahori convolution 结合性形式证明；affine simple reflection 局部模型；长度可加的标准对象卷积 | affine flag ind-projectivity、Iwahori orbit decomposition、Cartan decomposition、affine KL theorem | affine KL 多版本需分离 |
| 第十五章 | affine Kac-Moody residue cocycle 检查 | Kac-Moody localization、critical level、FLE | factorization categories 只作接口 |
| 第十六章 | Hecke correspondence 和 eigensheaf 类型定义 | 2024 GLC proof series、geometric Satake-Hecke action | 不进入证明链 |
| 第十七章 | quiver variety moment map 形式检查 | Nakajima representation construction | slice 同构后置 |
| 第十八章 | KLR induction product 结合性 | KLR/Rouquier categorification、canonical basis theorem | KLR relations 附录未写 |
| 第十九章 | conical symplectic resolution 和 quantization 定义 | BLPW-Losev category O、symplectic duality | 非全称定理 |
| 第二十章 | torus pure gauge 卷积计算；接受 BFN 后卷积结合性；loop rotation 基变换说明 | BFN Coulomb branch construction、finite generation、quantization flatness | 3d mirror 接口后置 |
| 第二十一章 | Hall multiplication 结合性的滤过计数证明 | CoHA、critical CoHA、DT wall crossing | orientation data 后置 |
| 第二十二章 | crystal string 有限性基本证明 | Lusztig/Kashiwara canonical bases、几何模型 | dual canonical convention 需锁定 |
| 第二十三章 | 前沿成熟度分级；结果进入正文流程；缺项不得作为内部证明输入 | 2024-2026 前沿定理均需 locator | 只作边界 |
| 附录 B/G/H/I/J | Coxeter convention、低秩例子、Soergel 低阶计算、Satake 检查表、前沿验证流程 | Bruhat-Schubert、MV、Soergel 等外部定理 | 附录 D locator 仍待扩展 |
| 正式化文件 | 完备矩阵、模型假设矩阵、内部证明核、低阶计算核、P0 locator 第一批 | 页码级 locator 和最终假设翻译仍需外部核验 | 不新增数学定理 |

## 2. 计划章节的外部输入预登记

| 主题 | 预期外部输入 | 当前状态 |
| --- | --- | --- |
| Springer correspondence | Springer、Borho-MacPherson、Kazhdan-Lusztig Steinberg variety 构造 | 第五、六章已写入骨架，待 locator |
| Beilinson-Bernstein localization | Localization theorem、acyclicity theorem、regular dominant 条件、twisted D-modules | 第八章已写入骨架，待 locator |
| Geometric Satake | Mirkovic-Vilonen 等价、convolution tensor category、Tannakian reconstruction | 第十二、十三章已写入骨架，待 locator |
| Soergel bimodules | Soergel categorification theorem、Elias-Williamson Hodge theory | 第十一章已写入骨架，待 locator |
| Quiver varieties | Nakajima construction、Kac-Moody action | 第十七章已写入骨架，待 locator |
| KLR/Rouquier categorification | KLR algebras、cyclotomic quotients、canonical bases | 第十八章已写入骨架，待 locator |
| Symplectic duality | BLPW category $\mathcal O$、Koszul duality、twisting/shuffling | 第十九章已写入骨架，待 locator |
| Coulomb branches | BFN convolution/Borel-Moore homology 构造、quantization | 第二十章已写入骨架，待 locator |
| Geometric Langlands | 2024 proof series、FLE、multiplicity one、ind-coherent/factorization formalism | 第十六章已写入边界骨架，需 locator |

## 3. 当前缺口

当前草稿已经达到主体教材化收口，但还不是出版终稿。主要缺口为：

1. 主体目录第一至第二十三章已有定义链、证明或外部输入标记、例子与练习；出版层仍需压缩证明文字并稳定交叉引用。
2. Kazhdan-Lusztig、Springer、BBD、Riemann-Hilbert、Beilinson-Bernstein、Borel-Weil-Bott、Soergel/Elias-Williamson、geometric Satake、Nakajima、KLR/Rouquier、BLPW、BFN、CoHA 和 canonical bases 均还需精确 theorem locator；第一批 P0 locator 已给定理包级入口。
3. 六函子、perverse sheaf 和 quotient stack 技术已有附录 C 初版，按模型分拆的假设表仍需扩展。
4. quiver varieties、Coulomb branches、symplectic duality 和 geometric Langlands 已有正文入口；作为外部定理使用时仍需 locator 和模型假设翻译。
