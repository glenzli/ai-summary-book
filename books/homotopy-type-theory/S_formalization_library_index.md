# 附录 S：形式化库索引

## S.0 目标

本附录给出本书当前版本可引用的形式化库入口。它不是形式化证明脚本，也不是完整依赖图，而是版本化的核查索引：每个条目至少固定仓库、commit、模块路径和若干入口定义或定理名。

本附录的作用有三点：

1. 防止正文只写“某结论已形式化”而没有可复核位置。
2. 区分库内定理、书内定理和可移植定理。
3. 给后续机器化本书时提供最小对照表。

## S.1 版本化形式化引用

**定义 S.1.1（形式化入口）。** 一个形式化入口是五元组
$$
(\mathcal L,c,m,\mathcal I,\mathcal A),
$$
其中：

- $\mathcal L$ 是库或仓库；
- $c$ 是 commit 或 release；
- $m$ 是模块路径；
- $\mathcal I$ 是该模块中用于定位结论的 identifier 列表；
- $\mathcal A$ 是该库使用的基础假设、语言选项和公理口径。

**规则 S.1.2（引用不自动迁移）。** 若库 $\mathcal L$ 在 commit $c$ 中证明命题 $P_{\mathcal L}$，本书不得直接把它写成书内命题 $P$ 的证明。可迁移使用需要补充：

1. 本书定义与库中定义的比较；
2. 所用公理、universe、HIT、截断、函数外延性和单值性假设；
3. 命题 $P_{\mathcal L}$ 与本书命题 $P$ 的翻译；
4. 若翻译不是 judgmental 相等，则给出等价、双向蕴含或结构等价。

**定义 S.1.3（当前快照）。** 本附录按 2026-06-29 核查以下公开仓库：

| 库 | 仓库 | commit |
|---|---|---|
| Coq-HoTT | `https://github.com/HoTT/Coq-HoTT` | `a030184c0bfc9d61f3bcd33c67660b800e106427` |
| UniMath | `https://github.com/UniMath/UniMath` | `9ed7661d3ad33c74e35824efccf861b4fdc17323` |
| Cubical Agda | `https://github.com/agda/cubical` | `92166033326aa59800a580b428125f3c654b5e45` |

后续若本书引用 release 而不是 commit，应在本表另增 release 标签、构建工具版本和检查命令。

## S.2 Coq-HoTT

Coq-HoTT 使用 Coq 中适配 HoTT 的基础设置。引用时必须记录是否使用 `Univalence`、`Funext`、HIT 公理化接口和库特定 tactic/typeclass 机制。

### S.2.1 基础路径代数

| 主题 | 模块 | 入口 |
|---|---|---|
| 路径复合、逆、单位律、结合律 | `theories/Basics/PathGroupoids.v` | `concat_1p`, `concat_p1`, `concat_p_pp`, `concat_pV`, `concat_Vp` |
| `ap` 与路径复合 | `theories/Basics/PathGroupoids.v` | `ap_pp`, `ap_V`, `ap_compose` |
| transport 计算 | `theories/Basics/PathGroupoids.v` | `transport_compose`, `transport_paths_FlFr`, `transport_pV`, `transport_Vp` |
| 函数路径的点态化 | `theories/Basics/PathGroupoids.v` | `ap10`, `apD10` |

这些入口对应本书第二章和附录 A、D 的路径代数约定。库中符号方向与本书 $\cdot$ 方向不一定逐字相同；机器化时必须逐条核对复合方向。

### S.2.2 等价、fiber 与同伦层级

| 主题 | 模块 | 入口 |
|---|---|---|
| 等价结构 | `theories/Basics/Overture.v`, `theories/Basics/Equivalences.v` | `IsEquiv`, `Build_Equiv`, `isequiv_adjointify` |
| 逆等价与等价复合 | `theories/Basics/Equivalences.v` | `equiv_inverse`, `isequiv_inverse`, `equiv_compose`, `isequiv_compose` |
| 等价上的路径与诱导 | `theories/Basics/Equivalences.v` | `equiv_ap`, `equiv_path_ind` |
| 同伦层级定义 | `theories/Basics/Trunc.v` | `IsTrunc`, `IsHProp`, `IsHSet` |
| 同伦层级保持 | `theories/Basics/Trunc.v` | `istrunc_succ`, `istrunc_isequiv_istrunc`, `istrunc_equiv_istrunc` |
| 同伦层级命题性 | `theories/Basics/Trunc.v` | `trunc_hprop`, `contr_forall`, `istrunc_forall` |

这些入口对应第五章、附录 E、G、O。特别地，`isequiv_adjointify` 是本书“准逆相干化为等价”的主要形式化对照。

### S.2.3 Universe 与单值性

| 主题 | 模块 | 入口 |
|---|---|---|
| 单值性公理 | `theories/Types/Universe.v` | `Univalence`, `isequiv_equiv_path` |
| 类型路径到等价 | `theories/Types/Universe.v` | `equiv_path`, `equiv_equiv_path` |
| 等价到类型路径 | `theories/Types/Universe.v` | `path_universe`, `path_universe_uncurried`, `equiv_path_universe` |
| transport 计算 | `theories/Types/Universe.v` | `transport_path_universe`, `transport_path_universe_V`, `path_universe_compose`, `path_universe_1` |
| 单值性推出函数外延性入口 | `theories/Types/Universe.v` | `Univalence_implies_Funext` |
| universe 非集合性 | `theories/Types/Universe.v` | `equiv_path2_universe`, `not_hset_Type` |

本书第六至七章采用 `ua` 风格记号；Coq-HoTT 中对应入口主要是 `path_universe`。

### S.2.4 截断、商与 HIT 接口

| 主题 | 模块 | 入口 |
|---|---|---|
| 命题/一般截断核心 | `theories/Truncations/Core.v` | 截断构造、递归和消去接口 |
| 集合商 | `theories/HIT/quotient.v` | quotient HIT 接口 |
| 自由整数商 | `theories/HIT/FreeIntQuotient.v` | free-integer quotient 相关入口 |

这些条目支撑第八、九章和附录 L、M。由于 Coq-HoTT 中部分 HIT 采用公理化或模块接口，本书引用它们时必须仍把 HIT 规则列为输入。

### S.2.5 整数、圆与基本群

| 主题 | 模块 | 入口 |
|---|---|---|
| 整数 successor/predecessor | `theories/Spaces/Int.v` | `int_succ`, `int_pred`, `int_pred_succ`, `int_succ_pred` |
| loop 幂 | `theories/Spaces/Int.v` | `loopexp`, `loopexp_succ_r`, `loopexp_pred_r`, `loopexp_add`, `loopexp_path_universe` |
| 圆 | `theories/Spaces/Circle.v` | `Circle`, `base`, `loop`, `Circle_ind`, `Circle_rec` |
| 圆的计算规则 | `theories/Spaces/Circle.v` | `Circle_ind_beta_loop`, `Circle_rec_beta_loop` |
| 圆的 code 覆盖 | `theories/Spaces/Circle.v` | `Circle_code`, `transport_Circle_code_loop` |
| encode-decode | `theories/Spaces/Circle.v` | `Circle_encode`, `Circle_decode`, `Circle_encode_loopexp`, `Circle_encode_isequiv` |
| 基点 loop 空间 | `theories/Spaces/Circle.v` | `equiv_loopCircle_int` |

这些入口对应第十一章和附录 M、N、V、W。`equiv_loopCircle_int` 是机器化对照入口；本书已在附录 W 补入整数加法群律证明核，在附录 V 补入基本群同构的群运算相容性证明核。

### S.2.6 范畴论

| 主题 | 模块 | 入口 |
|---|---|---|
| 集合范畴 | `theories/Categories/SetCategory.v`, `theories/Categories/SetCategory/Core.v` | Set category 相关结构 |
| Yoneda | `theories/Categories/Yoneda.v` | Yoneda lemma 和 Yoneda embedding 相关入口 |
| 函子范畴 | `theories/Categories/FunctorCategory/Core.v` | functor category 核心结构 |
| 自然变换路径 | `theories/Categories/NaturalTransformation/Paths.v` | natural transformation path 相关入口 |
| 结构等同性 | `theories/Categories/Structure/IdentityPrinciple.v` | category-level SIP 相关入口 |

这些入口对应第十三、十四章和附录 P、Q、U、X。

## S.3 UniMath

UniMath 采用单值基础的 Coq 库口径。引用 UniMath 时，不应默认其定义与本书或 Coq-HoTT judgmentally 相同；尤其要检查 `UU`、`isweq`、`hSet`、`category`、`z_iso` 和 truncation 的具体定义。

### S.3.1 基础与同伦层级

| 主题 | 模块 | 入口 |
|---|---|---|
| h-level 命题性 | `UniMath/Foundations/PartD.v` | `isapropiscontr`, `isapropisweq`, `isapropisofhlevel`, `isapropisaset` |
| 函数空间与命题性 | `UniMath/Foundations/PartD.v` | `impred_isaprop` |
| 集合基础 | `UniMath/Foundations/Sets.v` | `hSet`、集合相关入口 |
| 等价 | `UniMath/MoreFoundations/Equivalences.v` | weak equivalence 相关入口 |
| 单值性基础 | `UniMath/MoreFoundations/Univalence.v` | univalence 相关入口 |
| 单值性推出函数外延性 | `UniMath/Foundations/UnivalenceAxiom.v` | `univalenceStatement`, `funextsecImplication`, `funextfunPreliminaryUAH`, `funcontrUAH`, `funextcontrUAH`, `isweqtoforallpathsUAH`, `funextsecweqFromUnivalence` |

这些入口对应第四、第五、八章和附录 O。

### S.3.2 预范畴、单值范畴与同构

| 主题 | 模块 | 入口 |
|---|---|---|
| 预范畴与范畴 | `UniMath/CategoryTheory/Core/Categories.v` | `precategory`, `category`, `has_homsets`, `id_left`, `id_right`, `assoc` |
| 同构 | `UniMath/CategoryTheory/Core/Isos.v` | `iso`, `z_iso`, `weq_iso_z_iso`, `is_iso_qinv` |
| 单值范畴 | `UniMath/CategoryTheory/Core/Univalence.v` | `idtoiso`, `is_univalent`, `isotoid`, `idtoiso_isotoid`, `isotoid_idtoiso`, `idtoiso_concat`, `idtoiso_inv` |
| 函子与 fully faithful | `UniMath/CategoryTheory/Core/Functors.v` | `fully_faithful`, `weq_from_fully_faithful`, `fully_faithful_inv_hom`, `fully_faithful_reflects_iso_proof` |
| 函子范畴单值性 | `UniMath/CategoryTheory/FunctorCategory.v` | `is_univalent_functor_category` |
| HSET 范畴 | `UniMath/CategoryTheory/Categories/HSET/Univalence.v` | `is_univalent_HSET` |

这些入口是第十三章和附录 P 的主要 UniMath 对照。

### S.3.3 Yoneda 与范畴构造

| 主题 | 模块 | 入口 |
|---|---|---|
| Yoneda embedding | `UniMath/CategoryTheory/yoneda.v` | `yoneda_objects`, `yoneda`, `yoneda_weq`, `yoneda_fully_faithful` |
| 协变 Yoneda | `UniMath/CategoryTheory/covyoneda.v` | covariant Yoneda 相关入口 |
| Yoneda 与结构保存 | `UniMath/CategoryTheory/YonedaBinproducts.v`, `UniMath/CategoryTheory/YonedaExponentials.v` | `yoneda_preserves_binproduct`, `yoneda_preserves_exponentials` |

这些入口对应第十四章和附录 Q。

### S.3.4 结构等同性与代数结构

| 主题 | 模块 | 入口 |
|---|---|---|
| 抽象 SIP | `UniMath/MoreFoundations/StructureIdentity.v` | `Struc_univalence` |
| displayed SIP | `UniMath/CategoryTheory/DisplayedCats/SIP.v` | `is_univalent_disp_from_SIP_data` |
| displayed category 单值性 | `UniMath/CategoryTheory/DisplayedCats/Univalence.v`, `UniMath/CategoryTheory/DisplayedCats/Total.v` | `is_univalent_disp`, `is_univalent_total_category` |
| 通用代数结构 | `UniMath/CategoryTheory/Categories/Universal_Algebra/Algebras.v` | `is_univalent_algebras_disp`, `is_univalent_category_algebras` |
| 群、幺半群、Abelian group 范畴 | `UniMath/CategoryTheory/Categories/Group.v`, `Monoid.v`, `AbelianGroup.v` | `is_univalent_group_category`, `is_univalent_monoid_category`, `is_univalent_abelian_group_category` |

这些入口对应附录 I、J、P。

## S.4 Cubical Agda

Cubical Agda 在语言层提供 interval、path、Glue 和 cubical primitives。引用 `agda/cubical` 时必须记录 Agda 版本、cubical 选项、库 commit，以及是否使用实验模块。

### S.4.1 Cubical 基础、单值性与 h-level

| 主题 | 模块 | 入口 |
|---|---|---|
| 函数外延性 | `Cubical/Foundations/Prelude.agda` | `funExt`, `funExt⁻` |
| 等价 | `Cubical/Foundations/Equiv.agda` | `isEquiv`, `isoToEquiv`, `isEquiv≃isEquiv'` |
| 单值性 | `Cubical/Foundations/Univalence.agda` | `ua`, `uaIdEquiv`, `uaβ`, `uaη`, `uaCompEquiv`, `uaInvEquiv` |
| 命题外延性 | `Cubical/Foundations/Univalence.agda` | `hPropExt` |
| 同伦层级 | `Cubical/Foundations/HLevels.agda` | `isOfHLevel`, `isPropIsOfHLevel`, `isOfHLevelΠ`, `isOfHLevelΣ`, `isOfHLevelRespectEquiv` |

这些入口对应第六、七、十六章和附录 F、G、O、Z。

### S.4.2 HIT、截断与商

| 主题 | 模块 | 入口 |
|---|---|---|
| 命题截断 | `Cubical/HITs/PropositionalTruncation/Base.agda` | `∥_∥₁`, `∣_∣₁`, `squash₁` |
| 命题截断递归 | `Cubical/HITs/PropositionalTruncation/Properties.agda` | `rec`, `rec2`, `elim`, `isPropPropTrunc`, `propTruncIdempotent` |
| 一般截断 | `Cubical/HITs/Truncation/Properties.agda` | `isOfHLevelTrunc`, `PathIdTruncIso`, `PathIdTrunc`, `setTruncTrunc2Iso` |
| 集合商 | `Cubical/HITs/SetQuotients.agda` | `_/_`, `[_]`, `eq/`, `squash/` |
| 圆和球 | `Cubical/HITs/S1.agda`, `Cubical/HITs/Sn/Properties.agda` | `S¹`, sphere 相关 induction/elimination 入口 |

这些入口对应第八至十一章和附录 L、N、Z。

### S.4.3 单值范畴论与 Rezk 完备化

| 主题 | 模块 | 入口 |
|---|---|---|
| 单值集合范畴 | `Cubical/Categories/Instances/Sets.agda` | `isUnivalentSET`, `CatIsoToPath`, `univSetβ` |
| 预层范畴 | `Cubical/Categories/Presheaf/Base.agda` | `isUnivalentPresheafCategory` |
| 可表预层 | `Cubical/Categories/Presheaf/Representable.agda` | Yoneda/representable presheaf 相关入口 |
| Rezk 完备化：Yoneda 本质像 | `Cubical/Categories/RezkCompletion/Construction.agda` | `RezkByYoneda`, `YonedaImage`, `isUnivalentYonedaImage`, `ToYonedaImage`, `isWeakEquivalenceToYonedaImage`, `isRezkCompletionToYonedaImage` |
| Rezk 完备化：HIT 构造 | `Cubical/Categories/RezkCompletion/Construction.agda` | `RezkByHIT`, `RezkOb`, `inc`, `inc-ua`, `RezkHom` |

这些入口对应第十三、十四章和附录 P、R、AA。特别地，`RezkByYoneda` 是本书附录 R 的直接机器化对照，Rezk completion 的 weak-equivalence/universal-property 相关入口对应附录 AA 的证明架构。

### S.4.4 代数结构、群路径与 cohomology

| 主题 | 模块 | 入口 |
|---|---|---|
| 群的结构等同 | `Cubical/Algebra/Group/GroupPath.agda` | `GroupPath`, `uaGroup`, `uaGroupId`, `uaCompGroupEquiv` |
| 整数群 | `Cubical/Algebra/Group/Instances/Int.agda`, `Cubical/Algebra/AbGroup/Instances/Int.agda` | integer group/abelian group 实例 |
| Eilenberg-Mac Lane 上同调基础 | `Cubical/Cohomology/EilenbergMacLane/Base.agda` | `coHom`, `coHomGr`, `coHomRed`, `coHomRedGr`, `coHomFun`, `coHomHom` |
| cup product | `Cubical/Cohomology/EilenbergMacLane/CupProduct.agda` | `_⌣_`, `⌣-0ₕ`, `0ₕ-⌣`, `⌣-1ₕ`, `assoc⌣Dep`, `comm⌣Dep` |
| Eilenberg-Steenrod 性质 | `Cubical/Cohomology/EilenbergMacLane/EilenbergSteenrod.agda` | `satisfies-ES`, `Suspension`, `Exactness`, `Dimension` |
| 典型计算 | `Cubical/Cohomology/EilenbergMacLane/Groups/Sn.agda`, `Torus.agda`, `RP2.agda` | `H¹[S¹,G]≅G`, `Hⁿ[Sⁿ,G]≅G`, `H²[T²,G]≅G`, `H²[RP²,G]≅G/2` |
| cohomology ring | `Cubical/Cohomology/EilenbergMacLane/RingStructure.agda` | cohomology graded/ring structure 入口 |
| 整系数旧接口与论文入口 | `Cubical/ZCohomology/*.agda`, `Cubical/Papers/ZCohomology.agda`, `Cubical/Papers/CohomologyRings.agda` | `coHom`, `coHomGr`, `Hⁿ-Sⁿ≅ℤ`, cohomology ring examples |

这些入口对应第十二、十六、十七章和附录 Y。`Cubical/Cohomology/EilenbergMacLane/*` 是当前更结构化的入口；`Cubical/ZCohomology/*` 和 `Cubical/Experiments/*` 应按实验或历史接口谨慎引用。

## S.5 本书当前覆盖与剩余机器化义务

**命题 S.5.1（当前形式化覆盖等级）。** 在本附录的快照下，本书当前可把以下主题标为“有版本化形式化入口”：

1. 路径代数、transport、函数路径点态化；
2. 等价、逆等价、等价复合、同伦层级保持；
3. 单值性及其 transport 计算；
4. 命题截断、一般截断、集合商的库入口；
5. 整数群、圆的 encode-decode 与 $\Omega(\mathbb S^1)\simeq\mathbb Z$ 的 Coq-HoTT/Cubical Agda 对照；
6. 单值范畴、集合范畴、函子范畴、Yoneda、Rezk 完备化的 UniMath/Cubical Agda 对照；
7. Cubical Agda 中 Eilenberg-Mac Lane 上同调、cup product 和典型空间上同调计算入口。

**证明。** 逐项由 S.2-S.4 的模块路径和入口名给出。该命题仅断言“可定位形式化入口存在”，不宣称本书所有陈述已经逐行通过机器验证。$\square$

**剩余义务 S.5.2。** 若要把本书升级为逐行机器验证版本，还需补充：

1. 选择唯一目标系统，例如 Coq-HoTT、UniMath 或 Cubical Agda；
2. 把本书每章定义翻译为该系统中的定义；
3. 建立本书符号与库 identifier 的双向索引；
4. 对每个定理列出依赖脚本、导入顺序、universe 约束和公理列表；
5. 在 CI 中固定工具链版本并运行构建。

因此，本附录解决的是“可审计形式化来源”缺口，不替代逐行形式化工程。
