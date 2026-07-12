# 定理账本

本账本登记本书内部证明、外部输入和研究边界。`P0` 表示正文依赖，必须在出版前补精确 locator；`P1` 表示高级章节依赖；`R` 表示研究边界，只能作为前沿说明。

最近核查：2026-07-11。`located` 条目的详细版本与边界见
`P0_REFERENCE_LOCATORS_BATCH_1.md`、`P0_REFERENCE_LOCATORS_BATCH_2.md`、
`P0_REFERENCE_LOCATORS_BATCH_3.md`。

| 标签 | 类型 | 内容 | 默认假设 | 依赖源 | 状态 |
| --- | --- | --- | --- | --- | --- |
| CAT-A.5 | 外部输入 | presheaf free cocompletion/presentability | `\mathbb U`-小 infinity-category | Lurie HTT 5.1.5.6, 5.1.5.8 | located |
| CAT-A.8 | 外部输入 | higher sheafification 为 accessible left exact localization | `\mathbb U`-小 Grothendieck site；不含 hypercompletion | Lurie HTT 6.2.2.7 | located |
| CAT-A.10 | 外部输入 | 小生成 accessible localization 及泛性质 | presentable `\mathcal C`，`\mathbb U`-小 maps | Lurie HTT 5.5.4.15, 5.5.4.20 | located |
| CAT-C.4 | 外部输入 | pointed objects 的 presentable smash product | Cartesian presentable；乘积分变量保小余极限 | Lurie HA 4.8.1.15, 4.8.1.21, 4.8.2.11, 4.8.2.14 | located |
| CAT-C.8 | 外部输入 | symmetric monoidal object-inversion 与 3-symmetric spectra 比较 | presentably symmetric monoidal；谱模型需 3-symmetry | Robalo 4.10, 4.24, 4.29 | located |
| MH-1.9 | 内部推论 | `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` presentable | `\operatorname{Sm}_S` 为 `\mathbb U`-小站点 | CAT-A.8 | 已证明 |
| MH-1.13 | 内部命题 | Nisnevich sheaf 条件推出 elementary distinguished square 的拉回条件 | 使用该方块为覆盖生成数据 | Grothendieck topology 定义 | 已证明 |
| MH-2.4 | 内部推论 | `\mathbf H(S)` presentable | `W_{\mathbb A^1}` 为 `\mathbb U`-小态射集 | CAT-A.8, CAT-A.10 | 已证明 |
| MH-2.6 | 内部命题 | A1-局部对象由 `F(X)\to F(X\times A1)` 检测 | Yoneda + localization | 一般局部化理论 | 已证明 |
| MH-2.9 | 内部命题 | `X\times A1\to X` 在 `\mathbf H(S)` 中为等价 | 定义 | A1 localization | 已证明 |
| MH-3.5 | 内部命题 | `T\simeq S^{1,0}\wedge G_m` | `A1` contractible, pointed cofiber | pointed cofiber calculus | 已证明 |
| MH-3.8 | 外部输入 | symmetric monoidal `T`-反演、3-symmetry、symmetric spectra 比较 | `S` 为默认基概形 | Robalo 4.10, 4.24, 4.29; Hoyois 6.3, 6.4, 6.7 | located |
| MH-3.12 | 内部命题 | 悬挂谱函子保持小余极限 | 稳定化为左伴随 | presentable stabilization | 已证明 |
| MH-3.13 | 内部命题 | `T` 可逆推出 `S^{1,0}`、`G_m` 分别可逆并给出稳定性 | symmetric monoidal；MH-3.8 | 可逆因子引理；HA 1.4.2.27 | 已证明 |
| MH-3.17 | 外部输入 | `\mathbf{SH}(S)` 紧生成 | `S` qcqs；默认范围满足 | Hoyois Proposition 6.4(2)-(3) | located |
| MH-3.19 | 外部输入 | 几何构造性对象等于紧致对象 | MH-3.17 + stable compact generation | HTT 5.3.5.11, 5.5.7.8; HA 1.4.4.1 | located |
| SF-4.5 | 内部命题 | `f^*` 的右伴随 `f_*` 若存在则由映射空间唯一确定 | infinity-范畴伴随 | 一般范畴论 | 已证明 |
| SF-4.10 | 内部命题 | ordinary base-change mate 的构造 | Cartesian 方块和伴随 | mate calculus | 已证明 |
| SF-4.14 | 内部命题 | 模线性推出 `!`-投影公式 | module functor 定义 | 幺半范畴论 | 已证明 |
| SF-4.15 | 内部命题 | dualizable 基系数推出 ordinary projection formula | `B` dualizable；任意 `f` | duality + `f^*\dashv f_*` | 已证明 |
| SF-4.17 | 外部输入 | 默认基范畴上的 motivic 六操作 package | finite type over fixed Noetherian `B`; exceptional = separated | Hoyois 1.1, 6.18 | located |
| SF-4.20 | 内部命题 | localization recollement 推出局部化余纤维序列 | open-closed recollement | 稳定范畴 | 已证明 |
| MO-5.2 | 外部输入 | 六操作的精确定义域和方差 | finite type `B`-schemes；`f_!,f^!` only separated `f` | Hoyois 1.1, 6.18 | located |
| MO-5.6 | 外部输入 | proper compatibility `f_!\simeq f_*` | `f` proper | Hoyois 6.18(1) | located |
| MO-5.14 | 外部输入 | open-closed motivic localization recollement | closed/open complement | Hoyois 6.18(4)-(5) | located |
| MO-5.21 | 外部输入 | `f_*` 与 compactifiable `f` 的 `f^!` 保持小余极限 | 默认态射；`f` separated for `f^!` | Hoyois 6.4(4), 6.19 | located |
| PU-6.7 | 外部输入 | homotopy purity `X/(X-Z)\simeq Th(N)` | `X,Z` smooth over `S`; closed immersion | Morel-Voevodsky §3 Thm. 2.23 | located |
| PU-6.10 | 外部输入 | Thom twists 可逆并延拓到 K-theory | vector bundles/perfect twists | Hoyois 6.5, 6.7; motivic J | located |
| PU-6.12 | 外部输入 | `\operatorname{Ho}(\mathbf{SH})` 层的 smoothable-lci purity transformation `\Sigma^{\tau_f}f^*\to f^!` | smoothable lci separated finite-type `f`; composition organized as `\mathbf{Tri}`-valued pseudofunctors, BC as homotopy-category commuting squares; transverse BC requires Tor-independence | Deglise-Jin-Khan Proposition 2.5.4, Remark 2.5.5, Theorems 3.3.2, 4.1.4, §4.3.1 | located；不声称 infinity-enhancement |
| PU-6.13 | 外部输入 | smooth purity transformation 为等价 | `f` smooth separated | Hoyois 6.18(2); Deglise-Jin-Khan §4.3.1 | located |
| PU-6.14 | 定义/边界 | coefficientwise `f`-purity 与 absolute purity | smoothable lci between regular schemes | Deglise-Jin-Khan Definitions 4.3.7, 4.3.11 | located |
| AD-7.3 | 内部命题 | smooth ambidexterity `f_!\simeq f_\sharp\Sigma^{-T_f}` | `f` smooth separated；PU-6.13 | Thom duality + 左伴随唯一性 | 已证明 |
| AD-7.10 | 外部输入 | smooth proper dualizability；对偶为 `f_\sharp\Sigma^{-T_f}1\simeq f_*1` | `f` smooth proper | Hoyois Corollary 6.13 | located |
| BC-8.3 | 外部输入 | exceptional BC；proper ordinary BC；smooth ordinary BC | respectively `f` separated; `f` proper; base-change map smooth | Hoyois 6.18(3), 6.10；Proposition 4.2 stabilized after 6.4 | located |
| PF-8.5 | 外部输入 | exceptional projection formula | `f` separated；任意系数 | Hoyois 6.18(7) | located |
| PF-8.7 | 内部命题 | ordinary projection formula with dualizable base coefficient | 任意 `f`; `B` dualizable | SF-4.15 | 已证明 |
| TRI-F.1 | 外部输入 | stable infinity-category 的 canonical triangulated shadow | stable infinity-category | Lurie HA 1.1.2.14 | located |
| HZ-9.1 | 外部输入 | commutative `H\mathbb Z` construction/representation | mixed-characteristic Dedekind base；smooth test schemes | Spitzweck Theorem 7.18, Corollary 7.19; base change Theorem 8.25 | located |
| HZ-9.9 | 外部输入 | `H^{p,q}\cong CH^q(-,2q-p)` | perfect field；smooth separated scheme | MVW Theorem 19.1, Corollary 19.2 | located |
| HZ-9.10 | 外部输入 | `H^{n,n}(k)\cong K_n^M(k)` | 任意域 | MVW Theorem 5.1 | located |
| DM-10.8 | 外部输入 | `DM` 与 `H\mathbb Z`-modules 比较；三角/稳定 infinity 分层 | char 0 for integral triangulated；field and `1/e` for stable infinity | Röndigs-Ostvær Theorem 1.1; Elmanto-Kolderup Theorem 5.2, Corollary 5.3 | located |
| KG-11.1 | 外部输入 | strict commutative Bott model for `KGL` | Noetherian finite-dimensional | Röndigs-Spitzweck-Ostvær Lemma 2.5, Theorems 3.6, 4.1 | located |
| KG-11.6 | 外部输入 | `KGL` 表示 `KH` | Noetherian finite-dimensional | Cisinski Theorem 2.20 | located |
| KG-11.7 | 外部输入 | regular case `K\simeq KH` | regular Noetherian ring；quasi-projective regular Noetherian scheme | Weibel K-book IV Corollary 12.3.1, Lemma 12.8(3) | located |
| KG-11.12 | 外部输入 | `KH` satisfies cdh descent | Noetherian finite-dimensional | Cisinski Theorem 3.9 | located |
| MG-12.4 | 外部输入 | `MGL` orientation universality as monoid-map set bijection | field；homotopy category | Panin-Pimenov-Röndigs Theorem 2.3.1 | located |
| MG-12.11 | 外部输入 | Hopkins-Morel comparison | essentially smooth over field；invert exponent `c` | Hoyois Theorem 7.12 | located |
| SL-13.8 | 外部输入 | `s_0(1)\simeq HZ` and slices are `HZ`-modules | characteristic zero field | Voevodsky Theorem 6.6, Introduction pp.106-107 | located |
| TR-14.3 | 外部输入 | finite correspondences form additive category | 本章取 perfect field | MVW Lecture 1, Lemmas 1.4, 1.7, Definition 1.5 | located |
| TR-14.10 | 外部输入 | Nisnevich sheafification preserves transfers | 任意 field；本章取 perfect | MVW Theorem 13.1 | located |
| FR-15.x | 外部输入 | framed motivic spaces recognition principle | perfect field；very effective/effective 层级分开 | Elmanto-Hoyois-Khan-Sosnilo-Yakerson Theorems 1.2.3, 3.5.14 | located |
| FC-16.2 | 外部输入 | motivic fundamental classes 与 Gysin maps | smoothable lci separated finite type；transverse BC 需 Tor-independent | Deglise-Jin-Khan Definition 3.2.5; Proposition 2.5.4; Theorems 3.3.2, 4.1.4, 4.2.1 | located |
| FC-16.13 | 外部输入 | fundamental-class/coefficient Gysin excess formula | Cartesian square 中 `f,g` 均 smoothable lci s-morphisms；§3.3.3 的 excess bundle `\xi` 存在；系数版要求 unital associative commutative multiplication；push-pull 另要求两竖边 proper | Deglise-Jin-Khan §3.3.3, Propositions 3.3.4, 4.2.2 | located |
| NM-17.x | 外部输入 | unstable finite-locally-free norms；stable finite-etale norms；normed `HZ/KGL/MGL` | unstable construction 用 Theorem 3.3/localization 与 Corollary 3.11；Proposition 3.13 另要求 finite etale 且 Weil restriction 存在；其余定理基假设分别保留 | Bachmann-Hoyois Theorem 3.3; Corollary 3.11; Propositions 3.13, 4.5; Definition 7.1; Theorems 14.5, 15.22, 16.19 | located |
| MW-18.2 | 外部输入 | `End(1_k)\cong GW(k)` | perfect field | Morel Corollary 6.43 + Lemma 3.10 | located |
| MW-18.5 | 外部输入 | stable-range sphere maps are `K_*^{MW}` | perfect field；`n\ge2`, target weight positive | Morel Corollary 6.43 | located |
| MW-18.7 | 高级外部输入 | Chow-Witt/Milnor-Witt motives、枚举与 Gauss-Bonnet refinements | 模型/field/regularity/orientation 逐项指定 | Fasel, Deglise-Fasel 等 | P1 boundary；不参与 P0 主线 |
| EQ-19.3 | 外部输入 | quotient stacks 上 equivariant motivic six operations | linearly reductive 等 | Hoyois | P0 |
| ST-20.x | 外部输入 | scalloped algebraic stacks 上的六操作 | Khan-Ravi 假设 | Khan-Ravi | P0 |
| LG-21.2 | 外部输入 | log schemes 上的 A1-stable motivic homotopy | fs log schemes | Park | P1 |
| PF-21.5 | 研究边界 | perfect schemes 上 motivic homotopy | positive characteristic | Dahlhausen-Hekking-Wolters | R |
| RE-22.4 | 外部输入 | Betti realization | complex bases | Ayoub 等 | P0 |
| UF-23.2 | 外部输入 | universal six-functor formalism | coefficient systems | Drew-Gallauer | P0 |
| RB-24.1 | 研究边界 | pullback formalism 中更强 universal six-functor criterion | Magen 2025 假设 | Magen 2025 | R |
| RB-24.2 | 研究边界 | complex analytic stacks 的 localization theorem | Magen 2026 假设 | Magen 2026 | R |
