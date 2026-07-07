# 定理账本

本账本登记本书内部证明、外部输入和研究边界。`P0` 表示正文依赖，必须在出版前补精确 locator；`P1` 表示高级章节依赖；`R` 表示研究边界，只能作为前沿说明。

| 标签 | 类型 | 内容 | 默认假设 | 依赖源 | 状态 |
| --- | --- | --- | --- | --- | --- |
| MH-1.8 | 内部命题 | `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` presentable | `\operatorname{Sm}_S` 小站点 | presentable sheaf theory | 已证明 |
| MH-1.12 | 内部命题 | Nisnevich sheaf 条件推出 elementary distinguished square 的拉回条件 | 使用该方块为覆盖生成数据 | Grothendieck topology 定义 | 已证明 |
| MH-2.4 | 内部命题 | `\mathbf H(S)` presentable | accessible localization | HTT/presentable localization | 已证明 |
| MH-2.6 | 内部命题 | A1-局部对象由 `F(X)\to F(X\times A1)` 检测 | Yoneda + localization | 一般局部化理论 | 已证明 |
| MH-2.10 | 内部命题 | `X\times A1\to X` 在 `\mathbf H(S)` 中为等价 | 定义 | A1 localization | 已证明 |
| MH-3.5 | 内部命题 | `T\simeq S^{1,0}\wedge G_m` | `A1` contractible, pointed cofiber | pointed cofiber calculus | 已证明 |
| MH-3.8 | 外部输入 | `\mathbf{SH}(S)` 为 stable presentable symmetric monoidal infinity-category | 默认有限维 Noetherian 基；扩展见源 | Morel-Voevodsky, Jardine, Ayoub, Cisinski-Deglise, Hoyois | P0 |
| MH-3.12 | 内部命题 | 悬挂谱函子保持小余极限 | 稳定化为左伴随 | presentable stabilization | 已证明 |
| SF-4.5 | 内部命题 | `f^*` 的右伴随 `f_*` 若存在则由映射空间唯一确定 | infinity-范畴伴随 | 一般范畴论 | 已证明 |
| SF-4.9 | 内部命题 | ordinary base-change mate 的构造 | Cartesian 方块和伴随 | mate calculus | 已证明 |
| SF-4.12 | 内部命题 | 模线性推出投影公式 | module functor 定义 | 幺半范畴论 | 已证明 |
| SF-4.15 | 外部输入 | `\mathbf{SH}(-)` 的 motivic 六操作 | 相应基范畴和态射类 | Ayoub, Cisinski-Deglise, Drew-Gallauer | P0 |
| SF-4.18 | 内部命题 | localization recollement 推出局部化余纤维序列 | open-closed recollement | 稳定范畴 | 已证明 |
| MO-5.2 | 外部输入 | `\mathbf{SH}(-)` 支持 motivic 六操作 | 默认基范畴 | Ayoub, Cisinski-Deglise, Drew-Gallauer | P0 |
| MO-5.6 | 外部输入 | proper compatibility `f_!\simeq f_*` | `f` proper | Ayoub, Cisinski-Deglise | P0 |
| MO-5.14 | 外部输入 | open-closed motivic localization recollement | closed/open complement | Morel-Voevodsky, Ayoub, Cisinski-Deglise | P0 |
| PU-6.7 | 外部输入 | homotopy purity `X/(X-Z)\simeq Th(N)` | smooth closed immersion | Morel-Voevodsky | P0 |
| PU-6.11 | 外部输入 | smooth purity `f^!\simeq\Sigma^{T_f}f^*` | `f` smooth | Ayoub, Cisinski-Deglise, Hoyois | P0 |
| PU-6.12 | 外部输入 | closed immersion purity / absolute purity | regular closed immersion | Ayoub, Cisinski-Deglise, Deglise-Jin-Khan | P0 |
| AD-7.3 | 外部输入 | smooth ambidexterity `f_!\simeq f_\sharp\Sigma^{-T_f}` | `f` smooth | Ayoub, Hoyois | P0 |
| AD-7.10 | 外部输入 | smooth proper dualizability | `f` smooth proper | motivic Atiyah duality | P0 |
| BC-8.3 | 外部输入 | motivic base change equivalences | allowed Cartesian squares | Ayoub, Cisinski-Deglise, Drew-Gallauer | P0 |
| PF-8.5 | 外部输入 | motivic projection formula | allowed morphisms | Ayoub, Cisinski-Deglise | P0 |
| HZ-9.1 | 外部输入 | motivic Eilenberg-Mac Lane spectrum `H\mathbb Z` | 基和系数假设 | Voevodsky, Spitzweck, Cisinski-Deglise | P0 |
| HZ-9.9 | 外部输入 | motivic cohomology 与 Chow groups 比较 | smooth over field 等 | Bloch, Voevodsky, MVW | P0 |
| DM-10.8 | 外部输入 | `DM` 与 `H\mathbb Z`-modules 比较 | perfect field/系数假设 | Röndigs-Ostvær, Cisinski-Deglise | P0 |
| KG-11.1 | 外部输入 | `KGL` 表示 homotopy K-theory | Noetherian finite-dimensional 等 | Röndigs-Spitzweck-Ostvær | P0 |
| KG-11.9 | 外部输入 | `KH` satisfies cdh descent | finite type/noetherian 等 | Haesemeyer, Cisinski, Weibel | P0 |
| MG-12.4 | 外部输入 | `MGL` orientation universality | field/基假设 | Panin-Pimenov-Röndigs | P0 |
| MG-12.7 | 外部输入 | Hopkins-Morel 型比较 | invert characteristic exponent 等 | Hoyois, Spitzweck | P0 |
| SL-13.8 | 外部输入 | `s_0(1)\simeq HZ` | characteristic/base restrictions | Voevodsky and extensions | P0 |
| TR-14.3 | 外部输入 | finite correspondences form additive category | perfect field | Suslin-Voevodsky | P0 |
| TR-14.10 | 外部输入 | Nisnevich sheafification preserves transfers | perfect field | Voevodsky/MVW | P0 |
| FR-15.x | 外部输入 | framed motivic spaces recognition principle | perfect field 等假设 | Elmanto-Hoyois-Khan-Sosnilo-Yakerson | P0 |
| FC-16.2 | 外部输入 | motivic fundamental classes and formulas | lci 等 | Deglise-Jin-Khan | P0 |
| NM-17.x | 外部输入 | finite etale norm functors on `SH` | finite etale morphisms | Bachmann-Hoyois | P0 |
| MW-18.2 | 外部输入 | `End(1_k)\simeq GW(k)` | perfect field | Morel | P0 |
| MW-18.7 | 外部输入 | Chow-Witt/Milnor-Witt refinements | field/regularity assumptions | Fasel, Deglise-Fasel 等 | P1 |
| EQ-19.3 | 外部输入 | quotient stacks 上 equivariant motivic six operations | linearly reductive 等 | Hoyois | P0 |
| ST-20.x | 外部输入 | scalloped algebraic stacks 上的六操作 | Khan-Ravi 假设 | Khan-Ravi | P0 |
| LG-21.2 | 外部输入 | log schemes 上的 A1-stable motivic homotopy | fs log schemes | Park | P1 |
| PF-21.5 | 研究边界 | perfect schemes 上 motivic homotopy | positive characteristic | Dahlhausen-Hekking-Wolters | R |
| RE-22.4 | 外部输入 | Betti realization | complex bases | Ayoub 等 | P0 |
| UF-23.2 | 外部输入 | universal six-functor formalism | coefficient systems | Drew-Gallauer | P0 |
| RB-24.1 | 研究边界 | pullback formalism 中更强 universal six-functor criterion | Magen 2025 假设 | Magen 2025 | R |
| RB-24.2 | 研究边界 | complex analytic stacks 的 localization theorem | Magen 2026 假设 | Magen 2026 | R |
