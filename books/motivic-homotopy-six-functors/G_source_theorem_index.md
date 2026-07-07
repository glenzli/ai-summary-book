# 附录 G：资料源定理索引与 locator ledger

## 本附录目标

本附录把正文中的外部输入按主题重新索引，方便后续出版校对时补 theorem locator。它不同于 `SOURCES.md` 的资料清单；这里以“本书使用的定理”为单位。

## G.1 基础 motivic homotopy

| 本书标签 | 定理内容 | 资料源 | 使用位置 | locator 状态 |
| --- | --- | --- | --- | --- |
| MH-3.8 | `\mathbf{SH}(S)` 构造、稳定性、对称幺半性 | Morel-Voevodsky, Jardine, Ayoub, Cisinski-Deglise | 第三章 | pending |
| PU-6.7 | Homotopy purity | Morel-Voevodsky | 第六章 | pending |
| MW-18.2 | `End(1_k)\simeq GW(k)` | Morel | 第十八章 | pending |

## G.2 六操作

| 本书标签 | 定理内容 | 资料源 | 使用位置 | locator 状态 |
| --- | --- | --- | --- | --- |
| MO-5.2 | Motivic 六操作存在性 | Ayoub, Cisinski-Deglise, Drew-Gallauer | 第五章 | source-verified |
| MO-5.6 | Proper compatibility | Ayoub, Cisinski-Deglise | 第五章 | pending |
| MO-5.14 | Localization recollement | Morel-Voevodsky, Ayoub, Cisinski-Deglise | 第五章 | pending |
| PU-6.11 | Smooth purity | Ayoub, Hoyois, Cisinski-Deglise | 第六章 | pending |
| AD-7.3 | Smooth ambidexterity | Ayoub, Hoyois | 第七章 | pending |
| BC-8.3 | Base change | Ayoub, Cisinski-Deglise, Drew-Gallauer | 第八章 | pending |
| PF-8.5 | Projection formula | Ayoub, Cisinski-Deglise | 第八章 | pending |
| UF-23.2 | Universal six-functor formalism | Drew-Gallauer | 第二十三章 | located: Theorem 7.14; Proposition 7.13; Theorem 7.3 |

## G.3 Cohomology theories and spectra

| 本书标签 | 定理内容 | 资料源 | 使用位置 | locator 状态 |
| --- | --- | --- | --- | --- |
| HZ-9.1 | `H\mathbb Z` 构造 | Voevodsky, Spitzweck | 第九章 | source-verified |
| HZ-9.9 | Chow/motivic cohomology 比较 | Bloch, Voevodsky, MVW | 第九章 | pending |
| DM-10.8 | `DM` 与 `H\mathbb Z`-modules 比较 | Röndigs-Ostvær, Cisinski-Deglise | 第十章 | source-verified |
| KG-11.1 | `KGL` 表示 `KH` | Röndigs-Spitzweck-Ostvær | 第十一章 | source-verified |
| MG-12.4 | `MGL` orientation universality | Panin-Pimenov-Röndigs | 第十二章 | source-verified |
| MG-12.7 | Hopkins-Morel 型比较 | Hoyois, Spitzweck | 第十二章 | source-verified |
| SL-13.8 | Zero slice | Voevodsky | 第十三章 | source-verified |

## G.4 Transfers, framed structures, norms and refinements

| 本书标签 | 定理内容 | 资料源 | 使用位置 | locator 状态 |
| --- | --- | --- | --- | --- |
| TR-14.3 | Finite correspondences form category | Suslin-Voevodsky, MVW | 第十四章 | pending |
| FR-15.x | Framed recognition | Elmanto-Hoyois-Khan-Sosnilo-Yakerson | 第十五章 | located: Theorem 1.2.3; Theorem 3.5.14 |
| FC-16.2 | Fundamental classes | Deglise-Jin-Khan | 第十六章 | located: Definition 3.2.5; Theorems 3.3.2, 4.1.4, 4.2.1 |
| NM-17.x | Norm functors | Bachmann-Hoyois | 第十七章 | located: Proposition 4.5; Definition 7.1 |
| MW-18.7 | Chow-Witt/Milnor-Witt refinements | Fasel, Deglise-Fasel | 第十八章 | pending |

## G.5 Extensions and frontier

| 本书标签 | 定理内容 | 资料源 | 使用位置 | locator 状态 |
| --- | --- | --- | --- | --- |
| EQ-19.3 | Equivariant motivic six operations | Hoyois | 第十九章 | source-verified |
| ST-20.x | Stacks 上六操作 | Khan-Ravi | 第二十章 | source-verified |
| LG-21.2 | Log motivic homotopy | Park | 第二十一章 | source-verified |
| PF-21.5 | Perfect schemes motivic homotopy | Dahlhausen-Hekking-Wolters | 第二十一章 | R |
| RE-22.4 | Betti realization | Ayoub | 第二十二章 | pending |
| RB-24.1 | Pullback formalism criteria | Magen 2025 | 第二十四章 | R |
| RB-24.2 | Complex analytic stacks localization | Magen 2026 | 第二十四章 | R |

## G.6 出版闭合规则

**规则 G.1.** P0 条目必须在 `REFERENCE_LOCATOR_LEDGER.md` 中补精确 locator 后，才能在最终出版态中作为正文外部输入使用。

**规则 G.2.** P1 条目可以保留在高级章节，但若被主链证明调用，必须升级为 P0 并补 locator。

**规则 G.3.** R 条目只能作为研究边界描述，不得用于证明主链定理。

## G.7 本附录小结

本附录把外部输入按用途重排。下一轮工作不应继续增加大纲式章节，而应从本索引出发逐项补 locator、检查假设并把外部输入的边界写入正文。
