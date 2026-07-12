# Reference locator ledger

核查日期：2026-07-11

本账本用于把 `THEOREM_LEDGER.md` 中的外部输入定位到精确资料源。当前教材可读版允许外部输入以资料源和假设边界标记出现；出版级闭合前必须为 P0 条目补版本、章节、定理号、页码和假设。

## 状态标记

- `P0`：正文主链依赖；必须补 locator。
- `P1`：高级章节或比较章节依赖；出版前应补 locator。
- `R`：研究边界；只需版本和摘要级定位，不能作为无条件正文定理。
- `located`：已有精确章节/定理号/页码。
- `source-verified`：已联网确认题名、作者、日期或出版信息，但未补定理号。
- `pending`：尚未精确核查。

## P0 locator queue

| 账本标签 | 结果 | 资料源 | 当前状态 | 下一步 |
| --- | --- | --- | --- | --- |
| MH-3.8 | symmetric monoidal `T`-反演、3-symmetry、谱模型 | Robalo; Hoyois | located | Robalo 4.10, 4.24, 4.29；Hoyois 6.3, 6.4, 6.7；见 batch 2 |
| MH-3.17 | Compact generation | Hoyois | located | Proposition 6.4(2)-(3)，平凡群；见 batch 2 |
| MH-3.19 | Compact objects as thick closure | Lurie | located | HTT Propositions 5.3.5.11, 5.5.7.8；HA Proposition 1.4.4.1；见 batch 2 |
| MO-5.2 | Motivic 六操作的定义域/方差 | Hoyois; Ayoub | located | Hoyois Theorems 1.1, 6.18，平凡群；见 batch 2 |
| MO-5.14 | Motivic localization recollement | Hoyois | located | Theorem 6.18(4)-(5)，平凡群；见 batch 2 |
| MO-5.21 | Six-operation continuity | Hoyois | located | Proposition 6.4(4), Corollary 6.19；见 batch 2 |
| PU-6.7 | Homotopy purity | Morel-Voevodsky | located | Section 3, Theorem 2.23；见 batch 2 |
| PU-6.12 | `Ho(SH)`-level smoothable-lci purity transformation | Deglise-Jin-Khan | located | Proposition 2.5.4（复合与 BC 交换方块）；Remark 2.5.5（复合数据组织为 homotopy-category pseudofunctors，未完成 infinity-enhancement）；Theorems 3.3.2, 4.1.4；§4.3.1；见 batch 2 |
| PU-6.13 | Smooth purity equivalence | Hoyois; Deglise-Jin-Khan | located | smooth separated；Hoyois Theorem 6.18(2)；DJK §4.3.1；见 batch 2 |
| AD-7.3 | Smooth ambidexterity | Hoyois | located | 由 Theorem 6.18(2) 的 smooth purity 经伴随唯一性书内推出；见第七章证明 |
| AD-7.10 | Smooth proper Atiyah duality | Hoyois | located | Corollary 6.13；对偶为 `f_\sharp\Sigma^{-T_f}1\simeq f_*1`；见 batch 2 |
| BC-8.3 | Exceptional/proper/smooth base change | Hoyois | located | Theorem 6.18(3), Corollary 6.10；smooth ordinary 见 Proposition 4.2 及 Proposition 6.4 后的稳定化段落；见 batch 2 |
| PF-8.5 | Exceptional projection formula | Hoyois | located | Theorem 6.18(7)；ordinary proper 见 Corollary 6.11；见 batch 2 |
| HZ-9.1 | `H\mathbb Z` construction/representation | Spitzweck | located | Theorem 7.18, Corollary 7.19；mixed-characteristic Dedekind base；base change Theorem 8.25；见 batch 3 |
| HZ-9.9 | higher Chow comparison | Mazza-Voevodsky-Weibel | located | Theorem 19.1, Corollary 19.2；perfect `k`, smooth separated `X`；见 batch 3 |
| HZ-9.10 | diagonal Milnor `K` comparison | Mazza-Voevodsky-Weibel | located | Theorem 5.1；任意域；见 batch 3 |
| DM-10.8 | `DM` 与 `H\mathbb Z`-modules | Röndigs-Ostvær; Elmanto-Kolderup | located | RØ Theorem 1.1（char 0 triangulated）；EK Theorem 5.2/Corollary 5.3（`1/e` stable infinity）；见 batch 3 |
| KG-11.1 | strict commutative `KGL` model | Röndigs-Spitzweck-Ostvær | located | Lemma 2.5, Theorems 3.6, 4.1；Noetherian finite-dimensional；见 batch 3 |
| KG-11.6 | `KGL` 表示 `KH` | Cisinski | located | Theorem 2.20；Noetherian finite-dimensional；见 batch 3 |
| KG-11.7 | regular-case `K\simeq KH` | Weibel | located | K-book IV Corollary 12.3.1, Lemma 12.8(3)；见 batch 3 |
| KG-11.12 | `KH` satisfies cdh descent | Cisinski | located | Theorem 3.9；Noetherian finite-dimensional；见 batch 3 |
| MG-12.4 | `MGL` orientation universality | Panin-Pimenov-Röndigs | located | Theorem 2.3.1；field；homotopy-category monoid-map sets；见 batch 3 |
| MG-12.11 | Hopkins-Morel comparison | Hoyois | located | Theorem 7.12；essentially smooth over field；invert exponent `c`；见 batch 3 |
| SL-13.8 | zero slice and slice modules | Voevodsky | located | Theorem 6.6 and Introduction pp.106-107；char 0 field；见 batch 3 |
| TR-14.3 | finite correspondences 复合 | Mazza-Voevodsky-Weibel | located | Lecture 1, Lemmas 1.4, 1.7, Definition 1.5；见 batch 3 |
| TR-14.10 | Nisnevich sheafification preserves transfers | Mazza-Voevodsky-Weibel | located | Theorem 13.1；定理本身无需 perfectness；见 batch 3 |
| FR-15.x | framed recognition | Elmanto-Hoyois-Khan-Sosnilo-Yakerson | located | Theorems 1.2.3, 3.5.14；perfect field；见 batches 1, 3 |
| FC-16.2 | fundamental classes and Gysin maps | Deglise-Jin-Khan | located | Definition 3.2.5; Proposition 2.5.4; Theorems 3.3.2, 4.1.4, 4.2.1；见 batches 1, 3 |
| FC-16.13 | excess intersection formula | Deglise-Jin-Khan | located | §3.3.3；Propositions 3.3.4, 4.2.2；`f,g` 均 smoothable lci s-morphisms；系数乘法与 proper push-pull 条件分别保留；见 batches 1, 3 |
| NM-17.x | norm functors and normed examples | Bachmann-Hoyois | located | Theorem 3.3/localization、Corollary 3.11（finite-locally-free unstable）；Proposition 3.13（finite etale 且 Weil restriction 存在）；Proposition 4.5; Definition 7.1; Theorems 14.5, 15.22, 16.19；见 batches 1, 3 |
| MW-18.2 | `End(1_k)\cong GW(k)` | Morel | located | Corollary 6.43 + Lemma 3.10；perfect field；见 batch 3 |
| MW-18.5 | stable-range sphere maps and `K_*^{MW}` | Morel | located | Corollary 6.43；perfect field；见 batch 3 |
| EQ-19.3 | equivariant six operations | Hoyois | source-verified | 补 linearly reductive 假设 |
| ST-20.x | stacks 上六操作 | Khan-Ravi | source-verified | 补 scalloped stacks 定义和主定理 |
| RE-22.4 | Betti realization | Ayoub 等 | pending | 补 six-operation compatibility locator |
| UF-23.2 | universal six-functor formalism | Drew-Gallauer | located | Theorem 7.14; Proposition 7.13; Theorem 7.3; see P0 batch 1 |

## 第 9--18 章 P1 boundary queue

| 标签范围 | 内容 | 状态 | 不进入 P0 的理由 |
| --- | --- | --- | --- |
| HZ-9.12 | etale cycle map / Bloch--Kato truncation | P1 | arithmetic comparison；不参与 `HZ` 表示性、Chow 或 Milnor `K` 主线 |
| KG-11.17 | rational Chern character splitting | P1 | 需另定基、乘积/直和及有理系数版本 |
| SL-13.18 | motivic Adams calculations | P1 | 需素数、完备化与收敛假设 |
| FR-15.6, FR-15.8 | framed Hilbert models；framed/fundamental full compatibility | P1 | 几何模型和比较增强；recognition P0 不依赖它们 |
| NM-17.7, NM-17.9 | classical/Galois and Tambara/framed refinements | P1 | 逐理论比较，不由 Definition 7.1 形式推出 |
| MW-18.7 | Chow-Witt motives、local degree、enumerative/Gauss-Bonnet refinements | P1 | 多模型、多 twists 与额外 orientation/transversality 假设；不参与 `End(1)=GW` |

## R locator queue

| 账本标签 | 结果 | 资料源 | 当前状态 | 使用边界 |
| --- | --- | --- | --- | --- |
| PF-21.5 | perfect schemes motivic homotopy | Dahlhausen-Hekking-Wolters 2025 | source-verified | 研究边界，不作基础输入 |
| RB-24.1 | pullback formalism 几何判据 | Magen 2025 | source-verified | 研究边界，不作基础输入 |
| RB-24.2 | complex analytic stacks localization | Magen 2026 | source-verified | 研究边界，不作基础输入 |

## 下一轮 locator 工作流

1. 第九至第十八章教学主线的 P0 locators 已由 batch 3 闭合；后续不得把
   其中明确降级的 Adams、Chern character、framed Hilbert、Tambara 或
   quadratic-enumerative P1 边界反向当成主线输入。
2. 下一批 P0 只处理 extensions/realization：equivariant、stacky 与 Betti
   realization，并逐项保留其定义域和六操作相容假设。
3. Cisinski--Deglise 的一般基 `DM`/absolute-purity package 若进入正文证明，
   应作为独立模型定位；它不改变 batch 3 的域上 module comparison。
