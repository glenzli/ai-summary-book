# Reference locator ledger

核查日期：2026-07-08

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
| MH-3.8 | `\mathbf{SH}(S)` 构造与 symmetric monoidal stable structure | Morel-Voevodsky; Jardine; Ayoub; Cisinski-Deglise | pending | 区分模型范畴构造和 infinity-categorical 版本 |
| MO-5.2 | Motivic 六操作 | Ayoub; Cisinski-Deglise; Drew-Gallauer | source-verified | 补 theorem/section locator |
| MO-5.14 | Motivic localization recollement | Morel-Voevodsky; Ayoub; Cisinski-Deglise | pending | 补 closed-open localization 定理号 |
| PU-6.7 | Homotopy purity | Morel-Voevodsky | pending | 补 statement locator |
| PU-6.11 | Smooth purity | Ayoub; Hoyois; Cisinski-Deglise | pending | 固定 `T_f` 符号和 twist convention |
| AD-7.3 | Smooth ambidexterity | Ayoub; Hoyois | pending | 补 ambidexterity theorem locator |
| BC-8.3 | Base change | Ayoub; Cisinski-Deglise; Drew-Gallauer | pending | 分 ordinary/proper/extraordinary |
| PF-8.5 | Projection formula | Ayoub; Cisinski-Deglise | pending | 分 `f_!` 和 `f_*` 版本 |
| HZ-9.1 | `H\mathbb Z` | Voevodsky; Spitzweck; Cisinski-Deglise | source-verified | 补 spectrum construction locator |
| DM-10.8 | `DM` 与 `H\mathbb Z`-modules | Röndigs-Ostvær; Cisinski-Deglise; Elmanto-Kolderup | source-verified | 明确基域/系数假设 |
| KG-11.1 | `KGL` 表示 `KH` | Röndigs-Spitzweck-Ostvær; Cisinski-Deglise | source-verified | 补 strict ring model 和表示性定理 |
| MG-12.4 | `MGL` universality | Panin-Pimenov-Röndigs | source-verified | 补 orientation bijection 定理号 |
| MG-12.7 | Hopkins-Morel 型比较 | Hoyois; Spitzweck | source-verified | 补 invert characteristic exponent 条件 |
| SL-13.8 | Zero slice | Voevodsky | source-verified | 补 char 0/推广版本边界 |
| TR-14.3 | finite correspondences 复合 | Suslin-Voevodsky; MVW | pending | 补 construction locator |
| FR-15.x | framed recognition | Elmanto-Hoyois-Khan-Sosnilo-Yakerson | located | Theorem 1.2.3; Theorem 3.5.14; see P0 batch 1 |
| FC-16.2 | fundamental classes | Deglise-Jin-Khan | located | Definition 3.2.5; Theorems 3.3.2, 4.1.4, 4.2.1; see P0 batch 1 |
| NM-17.x | norm functors | Bachmann-Hoyois | located | Proposition 4.5; Definition 7.1; see P0 batch 1 |
| MW-18.2 | `End(1_k)\simeq GW(k)` | Morel | pending | 补 Morel 定理号 |
| EQ-19.3 | equivariant six operations | Hoyois | source-verified | 补 linearly reductive 假设 |
| ST-20.x | stacks 上六操作 | Khan-Ravi | source-verified | 补 scalloped stacks 定义和主定理 |
| RE-22.4 | Betti realization | Ayoub 等 | pending | 补 six-operation compatibility locator |
| UF-23.2 | universal six-functor formalism | Drew-Gallauer | located | Theorem 7.14; Proposition 7.13; Theorem 7.3; see P0 batch 1 |

## R locator queue

| 账本标签 | 结果 | 资料源 | 当前状态 | 使用边界 |
| --- | --- | --- | --- | --- |
| PF-21.5 | perfect schemes motivic homotopy | Dahlhausen-Hekking-Wolters 2025 | source-verified | 研究边界，不作基础输入 |
| RB-24.1 | pullback formalism 几何判据 | Magen 2025 | source-verified | 研究边界，不作基础输入 |
| RB-24.2 | complex analytic stacks localization | Magen 2026 | source-verified | 研究边界，不作基础输入 |

## 下一轮 locator 工作流

1. 先定位 Morel-Voevodsky、Ayoub、Cisinski-Deglise、Drew-Gallauer 四个基础源。
2. 再定位 `H\mathbb Z`、`DM`、`KGL`、`MGL`、slice filtration 五个谱/动机源。
3. 再定位 transfers/framed/fundamental/norm/Milnor-Witt。
4. 最后定位 equivariant/stacky/log/perfect/realization/universal formalisms 的扩展结果。
