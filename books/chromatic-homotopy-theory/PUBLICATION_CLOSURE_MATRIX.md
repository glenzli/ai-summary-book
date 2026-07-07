# 出版闭包矩阵

本文件判断《Chromatic Homotopy Theory》当前距离“正式教材内容范围、内部完整、细节完整”的差距。

## 1. 内容范围矩阵

| 模块 | 文件 | 覆盖状态 | 缺口 |
| --- | --- | --- | --- |
| 稳定谱和 Bousfield 局部化 | 01, H, E | 基本覆盖 | Bousfield localization 存在性 locator |
| 复定向和形式群 | 02, A | 正文细节已扩写 | Quillen/Landweber locator |
| BP、Morava K/E | 02, 03, J | 正文细节已扩写 | Hovey-Strickland、GHM locator |
| 有限谱和周期性 | 04, I | 正文细节已扩写 | Hopkins-Smith precise locator |
| Chromatic tower/fracture | 05 | 正文细节已扩写 | fracture square precise hypotheses |
| $K(n)$-local descent | 06, J | 正文细节已扩写 | Devinatz-Hopkins locator |
| Telescope/redshift | 07, G | 正文细节已扩写 | BHLŠ/HW/BSY theorem locator |
| Elliptic/tmf | 08, K | 正文细节已扩写 | tmf construction 和 level structure locator |
| Semiadditivity/character | 09 | 正文细节已扩写 | formal semiadditive height 和 HKR details |
| Splitting/duality/Picard | 10, L | 正文细节已扩写 | convention table 需 locator 填实 |
| Equivariant/motivic | 11 | 正文细节已扩写但仍为接口章 | 若要闭合需独立扩写多章 |
| 计算工具 | 12, B, C, M, N, Q | 正文细节已扩写 | 完整 ANSS 表格仍属扩展 |
| 前沿准入 | G, frontier audit, P1 locator | 覆盖 | 页码级 locator 属出版校对 |

## 2. 内部完整性矩阵

| 链条 | 当前状态 | 判定 |
| --- | --- | --- |
| 定义链 | 从谱到高度、type、tower、descent 已连通 | 基本闭合 |
| 证明链 | 形式范畴命题有证明；大型定理标外部输入 | 合格但未收口 |
| 符号链 | 主要符号已登记；$I_n$ 冲突已用 convention 处理 | 基本闭合 |
| 例子链 | 有低高度入口、fracture worked checks、chapter-level worked checks、综合习题提示和低 stem 表 | 内容闭合 |
| 前沿链 | 有 audit 和准入协议 | 基本闭合 |
| locator 链 | 所有外部输入有 bibliographic/frontier locator；无页码级 locator | 内容闭合，出版未闭合 |

## 3. 细节完整性矩阵

| 细节类型 | 当前状态 | 下一步 |
| --- | --- | --- |
| 低阶证明 | 稳定局部化、形式群、telescope 基本证明已补 | 压缩到章节正文 |
| 谱序列 | 有约定和风险表 | 补具体 $E_2$ 页样例 |
| Hopf algebroid | 有定义和 comodule 口径 | 补 $BP_*BP$ 结构公式 |
| Morava descent | 有 Morava module 和连续性 | 补 DH 定理定位 |
| tmf | 有层级约定 | 补 supersingular local decomposition 例子 |
| Picard/GH | 有 convention template | 补低高度计算 |
| Equivariant/motivic | 有模型警告 | 若要闭合需独立扩写多章 |

## 4. 当前结论

截至 2026-07-08，本书已达到“教材内容基本收口稿”。它尚未达到 camera-ready 出版态，剩余项主要是：

1. P0/P1 外部定理没有页码级 theorem locator；
2. 完整 ANSS 计算表仍属扩展内容；
3. tmf、Gross-Hopkins duality、Picard group 的低高度案例可继续加厚；
4. equivariant/motivic 方向按接口章收口，若要同等权重需独立成书级扩写。

## 5. 收口路线

下一轮应按以下顺序推进：

1. `P0_REFERENCE_LOCATORS_BATCH_1.md`：Quillen、BP、Landweber、DHS、Hopkins-Smith。
2. `P0_REFERENCE_LOCATORS_BATCH_2.md`：Hovey-Strickland、chromatic convergence、fracture、GHM、Devinatz-Hopkins。
3. `P1_REFERENCE_LOCATORS_FRONTIER.md`：BHLŠ、Hahn-Wilson、BSY、BCSY、BSSW、CSY。
4. 低阶计算批：填 Adams-Novikov 和 $K(1)$/$K(2)$ 样例。
5. 出版校对批：统一 theorem numbering、交叉引用和 bibliography。
