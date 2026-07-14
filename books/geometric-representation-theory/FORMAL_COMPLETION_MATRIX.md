# 正式教材完备矩阵

本文件给出《Geometric Representation Theory》从“主体教材化收口”走向“出版终稿”的判定矩阵。它不新增定理，而是规定每一部分达到正式教材状态需要满足的内部闭合条件。

## 0. 状态词

- **主体收口**：已有定义链、主要构造、内部命题证明、外部输入标记、例子和练习。
- **源级引用闭合**：外部输入能追溯到 `SOURCES.md`、附录 D 和 locator 批次；但尚非页码级出版 locator。
- **内部闭合**：所有本书自行证明的命题已经给出类型正确的证明，且依赖只指向更早章节或明确附录。
- **外部输入闭合**：大型定理已经有精确来源、版本、定理编号或章节定位，并且假设已翻译到本书符号。
- **正式教材态**：章节包含定义、动机、例子、非例子、证明、习题、交叉引用和外部输入定位，且不存在未声明模型切换。

## 1. 主体章节矩阵

| 范围 | 当前状态 | 正式教材还需 |
| --- | --- | --- |
| 第 0 章 | 主体收口 | 附录 J 交叉引用稳定化 |
| 第 1 章 | 主体收口 | Borel/Bruhat 页码级 locator |
| 第 2 章 | 主体收口 | PBW、HC、BGG 页码级 locator |
| 第 3 章 | 主体收口 | Betti、etale、mixed Hodge、D-module 模型分拆 |
| 第 4 章 | 主体收口 | KL-IC 页码级 locator |
| 第 5 章 | 主体收口 | Springer action 页码级 locator |
| 第 6 章 | 主体收口 | generalized Springer 和 character sheaf 页码级 locator |
| 第 7 章 | 主体收口 | Bernstein inequality 和 RH 页码级 locator |
| 第 8 章 | 主体收口 | BB theorem 页码级 locator；$\rho$ shift 出版锁定 |
| 第 9 章 | 主体收口 | BWB 和 wall crossing 页码级 locator |
| 第 10 章 | 主体收口 | Joseph/Borho-Brylinski 页码级 locator |
| 第 11 章 | 主体收口 | Soergel/EW 页码级 locator |
| 第 12 章 | 主体收口 | ind-projectivity 和 orbit 页码级 locator |
| 第 13 章 | 主体收口 | MV theorem、Tannakian、commutativity constraints 页码级 locator |
| 第 14 章 | 主体收口 | affine KL locator 和 mixed/monodromic convention 出版锁定 |
| 第 15 章 | 主体收口 | critical level、opers、FLE 边界 locator |
| 第 16 章 | 主体收口 | 保持边界；不进入基础定理链 |
| 第 17 章 | 主体收口 | Nakajima correspondences 页码级 locator |
| 第 18 章 | 主体收口 | KLR categorification 页码级 locator |
| 第 19 章 | 主体收口 | BLPW/Losev 页码级 locator；具体例子边界 |
| 第 20 章 | 主体收口 | BFN construction、finite type、quantization 页码级 locator |
| 第 21 章 | 主体收口 | KS/Davison-Meinhardt locator；critical/ordinary convention |
| 第 22 章 | 主体收口 | Lusztig/Kashiwara locator；canonical/dual canonical convention |
| 第 23 章 | 主体收口 | 保持五类数学障碍边界；时效性结果与核验流程只交叉到附录 J 和 locator 记录 |

## 2. 内部闭合最小标准

一个章节只有在满足下列条件时才可标为内部闭合：

1. 每个自定义对象都有所在范畴、底域、系数和群作用。
2. 每个自称“命题”的陈述有证明；证明不得调用未声明的大型定理。
3. 每个“外部输入定理”在附录 D 或 locator 批次中有入口。
4. 每个卷积都有 correspondence、properness 或 compact-support 说明。
5. 每个 sheaf/D-module 结果说明模型：Betti、etale、mixed Hodge、regular holonomic 或 ind-coherent。
6. 每个高级边界结果不得用于基础章节证明。

## 3. 当前判定

当前书稿达到“主体教材化收口”和“源级引用闭合”。主体章节已经具备正式教材的基本部件；外部输入闭合停留在定理包级或资料源级 locator，下一阶段的关键不是扩展更多方向，而是：

- P0 theorem locator；
- 模型假设表；
- 内部证明核并入正文；
- 低阶计算；
- 交叉引用稳定化。
