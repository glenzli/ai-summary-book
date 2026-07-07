# 定理账本：内部证明、外部输入与研究边界

本文件把当前《Chromatic Homotopy Theory》草稿中的结论分成三类：

1. **内部证明**：正文已经给出稳定范畴或代数层面的证明。
2. **外部输入**：正文可引用，但必须在资料源中定位，后续需要 theorem/section locator。
3. **边界说明**：只用于范围、失败模式或研究方向，不得作为证明步骤。

## 1. 基础稳定范畴

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第一章 | finite spectra compact；acyclic 类对 cofiber 封闭；localization 泛性质；smashing localization 保持 colimit | Bousfield localization 存在性的一般定理 | 不同文献 Bousfield 偏序方向不同 |
| 第五章 | $M_nX$ fiber 序列；fracture square 退化时的乘积结论 | $\langle E(n)\rangle=\langle K(0)\vee\cdots\vee K(n)\rangle$；chromatic fracture square；chromatic convergence | $M_nX\ne L_{K(n)}X$；一般谱不自动 convergence |

## 2. 形式群和复定向

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 序章/第二章 | 复定向给形式群律的公理检查；$K(n)_*$ 是 graded field | projective bundle formula；Quillen theorem；$BP$ splitting；Landweber exactness | 系数环商不自动给结构化 ring quotient |
| 附录 A | 加法/乘法形式群公理；$p$-series 计算；乘法形式群高度一 | 坐标改变下高度不变的完整一般证明 | Hazewinkel 坐标不可与任意坐标混用 |

## 3. Morava theories

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第三章 | 零谱被所有 $K(n)$ 检测；Bousfield 等价推出相同 acyclic 类 | finite spectra 被 Morava K theories 检测；Lubin-Tate deformation；Goerss-Hopkins-Miller；$E_n/\mathfrak m$ 与 $K(n)$ Bousfield 等价 | $E_n$ 的 $\mathbb E_\infty$ 结构不能由系数环推出 |
| 第六章 | $K(n)$-local 对等价封闭；$S_{K(n)}\simeq E_n^{hG_n}$ 由 descent 取 $X=S$ 推出 | Devinatz-Hopkins homotopy fixed point theorem；Morava descent spectral sequence | profinite group action 不能离散化 |

## 4. 有限谱和周期性

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第四章 | $\mathcal C_n$ 是 thick 子范畴 | nilpotence theorem；periodicity theorem；thick subcategory theorem；finite detection | type 是首次非消失高度，不是唯一被检测高度 |

## 5. 前沿和近期研究

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第七章 | redshift 假设清单；telescope 旧表述改写规则 | Hahn-Wilson redshift；Chromatic Nullstellensatz；cyclotomic redshift/descent | BHLŠ telescope 反例改变默认口径；2026 syntomic/K-theory 预印本暂作前沿 |
| 第八章 | tmf 与 $E_2$ 类型区别说明 | 椭圆曲线形式群高度；tmf sheaf 构造；Goerss-Hopkins-Miller/Lurie derived moduli | tmf 不是单个 Morava E-theory |
| 附录 B | spectral sequence 使用检查表 | Adams-Novikov、Morava descent、Tate spectral sequences 的收敛定理 | hidden extensions 不可忽略 |
| 第九章 | 0-semiadditivity；有限集合 cardinality 标量 | Hopkins-Lurie/CSY higher semiadditivity；HKR character theory；Ben-Moshe transchromatic/semiadditive results | redshift 不自动推出 semiadditivity |
| 第十章 | splitting square 退化判据；Picard/exotic 定义 | rational $K(n)$-local sphere；Gross-Hopkins duality；Picard descent | dualizing object convention 不可省略 |
| 第十一章 | genuine/geometric fixed point 基础检查 | equivariant Balmer/formal group laws；Behrens-Carlisle equivariant periodicity；motivic $MGL$ 基础；synthetic reconstruction | naive/genuine、topological/motivic 不可混用 |
| 第十二章 | 计算检查表；Tate construction 定义 | Adams-Novikov SS；chromatic SS；Morava change-of-rings；HFPSS/Tate SS | $E_\infty$ 页不等于 abutment |
| 附录 C | invariant ideal quotient 的证明草图；comodule Ext 口径 | $BP_*BP$ 结构公式；change-of-rings | Hopf algebroid Ext 不是普通环 Ext |
| 附录 E | failure modes 清单 | 无新增定理 | 作为审稿约束 |
| 附录 F | 高度 0、Moore spectrum 有理同调样例 | 低高度 $K(1)$、tmf 局部模型 | 低高度直觉不能替代一般定理 |
| 附录 G | frontier 升级协议 | 无新增定理 | 约束预印本使用 |
| 附录 H | acyclics localizing；locals 对 limits 封闭；smashing tensor ideal；$K(n)$-module field-like 证明草图 | $K(n)$-module category classification | field-like 不适用于普通谱 |
| 附录 I | telescope 的 filtered colimit 计算；低高度消失 | periodicity theorem；telescope Bousfield class 选择无关 | $T(n)\ne K(n)$ 默认 |
| 附录 J | Morava module semilinear action；descent spectral sequence 目标推导 | Devinatz-Hopkins；Morava change-of-rings | 连续性不可省略 |
| 附录 K | elliptic/tmf/version convention | tmf sheaf；supersingular local model；power operations | weak elliptic datum 不给 $\mathbb E_\infty$ sheaf |
| 附录 L | GH/Picard convention template | Gross-Hopkins duality；Picard group computations | 未定位公式不得简化 |
| 附录 M | hidden extension 定义；计算协议 | ANSS 低阶表来源 | 模板不是计算结果 |
| 附录 N | 高度 0、height-1 fracture、Moore spectrum type 检查 | $K(1)$ Adams operation model；$K(2)$ tmf local model | 低高度公式需 convention |
| 出版闭包矩阵 | 内容范围和内部完整性判定 | 无新增定理 | 指导后续收口 |

## 6. 当前 P0 locator 缺口

1. DHS nilpotence theorem 的精确版本和 theorem number。
2. Hopkins-Smith thick subcategory theorem 和 periodicity theorem 的精确定位。
3. Hopkins-Ravenel chromatic convergence theorem 的精确定位。
4. Hovey-Strickland Morava K-theory localization 的 theorem locator。
5. Goerss-Hopkins-Miller theorem 的可引用定位包。
6. Devinatz-Hopkins homotopy fixed point theorem 和 spectral sequence locator。
7. BHLŠ telescope counterexample 的 precise theorem statement。
8. Hahn-Wilson redshift theorem 和 multiplication theorem locator。
9. Burklund-Schlank-Yuan Chromatic Nullstellensatz 的 redshift corollary locator。
10. 2026 Angelini-Knoll BP<n> syntomic/K-theory 预印本的版本和 theorem locator。
11. Higher semiadditivity 的 Hopkins-Lurie/CSY 定理定位。
12. Hopkins-Kuhn-Ravenel character theory 和 transchromatic character locator。
13. Gross-Hopkins duality 的 convention table。
14. Picard descent spectral sequence 与 exotic Picard group locator。
15. Equivariant chromatic periodicity 和 motivic synthetic reconstruction locator。

## 7. 当前状态

当前草稿达到“正式教材扩展初稿”状态，未达到数学收口。主体章节已经覆盖 stable spectra、formal groups、Morava theory、finite spectra、chromatic tower、descent、redshift、tmf、semiadditivity、splitting/duality/Picard、equivariant/motivic 和计算工具；仍需补齐 P0/P1 theorem locators、详细计算样例和 convention tables。
