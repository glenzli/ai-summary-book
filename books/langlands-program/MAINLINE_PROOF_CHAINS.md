# 主线最短证明链

本文档把 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 的阅读路径收口为可审查的最短证明链。状态标记沿用 [THEOREM_INDEX.md](THEOREM_INDEX.md)：

- `P`：本书给出证明。
- `S`：本书给出外部输入的证明路线。
- `E`：外部输入定理。
- `C`：猜想或纲领性预期。

本文件的作用是防止主线继续横向发散：若某材料不能进入以下四条链之一，或者不能修正其中的缺口，则默认应作为外部输入或另卷处理。

四条链共同通过以下归一化闸门：局部 reciprocity 把 uniformizer 送到几何 Frobenius；Fourier 变换使用 $\psi_v$-自对偶加法测度，而 $d^\times x_v$ 另按 $\operatorname{vol}(\mathcal O_v^\times)=1$ 归一化；经典模形式根与 unitary Satake 根相差 $p^{(k-1)/2}$；$v\nmid\ell$ 的局部-整体相容统一采用 Galois dual、Frobenius semisimplification 和显式 $|\cdot|^{w/2}$ twist。任何步骤若改用算术 Frobenius、algebraic automorphic normalization 或另一 Hodge-Tate 符号，必须先在 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 中换算。

## 1. `GL(1)` 到类域论

目标：解释 `GL(1)` Langlands 的参数对应由类域论给出，而 L 函数解析接口由 Tate thesis 给出。

| 步骤 | 状态 | 结果 | 位置 | 收口说明 |
|---|---:|---|---|---|
| 1 | `P/S/E` | 整体域、局部域、adeles、ideles 和 idele class group 建立 | 第 1 章 | 紧性和 Pontryagin 自对偶作为外部输入 |
| 2 | `E/P` | Fourier 变换、Poisson summation 和 Tate zeta integral 接口 | 第 2 章、附录 F | 固定 $\mathfrak c(\psi_v)$、自对偶 $dx_v$ 与另行归一化的 $d^\times x_v$；完整 Tate thesis 保持外部输入 |
| 3 | `P` | Hecke 特征分解为局部分量，整体 zeta integral 分解为 Euler product | 2.5、2.11 | 这是 `GL(1)` L 函数进入 Langlands 语言的本书证明部分 |
| 4 | `E` | 局部和全局类域论给出 reciprocity map | 3.2、3.11、附录 V | $F^\times\simeq W_F^{ab}$；到 $G_F^{ab}$ 和全局 $G_K^{ab}$ 使用 profinite completion/有限商，不重证 class formation |
| 5 | `P` | 非分歧局部特征对应一维 Weil 参数，且 L 因子相容 | 3.5、3.6、5.14 | 使用几何 Frobenius convention |
| 6 | `P/E` | 有限阶 Hecke 特征对应有限像一维 Galois 表示 | 3.14、3.15、3.16 | 有限阶 Galois 形式为本书闭环；一般 quasi-character 放在 Weil 侧 |
| 7 | `E` | 一般 `GL(1)` Langlands 的 Weil 形式 | 3.17 | 作为类域论的 Weil 群版本引用 |

闭环结论：本书证明 `GL(1)` Langlands 如何由 Tate thesis 和类域论翻译出来；不把 Tate thesis 和 class formation 改写为完整证明卷。

## 2. `GL(2)`、椭圆曲线和费马应用

目标：证明“接受 Frey 曲线性质、半稳定模性和 Ribet 降层，则费马大定理成立”。

| 步骤 | 状态 | 结果 | 位置 | 收口说明 |
|---|---:|---|---|---|
| 1 | `P` | 费马大定理归约到奇素数指数和指数 `4` | 90.2 | 应用章内部证明 |
| 2 | `E` | Frey 曲线由 primitive 反例构造，满足半稳定性和导子性质 | 90.5、附录 AD | 局部导子只保留 Frey 应用所需接口 |
| 3 | `E` | 半稳定椭圆曲线模性定理 | 90.7、9.23 | Taylor-Wiles 机器作为外部输入 |
| 4 | `E` | Ribet 降层把 Frey 曲线 residual representation 降到权 `2`、级 `2` newform | 90.8、10.8、10.10、10.13 | 第十章分别列绝对不可约、$q\parallel N$、局部非分歧/有限平坦和残余导子假设；深层 Ihara/Jacobian 输入不在正文伪证 |
| 5 | `P/E` | $S_2(\Gamma_0(2))=0$，故级 `2` 权 `2` newform 不存在 | 90.9、附录 D、附录 W | genus 计算在附录中给出，完整模曲线理论为外部输入 |
| 6 | `P` | 上述矛盾推出费马大定理 | 90.10 | 这是应用章的逻辑闭环 |

闭环结论：本书的费马章不是 Wiles-Taylor-Wiles 证明，而是 Langlands 主线中 `GL(2)/\mathbb Q` 模性输入的应用链。

## 3. 一般算术 Langlands

目标：给出从局部参数、L 群、自守表示到函子性、trace formula 和 Arthur 参数的对象链。

| 步骤 | 状态 | 结果 | 位置 | 收口说明 |
|---|---:|---|---|---|
| 1 | `P/E` | 局部紧群、Hecke 代数、光滑表示和 Satake 参数语言 | 第 4、5 章，附录 P/Z/AA | $e_J=\operatorname{vol}(J)^{-1}1_J$；split Satake 类位于 $\widehat G$，仅 unramified 时位于 $\widehat G\rtimes\operatorname{Fr}$ |
| 2 | `P/E` | 还原群、根资料、对偶群和 L 群 | 第 11 章，附录 G | 代数群结构定理作外部输入；root datum 计算在本书展开 |
| 3 | `C/E/P` | 局部 Langlands packet 形式；`GL(n)` 为定理，一般还原群为猜想或已知特殊情形 | 第 5、12、14 章，附录 N/Q/AE | 5.8.1 证明 $W_F\times\mathrm{SL}_2$ 到 WD 的方向；逆向、packet、enhancement 和 inner form 分别标界 |
| 4 | `P/E/C` | 全局 unitary 自守表示、非分歧局部因子、部分 L 函数和解析性质 | 第 13 章，附录 I/L/M | unitary `GL(n)` Euler product 初始域为 $\Re(s)>1$；函数方程只在 $(\pi,r^\vee)$ 与 $(\pi^\vee,r)$ 两种等价写法中择一，不双重取对偶 |
| 5 | `E/P` | `GL(n)` 的局部 LLC、强重数一、Rankin-Selberg、函数域 Lafforgue 定理和数域已知 Galois 表示接口 | 第 14 章 | 数域完整全局 Langlands 仍是纲领 |
| 6 | `C/E/P` | 函子性由 L 同态诱导局部参数推前和全局转移 | 第 15 章 | 强/弱转移分开；已知 lifts 与一般猜想分开 |
| 7 | `E/P` | trace formula、endoscopy 和 stable transfer 解释函子性证明机制 | 第 16 章，附录 R/N | 固定测试函数、中心/商测度与 truncation data；只在已稳定化的群和版本中调用，不重证 Arthur trace formula 和 fundamental lemma |
| 8 | `E/P` | Arthur 参数解释 classical groups 离散谱中的非 tempered 现象 | 第 17 章，附录 X | Arthur-Mok 分类作为外部输入 |

闭环结论：一般算术 Langlands 主线已成型，但收口状态是“对象链闭合、证明层依赖外部输入”。后续只补对象字典、归一化回指和状态表，不开新的大方向。

## 4. 几何 Langlands 和函数域桥梁

目标：解释几何 Satake、Hecke eigensheaves、谱侧范畴和函数域 Langlands 的桥梁。

| 步骤 | 状态 | 结果 | 位置 | 收口说明 |
|---|---:|---|---|---|
| 1 | `P/E` | 曲线、$\operatorname{Bun}_G$、Hecke stack 和 Hecke 修改 | 第 18 章 | 代数栈深层技术作为外部输入 |
| 2 | `E/P` | 几何 Satake 把 affine Grassmannian 上的 perverse sheaves 识别为 $\operatorname{Rep}(\widehat G)$ | 第 19 章、附录 Y | 只证明形式后果，完整定理外部输入 |
| 3 | `P/C/E` | Hecke eigensheaf 的定义和几何 Langlands 朴素形式 | 第 20 章 | `GL(1)` 几何类域论为外部输入接口 |
| 4 | `E/C` | 范畴化几何 Langlands 的谱侧和自动侧 | 第 21 章、附录 O/AB | 特征零 unramified 版本登记为五篇 proof-series 的外部 preprint theorem；正特征只登记某些参数连通分支上的部分结果，其余版本保持边界 |
| 5 | `E/P` | sheaf-function dictionary 把 Hecke eigensheaf 的 trace 变成函数域 Hecke eigenfunction | 20.11、22.3、22.5 | 必须声明 Frobenius trace convention |
| 6 | `E` | Drinfeld-Lafforgue 和 shtuka cohomology 给出函数域 `GL(n)` 全局 Langlands | 22.6、22.7、22.11、附录 S | 函数域定理作为桥梁，不重证 shtuka 技术 |
| 7 | `E` | Fargues-Fontaine 和 Fargues-Scholze 给出局部几何化接口 | 附录 AC | 只调用 spectral action 与 semisimple LLC map；不由此恢复 WD monodromy、enhanced packets 或 $p$-adic Banach correspondence |

闭环结论：几何主线的收口目标是对象字典和 sheaf-function 桥梁闭合；完整 D-module、derived stack、shtuka 和 diamond 理论后置或另卷。

## 收口缺口表

| 缺口 | 阻塞程度 | 处理方式 |
|---|---|---|
| 各章到 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 的回指 | 高 | 高风险主章和高风险附录已加入回指；后续新增段落必须维护 |
| 定理索引与外部输入索引一致性 | 高 | 已完成 [INDEX_CONSISTENCY_AUDIT.md](INDEX_CONSISTENCY_AUDIT.md)；后续做来源粒度精校 |
| 习题覆盖表 | 中 | 见 [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md) |
| `GL(1)` 中 class formation 的完整证明 | 低 | 后置或另卷 |
| Taylor-Wiles、Ribet、Arthur trace formula、几何范畴技术完整证明 | 低 | 后置或另卷 |

基本收口要求不是消灭所有外部输入，而是让每个外部输入的位置、作用和状态可检查。
