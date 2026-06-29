# 最终概念审定

本文档记录《Langlands 纲领》审定前闭合版的概念层审定结论。它不新增数学内容，只规定全书核心概念的最终口径，防止出版前审定时把口号、类比、猜想、外部输入和已证命题混用。

## 总体判定

本书的核心概念链已经闭合：

1. 数论侧对象：Galois 群、Weil 群、Weil-Deligne 参数、$\ell$-adic Galois 表示和 conjectural Langlands group。
2. 自守侧对象：局部光滑表示、adelic 自守形式、自守表示、Hecke 本征数据和自守 L 函数。
3. 结构桥梁：局部 Langlands 参数、L 群、Satake 参数、局部因子、Euler 乘积、epsilon 因子和局部-整体相容。
4. 机制层：函子性、converse theorem、trace formula、endoscopy、Arthur 参数和稳定转移。
5. 几何层：$\operatorname{Bun}_G$、Hecke 修改、几何 Satake、Hecke eigensheaves、谱侧范畴和 sheaf-function dictionary。
6. 应用层：费马大定理只作为 `GL(2)/\mathbb Q` 模性、Frey 曲线和 Ribet 降层的应用链，不作为完整 Langlands 纲领的直接推论。

当前审定结论：概念层达到审定前闭合。后续只允许修正局部概念误差、术语不统一、状态标记错误或来源缺口，不允许新增第五条同级主线。

## 概念分层规则

### 参数不是表示本身

允许说法：

- 局部 Langlands 把表示组织为由 L 参数控制的 packets。
- 对 `GL(n)`，局部 packet 是单点，因此可写成参数和不可约可容许表示的双射。
- 对一般还原群，一个参数通常给出 L-packet；packet 内部需要 component group、enhancement 和 inner form 数据。

禁止误读：

- 不得把一般 reductive group 的 LLC 写成 $\operatorname{Irr}(G(F))$ 与参数集合的普通双射。
- 不得把 enhanced parameter、rigid inner twist 和 transfer factor normalization 视为装饰性数据。

### 全局对应不是无条件定理

允许说法：

- 函数域 `GL(n)` 全局 Langlands 是 Drinfeld-Lafforgue 定理层结果。
- 数域 regular algebraic、polarized 或 cohomological 情形中存在深刻的 Galois 表示构造和局部-整体相容定理。
- 一般数域全局 Langlands 仍是猜想或纲领性结构。

禁止误读：

- 不得把数域完整 `GL(n)` 或一般 reductive group 全局 Langlands 写成已证对应。
- 不得把数域上的 conjectural Langlands group 当作本书中已经构造的对象。

### L 函数不是单独的对象

允许说法：

- L 函数必须相对于表示 $r:{}^LG\to\operatorname{GL}(V)$ 定义。
- Euler 乘积、局部因子、ramified 因子、Archimedean 因子、解析延拓和函数方程是不同层次的数据。
- 对一般 $G$ 和一般 $r$，解析性质属于外部输入或猜想。

禁止误读：

- 不得把形式 Euler 乘积自动视为全平面解析函数。
- 不得写“$G$ 的 L 函数”而不说明所选 L 群表示和归一化。

### 函子性由 L 群同态控制

允许说法：

- 函子性是 L 群同态诱导参数推前，进而预期给出自守表示转移。
- 弱转移由几乎所有非分歧 Satake 参数控制；强转移要求局部参数逐处相容。
- Converse theorem 可把若干 `GL(N)` 目标问题转化为 L 函数解析性质。

禁止误读：

- 不得把原群之间的同态直接等同于 Langlands functorial transfer。
- 不得由少数 L 函数相等直接推出一般函子性。

### Trace formula 是比较机制，不是对应本身

允许说法：

- Trace formula 比较谱侧和几何侧，用于稳定化、endoscopy、base change 和 Arthur 分类。
- Fundamental lemma、transfer factor 和 stable orbital integrals 是让比较成立的必要局部输入。

禁止误读：

- 不得把 trace formula 本身写成 Langlands 对应。
- 不得把 endoscopy 说成普通 subgroup restriction 或 representation restriction。

### 几何 Langlands 不是数论 Langlands 的简单翻译

允许说法：

- 几何 Langlands 的现代形式是谱侧范畴和自动侧范畴之间的等价。
- Hecke eigensheaf 是范畴等价在点对象上的影子。
- 有限域上可经 sheaf-function dictionary 得到函数域自守函数；代数闭域上则保留范畴结构。

禁止误读：

- 不得把几何 Langlands 仅写成“每个 local system 对应一个 sheaf”。
- 不得把 Fargues-Fontaine 曲线视为数域的全局曲线替代物；它是 $p$-adic 局部域的几何化接口。

### 费马应用是 `GL(2)` 模性的实例

允许说法：

- 费马大定理由 Frey 曲线、半稳定模性、Ribet 降层和 $S_2(\Gamma_0(2))=0$ 的矛盾推出。
- 该链体现 `GL(2)/\mathbb Q` 模性思想和二维 Galois 表示到自守表示的关系。

禁止误读：

- 不得写“完整 Langlands 纲领证明费马大定理”。
- 不得把 Wiles-Taylor-Wiles patching、Ribet 降层或 Frey 曲线局部计算当作本书内部已证内容。

## 四条主线的最终概念状态

| 主线 | 概念状态 | 审定结论 |
|---|---|---|
| `GL(1)` | 类域论给出参数对应；Tate thesis 给出 L 函数解析接口 | 闭合 |
| `GL(2)` 与费马应用 | 模性输入、降层输入和 Frey 曲线输入分离 | 闭合 |
| 一般算术 Langlands | 对象链闭合；证明层依赖外部输入和猜想 | 审定前闭合 |
| 几何 Langlands | 几何对象链、范畴接口和函数域桥梁闭合 | 审定前闭合 |

## 出版前概念检查清单

出版前若改动正文或附录，必须检查：

1. 是否把 conjecture 写成 theorem。
2. 是否把 external input 写成 internal proof。
3. 是否把 local statement 写成 global statement。
4. 是否把 `GL(n)` 的单点 packet 现象推广到一般群。
5. 是否混用 arithmetic Frobenius 和 geometric Frobenius。
6. 是否混用 classical normalization 和 unitary automorphic normalization。
7. 是否把 L 函数写成未指定 $r:{}^LG\to\operatorname{GL}(V)$ 的对象。
8. 是否把 geometric Langlands 的 categorical form 降格成点到对象的朴素对应。
9. 是否把函数域定理误读为数域定理。
10. 是否把费马应用链误写成完整 Langlands 纲领的直接证明。

## 最终概念结论

本书已经达到概念审定前闭合：核心词语均已分层，四条主线的对象边界清楚，猜想、外部输入、证明草图和本书内证明的状态可追踪。后续工作应是出版前审校，而不是概念体系重建。
