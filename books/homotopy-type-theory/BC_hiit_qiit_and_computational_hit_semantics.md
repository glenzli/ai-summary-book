# 附录 BC：HIIT、QIIT 与计算 HIT 语义

本附录补齐高阶归纳类型方向的另一条主线：higher inductive-inductive types、quotient inductive types、quotient inductive-inductive types 以及它们的计算语义。第九章和附录 L 只列出本书直接使用的 HIT 规则；本附录说明更一般的签名、消去原则、初始性和 cubical 实现边界。

## BC.1 HIT 签名的层级

**定义 BC.1（ordinary HIT signature）。** 一个普通 HIT 签名由一个新类型符号 $H$、点构造子、路径构造子和更高路径构造子组成。构造子的目标可以包含 $H$，但必须满足严格正性。

**定义 BC.2（higher inductive-inductive type, HIIT）。** HIIT 同时生成多个相互依赖的类型或类型族。例如生成类型 $A$ 的同时生成族
$$
B:A\to\mathcal U
$$
并允许构造子同时作用于 $A$ 和 $B$。

**定义 BC.3（quotient inductive type, QIT）。** QIT 是带路径方程和截断构造子的归纳类型。集合商、有限多重集、语法商和自由代数商都是典型例子。

**定义 BC.4（quotient inductive-inductive type, QIIT）。** QIIT 同时具有 inductive-inductive 依赖生成和 quotient/path constructors。它可表达“语法与等式理论同时生成”的对象，例如上下文、类型、项和转换规则的商。

## BC.2 消去原则

**原则 BC.5（代数解释）。** 给定 HIT/QIIT 签名 $\Sigma$，其递归原则应说：若 $X$ 是 $\Sigma$-代数，则存在结构保持映射
$$
\mathsf{rec}_X:H_\Sigma\to X.
$$
其依赖消去原则应说：若 $P$ 是 $H_\Sigma$ 上的依赖 $\Sigma$-代数，则存在依赖截面
$$
\mathsf{ind}_P:\prod_{h:H_\Sigma}P(h).
$$

**规则 BC.6（路径构造子的消去数据）。** 若签名含路径构造子
$$
c:\prod_{\vec x} u(\vec x)=v(\vec x),
$$
则递归到 $X$ 时必须给出
$$
c_X:\prod_{\vec x} u_X(\vec x)=v_X(\vec x).
$$
依赖消去时，该数据变成 transport 后的依赖路径。

**规则 BC.7（截断构造子的目标限制）。** 若 QIT 加入 0-truncation 构造子，则非依赖递归到 $X$ 时通常需要 $X$ 是集合；依赖消去到 $P$ 时需要每个 fiber $P(h)$ 满足相应截断层级。

## BC.3 初始性口径

**定义 BC.8（$\Sigma$-代数范畴）。** 对签名 $\Sigma$，令 $\mathsf{Alg}_\Sigma$ 为其代数和结构同态构成的范畴或高阶范畴。

**原则 BC.9（初始代数语义）。** HIT/QIIT $H_\Sigma$ 的递归和唯一性可由初始性表达：
$$
\mathsf{isInitial}_{\mathsf{Alg}_\Sigma}(H_\Sigma).
$$

**命题 BC.10（初始性推出递归唯一性，书内证明核）。** 若 $H$ 是 $\mathsf{Alg}_\Sigma$ 中初始对象，则对任意代数 $X$，结构保持映射类型
$$
\mathsf{Hom}_{\mathsf{Alg}_\Sigma}(H,X)
$$
可收缩。

**证明.** 初始对象的定义正是对每个目标对象 $X$，Hom 类型可收缩。其中心给出递归函数；收缩路径给出递归函数唯一性。$\square$

**边界 BC.11.** 初始性本身不自动给出 judgmental computation。它给出 propositional uniqueness。若要求点构造子上的 judgmental beta rule，需要语法或实现语言的额外计算规则。

## BC.4 标准例子

**例 BC.12（集合商）。** 给定集合 $A$ 和关系 $R:A\to A\to\mathsf{Prop}$，集合商 $A/R$ 是 QIT：
$$
[-]:A\to A/R,
$$
$$
\mathsf{glue}_{x,y}:R(x,y)\to [x]=[y],
$$
并加入 $\mathsf{isSet}(A/R)$ 构造子。其递归到集合 $X$ 时等价于给出 $f:A\to X$ 且 $R$-相容。

**例 BC.13（Cauchy 实数 HIIT）。** 附录 AK 的 Cauchy 实数可看作 HIIT/HIIT-like 规范：它同时生成实数对象、近似关系或等价关系、极限构造和路径等式。Brough 2026 验证了 HoTT Book 口径下的核心构造路线。

**例 BC.14（类型论语法商）。** 上下文、类型、项、替换和 definitional equality 常可作为 QIIT 生成：对象层语法由点构造子给出，转换规则和方程由路径构造子给出，良构性依赖关系由 inductive-inductive 结构给出。

## BC.5 Cubical 计算语义

**事实 BC.15（cubical HIT computation）。** Cubical type theory 通过区间、composition、filling 和 Glue 等结构，使许多 HIT 的路径构造子具有更直接的计算解释，覆盖圆、悬挂、pushout、截断和若干高阶构造。

**边界 BC.16（非所有签名自动可计算）。** “HIT 有 cubical 实现”不是一条无条件定理。每类签名需要分别证明或实现：

1.  严格正性或可接受签名条件；
2.  构造子的边界合法性；
3.  composition/filling 操作；
4.  canonicity 或 normalization；
5.  与 universe、resizing 和截断的相容性。

**原则 BC.17（本书采用的保守口径）。** 本书在对象语言中只使用附录 L 明确列出的 HIT 输入规则。更一般 HIIT/QIIT 只作为研究边界或外部来源；若某定理依赖它，必须记录具体签名、消去原则、计算规则和来源。

## BC.6 QIIT 与元理论风险

**风险 BC.18（严格正性）。** 若构造子在负位置使用待生成类型，可能破坏一致性或初始代数语义。

**风险 BC.19（universe level）。** QIIT 常同时生成语法、族和等式，容易触发 universe 提升。若正文隐去 universe 参数，逐项展开时会失败或改变定理强度。

**风险 BC.20（propositional 与 judgmental 混淆）。** QIIT 的路径方程通常只给出 propositional equality。把它当作 definitional equality 会改变替换、归约和 canonicity。

**风险 BC.21（元理论不可内部化）。** QIIT 的语法模型、初始性证明和 canonicity 是元理论结果。它们支持对象语言规则，但不是对象语言中的普通项。

## BC.7 与本书其他章节的连接

1.  附录 L 的 HIT 规则表是本书实际使用的最小输入。
2.  附录 AK、AW、BA 的 Cauchy/Dedekind 实数方向依赖 HIIT/QIT 口径，但具体分析定理仍需有理数误差证明。
3.  附录 Z 讨论 cubical/HIT 元理论边界；本附录给出更细的签名和计算风险。
4.  每个具体 HIT/QIIT 应先给出签名，再证明递归/消去原则与所采用的语义一致。

## BC.8 具体签名的边界

签名、代数、初始性、消去原则与计算规则是彼此独立的数据。只列出这些栏目并不证明任意 HIT/QIIT 存在；对每个具体签名仍须给出严格正性、语法、模型和所声称的 canonicity 定理。附录 L 之外的构造在本书中因此保持外部输入或研究边界身份。
