# 附录 D：第二卷输入定理登记表

## D.0 目标

本附录把第二卷使用的外部输入定理拆成较小颗粒，避免“Scholze 结构定理”这样过粗的引用。每个条目包含：精确形式、使用位置、不可省略的假设和本书不证明的部分。

## D.1 solidification 存在

**输入定理 D.1.** 设 $K_S$ 为

$$
\operatorname{Cone}(\mathbb Z[\underline S]\to\mathbb Z^\square[S])
$$

其中 $S$ 遍历 profinite 集合。全子范畴

$$
D_\square(\mathbb Z)
=
\{C\in D(\mathbf{CondAb})\mid
R\operatorname{Hom}(K_S,C)=0\ \forall S\}
$$

的包含函子有左伴随 $L^\square$。

**使用位置.** 第一章定义派生 solidification；第四章与 analyticization 类比。

**不可省略的假设.** $S$ 取自一个集合大小控制下的测试对象族；$D(\mathbf{CondAb})$ 的可展示性和稳定性被使用。

**本书不证明.** presentable stable category 层面的局部化存在性和 condensed derived category 的可展示性。

**证明模块位置.** 附录 V 证明集合生成局部化的稳定形式后果、\(K_S\)-正交判别和 Dirac-to-measure cone 局部对象口径；与 Scholze solidification 的识别仍为输入。

## D.2 solid tensor 下降

**输入定理 D.2.** D.1 中 localization 的核是张量理想。因此

$$
M\otimes^{L,\square}N=L^\square(M\otimes^LN)
$$

给出 $D_\square(\mathbb Z)$ 上的闭对称幺半结构。

**使用位置.** 第二章 solid 环、solid 模、solid 张量积。

**不可省略的假设.** 张量积必须是 $D(\mathbf{CondAb})$ 中的派生张量；无限乘积对象必须在 solid localization 后处理。

**本书不证明.** $K_S$ 生成的核对张量稳定。

**证明模块位置.** 附录 W 证明张量理想性的生成元判别，并把本条归约到 profinite 测度张量计算和 \(\ker L^\square=\mathcal N_\square\)。

## D.3 profinite solid tensor 公式

**输入定理 D.3.** 对 profinite 集合 $S,T$，有自然等价

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

**使用位置.** 第二章、第四卷 solid 张量例子。

**不可省略的假设.** $S,T$ 是 profinite；张量积是导出 solid 张量积。

**边界反例.** 普通阿贝尔群中

$$
\left(\prod_n\mathbb Z\right)\otimes\mathbb Q
\to
\prod_n\mathbb Q
$$

不是满射；因此不能用普通张量积推导 D.3。

**证明模块位置.** 附录 W 说明本公式如何推出 \(K_S\otimes \mathbb Z[\underline T]\) 被 solidification 杀掉，并由此推出 solid kernel 张量理想性。

## D.4 analytic ring localization

**输入定理 D.4.** 若 $(A,\mathcal M)$ 是 analytic ring，则由

$$
K_S^{\mathcal M}=
\operatorname{Cone}(A[\underline S]\to\mathcal M[S])
$$

定义的解析对象范畴

$$
D(A,\mathcal M)
=
\{C\in D(A)\mid R\operatorname{Hom}_A(K_S^{\mathcal M},C)=0\ \forall S\}
$$

是反射局部子范畴，且 localization 与张量积相容。

**使用位置.** 第三、四章。

**不可省略的假设.** $(A,\mathcal M)$ 必须满足 analytic ring 公理；任意测度赋值不够。

**本书不证明.** analytic ring 公理推出 Bousfield localization 与张量相容。

**证明模块位置.** 附录 X 证明 analytic cone 局部对象、analyticization 泛性质、analytic tensor 下降的形式部分，并说明 solidification 是特例。

## D.5 liquid analytic ring

**输入定理 D.5.** 对 \(0<p\le 1\)，$(\mathbb R,\mathcal M_{<p})$ 是 analytic ring，其解析模范畴给出 $p$-liquid 实向量空间。

**使用位置.** 第五章、第三卷 Dolbeault/liquid 模型。

**不可省略的假设.** 必须固定 $p$ 或 $<p$；$\mathcal M_{<p}[S]$ 的增长条件参与定义。

**本书不证明.** $\mathcal M_{<p}$ 满足 analytic ring 条件。

**证明模块位置.** 附录 S、Z 证明接受该 analytic ring 输入后，finite-dimensional realization、Fredholm cohomology 和 Dolbeault 类型检查的形式后果。

## D.6 liquid realization

**输入定理 D.6.** 对文献中指定、并在本书作为输入范围登记的拓扑向量空间子范畴 \(\mathcal T_p\)，存在 realization 过程

$$
\mathcal L_p:E\mapsto E_{\mathrm{liq}}
$$

进入 $p$-liquid analytic 模范畴。本文只在 \(\mathcal T_p\) 的下列性质内使用它：\(\mathcal T_p\) 包含有限维实向量空间和本文使用的核 Fréchet/Dolbeault 型对象；\(\mathcal L_p\) 与有限极限、有限直和、闭子空间 kernel、有限维 quotient 和闭值域短正合列相容。

**使用位置.** 第五章、附录 J、附录 P、附录 S；第三卷 Dolbeault 复形和 Hodge/Fredholm 有限性接口。

**不可省略的假设.** 对象必须处于 realization 的适用范围；单纯是 Banach 或 Fréchet 空间不够。

**本书不证明.** liquid realization 的构造、Hom 判别、与经典连续线性算子的相容性。

**证明模块位置.** 附录 Z 证明拓扑向量空间凝聚化和闭值域 Fréchet cohomology 的书内部分，并把 realization exactness 精确隔离为输入。

## D.7 Huber pair 的解析化

**输入定理 D.7.** 离散 Huber pair $(A,A^+)$ 给出 analytic ring $(A,A^+)^\square$；rational localization 与解析模范畴相容，并满足 rational Čech 下降。

**使用位置.** 第六章。

**不可省略的假设.** $A^+$ 不能省略；它控制有界元素和 rational domain。

**本书不证明.** 从 Huber pair 构造测度对象、验证 rational localization 下降。

**证明模块位置.** 附录 Y 证明 rational descent 的 Čech nerve、mapping-space descent、对象 gluing 和 compact generation descent 的形式层，把 Huber rational acyclicity 隔离为输入。

## D.8 Fréchet-Fredholm/Hodge 输入

**输入定理 D.8.** 紧光滑流形或紧复流形上的椭圆 Fréchet 复形满足闭值域、Fredholm 有限性和 Hodge 分解；Dolbeault 复形的 Fréchet cohomology 与 sheaf cohomology 相容。

**使用位置.** 附录 P、附录 S；第三卷第四、五章和 Hodge/Fredholm 有限性附录。

**不可省略的假设.** 需要紧性、椭圆性、合适的 Fréchet 拓扑和闭值域；代数复形本身不包含这些拓扑信息。

**本书不证明.** parametrix、椭圆估计、Hodge decomposition、Dolbeault-Fredholm 定理。

**证明模块位置.** 附录 P、Z 和第三卷附录 Z、AA 证明闭值域、Fredholm 和 Hodge 输入被接受后的 liquid/analytic 类型后果。

## D.9 紧支撑推前与右伴随

**输入定理 D.9.** 对 Scholze 讲义中的有限型仿射情形，存在

$$
f_!:D(A^\square)\to D(\mathbb Z^\square)
$$

满足投影公式，并在 compact generation 假设下有右伴随 $f^!$。

**使用位置.** 第七章和第三卷 Serre duality 语言。

**不可省略的假设.** 非 proper 情形必须包含无穷远边界项；$f_!$ 不是忘记结构后的普通推前。

**本书不证明.** 边界项构造、compact generation 验证和投影公式。

## D.10 与附录 T 的对应

| 附录 T 编号 | 本附录条目 | 说明 |
| --- | --- | --- |
| T-I | D.1 | solid 反射局部化 |
| T-II | D.2 | solid kernel 张量理想性 |
| T-III | D.3 | profinite 测度张量公式 |
| T-IV | D.4 | analytic ring 反射局部化 |
| T-V | D.4 | analytic kernel 张量理想性包含在 analytic ring localization 条件中 |
| T-VI | D.7 | Huber pair rational localization 与 Čech descent |
| T-VII | D.5 | \(p\)-liquid analytic ring |
| T-VIII | D.6 | liquid realization |
| T-IX | D.8 | Fréchet Fredholm/Hodge 输入 |

## D.11 引用规则

正文以后引用本附录时，应写成：

- “由输入定理 D.1”而不是“由 Scholze 定理”。
- “由输入定理 D.3 的 profinite 假设”而不是“由 solid 张量积公式”。
- “由输入定理 D.4 的 analytic ring 条件”而不是“由解析化存在”。
- “由输入定理 D.6 的 realization 范围”而不是“Fréchet 空间自然是 liquid 对象”。

这样可以强制读者看到每个结论的精确假设。
