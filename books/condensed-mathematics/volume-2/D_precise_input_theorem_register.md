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

## D.2 solid tensor 下降

**输入定理 D.2.** D.1 中 localization 的核是张量理想。因此

$$
M\otimes^{L,\square}N=L^\square(M\otimes^LN)
$$

给出 $D_\square(\mathbb Z)$ 上的闭对称幺半结构。

**使用位置.** 第二章 solid 环、solid 模、solid 张量积。

**不可省略的假设.** 张量积必须是 $D(\mathbf{CondAb})$ 中的派生张量；无限乘积对象必须在 solid localization 后处理。

**本书不证明.** $K_S$ 生成的核对张量稳定。

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

## D.5 liquid analytic ring

**输入定理 D.5.** 对指定范围的 $p$，$(\mathbb R,\mathcal M_{<p})$ 是 analytic ring，其解析模范畴给出 $p$-liquid 实向量空间。

**使用位置.** 第五章、第三卷 Dolbeault/liquid 模型。

**不可省略的假设.** 必须固定 $p$ 或 $<p$；$\mathcal M_{<p}[S]$ 的增长条件参与定义。

**本书不证明.** $\mathcal M_{<p}$ 满足 analytic ring 条件。

## D.6 Huber pair 的解析化

**输入定理 D.6.** 离散 Huber pair $(A,A^+)$ 给出 analytic ring $(A,A^+)^\square$；rational localization 与解析模范畴相容，并满足 rational Čech 下降。

**使用位置.** 第六章。

**不可省略的假设.** $A^+$ 不能省略；它控制有界元素和 rational domain。

**本书不证明.** 从 Huber pair 构造测度对象、验证 rational localization 下降。

## D.7 紧支撑推前与右伴随

**输入定理 D.7.** 对 Scholze 讲义中的有限型仿射情形，存在

$$
f_!:D(A^\square)\to D(\mathbb Z^\square)
$$

满足投影公式，并在 compact generation 假设下有右伴随 $f^!$。

**使用位置.** 第七章和第三卷 Serre duality 语言。

**不可省略的假设.** 非 proper 情形必须包含无穷远边界项；$f_!$ 不是忘记结构后的普通推前。

**本书不证明.** 边界项构造、compact generation 验证和投影公式。

## D.8 引用规则

正文以后引用本附录时，应写成：

- “由输入定理 D.1”而不是“由 Scholze 定理”。
- “由输入定理 D.3 的 profinite 假设”而不是“由 solid 张量积公式”。
- “由输入定理 D.4 的 analytic ring 条件”而不是“由解析化存在”。

这样可以强制读者看到每个结论的精确假设。
