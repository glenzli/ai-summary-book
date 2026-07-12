# 附录 I：解析环公理检查表与失败模式

## I.0 目标

第三章已按 S26 Definition 7.4 给出 analytic ring 的复形级定义。本附录用于实际审查一套
测度数据：哪些是定义义务，哪些是 Proposition 7.5 的结构后果，哪些又是 rational
descent 等额外几何输入。

## I.1 定义数据的逐项类型检查

固定凝聚有单位结合环 \(A\) 和
\(\mathbf{ED}_\kappa\)。

**检查表 I.1（theory of measures）.** 必须逐项给出：

1. 协变函子
   $$
   \mathcal M:\mathbf{ED}_\kappa\to A\text{-}\mathbf{Mod};
   $$
2. 对每个连续映射 \(f:S\to T\) 的 \(A\)-线性态射
   \(f_*:\mathcal M[S]\to\mathcal M[T]\)，以及恒等与复合函子性；
3. 对有限不交并的自然同构，包括
   \(\mathcal M[\varnothing]=0\)；
4. 自然 Dirac 映射 \(\underline S\to U(\mathcal M[S])\)，等价地自由延拓
   \(A[\underline S]\to\mathcal M[S]\)。

定义域是小骨架 \(\mathbf{ED}_\kappa\)，值域是凝聚 \(A\)-模；不能把
\(\mathcal M[S]\) 只定义成普通抽象群，也不能让 \(S\) 遍历未截断的 proper class。

**命题 I.2（有限离散对象）.** 若 \(S=\{s_1,\ldots,s_n\}\) 有限离散，则

$$
\mathcal M[S]\cong\prod_{i=1}^n\mathcal M[*],
$$

其中 \(n=0\) 时右侧为零对象。

**证明.** 把 \(S\) 写成 \(n\) 个单点的不交并，反复使用 I.1(3)。空情形正是
\(\mathcal M[\varnothing]=0\)。证毕。

## I.2 真正的 analytic 义务

**检查表 I.3（S26 Definition 7.4）.** 在 I.1 之外，必须验证：对每个复形

$$
C_\bullet:\cdots\to C_2\to C_1\to C_0\to0
$$

以及每个 \(S\in\mathbf{ED}_\kappa\)，若每项 \(C_i\) 是对象
\(\mathcal M[T]\) 的任意允许直和，则

$$
R\underline{\operatorname{Hom}}_A(\mathcal M[S],C_\bullet)
\longrightarrow
R\underline{\operatorname{Hom}}_A(A[\underline S],C_\bullet)
$$

在 \(D(\mathbf{CondAb}_\kappa)\) 中为等价。这里有四个不可删除的量词：所有
\(S\)、所有次数、所有允许直和以及所有此类微分。

**边界 I.4（finite tests 不够）.** 若 Dirac 映射在有限离散 \(S\) 上是同构，则这些
\(S\) 的 analytic 检验自动成立，但这不控制任何无限 profinite 或 ED 对象。Solid 的
\(S=\mathbb N\cup\{\infty\}\) 计算已显示有限测试会漏掉乘积型测度。

**边界 I.5（cone 正交不证明 analytic 公理）.** 对任意 I.1 数据都能定义

$$
K_S^{\mathcal M}=\operatorname{cofib}(A[\underline S]\to\mathcal M[S])
$$

及其正交对象类。即使一般 presentable localization 定理给出该正交类的反射，也没有
证明 I.3 对“由 \(\mathcal M[T]\) 组成的所有复形”成立，更没有识别反射的心脏或紧投射
生成元。因此“localization 存在”不是 analytic ring 的替代定义。

## I.3 定义之后才能引用的结构定理

**外部输入定理 I.6（S26 Proposition 7.5）.** 若 I.3 已验证，则可引用：

1. 解析模构成对所有极限、余极限和扩张封闭的阿贝尔范畴；
2. \(\mathcal M[S]\) 是紧投射生成元；
3. 心脏层解析化是 \(A[\underline S]\mapsto\mathcal M[S]\) 的余极限延拓；
4. 解析模导出范畴全忠实嵌入 \(D(A)\)，且由 cohomology objects 检测；
5. 导出包含有左伴随；交换底环时存在唯一兼容的对称幺半结构。

本书附录 C、E、K、O、X 只证明这些结论中的一般局部化形式，不重证从 I.3 到 I.6
的 Scholze 核心论证。

**警告 I.7（S26 Warning 7.6）.** “导出范畴有 analytic tensor”不自动等于“它是
心脏层 analytic tensor 的总左导出”。后一个结论还要验证 simplicial resolution 所算的
\(\mathcal M[S\times T]\) 集中在 degree zero。任何把 underived 公式直接用于无界复形
的计算都必须引用这个额外集中性。

**边界 I.8（几何下降是另一层输入）.** Huber pair 给出 analytic ring、rational
localization、rational Čech acyclicity 和 category-valued descent 是彼此相关但不同的
定理。Definition 7.4 与 Proposition 7.5 本身不蕴含某个给定 rational cover 的
acyclicity；本书把后者单独登记为输入定理 D.7。

## I.4 失败模式与反例

**失败模式 I.9（普通完备化）.** 一个拓扑环的 adic、Banach 或 sequential completion
只控制指定滤过或 Cauchy 序列。analytic 公理控制所有 ED 测试对象上的测度以及由自由
测度模组成的复形；前者不能推出后者。

**失败模式 I.10（普通 Radon 测度）.** 在通常拓扑的 \(\mathbb R\) 上取 bounded
signed Radon measures 确实满足 I.1，却不满足 I.3；Ribe 的非局部凸扩张提供障碍。
这是 S26 Example 7.10 的反例。S26 Theorem 7.11 的
\((\mathbb R,\mathcal M_{<p})\)、\(0<p\le1\) 才是 analytic ring。

**失败模式 I.11（只检查全局点）.** 只验证

$$
R\operatorname{Hom}_A(\mathcal M[S],C_\bullet)
\to R\operatorname{Hom}_A(A[\underline S],C_\bullet)
$$

在单点取值后的等价，弱于 I.3 的内部凝聚 Hom 等价。内部等价须在所有测试对象上成立；
忘掉这一区别会遗漏参数族中的失败。

**失败模式 I.12（普通张量已经解析）.** 即使 \(M,N\) 是解析对象，也只有

$$
M\otimes^L_{(A,\mathcal M)}N
=L_{(A,\mathcal M)}(M\otimes_A^LN)
$$

在一般性下有定义。若要删去最外层解析化，必须证明普通张量保持局部对象；这正是张量
理想性/幺半局部化义务，不能从“两个输入对象都解析”推出。

## I.5 使用规则

正文声称 \((A,\mathcal M)\) analytic 时，必须按下列顺序引用：

1. 先给 I.1 的数据；
2. 再给 I.3 的验证或精确外部定理；
3. 然后才能调用 I.6 的 localization、生成元和 tensor；
4. rational descent、underived/derived tensor 比较及几何 realization 另列输入。

## 练习

**练习 I.1.** 对有限二点集写出命题 I.2 的同构与两个 Dirac 映射。

**练习 I.2.** 解释为什么 I.5 中反射存在仍不能推出 \(\mathcal M[S]\) 投射。

**练习 I.3.** 在失败模式 I.11 中写出内部 Hom 在测试对象 \(T\) 上的取值。

**练习 I.4.** 分别指出 Definition 7.4、Proposition 7.5、Warning 7.6 和 rational
descent 在本附录中的编号。
