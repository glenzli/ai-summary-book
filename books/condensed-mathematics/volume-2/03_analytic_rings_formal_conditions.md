# 第三章：解析环的正式条件

## 本章目标

第一卷第十四章只给出 analytic ring 的入口。本章区分三件不能互相替代的事：

1. theory of measures 的数据；
2. analytic ring 的复形级公理；
3. 由该公理推出的阿贝尔范畴、导出全嵌入、解析化与幺半结构定理。

旧写法把第 3 项当作第 2 项的定义，造成循环。本章按 S26 Lecture VII 的顺序修正。

## 依赖

需要第一卷第十四章、第二卷第一章，以及第一卷附录 A 的固定
\(\kappa\)-层级约定。

## 3.0 类型与方差

固定一个有单位结合凝聚环 \(A\in\mathbf{CondRing}_\kappa\)。记
\(A\text{-}\mathbf{Mod}\) 为凝聚阿贝尔群中的左 \(A\)-模范畴。只有在讨论对称
幺半结构时才额外假设 \(A\) 交换。

对 \(S\in\mathbf{ED}_\kappa\)，\(A[\underline S]\) 表示由凝聚集合
\(\underline S\) 生成的自由凝聚左 \(A\)-模。若 \(M,N\in D(A)\)，则
\(R\underline{\operatorname{Hom}}_A(M,N)\) 取值于
\(D(\mathbf{CondAb}_\kappa)\)；去掉下划线的 \(R\operatorname{Hom}_A\) 才表示
取全局截面后的普通导出 Hom 复形。

## 3.1 Theory of measures

**定义 3.1（测度理论；S26 Definition 7.1）.** 凝聚环 \(A\) 上的 theory of
measures 是以下数据：

1. 一个协变函子
   $$
   \mathcal M:\mathbf{ED}_\kappa\longrightarrow A\text{-}\mathbf{Mod},
   \qquad S\longmapsto\mathcal M[S];
   $$
2. 对极不连通空间的每个有限不交并，包括空不交并，有自然同构
   $$
   \mathcal M[S\sqcup T]\cong\mathcal M[S]\times\mathcal M[T],
   \qquad \mathcal M[\varnothing]\cong0;
   $$
3. 一个对 \(S\) 自然的 Dirac 映射
   $$
   \delta_S:\underline S\longrightarrow U(\mathcal M[S]),
   $$
   其中 \(U\) 忘掉 \(A\)-模结构。

由自由模伴随，\(\delta_S\) 唯一延拓为 \(A\)-线性态射

$$
A[\underline S]\longrightarrow\mathcal M[S].
$$

组合 \((A,\mathcal M)\) 在验证下一节公理之前只称为**预解析数据**。

**边界 3.2（非离散底环）.** 当 \(A\) 离散时，有限不交并条件可把
\(S\mapsto\mathcal M[S]\) 延拓为自由模 \(A[\underline S]\) 之间态射上的加性
函子。一般凝聚环上，这个延拓不是定义 3.1 的形式后果；不能把“对连续映射协变”无标记地
升级为“对所有自由 \(A\)-模态射协变”。

## 3.2 Analytic 公理

**定义 3.3（analytic ring；S26 Definition 7.4）.** 预解析数据
\((A,\mathcal M)\) 称为 analytic ring，如果对任意非负同调次数的复形

$$
C_\bullet:\quad \cdots\longrightarrow C_2\longrightarrow C_1
\longrightarrow C_0\longrightarrow0,
$$

只要每个 \(C_i\) 都是若干对象 \(\mathcal M[T]\)、
\(T\in\mathbf{ED}_\kappa\) 的集合指标直和，就对每个
\(S\in\mathbf{ED}_\kappa\) 有等价

$$
R\underline{\operatorname{Hom}}_A(\mathcal M[S],C_\bullet)
\xrightarrow{\ \sim\ }
R\underline{\operatorname{Hom}}_A(A[\underline S],C_\bullet)
$$

于 \(D(\mathbf{CondAb}_\kappa)\) 中成立。箭头由
\(A[\underline S]\to\mathcal M[S]\) 在第一变量反变地诱导。

**量词说明.** 公理量化的是上述整类可能无界向左的复形，不只是单个
\(\mathcal M[T]\)、有限直和或 bounded complex。只验证 degree zero Hom 判别不足以
证明预解析数据是 analytic ring。

## 3.3 心脏与派生局部对象

**定义 3.4.** 凝聚左 \(A\)-模 \(N\) 称为 \((A,\mathcal M)\)-解析模，如果对每个
\(S\in\mathbf{ED}_\kappa\)，自然映射

$$
\operatorname{Hom}_A(\mathcal M[S],N)
\longrightarrow
\operatorname{Hom}_A(A[\underline S],N)
\cong N(S)
$$

是同构。记这些对象的满子范畴为
\((A,\mathcal M)\text{-}\mathbf{Mod}\)。

**外部输入定理 3.5（analytic 结构定理；Scholze）.** 若
\((A,\mathcal M)\) 满足定义 3.3，则：

1. \((A,\mathcal M)\text{-}\mathbf{Mod}\) 是对所有极限、余极限和扩张封闭的
   阿贝尔范畴；\(\mathcal M[S]\) 构成紧投射生成族。
2. 包含函子有左伴随
   $$
   L^0_{(A,\mathcal M)}:A\text{-}\mathbf{Mod}
   \longrightarrow(A,\mathcal M)\text{-}\mathbf{Mod},
   $$
   它是把 \(A[\underline S]\mapsto\mathcal M[S]\) 余极限延拓所得的唯一函子。
3. 自然函子
   $$
   D((A,\mathcal M)\text{-}\mathbf{Mod})\longrightarrow D(A)
   $$
   全忠实。本质像由满足
   $$
   R\underline{\operatorname{Hom}}_A(\mathcal M[S],C)
   \xrightarrow{\sim}
   R\underline{\operatorname{Hom}}_A(A[\underline S],C)
   $$
   的 \(C\) 构成；等价地，每个 \(H^n(C)\) 都是解析模。
4. 该导出满子范畴的包含有左伴随
   $$
   L_{(A,\mathcal M)}:D(A)\longrightarrow D(A,\mathcal M),
   $$
   即 \(L^0_{(A,\mathcal M)}\) 的总左导出函子。
5. 若 \(A\) 交换，则心脏和导出范畴各有唯一的对称幺半张量，使相应解析化函子为
   对称幺半函子。

**来源与外部边界.** 这是 S26 Proposition 7.5。本书在附录 C、E、K、O、X 中证明
反射局部化和张量下降的一般形式，但不重证“定义 3.3 的公理推出上述全部结论”的核心
论证。

**警告 3.6（underived 与 derived tensor 的边界）.** S26 Warning 7.6 指出，在最大
一般性下，导出 analytic tensor 未必已知是心脏层 analytic tensor 的总左导出函子。
这等价于要求：对所有极不连通 \(S,T\)，把 \(S\times T\) 由极不连通对象作 simplicial
resolution 后计算的 \(\mathcal M[S\times T]\) 集中在 degree zero。后文章节若使用
underived tensor 计算 derived tensor，必须另行验证这个集中性；定理 3.5 的幺半存在性
本身不消除该义务。

## 3.4 Cone 判别是推论，不是 analytic 公理

对 \(S\in\mathbf{ED}_\kappa\)，在稳定增强中定义

$$
K_S^{\mathcal M}
=\operatorname{cofib}\left(A[\underline S]\to\mathcal M[S]\right).
$$

**命题 3.7（局部对象的 cone 判别）.** 对 \(C\in D(A)\)，以下条件等价：

1. 对每个 \(S\)，
   $$
   R\underline{\operatorname{Hom}}_A(\mathcal M[S],C)
   \to R\underline{\operatorname{Hom}}_A(A[\underline S],C)
   $$
   是等价；
2. 对每个 \(S\)，
   $$
   R\underline{\operatorname{Hom}}_A(K_S^{\mathcal M},C)\simeq0.
   $$

**证明.** 对 defining cofiber sequence 应用第一变量反变的
\(R\underline{\operatorname{Hom}}_A(-,C)\)，得到 fiber sequence

$$
R\underline{\operatorname{Hom}}_A(K_S^{\mathcal M},C)
\longrightarrow
R\underline{\operatorname{Hom}}_A(\mathcal M[S],C)
\longrightarrow
R\underline{\operatorname{Hom}}_A(A[\underline S],C).
$$

稳定范畴中后一箭头为等价当且仅当其 fiber 为零。对每个 \(S\) 取交即得。证毕。

**逻辑边界 3.8.** 对任意预解析数据都能写出 \(K_S^{\mathcal M}\) 和正交对象类；
命题 3.7 只证明两个正交表述等价。反射函子存在、心脏为阿贝尔范畴及核为张量理想仍需
定义 3.3 与外部输入定理 3.5，不能由 cone 记号推出。

## 3.5 Solid 与 liquid 接口

**例 3.9（solid 是 analytic ring）.** 取离散凝聚环 \(A=\underline{\mathbb Z}\)，
并令

$$
\mathcal M[S]=\mathbb Z^\square[S].
$$

S26 Theorem 5.8 与 Proposition 7.5 断言这是 analytic ring，且

$$
(\mathbb Z,\mathbb Z^\square)\text{-}\mathbf{Mod}=\mathbf{Solid},
\qquad
D(\mathbb Z,\mathbb Z^\square)=D_\square(\mathbb Z).
$$

这两个等号是外部结构定理的识别，不是把符号代入定义后便自动成立。

**反例 3.10（普通 Radon 测度不够）.** 在通常拓扑实数
\(\underline{\mathbb R}\) 上取有界 signed Radon measures
\(\mathcal M_1[S]\)，会得到一套 theory of measures，但它不是 analytic ring。
Ribe 的非局部凸扩张给出 analytic 公理失败的障碍。S26 Example 7.10 记录此失败，
Theorem 7.11 则以 \(\mathcal M_{<p}\)、\(0<p\le1\) 修复它。第五章将精确定义相应
\(p\)-liquid 模。

## 3.6 本章小结

analytic ring 的定义是复形级 Hom 公理。反射局部化、cohomology 判别和幺半结构是
S26 Proposition 7.5 的外部结构定理；cone 判别只是接受定义后的形式改写。Solid 是一个
特例，liquid 则来自普通 Radon 测度失败后的 \(<p\)-测度修正。

## 练习

**练习 3.1.** 检查定义 3.1 中空不交并为何强制
\(\mathcal M[\varnothing]=0\)。

**练习 3.2.** 证明命题 3.7，并标明使用的是 internal derived Hom。

**练习 3.3.** 说明只对 \(C=\mathcal M[T][0]\) 检查 Hom 等价为何弱于定义 3.3。

**练习 3.4.** 列出从 solid theory 得到例 3.9 时使用的 S26 Theorem 5.8 的两项结论。

**练习 3.5.** 解释警告 3.6 中 degree-zero 集中性在何处进入
underived/derived tensor 比较。
