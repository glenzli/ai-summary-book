# 附录 T：Solid、Analytic 与 Liquid 的统一闭包

## T.0 目标

第二卷的出版级目标不是分别介绍三个名词，而是说明它们共享一个局部化框架，同时避免
把不同底环上的理论误写成一条无条件的函子链。正确的结构图是

$$
\begin{array}{ccccc}
&&\text{analytic rings }(A,\mathcal M)&&\\
&\swarrow&&\searrow&\\
(\mathbb Z,\mathbb Z^\square)&&&&(\underline{\mathbb R},\mathcal M_{<p})\\
\text{solid}&&&&p\text{-liquid}.
\end{array}
$$

两端都构造在 condensed 模的派生环境中，但底环和测度理论不同。除非给出 analytic
rings 的态射及其测度相容性，不存在由“solid”自动通向“liquid”的 scalar-extension
函子。

本附录把附录 Q、R、S 合成统一 theorem package，说明第二卷在输入定理型标准下已经闭合在哪里、还把哪些内容诚实保留为 Scholze/Clausen-Scholze 输入。

## T.1 四类范畴

**定义 T.1（四类带类型的环境）.** 本书的凝聚数学主线使用以下四类环境：

1. \(\mathbf{CondAb}\) 和 \(D(\mathbf{CondAb})\)；
2. \(D_\square(\mathbb Z)\) 及其 solid 环、solid 模；
3. \(D(A,\mathcal M)\) 形式的 analytic 模范畴；
4. 固定 \(0<p\le1\) 后，已在定理 5.3 与 S.1 定义的
   \(D(\mathbf{Liquid}_p)\)。

需要复结构时不另引入未定义的“复 liquid 派生范畴”：复 liquid 对象指一对
\((K,J)\)，其中 \(K\in D(\mathbf{Liquid}_p)\)、
\(J\in\operatorname{End}(K)\) 且 \(J^2=-\operatorname{id}_K\)；复线性态射按定义
与相应的 \(J\) 交换。这正是 P.10 中“乘以 \(i\)”的内自同态口径。

**原则 T.2（接口关系）.** 各环境之间只使用下列有类型的关系：

1. solid 是 \(D(\mathbf{CondAb})\) 对 Dirac-to-measure cone 的局部化；
2. analytic 是 \(D(A)\) 对 analytic measure cone 的局部化；
3. solid theory 是 analytic ring \((\mathbb Z,\mathbb Z^\square)\) 的特例；
4. \(p\)-liquid theory 是另一 analytic ring
   \((\underline{\mathbb R},\mathcal M_{<p})\) 的解析模理论，其派生范畴按 S.1
   记为 \(D(\mathbf{Liquid}_p)\)；
5. 对经典完备拓扑向量空间 \(E\)，记号
   \(\mathcal L_p(E)=\underline E\) 只记录其 liquid membership；它不是从 solid
   范畴出发的局部化。

## T.2 输入定理总表

第二卷主线需要以下输入定理：

| 编号 | 输入 | 所属主线 | 书内证明的部分 |
| --- | --- | --- | --- |
| T-I | solid 反射局部化 | solid | 局部对象、泛性质、形式正交 |
| T-II | solid kernel 张量理想性 | solid | 幺半下降、solid 张量、solid 环与模 |
| T-III | profinite 测度张量公式 | solid | 生成元检验和计算使用边界 |
| T-IV | analytic ring 反射局部化 | analytic | analyticization 泛性质 |
| T-V | analytic kernel 张量理想性 | analytic | analytic 张量、相对张量 |
| T-VI | Huber pair rational localization | analytic | 对象与态射 descent 后果 |
| T-VII | \(p\)-liquid analytic ring | liquid | liquid 范畴位置 |
| T-VIII | 经典完备空间的 liquid membership | liquid | 凝聚化、Fréchet/Banach 类型检查、局部提升判据 |
| T-IX | Fréchet Fredholm/Hodge 输入 | liquid/application | finite-dimensional cohomology 的 perfect 性 |

这些输入全部应在 [../INPUT_THEOREM_REGISTER.md](../INPUT_THEOREM_REGISTER.md) 和本卷附录 D 中可追踪。

## T.3 统一主定理

**定理 T.3（第二卷主线闭包）。** 接受 T-I 至 T-IX 后，第二卷建立以下闭合结构：

1. \(D_\square(\mathbb Z)\) 是闭对称幺半稳定范畴；
2. solid 环和 solid 模范畴有相对张量积、内部 Hom 和生成元检验；
3. analytic ring \((A,\mathcal M)\) 给出反射局部子范畴 \(D(A,\mathcal M)\)；
4. analyticization 满足泛性质，并与张量积相容；
5. Huber pair 的 rational localization 满足 Čech descent；
6. \(p\)-liquid 向量空间形成 \(\mathbf{Liquid}_p\)，其派生环境为
   \(D(\mathbf{Liquid}_p)\)；复结构在需要时由满足
   \(J^2=-\operatorname{id}\) 的内自同态记录；
7. 逐项 liquid 且相关 quotient 映射满足 profinite 参数族局部提升的 Fréchet 复形，
   其拓扑 cohomology 与 liquid cohomology 相容；连续 Hodge splitting 是充分条件；
8. Dolbeault 型对象进入第三卷时，类型、finite-dimensionality 和 perfect 性已由第二卷给出检查表。

**证明.** 第 1-2 项由附录 Q 的 Solid 主闭包定理 Q.15 给出。第 3-5 项由附录 R 的
Analytic 主闭包定理 R.17 给出。第 6-8 项由附录 S 的 Liquid 主闭包定理 S.14
给出。三组结论共享同一 analytic-localization 形式。Solid 与 liquid 是不同底环和
不同测度理论上的两个实例，并非先后完成化；经典空间的对象接口和 exactness 接口由
T.1-T.2 及第二卷命题 5.9 分开控制。证毕。

## T.4 与第三卷的接口

第三卷使用第二卷时，只能使用以下形式：

1. 若出现 solid tensor，必须说明张量在 \(D_\square\) 或 solid 模范畴中计算；
2. 若出现 analytic module，必须说明 analytic ring \((A,\mathcal M)\) 和 localization；
3. 若出现 rational cover，必须引用 rational Čech descent；
4. 若出现 Fréchet、Banach、Dolbeault 对象，必须说明逐项凝聚化的 liquid membership；
   若还比较 cohomology，必须验证 quotient 的局部提升或给出连续 splitting；若对象带
   复结构，则在 \(D(\mathbf{Liquid}_p)\) 中记录满足
   \(J^2=-\operatorname{id}\) 的内自同态；
5. 若声称 cohomology finite-dimensional 或 perfect，必须说明使用 Fredholm/Hodge、Grauert 或其他输入。

**命题 T.4（应用卷不重建主线）。** 第三卷中的 Dolbeault、Serre duality、GAGA 和 HRR/GRR 不能替代第二卷的 solid/analytic/liquid 主线证明；它们只能在 T.3 的框架内使用这些结构。

**证明.** 第三卷的对象属于复几何应用。其 analytic/liquid 类型来自第二卷；其经典
有限性和对偶来自复几何输入。若不先建立 T.3 的范畴框架，第三卷公式中的张量、Hom、
局部化、对象 membership 和 cohomology 比较均无定义或无类型保证。证毕。

## T.5 当前闭合度

按本书的输入定理型标准，第二卷已经闭合：

1. 定义闭合：每个主线对象都有所在范畴；
2. 输入闭合：每个深层结构定理都有登记；
3. 形式推论闭合：接受输入后，张量、Hom、模、下降、经典空间 membership 与严格
   cohomology 比较有证明；
4. 应用接口闭合：第三卷使用第二卷对象时有类型检查；
5. 反例边界闭合：普通张量、普通完备化、Banach/Fréchet 直接替代 liquid 等误用均已标明。

它尚未在完全自足意义下闭合：

1. solid 反射存在性未书内重证；
2. solid kernel 张量理想性未书内重证；
3. analytic ring 公理推出 localization 未书内重证；
4. rational descent 未书内重证；
5. liquid measure theory 与经典完备空间的 membership 定理未书内重证。

## T.6 出版级扩写路线

若继续推进到更接近出版级证明版，优先顺序应为：

1. 把 Q.4-Q.6 展开成 solid theory 专章；
2. 把 R.4-R.13 展开成 analytic rings 专章；
3. 把 S.1-S.3 展开成 liquid measure theory、经典空间 membership 与严格性专章；
4. 为每个输入定理补精确文献位置和证明概要；
5. 为 Q、R、S、T 的练习补完整解答。

## 练习

1. 说明 T.3 中第 4 项如何依赖 R.9。
2. 说明 T.3 中第 7 项为什么局部提升条件强于只要求闭值域。
3. 找出第三卷中一个 Dolbeault 公式，并标注它使用 T.3 的哪些部分。
4. 解释为什么 solid/analytic/liquid 不能被移到“应用附录”。
