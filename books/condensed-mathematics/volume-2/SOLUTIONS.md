# 第二卷练习答案与教师手册补充

作者：Dr. Stochastic Parrot

## 使用说明

全书统一答案见 [../SOLUTIONS.md](../SOLUTIONS.md)。本文件补第二卷中最关键的局部化、solid、analytic 和 liquid 题目。

## 1. 局部对象判别

**命题。** 对反射局部化 \(L:\mathcal C\to\mathcal C_{\mathrm{loc}}\)，态射 \(f:X\to Y\) 为局部等价，当且仅当 \(Lf\) 为等价。

**详解。** 若 \(Lf\) 为等价，则对任意 local 对象 \(Z\)，由伴随有

$$
\operatorname{Map}(Y,Z)\simeq\operatorname{Map}(LY,Z)
$$

和

$$
\operatorname{Map}(X,Z)\simeq\operatorname{Map}(LX,Z).
$$

于是 \(f\) 对所有 local 对象诱导映射空间等价。反向，取 \(Z=LX,LY\)，由 Yoneda 判别，\(Lf\) 在 local 子范畴中为等价。

## 2. 张量理想与幺半下降

**命题。** 若 \(\ker L\) 是张量理想，则局部范畴继承张量积

$$
X\otimes_LY=L(X\otimes Y).
$$

**详解。** 需证明该定义与代表元无关。若 \(X\to X'\) 是局部等价，则 cofiber \(N\in\ker L\)。张量 \(Y\) 后 cofiber 为 \(N\otimes Y\)，由张量理想性仍在 \(\ker L\)。故

$$
X\otimes Y\to X'\otimes Y
$$

仍为局部等价。对第二变量同理。结合律和交换律由原范畴中的约束经 \(L\) 得到。

## 3. Solid 与普通张量积的边界

**问题。** 为什么 solid tensor product 不能由普通张量积逐点计算？

**答案。** 普通张量积不保持无限乘积。例如

$$
(\prod_{n\ge1}\mathbb Z)\otimes\mathbb Q
\to
\prod_{n\ge1}\mathbb Q
$$

不是满射；序列 \((1,1/2,1/3,\ldots)\) 不来自左侧，因为左侧元素有统一分母。solid 对象正是为了控制 profinite 测度对象和无限乘积行为而引入。

## 4. Analytic ring cone 判别

**命题。** \(C\) 是 \((A,\mathcal M)\)-analytic 对象，当且仅当对所有测试对象 \(S\)，

$$
R\operatorname{Hom}_A(K_S^\mathcal M,C)\simeq0.
$$

**详解。** \(K_S^\mathcal M\) 是 Dirac map

$$
A[\underline S]\to\mathcal M[S]
$$

的 cone。局部对象定义为对所有被杀掉的 cone 正交。正交性用 derived Hom 表达，即上式。

## 5. Liquid 与 Fréchet 闭值域

**命题。** 若 Fréchet 复形在次数 \(q\) 有闭像，则其拓扑 cohomology 是 Hausdorff Fréchet 空间。

**详解。** \(\ker d^q\) 是 Fréchet 空间 \(E^q\) 的闭子空间。若 \(\operatorname{im}d^{q-1}\) 在 \(\ker d^q\) 中闭，则商

$$
\ker d^q/\operatorname{im}d^{q-1}
$$

是闭子空间商，故 Hausdorff 且完备。若像不闭，商拓扑非 Hausdorff，不能作为 Fredholm/Hodge 理论中的 cohomology 空间。

## 6. Solid 主定理包练习

**Q.1.** Q.3 中的对象类对小余极限封闭。

**答案。** 固定 solid 对象 \(M\)。若 \(N_\alpha\) 都满足 \(R\operatorname{Hom}(N_\alpha,M)\simeq0\)，则

$$
R\operatorname{Hom}(\operatorname*{colim}_\alpha N_\alpha,M)
\simeq
\operatorname*{lim}_\alpha R\operatorname{Hom}(N_\alpha,M)
\simeq0.
$$

所以该类对小余极限封闭。

**Q.2.** Q.8 中 associativity constraint 如何下降？

**答案。** 普通张量积有自然同构 \((X\otimes Y)\otimes Z\simeq X\otimes(Y\otimes Z)\)。局部化后两边分别变成

$$
L^\square(L^\square(X\otimes Y)\otimes Z)
$$

和

$$
L^\square(X\otimes L^\square(Y\otimes Z)).
$$

张量理想性保证把某个因子替换为其局部化不会改变最终 \(L^\square\) 后的对象，因此普通 associativity 给出 solid associativity。

**Q.3.** Q.9 的 cofiber 计算。

**答案。** 若 \(f:M\to M'\) 是局部等价，则 \(C=\operatorname{cofib}(f)\in\mathcal N_\square\)。张量 \(N\) 后得到 cofiber sequence

$$
M\otimes N\to M'\otimes N\to C\otimes N.
$$

由张量理想性 \(C\otimes N\in\mathcal N_\square\)，局部化后为零，所以第一箭头局部化后为等价。

## 7. Analytic 主定理包练习

**R.1.** R.3 的证明与 solid 情形相同：用所有 \(R\operatorname{Hom}_A(K_S^\mathcal M,-)\) 同时检测，零对象、shift、fiber、cofiber、小极限和 retract 都保持消没条件。

**R.2.** R.9 的单位约束由普通单位 \(A\otimes_A^LX\simeq X\) 经 \(L_{(A,\mathcal M)}\) 得到：

$$
L(A)\otimes_A^{L,\mathcal M}LX
=
L(A\otimes_A^LX)
\simeq
LX.
$$

**R.3.** 二开 rational cover \(X=U\cup V\) 的 descent datum 是对象 \(M_U,M_V\)，交叠 \(U\cap V\) 上的等价

$$
M_U|_{U\cap V}\simeq M_V|_{U\cap V},
$$

以及三重交叠上自动相容的 cocycle 条件；在高阶范畴中还要保留完整 Čech nerve 的高阶相容。

## 8. Liquid 主定理包练习

**S.1.** S.8：\(\ker d^q\) 是 Fréchet 空间的闭子空间；闭值域假设说明 \(\operatorname{im}d^{q-1}\) 是 \(\ker d^q\) 的闭子空间。闭子空间商为 Hausdorff Fréchet 空间。

**S.2.** S.9 的 exact triangle 来自短正合列

$$
0\to\operatorname{im}d^{q-1}\to\ker d^q\to H^q_{\mathrm{top}}(E^\bullet)\to0.
$$

若 \(E^{q-1}\twoheadrightarrow\operatorname{im}d^{q-1}\) 与
\(\ker d^q\twoheadrightarrow H^q_{\mathrm{top}}(E^\bullet)\) 对 profinite 参数族局部可
提升，则第五章命题 5.9 说明其凝聚化是短正合列，故液化复形的第 \(q\) 个 cohomology
由同一 quotient 表示。闭值域只保证这些商是 Hausdorff Fréchet 空间，不提供局部提升。

**S.3.** 若连续线性映射像不闭，例如某些紧算子在无限维 Banach 空间中的像，则 quotient 非 Hausdorff。此时没有 Fréchet 短正合列，S.3 的 exactness 假设不能应用。

## 9. 统一闭包练习

**T.1.** T.3 的第 4 项依赖 R.9：analytic 张量定义为 ordinary tensor 后 analyticization，因此 analyticization 与张量相容来自 analytic kernel 的张量理想性。

**T.2.** 闭值域使拓扑 cohomology 成为 Hausdorff Fréchet 空间；第 7 项还需要相关
quotient 对 profinite 参数族局部可提升。后者严格强于闭值域，连续 Hodge/Green
splitting 是同时满足两项要求的充分条件。

**T.3.** 第三卷 Dolbeault 复形使用 T.3 的第 6-8 项：Fréchet 项的 liquid
membership、连续 Hodge splitting 给出的凝聚严格性、Fredholm cohomology 和
perfect 性。

## 10. Solidification 反射存在性练习

**V.1.** 局部对象对 retract 封闭。

**答案。** 若 \(X\) 是 \(Y\) 的 retract，存在 \(i:X\to Y\) 和 \(r:Y\to X\)，且 \(r i=\operatorname{id}_X\)。对任意 \(K_f\)，

$$
R\operatorname{Hom}(K_f,X)
$$

是 \(R\operatorname{Hom}(K_f,Y)\) 的 retract。若后者为零，则其 retract 也为零，因此 \(X\) 局部。

**V.2.** 在稳定范畴中，\(f\) 被局部化为等价当且仅当 \(\operatorname{cofib}(f)\) 被局部化为零。

**答案。** 对 cofiber sequence

$$
X\to Y\to \operatorname{cofib}(f)
$$

应用 exact functor \(L\)。在稳定范畴中，\(Lf\) 为等价当且仅当其 cofiber 为零，而该 cofiber 正是 \(L(\operatorname{cofib}(f))\)。

**V.3.** 必须固定 profinite 空间小骨架的原因。

**答案。** 集合生成局部化定理要求被倒置的态射是一组而不是 proper class。所有 profinite 空间本身形成大类；固定 universe 后取同构类代表的小骨架，才得到一组 Dirac-to-measure maps。

**V.4.** V.9 不足以推出 solid tensor product。

**答案。** V.9 只给出反射局部化和 exact 左伴随。要把

$$
L^\square(M\otimes^LN)
$$

做成局部范畴上的张量积，还需要 kernel 对张量封闭。否则替换 \(M\) 为局部等价对象时，张量 \(N\) 后可能不再是局部等价。

## 11. Solid 核张量理想性练习

**W.1.** \(\mathcal A_X\) 对 cofiber 封闭。

**答案。** 若 \(M'\to M\to M''\) 是 cofiber sequence，且 \(M'\otimes X\) 与 \(M\otimes X\) 在 \(\mathcal N\) 中，则

$$
M'\otimes X\to M\otimes X\to M''\otimes X
$$

仍是 cofiber sequence。因为 \(\mathcal N\) 是 localizing subcategory，对 cofiber 封闭，故 \(M''\otimes X\in\mathcal N\)。

**W.2.** 自由凝聚对象的张量公式。

**答案。** 对任意凝聚阿贝尔群 \(A\)，双线性映射

$$
\mathbb Z[\underline S]\times\mathbb Z[\underline T]\to A
$$

等价于集合映射 \(\underline S\times\underline T\to A\)，也等价于凝聚集合映射 \(\underline{S\times T}\to A\)。由自由对象泛性质，代表对象为 \(\mathbb Z[\underline{S\times T}]\)。

**W.3.** 普通张量积不保持无限乘积的障碍。

**答案。** 例如 \((\prod_n\mathbb Z)\otimes\mathbb Q\to\prod_n\mathbb Q\) 不是满射，因为左侧元素有统一分母，而右侧序列 \((1,1/2,1/3,\ldots)\) 没有统一分母。profinite 测度对象本质涉及无限乘积，因此必须使用 solid 修正。

**W.4.** W.6 推出 Q.5。

**答案。** Q.5 的内容是 solid kernel \(\mathcal N_\square\) 为张量理想。W.6 在三个输入假设下正证明了这件事：凝聚生成族、profinite 测度张量计算和 kernel 等于由 \(K_S\) 生成的 localizing subcategory。

## 12. Analytic localization 练习

**X.1.** 证明 X.2。

**答案。** 对 cofiber sequence

$$
A[\underline S]\to\mathcal M[S]\to K_S^{\mathcal M}
$$

取内部派生 Hom \(R\underline{\operatorname{Hom}}_A(-,C)\)，得到
$D(\mathbf{CondAb})$ 中的 fiber sequence。映射

$$
R\underline{\operatorname{Hom}}_A(\mathcal M[S],C)
\to R\underline{\operatorname{Hom}}_A(A[\underline S],C)
$$

是等价，当且仅当其 fiber
\(R\underline{\operatorname{Hom}}_A(K_S^{\mathcal M},C)\) 为零。

**X.2.** analytic 张量代表元无关。

**答案。** 若 \(M\to M'\) 是 analytic 局部等价，其 cofiber \(N\) 属于 kernel。若 kernel 是张量理想，则 \(N\otimes_A^LX\) 仍在 kernel 中，所以

$$
M\otimes_A^LX\to M'\otimes_A^LX
$$

局部化后为等价。第二变量同理。

**X.3.** solid 特例中的 \(K_S^{\mathcal M}\)。

**答案。** 取 \(A=\mathbb Z\)，\(\mathcal M[S]=\mathbb Z^\square[S]\)。则

$$
K_S^{\mathcal M}
=\operatorname{cofib}(\mathbb Z[\underline S]\to\mathbb Z^\square[S]),
$$

这正是 solid cone \(K_S\)。

**X.4.** pre-analytic datum 不足以推出 rational descent。

**答案。** pre-analytic datum 只给出对象 \(A[\underline S]\to\mathcal M[S]\) 和形式局部化候选；它不保证 rational localization 存在、不保证 rational intersections 与 iterated localization 相容，也不保证 Čech 复形 acyclic。descent 需要这些几何相容性。

## 13. Rational descent 练习

**Y.1.** 二开覆盖的 descent datum。

**答案。** 对 \(X=U\cup V\)，一个 datum 包含 \(M_U\in D(U)\)、\(M_V\in D(V)\)，以及交叠上的等价

$$
\varphi:M_U|_{U\cap V}\simeq M_V|_{U\cap V}.
$$

在高阶范畴中还要包含三重交叠上的相容同伦；二开覆盖的三重交叠仍是 \(U\cap V\)，条件退化为 \(\varphi\) 与自身限制的同伦相容。

**Y.2.** Y.2 的证明。

**答案。** totalization 范畴中的 mapping space 等于 cosimplicial mapping spaces 的 totalization：

$$
\operatorname{Map}_{\operatorname{Tot}\mathcal C^\bullet}(M,N)
\simeq
\operatorname{Tot}\operatorname{Map}_{\mathcal C^\bullet}(M^\bullet,N^\bullet).
$$

若全局 mapping space 到右侧为等价，则限制函子在任意两个对象之间诱导 mapping space 等价，因此全忠实。

**Y.3.** 为什么需要 compact generation。

**答案。** descent 经常先在一组生成对象上验证。若没有生成性假设，从生成对象的 mapping descent 不能推广到任意 colimit、cofiber 和 retract 构造出的对象，也无法保证 totalization 范畴由可 glue 的对象生成。

**Y.4.** rational descent 与普通 sheaf descent 的差别。

**答案。** 普通 sheaf descent 只讨论固定站点上的对象值如何沿覆盖 glue。rational descent 还要证明 rational localization 后的 analytic module categories 本身组成可下降的范畴值对象，并且 analytic tensor、局部化和生成对象都与 rational intersections 相容。

## 14. Liquid realization 练习

**Z.1.** compact Hausdorff 有限联合满射是 quotient map。

**答案。** 若 \(q:\coprod_iS_i\to S\) 是有限联合满射，源空间紧而目标 Hausdorff。连续满射从紧空间到 Hausdorff 空间是闭映射，因此是 quotient map。

**Z.2.** Z.1 的 sheaf 条件。

**答案。** 给定连续映射 \(f_i:S_i\to E\) 在 \(S_i\times_SS_j\) 上相容，它们唯一拼成集合映射 \(f:S\to E\)。因为 \(\coprod_iS_i\to S\) 是 quotient map，且 \(f\circ q\) 连续，所以 \(f\) 连续。这证明 \(\underline E\) 满足覆盖等化子条件。

**Z.3.** 像不闭导致 quotient 非 Hausdorff 的例子。

**答案。** 取紧算子

$$
T:\ell^2\to\ell^2,\qquad T(x_1,x_2,\ldots)=(x_1,x_2/2,x_3/3,\ldots).
$$

其像稠密但不闭。于是 \(\ell^2/\operatorname{im}T\) 的零点闭包不是零点本身，商空间非 Hausdorff。

**Z.4.** Z.11 中 perfect 性的最后一步。

**答案。** 若 \(H^q(X,E)\) 有限维，则由 Z.4

$$
\mathcal L_p(H^q(X,E))\simeq\mathbb R_{\mathrm{liq}}^{\oplus n}.
$$

单位对象 perfect，而 perfect 对象对有限直和封闭，所以该 realization perfect。由 Z.11 的 cohomology 比较，Dolbeault liquid cohomology 也 perfect。

## 15. Scholze 与 Clausen-Scholze 核心定理图谱练习

**AA.1.** 为什么 solid kernel 张量理想性是 solid tensor product 的必要条件？

**答案。** solid tensor 定义为

$$
M\otimes^{L,\square}N=L^\square(M\otimes^LN).
$$

若 \(M\to M'\) 是 solid 局部等价，其 cofiber \(K\) 在 \(\ker L^\square\)。为了使张量只依赖 \(L^\square M\) 而非代表元，必须有 \(K\otimes^LN\in\ker L^\square\)。这正是 kernel 为张量理想的要求。

**AA.2.** ordinary sheaf descent 与 category-valued descent 的区别。

**答案。** ordinary sheaf descent 断言一个 sheaf 的截面可由覆盖上的截面和交叠相容数据恢复。category-valued descent 断言整个范畴 \(D(A,\mathcal M)\) 可由 rational cover 上的一族范畴及其高阶交叠 totalization 恢复；它同时要求对象、态射空间、张量和局部化结构都满足 descent。

**AA.3.** 为什么“\(\underline E\) 是 liquid”不能推出每个连续满射
\(E\to F\) 凝聚化后都是 epimorphism？

**答案。** 凝聚化只给 sheaf

$$
S\mapsto\operatorname{Cont}(S,E),
$$

它记录连续族。CS26 的 membership 定理可进一步证明 \(\underline E\) 满足
\(\mathcal M_{<p}\) 的 Hom 判别，但 epimorphism 是关于态射的局部提升性质：每个
\(S\to F\) 必须在有限覆盖后提升到 \(E\)。对象 membership 不含这个量词，因此不能
推出 exactness；有连续截面时局部提升才自动成立。

**AA.4.** 以 Serre duality 为例列出三段式。

**答案。** classical 输入：紧复流形或相干层上的 Serre/Grothendieck duality 和 trace theorem。condensed/analytic 输入：Clausen-Scholze 建模、\(f^!\) 与 trace 在 analytic/liquid 范畴中的相容。书内形式后果：第三卷附录 J、AD、AQ 证明接受输入后，链级配对、derived duality、Ext-Serre pairing 和有限维 perfect pairing 如何互相推出。
