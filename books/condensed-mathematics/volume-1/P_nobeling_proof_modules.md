# 附录 P：Nöbeling 定理的证明模块

## P.0 目标

solid 理论使用 Nöbeling 定理：

$$
C(S,\mathbb Z)
$$

对任意 profinite 空间 \(S\) 是自由阿贝尔群。附录 F 已说明该定理在 solid 计算中的位置。本附录把证明拆成可读模块：有限层、可数层、一般层的超限归纳，以及该定理进入 solid 对象的具体公式。

完整一般情形仍引用 Asgeirsson 的形式化证明；本附录给出有限、可数情形的书内证明，并说明一般情形的归纳义务。

## P.1 有限与可数 profinite 情形

若 \(S\) 是有限离散集合，则

$$
C(S,\mathbb Z)\cong\mathbb Z^S
$$

是有限自由阿贝尔群。

设

$$
S=\varprojlim_n S_n
$$

为可数逆极限，其中每个 \(S_n\) 有限离散，过渡映射 \(S_{n+1}\to S_n\) 满射。则

$$
C(S,\mathbb Z)=\varinjlim_n C(S_n,\mathbb Z),
$$

因为从紧空间到离散 \(\mathbb Z\) 的连续函数有有限像，其纤维 clopen，故通过某个有限商 \(S_n\) 分解。

**命题 P.1（可数层自由性）.** 在上述可数表示下，\(C(S,\mathbb Z)\) 是自由阿贝尔群。

**证明.** 每个

$$
C(S_n,\mathbb Z)
$$

有限自由。满射 \(S_{n+1}\to S_n\) 诱导单射

$$
C(S_n,\mathbb Z)\hookrightarrow C(S_{n+1},\mathbb Z).
$$

该单射在阿贝尔群中分裂：选择每个 fiber 中一点，得到重traction \(S_n\leftarrow S_{n+1}\)，从而得到群同态

$$
C(S_{n+1},\mathbb Z)\to C(S_n,\mathbb Z)
$$

左逆于拉回。于是

$$
C(S_{n+1},\mathbb Z)\cong C(S_n,\mathbb Z)\oplus Q_n
$$

且 \(Q_n\) 有限自由。归纳得

$$
\varinjlim_nC(S_n,\mathbb Z)
\cong
C(S_0,\mathbb Z)\oplus\bigoplus_{n\ge0}Q_n,
$$

故自由。证毕。

## P.2 一般 profinite 空间的超限过滤

一般 profinite 空间可写为有限离散商的 cofiltered inverse limit。对应函数群可写为有限商函数群的 filtered colimit：

$$
C(S,\mathbb Z)=\varinjlim_{S\to T,\ T\ \mathrm{finite}}C(T,\mathbb Z).
$$

困难在于该 filtered system 不再是可数链，不能直接用 P.1。

**输入定理 P.2（Nöbeling-Asgeirsson 超限过滤）.** 对任意 profinite \(S\)，存在 ordinal \(\lambda\) 与子群过滤

$$
0=F_0\subset F_1\subset\cdots\subset F_\alpha\subset\cdots\subset C(S,\mathbb Z)
$$

满足：

1. \(\bigcup_{\alpha<\lambda}F_\alpha=C(S,\mathbb Z)\)；
2. 极限序数处 \(F_\beta=\bigcup_{\alpha<\beta}F_\alpha\)；
3. 每个后继商 \(F_{\alpha+1}/F_\alpha\) 是自由阿贝尔群；
4. 每个包含 \(F_\alpha\subset F_{\alpha+1}\) 是纯嵌入或带有可控制的分裂数据。

由此 \(C(S,\mathbb Z)\) 自由。

**证明边界.** 核心是构造 well-ordered clopen 分解和支撑控制，使每一步新增函数可由新的 clopen 层生成，并保证旧关系不被新层破坏。Asgeirsson 的形式化证明把该过程写成 ordinal induction。

## P.3 从过滤到自由性的代数引理

**引理 P.3（超限自由扩张）.** 设 \(A=\bigcup_{\alpha<\lambda}F_\alpha\)，其中过滤满足 P.2 的 2-3，并且每个短正合列

$$
0\to F_\alpha\to F_{\alpha+1}\to F_{\alpha+1}/F_\alpha\to0
$$

分裂。则 \(A\) 自由。

**证明.** 对 \(\alpha\) 超限归纳选择 \(F_\alpha\) 的基。后继步中，由分裂性有

$$
F_{\alpha+1}\cong F_\alpha\oplus Q_\alpha,
$$

其中 \(Q_\alpha\) 自由。取 \(F_\alpha\) 的基与 \(Q_\alpha\) 的基并为 \(F_{\alpha+1}\) 的基。极限步中，取此前所有新增基的并集；每个元素属于某个早期 \(F_\alpha\)，线性关系也落在某个 \(F_\alpha\)，故基性质保持。最终所有新增基的并集生成 \(A\)，且线性无关。证毕。

**边界 P.4.** P.2 中实际得到的嵌入控制比“每步分裂”更精细；P.3 说明只要有足够的分裂或纯性控制，超限归纳会推出自由性。

## P.4 solid 计算中的使用

对 profinite \(S\)，Scholze 定义的 solid 自由对象满足

$$
\mathbb Z^\square[S]\simeq \prod_{i\in I}\underline{\mathbb Z}
$$

其中 \(I\) 是 \(C(S,\mathbb Z)\) 的一组基。

**命题 P.5（基选择后的 Hom 公式）.** 若 \(C(S,\mathbb Z)\cong\bigoplus_{i\in I}\mathbb Z\)，则对任意凝聚阿贝尔群 \(A\)，有自然映射

$$
\operatorname{Hom}\left(\prod_{i\in I}\underline{\mathbb Z},A\right)
\to
\operatorname{Hom}_{\mathbb Z}(C(S,\mathbb Z),A(*))
$$

其构造依赖于乘积对象的泛性质和基的选择。

**证明.** 从 \(\prod_I\underline{\mathbb Z}\) 到 \(A\) 的态射等价于与所有坐标投影相容的族。基 \(e_i\) 把 \(C(S,\mathbb Z)\) 写成 \(\bigoplus_I\mathbb Z\)。给定态射，可取每个坐标的像，得到 \(A(*)\) 中的族，从而给 \(C(S,\mathbb Z)\to A(*)\) 的群同态。反向由自由基给出坐标族，再由乘积泛性质给态射。证毕。

**边界 P.6.** \(C(S,\mathbb Z)\) 的基不是典范的，因此 \(\mathbb Z^\square[S]\) 与某个乘积 \(\prod_I\underline{\mathbb Z}\) 的同构依赖选择。solid 理论中的构造应以泛性质为准，而非以某个基为准。

## 练习

1. 证明连续映射 \(S\to\mathbb Z\) 通过有限离散商分解。
2. 在 P.1 中写出 \(Q_n\) 的秩。
3. 证明 P.3 的极限步中线性无关性保持。
4. 说明为什么 Nöbeling 定理对构造非零 solid 对象是必要输入。
