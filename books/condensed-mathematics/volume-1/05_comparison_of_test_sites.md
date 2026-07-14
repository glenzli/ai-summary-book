# 第五章：测试站点的比较

定义凝聚对象时，任意紧 Hausdorff 空间都是自然测试对象；实际计算时，这个范畴却
过于宽广。profinite 空间能由有限离散商逼近，覆盖和纤维积因而更容易展开。真正的
问题不是把一个函子随意限制到子范畴，而是证明限制后仍保留全部 sheaf 数据，并能从
较小站点唯一重建原对象。

第一章的站点语言、第二章的两类测试空间以及第三章的凝聚定义在这里汇合。我们使用
稳定基形式的站点比较定理，检查 profinite 对象覆盖每个紧 Hausdorff 对象且拉回后仍
局部落回该基，从而得到 sheaf 范畴等价。比较定理的一般证明展开在
[附录 B](B_site_comparison_theorem.md)，第四卷第二章则把同一检查拆成可形式化步骤。

## 5.1 为什么要换测试站点

第三章采用定义

$$
\mathbf{CondSet}
=
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}}).
$$

这个定义自然，因为任意拓扑空间 $T$ 都给出

$$
\underline T(S)=\operatorname{Cont}(S,T),
\qquad S\in \mathbf{CHaus}.
$$

但是在计算中，$\mathbf{CHaus}$ 太大。紧 Hausdorff 空间可以很复杂，纤维积、覆盖和连续映射的结构不总是容易掌控。相比之下，profinite 空间有更强的离散近似：

$$
S\simeq \varprojlim_i S_i,
$$

其中 $S_i$ 是有限离散集合。这使它们更接近代数对象。

因此，我们希望知道：是否可以只在 profinite 空间上测试 sheaf？

答案是肯定的，但严格说法不是“随便取一个大子范畴”。本章固定第一卷附录 A 的 universe 约定，并在该 universe 内取 $\mathbf{CHaus}_{\mathcal U}$ 与 $\mathbf{ProFin}_{\mathcal U}$ 的小骨架。包含

$$
i:\mathbf{ProFin}_{\mathcal U}\hookrightarrow \mathbf{CHaus}_{\mathcal U}
$$

在有限联合满射拓扑下满足稳定基条件：每个紧 Hausdorff 空间有 profinite 满射覆盖，profinite 对象在紧 Hausdorff 底上的纤维积仍是 profinite，并且任意覆盖在 profinite 底上可由 profinite 覆盖共同细化。

## 5.2 基与站点比较

**定义 5.1（稳定基子站点）.** 设 $(\mathcal C,J)$ 为小站点，$\mathcal D\subset\mathcal C$ 为全子范畴。称 $\mathcal D$ 是 $\mathcal C$ 的稳定基子站点，如果满足：

1. $\mathcal D$ 的拓扑由 $\mathcal C$ 限制得到。
2. 对每个 $U\in\mathcal C$，存在覆盖族 $\{D_i\to U\}$，其中 $D_i\in\mathcal D$。
3. 若 $D_1,D_2\in\mathcal D$ 且有映射 $D_1\to U\leftarrow D_2$，其中 $U\in\mathcal C$，则 $D_1\times_UD_2$ 存在且属于 $\mathcal D$。
4. 若 $D\in\mathcal D$ 且 $\{U_a\to D\}$ 是 $\mathcal C$ 中覆盖，则存在 $\mathcal D$ 中覆盖 $\{D_b\to D\}$ 共同细化它。

第四条的意思是：在 $\mathcal D$ 中检验 sheaf 条件时，不会丢失 $\mathcal C$ 中覆盖的交叠信息。若只知道 $\mathcal D$ 的对象覆盖所有对象，而没有纤维积和共同细化条件，则限制到 $\mathcal D$ 的数据一般不足以恢复 $\mathcal C$ 上的 sheaf。

**定理 5.2（站点比较定理，稳定基版本）.** 设 $\mathcal D\subset \mathcal C$ 是站点 $(\mathcal C,J)$ 的稳定基子站点。则限制函子

$$
\operatorname{Sh}(\mathcal C,J)
\longrightarrow
\operatorname{Sh}(\mathcal D,J|_{\mathcal D})
$$

是范畴等价。

**证明.** sheaf 由其在覆盖基上的取值决定。给定 $\mathcal C$ 上的 sheaf，限制到 $\mathcal D$ 后，对 $\mathcal D$ 中的覆盖族仍满足同一等化子条件，因此得到 $\mathcal D$ 上的 sheaf。反过来，给定 $\mathcal D$ 上的 sheaf $F$，对 $U\in\mathcal C$ 选择 $\mathcal D$-覆盖 $\{D_i\to U\}$，并定义 $F(U)$ 为匹配族集合：

$$
F(U)=
\operatorname{Eq}
\left(
\prod_i F(D_i)
\rightrightarrows
\prod_{i,j}F(D_i\times_U D_j)
\right),
$$

其中 $D_i\times_UD_j\in\mathcal D$ 由稳定基条件保证。若采用附录 B 的较弱版本，则该交对象只需能被 $\mathcal D$-对象覆盖，公式改写为切片范畴上的极限。共同细化条件保证该定义与覆盖选择无关，并满足 $\mathcal C$ 上的 sheaf 条件。完整证明需要检查自然性、覆盖独立性和拟逆，见附录 B。证毕。

**注 5.3.** 定理 5.2 是 sheaf 理论的标准工具。本书后续使用它时，会明确指出所用子范畴是否确实构成稳定基，或引用附录 B 中允许交对象再被覆盖的弱版本。

## 5.3 Profinite 空间是一个基

要把 $\mathbf{CHaus}$ 换成 $\mathbf{ProFin}$，需要一个非平凡拓扑事实。

**定理 5.4（profinite 覆盖）.** 对每个紧 Hausdorff 空间 $K$，存在 profinite 空间 $P$ 和满射连续映射

$$
P\to K.
$$

此外，也存在极不连通紧 Hausdorff 空间 $E$ 和满射 $E\to K$。

**证明说明.** profinite 覆盖可由 Stone 型表示定理得到；极不连通覆盖来自 Gleason 关于 compact Hausdorff 范畴中投射覆盖的理论。完整证明超出本章范围，后续第六章会讨论极不连通空间与投射性。本书把该定理作为引用结果使用，依赖来源见 [SOURCES.md](SOURCES.md)。

**引理 5.5.** 若 $P,Q$ 是 profinite 空间，且有连续映射

$$
P\to K,\qquad Q\to K
$$

到紧 Hausdorff 空间 $K$，则纤维积 $P\times_K Q$ 是 profinite 空间。

**证明.** 纤维积可看作 $P\times Q$ 的闭子空间：

$$
P\times_K Q
=
\{(p,q)\in P\times Q\mid f(p)=g(q)\}.
$$

因为 $K$ Hausdorff，对角线 $\Delta_K\subset K\times K$ 闭，故上述集合是连续映射

$$
P\times Q\to K\times K
$$

下 $\Delta_K$ 的逆像，因此闭。$P\times Q$ 是 profinite 空间，闭子空间仍 profinite。证毕。

**命题 5.6.** 在固定 universe 和小骨架约定下，$\mathbf{ProFin}_{\mathcal U}$ 是 $\mathbf{CHaus}_{\mathcal U}$ 在有限联合满射拓扑下的稳定基子站点。

**证明.** 拓扑由限制得到，这是定义。

覆盖性由定理 5.4 给出：任意 $K\in\mathbf{CHaus}_{\mathcal U}$ 有 profinite 空间 $P$ 满射到 $K$，于是 $\{P\to K\}$ 是覆盖。

纤维积稳定性由引理 5.5 给出：若 $P,Q$ 是 profinite，且 $P\to K\leftarrow Q$ 是紧 Hausdorff 底上的两条映射，则 $P\times_KQ$ 是 profinite。

共同细化条件如下。设 $P\in\mathbf{ProFin}_{\mathcal U}$，且

$$
\{K_a\to P\}_{a=1}^n
$$

是 $\mathbf{CHaus}_{\mathcal U}$ 中有限联合满射覆盖。对每个 $a$，由定理 5.4 取 profinite 满射 $Q_a\to K_a$。则复合族

$$
\{Q_a\to K_a\to P\}_{a=1}^n
$$

仍是有限联合满射覆盖，并且所有 $Q_a$ 都属于 $\mathbf{ProFin}_{\mathcal U}$。它共同细化原覆盖。因此 $\mathbf{ProFin}_{\mathcal U}$ 满足稳定基条件。证毕。

## 5.4 凝聚集合的 profinite 定义

由定理 5.2 得到：

**定理 5.7.** 限制函子给出范畴等价

$$
\operatorname{Sh}(\mathbf{CHaus}_{\mathcal U},J_{\operatorname{surj}})
\simeq
\operatorname{Sh}(\mathbf{ProFin}_{\mathcal U},J_{\operatorname{surj}}).
$$

**证明.** 定理 5.6 说明全子范畴
$\mathbf{ProFin}_{\mathcal U}\subset\mathbf{CHaus}_{\mathcal U}$
满足定理 5.2 的稳定基条件：每个紧 Hausdorff 空间有来自 profinite
空间的覆盖，并且任意有限覆盖在拉回后仍可由 profinite 覆盖共同细化。
因此定理 5.2 的限制函子正是上式中的等价。证毕。

因此也可以定义

$$
\mathbf{CondSet}
\simeq
\operatorname{Sh}(\mathbf{ProFin},J_{\operatorname{surj}}).
$$

在 profinite 口径下，一个凝聚集合是反变函子

$$
X:\mathbf{ProFin}^{\operatorname{op}}\to\mathbf{Set}
$$

满足有限联合满射覆盖的 sheaf 条件：

$$
X(S)\to \prod_i X(S_i)
\rightrightarrows
\prod_{i,j}X(S_i\times_S S_j)
$$

是等化子。

**注 5.8.** 有些资料直接把凝聚集合定义为 profinite 集合站点上的 sheaf。本书从 $\mathbf{CHaus}$ 开始，是为了让拓扑空间 $T\mapsto\underline T$ 的构造更直观；定理 5.7 说明这不会改变最终范畴。

## 5.5 口径差异如何使用

后续章节采用如下约定：

- 写定义和直观例子时，可以使用 $\mathbf{CHaus}$。
- 做计算时，优先使用 $\mathbf{ProFin}$ 或极不连通空间。
- 若一个断言只在 profinite 测试对象上验证，必须说明它如何由站点比较提升到 $\mathbf{CondSet}$。

例如，对拓扑空间 $T$，其凝聚集合可写为

$$
\underline T(S)=\operatorname{Cont}(S,T),
\qquad S\in\mathbf{CHaus},
$$

也可限制为

$$
\underline T(P)=\operatorname{Cont}(P,T),
\qquad P\in\mathbf{ProFin}.
$$

第二种写法通常更便于计算，但第一种写法更清楚地说明它来自拓扑空间。

## 5.6 从大站点到可计算测试

profinite 覆盖与稳定基条件给出等价

$$
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}})
\simeq
\operatorname{Sh}(\mathbf{ProFin},J_{\operatorname{surj}}).
$$

因此在 $\mathbf{CHaus}$ 上定义对象、在 $\mathbf{ProFin}$ 上计算取值并不产生两套
理论；限制函子携带全部粘合信息。证明中的关键拓扑输入是每个紧 Hausdorff 空间都有
profinite 满射覆盖。若继续要求覆盖本身具有提升性质，就自然得到下一章的极不连通
空间，它们会把 sheaf 意义的局部提升变成真正的全局提升。

## 练习

**练习 5.1.** 证明引理 5.5 中 $P\times_K Q$ 是 $P\times Q$ 的闭子空间。

**练习 5.2.** 假设定理 5.4，证明任意紧 Hausdorff 空间 $K$ 的可表凝聚集合 $\underline K$ 由其在 profinite 空间上的取值决定。

**练习 5.3.** 查阅一般 sheaf 理论，写出站点比较定理的完整证明。

**练习 5.4.** 解释为什么“全子范畴对象覆盖所有对象”还不足以推出 sheaf 范畴等价，还需要拉回后的覆盖条件。
