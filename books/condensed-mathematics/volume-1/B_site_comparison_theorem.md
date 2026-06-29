# 附录 B：站点比较定理

## 本附录目标

第五章使用了站点比较定理，把 $\mathbf{CHaus}$ 上的 sheaf 与 $\mathbf{ProFin}$ 上的 sheaf 进行比较。本附录给出一个可用于本书的证明版本。

## B.1 基子站点

设 $(\mathcal C,J)$ 是站点，$\mathcal D\subset\mathcal C$ 是全子范畴。假设：

1. 对任意 $U\in\mathcal C$，存在覆盖族 $\{D_i\to U\}$，其中 $D_i\in\mathcal D$。
2. 若 $D\in\mathcal D$ 且 $U\to D$ 是 $\mathcal C$ 中态射，则存在覆盖族 $\{D_j\to U\}$，其中 $D_j\in\mathcal D$。
3. $\mathcal D$ 上的拓扑由 $\mathcal C$ 中覆盖族诱导。

称 $\mathcal D$ 为 $\mathcal C$ 的基子站点。

## B.2 限制函子

限制给出函子

$$
\rho:\operatorname{Sh}(\mathcal C,J)\to
\operatorname{Sh}(\mathcal D,J|_{\mathcal D}).
$$

若 $F$ 是 $\mathcal C$ 上的 sheaf，则 $\rho F=F|_{\mathcal D}$。因为 $\mathcal D$ 的覆盖族来自 $\mathcal C$，所以 $F|_{\mathcal D}$ 满足相同等化子条件。

## B.3 从基上延拓 sheaf

设 $G$ 是 $\mathcal D$ 上的 sheaf。对 $U\in\mathcal C$，选取 $\mathcal D$-覆盖

$$
\{D_i\to U\}_{i\in I}.
$$

定义

$$
\tilde G(U)=
\operatorname{Eq}
\left(
\prod_i G(D_i)
\rightrightarrows
\prod_{i,j}G(D_i\times_U D_j)
\right).
$$

这里若 $D_i\times_U D_j$ 不属于 $\mathcal D$，用基条件选取其 $\mathcal D$-覆盖，并用 $G$ 的 sheaf 条件计算该对象上的值。

更概念化地说，$\tilde G(U)$ 是所有从 $\mathcal D/U$ 中对象到 $G$ 的相容截面的集合，即

$$
\tilde G(U)=\varprojlim_{(D\to U)\in(\mathcal D/U)^{\operatorname{op}}}G(D).
$$

这个写法避免了覆盖选择。

## B.4 覆盖选择无关性

若 $\{D_i\to U\}$ 与 $\{D'_j\to U\}$ 是两个 $\mathcal D$-覆盖，则共同细化由纤维积

$$
D_i\times_U D'_j
$$

的 $\mathcal D$-覆盖给出。由于 $G$ 是 sheaf，匹配族在共同细化上相同当且仅当它们分别在原覆盖上相同。因此由不同覆盖定义的 $\tilde G(U)$ 自然同构。

这是基条件第二条发挥作用的地方：它保证共同细化仍可由 $\mathcal D$ 中对象覆盖。

更详细地说，记第一组覆盖给出的匹配族集合为 $M(\{D_i\})$，第二组给出的匹配族集合为 $M(\{D'_j\})$。共同细化给出两个限制映射

$$
M(\{D_i\})\longrightarrow M(\{D_i\times_U D'_j\}),
\qquad
M(\{D'_j\})\longrightarrow M(\{D_i\times_U D'_j\}).
$$

这两个映射都是双射。以第一组为例，若一个匹配族在细化上为零，则它在每个 $D_i$ 的覆盖 $\{D_i\times_U D'_j\to D_i\}_j$ 上局部为零；由于 $G$ 是 sheaf，它在 $D_i$ 上为零。故限制映射单射。若给定细化上的匹配族，则对固定 $i$，它在覆盖 $\{D_i\times_U D'_j\to D_i\}_j$ 上满足 cocycle 条件，因此由 sheaf 条件唯一粘合为 $D_i$ 上的截面；再用细化的二重交验证这些 $D_i$ 上的截面彼此相容。这给出满射。第二组同理。

## B.5 Sheaf 条件

需要证明 $\tilde G$ 是 $\mathcal C$ 上的 sheaf。设 $\{U_a\to U\}$ 是 $\mathcal C$ 中覆盖。对每个 $U_a$ 取 $\mathcal D$-覆盖 $\{D_{a\alpha}\to U_a\}$。复合族

$$
\{D_{a\alpha}\to U_a\to U\}
$$

是 $U$ 的 $\mathcal D$-覆盖。

于是 $\tilde G(U)$ 可用该复合覆盖上的匹配族计算，而每个 $\tilde G(U_a)$ 也可用 $\{D_{a\alpha}\}$ 计算。把等化子展开后，$\tilde G$ 对 $\{U_a\to U\}$ 的 sheaf 条件正是 $G$ 在 $\mathcal D$-覆盖上的 sheaf 条件。故 $\tilde G$ 是 sheaf。

函子性由切片范畴上的限制给出。若 $f:V\to U$ 是 $\mathcal C$ 中态射，则拉回给出函子

$$
\mathcal D/V\longrightarrow \mathcal D/U,
\qquad
(D\to V)\mapsto(D\to V\to U).
$$

对极限取反变限制，得到

$$
\tilde G(U)=\varprojlim_{\mathcal D/U}G(D)
\longrightarrow
\varprojlim_{\mathcal D/V}G(D)=\tilde G(V).
$$

恒等态射和复合态射的相容性来自切片范畴拉回函子的相容性，因此 $\tilde G$ 是 $\mathcal C$ 上的预层；上面的论证说明它是 sheaf。

## B.6 等价

**定理 B.1.** 在上述假设下，限制函子

$$
\rho:\operatorname{Sh}(\mathcal C,J)\to
\operatorname{Sh}(\mathcal D,J|_{\mathcal D})
$$

是范畴等价。

**证明.** 对 $G\in\operatorname{Sh}(\mathcal D)$，构造 $\tilde G$ 如上。若 $D\in\mathcal D$，则可取覆盖 $\{D\to D\}$，由 sheaf 条件得

$$
\tilde G(D)\cong G(D).
$$

故 $\rho\tilde G\simeq G$。

反过来，若 $F\in\operatorname{Sh}(\mathcal C)$，则由 $F$ 的 sheaf 条件，$F(U)$ 可由任意 $\mathcal D$-覆盖 $\{D_i\to U\}$ 上的匹配族恢复。因此由 $\rho F$ 延拓得到的 $\widetilde{\rho F}$ 与 $F$ 自然同构。两边同构与态射自然相容，故 $\rho$ 是等价。证毕。

**推论 B.2.** 若两个基子站点 $\mathcal D_1,\mathcal D_2\subset\mathcal C$ 都满足 B.1 的条件，则

$$
\operatorname{Sh}(\mathcal D_1,J|_{\mathcal D_1})
\simeq
\operatorname{Sh}(\mathcal C,J)
\simeq
\operatorname{Sh}(\mathcal D_2,J|_{\mathcal D_2}).
$$

因此 sheaf 范畴不依赖所选测试基，只依赖它生成的 Grothendieck 拓扑。

**证明.** 分别对两个包含函子使用定理 B.1。证毕。

## B.7 用于凝聚集合

在本书中先固定第一卷附录 A 的 universe，并取小骨架。于是：

$$
\mathcal C=\mathbf{CHaus}_{\mathcal U},
\qquad
\mathcal D=\mathbf{ProFin}_{\mathcal U}.
$$

第五章证明或引用了 $\mathbf{ProFin}_{\mathcal U}$ 是 $\mathbf{CHaus}_{\mathcal U}$ 的稳定基子站点，因此

$$
\operatorname{Sh}(\mathbf{CHaus}_{\mathcal U},J_{\operatorname{surj}})
\simeq
\operatorname{Sh}(\mathbf{ProFin}_{\mathcal U},J_{\operatorname{surj}}).
$$

同理，若 $\mathbf{ED}_{\mathcal U}$ 表示极不连通紧 Hausdorff 空间的小骨架，则 Gleason cover 给出 $\mathbf{ED}_{\mathcal U}$ 对 $\mathbf{CHaus}_{\mathcal U}$ 的覆盖基。于是也有

$$
\operatorname{Sh}(\mathbf{CHaus}_{\mathcal U},J_{\operatorname{surj}})
\simeq
\operatorname{Sh}(\mathbf{ED}_{\mathcal U},J_{\operatorname{surj}}).
$$

这解释了为什么正文可以在 $\mathbf{CHaus}$、$\mathbf{ProFin}$ 与 $\mathbf{ED}$ 之间切换：切换的是测试站点，不是 sheaf 范畴本身。

## 练习

**练习 B.1.** 详细证明 B.4 中共同细化导致的覆盖选择无关性。

**练习 B.2.** 在偏序范畴开集站点的情形下，把定理 B.1 解释为“sheaf 由开集基上的取值决定”。

**练习 B.3.** 检查 $\mathbf{ProFin}$ 对 $\mathbf{CHaus}$ 的基条件中，纤维积为什么仍可由 profinite 空间覆盖。
