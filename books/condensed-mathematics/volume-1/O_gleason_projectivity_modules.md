# 附录 O：Gleason 投射性定理的证明模块

## O.0 目标

第一卷第六、七章使用 Gleason 定理：极不连通紧 Hausdorff 空间正是 \(\mathbf{CHaus}\) 中关于满射的投射对象。附录 J 已构造 regular open Stone cover。本附录把 Gleason 定理的证明拆成 Boolean algebra、regular open algebra 和选择提升三个模块。

完整 Gleason 定理仍作为拓扑输入；本附录证明可在书内完成的代数部分，并精确标出剩余拓扑选择定理。

## O.1 Regular open algebra 的完备性

设 \(X\) 为紧 Hausdorff 空间。记

$$
\operatorname{RO}(X)=\{U\subset X\mid U=\operatorname{int}\overline U\}.
$$

布尔运算定义为

$$
U\wedge V=U\cap V,\qquad
U^\complement=\operatorname{int}(X\setminus U),
$$

以及

$$
U\vee V=\operatorname{int}\overline{U\cup V}.
$$

**命题 O.1.** \(\operatorname{RO}(X)\) 是完备 Boolean algebra。

**证明.** 有限布尔运算已在附录 J 证明。对任意族 \(\{U_i\}_{i\in I}\subset\operatorname{RO}(X)\)，定义

$$
\bigvee_{i\in I}U_i
=
\operatorname{int}\overline{\bigcup_{i\in I}U_i}.
$$

该集合 regular open。它含每个 \(U_i\)。若 \(V\in\operatorname{RO}(X)\) 且 \(U_i\subset V\) 对所有 \(i\) 成立，则

$$
\bigcup_iU_i\subset V\subset\overline V,
$$

从而

$$
\overline{\bigcup_iU_i}\subset\overline V.
$$

取内部得 \(\bigvee_iU_i\subset\operatorname{int}\overline V=V\)。所以这是最小上界。任意 meet 由 De Morgan 公式定义：

$$
\bigwedge_iU_i=\left(\bigvee_iU_i^\complement\right)^\complement.
$$

证毕。

## O.2 极不连通时 regular open 等于 clopen

**命题 O.2.** 若 \(E\) 是极不连通紧 Hausdorff 空间，则

$$
\operatorname{RO}(E)=\operatorname{Clop}(E).
$$

**证明.** 若 \(C\) clopen，则 \(\operatorname{int}\overline C=C\)，故 \(C\in\operatorname{RO}(E)\)。反向，取 \(U\in\operatorname{RO}(E)\)。因 \(E\) 极不连通，\(\overline U\) 开。它又是闭集，所以 \(\overline U\) clopen。由 regular open 性，

$$
U=\operatorname{int}\overline U=\overline U.
$$

因此 \(U\) clopen。证毕。

**推论 O.3.** 若 \(E\) 极不连通紧 Hausdorff，则 \(\operatorname{Clop}(E)\) 是完备 Boolean algebra。

**证明.** 由 O.1 与 O.2。证毕。

## O.3 Sikorski extension 与 Stone 端

**输入定理 O.4（Sikorski extension theorem）.** 设 \(B\) 是 Boolean algebra，\(A\subset B\) 是子 Boolean algebra。若 \(C\) 是完备 Boolean algebra，则任意 Boolean homomorphism

$$
\varphi:A\to C
$$

延拓为 Boolean homomorphism

$$
\widetilde\varphi:B\to C.
$$

该定理是 Gleason 投射性证明中的代数核心。

**命题 O.5（Stone 空间中的投射性）.** 若 \(E=\operatorname{Stone}(C)\)，其中 \(C\) 是完备 Boolean algebra，则 \(E\) 在 Stone 空间范畴中关于满射投射。

**证明.** Stone 对偶把 Stone 空间中的连续满射

$$
p:Y\to X
$$

变为 Boolean algebra 的单射

$$
p^\ast:\operatorname{Clop}(X)\hookrightarrow\operatorname{Clop}(Y).
$$

给定 \(f:E\to X\)，对偶为

$$
f^\ast:\operatorname{Clop}(X)\to C.
$$

由 O.4，\(f^\ast\) 沿 \(p^\ast\) 延拓为

$$
g^\ast:\operatorname{Clop}(Y)\to C.
$$

再由 Stone 对偶，\(g^\ast\) 对应 \(g:E\to Y\)，且

$$
p\circ g=f.
$$

证毕。

## O.4 从 Stone 端到 compact Hausdorff 端

一般 compact Hausdorff 空间不由 clopen 代数恢复，因此 O.5 还不是完整 Gleason 定理。需要 regular open cover 与 compact Hausdorff 表示的相容性。

**输入定理 O.6（Gleason lifting theorem）.** 设 \(E\) 是极不连通紧 Hausdorff 空间，\(p:Y\to X\) 是紧 Hausdorff 空间中的连续满射，\(f:E\to X\) 连续。则存在连续 \(g:E\to Y\)，使

$$
p\circ g=f.
$$

**证明边界.** 一条证明路线是：把 \(X,Y\) 的 regular open algebra 与其 Stone cover 联系起来，用 O.4 在完备 Boolean algebra 层面延拓，再证明所得 Stone 端映射下降到给定紧 Hausdorff 空间。下降步骤需要处理 regular open cover 的纤维等价关系和紧 Hausdorff 商。

本书不在第一卷重证该下降定理。

## O.5 反向：投射推出极不连通

**定理 O.7（反向证明）.** 若 \(E\) 是紧 Hausdorff 空间，并且 \(E\) 关于紧 Hausdorff 满射投射，则 \(E\) 极不连通。

**证明.** 取附录 J 构造的 Gleason cover

$$
\pi:E_E\to E,
$$

其中 \(E_E=\operatorname{Stone}(\operatorname{RO}(E))\) 极不连通。由投射性，存在

$$
s:E\to E_E
$$

使

$$
\pi\circ s=\operatorname{id}_E.
$$

所以 \(E\) 是极不连通空间 \(E_E\) 的 retract。极不连通性对紧 Hausdorff retract 封闭：若 \(r:Z\to E\)、\(s:E\to Z\) 且 \(r s=\operatorname{id}\)，对 \(E\) 中开集 \(U\)，有

$$
\overline U=r(\overline{s(U)})
$$

并且 \(\overline{s(U)}\) 在 \(Z\) 中开；用商映射 \(r\) 的闭性和 retract 等式可验证 \(\overline U\) 在 \(E\) 中开。故 \(E\) 极不连通。证毕。

## O.6 本书使用形式

第一卷实际需要的是以下推论。

**推论 O.8.** 若 \(E\) 极不连通紧 Hausdorff，则自由凝聚阿贝尔群

$$
\mathbb Z[\underline E]
$$

是 \(\mathbf{CondAb}\) 中投射对象。

**证明.** 对 sheaf 满射 \(A\to B\) 和态射 \(\mathbb Z[\underline E]\to B\)，由自由对象泛性质等价于给 \(b\in B(E)\)。sheaf 满射给覆盖 \(p:E'\to E\) 使 \(b|_{E'}\) 提升到 \(A(E')\)。取 \(E'\) 的有限不交并替换，可令 \(p\) 为紧 Hausdorff 满射。由 O.6，存在截面 \(s:E\to E'\)。沿 \(s\) 拉回提升，得 \(A(E)\) 中提升。证毕。

## 练习

1. 证明 \(\operatorname{RO}(X)\) 中任意 join 的公式满足最小上界性质。
2. 证明极不连通紧 Hausdorff 空间的 clopen algebra 完备。
3. 在 Stone 空间范畴中，说明满射对应 Boolean algebra 单射。
4. 检查 O.8 中 sheaf 满射到覆盖提升的使用位置。
