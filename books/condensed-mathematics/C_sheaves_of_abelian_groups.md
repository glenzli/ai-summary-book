# 附录 C：阿贝尔群值 sheaf 的范畴性质

## 本附录目标

第四章和第八章使用了 sheaf of abelian groups 的标准性质：它们形成阿贝尔范畴，核逐点计算，余核需 sheafification，满射具有局部提升性质。本附录给出这些事实在第一卷所需范围内的证明。

## C.1 预层阿贝尔范畴

设 $(\mathcal C,J)$ 是站点。阿贝尔群值预层范畴

$$
\operatorname{PSh}(\mathcal C;\mathbf{Ab})
=
\operatorname{Fun}(\mathcal C^{\operatorname{op}},\mathbf{Ab})
$$

是阿贝尔范畴。

核、余核、像、余像都逐点计算。例如若 $f:F\to G$，则

$$
(\ker f)(U)=\ker(F(U)\to G(U)),
$$

$$
(\operatorname{coker} f)(U)=\operatorname{coker}(F(U)\to G(U)).
$$

这是因为函子范畴到阿贝尔范畴的对象可以逐点做有限极限和余极限。

## C.2 Sheafification

设

$$
i:\operatorname{Sh}(\mathcal C;\mathbf{Ab})
\hookrightarrow
\operatorname{PSh}(\mathcal C;\mathbf{Ab})
$$

为包含函子。它有左伴随 sheafification：

$$
a:\operatorname{PSh}(\mathcal C;\mathbf{Ab})
\to
\operatorname{Sh}(\mathcal C;\mathbf{Ab}).
$$

即对预层 $F$ 和 sheaf $A$，有自然同构

$$
\operatorname{Hom}_{\operatorname{Sh}}(aF,A)
\cong
\operatorname{Hom}_{\operatorname{PSh}}(F,iA).
$$

本书使用以下标准事实：

**定理 C.1.** 阿贝尔群值 sheafification 是正合函子。

**证明说明.** 可通过加法 sheafification 的加号构造证明：先把预层局部化成 separated presheaf，再粘合局部截面。该过程保持有限极限，并作为左伴随保持余极限；在阿贝尔群值情形中可推出正合性。完整证明见标准 sheaf theory 教材或 Stacks Project。证毕。

## C.3 Sheaf 范畴是阿贝尔范畴

**定理 C.2.** $\operatorname{Sh}(\mathcal C;\mathbf{Ab})$ 是阿贝尔范畴。

**证明.** 设 $f:A\to B$ 为 sheaf 态射。

核逐点计算：

$$
\ker_{\operatorname{Sh}}(f)=\ker_{\operatorname{PSh}}(f),
$$

因为 kernel 是极限，而 sheaf 条件对极限稳定。

余核定义为预层余核的 sheafification：

$$
\operatorname{coker}_{\operatorname{Sh}}(f)
=
a(\operatorname{coker}_{\operatorname{PSh}}(f)).
$$

像与余像同理。由于预层范畴是阿贝尔范畴，且 sheafification 正合，可推出自然映射

$$
\operatorname{coim}(f)\to\operatorname{im}(f)
$$

为同构。因此 sheaf 范畴是阿贝尔范畴。证毕。

## C.4 满射的局部提升判据

**命题 C.3.** 设 $f:A\to B$ 是阿贝尔群值 sheaf 的态射。则 $f$ 是 sheaf 范畴中的满射，当且仅当对任意对象 $U$ 和任意 $b\in B(U)$，存在覆盖族 $\{U_i\to U\}$，使得每个限制

$$
b|_{U_i}\in B(U_i)
$$

来自某个 $a_i\in A(U_i)$。

**证明.** 若 $f$ 是满射，则其余核为零。余核是预层余核 sheafification。一个截面在 sheafification 中为零，等价于它局部为零。因此 $b$ 在余核中的像局部为零，即局部来自 $A$。

反过来，若每个 $b$ 都局部来自 $A$，则预层余核的每个局部截面在某个覆盖上为零，故其 sheafification 为零。因此 sheaf 余核为零，$f$ 是满射。证毕。

## C.5 正合性的局部判据

由命题 C.3 可得：复形

$$
A\to B\to C
$$

在 $B$ 处正合，当且仅当任意 $b\in B(U)$ 若映到 $0\in C(U)$，则存在覆盖 $\{U_i\to U\}$，使得 $b|_{U_i}$ 来自 $A(U_i)$。

这就是第六章和第八章中“局部提升”的形式依据。

## C.6 本附录小结

本附录解释了凝聚阿贝尔群范畴的基本代数性质为何成立。凝聚阿贝尔群只是特定站点上的阿贝尔群值 sheaf，因此所有这些结果都适用于

$$
\mathbf{CondAb}
=
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}};\mathbf{Ab}).
$$

## 练习

**练习 C.1.** 证明 sheaf 条件对任意小极限稳定。

**练习 C.2.** 给出一个 sheaf 满射不是逐点满射的例子。

**练习 C.3.** 用命题 C.3 重新证明第六章定理 6.11 中满射在 ED 空间上取值仍满。
