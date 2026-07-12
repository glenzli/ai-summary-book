# 附录 G：根资料、对偶群和 L 群计算表

本附录补充第十一章的计算层。目标是让读者能从矩阵群直接算出 root datum、dual group 和若干基本 L 群，而不是只引用分类定理。

收口归一化回指：本附录涉及 dual group、L group、L homomorphism、Satake 参数和 L 群表示给出的局部因子；与第十一至十五章比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、4、8 节。

## G.1 Split Torus

**定义 G.1.** 设 $T=\mathbb G_m^r$。其 character lattice 和 cocharacter lattice 为
$$
X^*(T)=\operatorname{Hom}(T,\mathbb G_m),\qquad
X_*(T)=\operatorname{Hom}(\mathbb G_m,T).
$$

**命题 G.2.** 对 $T=\mathbb G_m^r$，有自然同构
$$
X^*(T)\simeq\mathbb Z^r,\qquad X_*(T)\simeq\mathbb Z^r,
$$
且配对为标准点积。

**证明.** 任一 character 形如
$$
(t_1,\ldots,t_r)\mapsto t_1^{a_1}\cdots t_r^{a_r},
$$
其中 $a_i\in\mathbb Z$。任一 cocharacter 形如
$$
z\mapsto(z^{b_1},\ldots,z^{b_r}),
$$
其中 $b_i\in\mathbb Z$。复合后得到
$$
z\mapsto z^{\sum_i a_ib_i},
$$
故配对为 $\sum_i a_ib_i$。$\square$

## G.2 `GL(n)`

设 $G=\operatorname{GL}_n$，$T$ 为 diagonal torus，$B$ 为 upper triangular Borel subgroup。令 $e_i\in X^*(T)$ 为
$$
e_i(\operatorname{diag}(t_1,\ldots,t_n))=t_i.
$$
令 $e_i^\vee\in X_*(T)$ 为
$$
e_i^\vee(z)=\operatorname{diag}(1,\ldots,1,z,1,\ldots,1).
$$

**命题 G.3.** `GL(n)` 的根为
$$
\Phi(\operatorname{GL}_n,T)=\{e_i-e_j:i\ne j\}.
$$
相对于 $B$ 的 simple roots 为
$$
\Delta=\{\alpha_i=e_i-e_{i+1}:1\le i\le n-1\}.
$$
对应 coroots 为
$$
\alpha_i^\vee=e_i^\vee-e_{i+1}^\vee.
$$

**证明.** Lie algebra $\mathfrak{gl}_n$ 有矩阵单位基 $E_{ij}$。对 $t=\operatorname{diag}(t_1,\ldots,t_n)$，
$$
\operatorname{Ad}(t)E_{ij}=t_it_j^{-1}E_{ij}.
$$
当 $i\ne j$ 时，该权为 $e_i-e_j$；对角部分给出零权。上三角 Borel 选择 $i<j$ 为正根，因此 simple roots 为相邻差。coroot 由对应的 $\operatorname{SL}_2$-子群嵌入
$$
\begin{pmatrix}a&b\\c&d\end{pmatrix}
\mapsto
\text{在 }(i,i+1)\text{ 块上放入该矩阵}
$$
给出，即 $e_i^\vee-e_{i+1}^\vee$。$\square$

**命题 G.4.** `GL(n)` 的 Weyl group 为 $S_n$，按置换 diagonal entries 作用于 $X^*(T)$。

**证明.** Normalizer $N_G(T)$ 由 monomial matrices 组成，即每行每列恰有一个非零元。商去 $T$ 后只剩置换矩阵，因此 $N_G(T)/T\simeq S_n$。其共轭作用把 $e_i$ 送到 $e_{\sigma(i)}$。$\square$

**推论 G.5.** `GL(n)` 的对偶群仍为
$$
\widehat{\operatorname{GL}_n}=\operatorname{GL}_n(\mathbb C).
$$

**证明.** `GL(n)` 的 root datum 为
$$
\left(\mathbb Z^n,\{e_i-e_j\},\mathbb Z^n,\{e_i^\vee-e_j^\vee\}\right).
$$
交换 character lattice 与 cocharacter lattice 后得到同构的根资料。因此对偶群为 $\operatorname{GL}_n(\mathbb C)$。$\square$

## G.3 `SL(n)` 和 `PGL(n)`

**定义 G.6.** 对 $G=\operatorname{SL}_n$，取 diagonal torus
$$
T_{\operatorname{SL}_n}=\{\operatorname{diag}(t_1,\ldots,t_n):\prod_i t_i=1\}.
$$
对 $G=\operatorname{PGL}_n$，取 $\operatorname{GL}_n$ diagonal torus 对中心的商。

**命题 G.7.** `SL(n)` 的 character lattice 为
$$
X^*(T_{\operatorname{SL}_n})=\mathbb Z^n/\mathbb Z(e_1+\cdots+e_n),
$$
根仍由 $e_i-e_j$ 给出。其 cocharacter lattice 为
$$
\{(b_1,\ldots,b_n)\in\mathbb Z^n:\sum_i b_i=0\}.
$$

**证明.** 限制 $\operatorname{GL}_n$ diagonal torus 的 character 到行列式为 $1$ 的子 torus。character $e_1+\cdots+e_n$ 即 determinant，在 $T_{\operatorname{SL}_n}$ 上平凡，所以 character lattice 是相应商。cocharacter $z\mapsto\operatorname{diag}(z^{b_i})$ 落在 $\operatorname{SL}_n$ 中当且仅当 $\sum_i b_i=0$。根来自 adjoint action 对 $E_{ij}$ 的权，故仍为 $e_i-e_j$。$\square$

**命题 G.8.** 有对偶关系
$$
\widehat{\operatorname{SL}_n}=\operatorname{PGL}_n(\mathbb C),\qquad
\widehat{\operatorname{PGL}_n}=\operatorname{SL}_n(\mathbb C).
$$

**证明.** `SL(n)` 的 character lattice 是 weight lattice，而 coroot lattice 为 $\sum b_i=0$ 的根格。对偶根资料交换二者。`PGL(n)` 的 character lattice 为根格，cocharacter lattice 为 coweight lattice。两者互为对偶。$\square$

**例 G.9.** 对 $n=2$，
$$
\widehat{\operatorname{SL}_2}=\operatorname{PGL}_2(\mathbb C),\qquad
\widehat{\operatorname{PGL}_2}=\operatorname{SL}_2(\mathbb C).
$$
因此 `SL(2)` 的 L-parameters 进入 $\operatorname{PGL}_2(\mathbb C)$，这会丢失 `GL(2)` 参数的 determinant 信息。

## G.4 Symplectic Groups

设
$$
J=\begin{pmatrix}0&I_n\\-I_n&0\end{pmatrix}.
$$
定义
$$
\operatorname{Sp}_{2n}=\{g\in\operatorname{GL}_{2n}:g^tJg=J\}.
$$

**定义 G.10.** 取 split maximal torus
$$
T=\{\operatorname{diag}(t_1,\ldots,t_n,t_1^{-1},\ldots,t_n^{-1})\}.
$$
令 $e_i\in X^*(T)$ 为第 $i$ 个坐标 character。

**命题 G.11.** $\operatorname{Sp}_{2n}$ 的根系统为 type $C_n$：
$$
\Phi=\{\pm e_i\pm e_j:i\ne j\}\cup\{\pm2e_i\}.
$$
一组 simple roots 为
$$
\alpha_i=e_i-e_{i+1}\quad(1\le i<n),\qquad \alpha_n=2e_n.
$$

**证明.** 把 $X\in\mathfrak{sp}_{2n}$ 写成块矩阵
$$
X=\begin{pmatrix}A&B\\ C&-A^t\end{pmatrix},
\qquad B=B^t,\quad C=C^t.
$$
对 $t=\operatorname{diag}(t_1,\ldots,t_n,t_1^{-1},\ldots,t_n^{-1})$，
共轭作用在 $A_{ij}$ 上的权为 $e_i-e_j$，在 $B_{ij}$ 上的权为
$e_i+e_j$，在 $C_{ij}$ 上的权为 $-e_i-e_j$。当 $i=j$ 时后两类权分别
为 $2e_i$ 与 $-2e_i$。去掉 $A$ 的对角零权部分，恰得到显示的根集。
选择使 $A$ 上三角、$B$ 取相应正根空间的 Borel 后，正根为
$e_i-e_j$（$i<j$）、$e_i+e_j$（$i<j$）和 $2e_i$。逐次相减可知不可再
分解为正根和的正根恰为
$e_i-e_{i+1}$（$i<n$）与 $2e_n$，故它们构成 simple roots。$\square$

**外部输入定理 G.12（classical groups 的对偶）.** Split classical groups 的对偶群满足
$$
\widehat{\operatorname{Sp}_{2n}}=\operatorname{SO}_{2n+1}(\mathbb C),
$$
$$
\widehat{\operatorname{SO}_{2n+1}}=\operatorname{Sp}_{2n}(\mathbb C),
$$
并且 split even orthogonal group 的对偶仍为同型 even orthogonal complex group，需按根资料和中心形式精确区分。

## G.5 L 群的基本例子

**定义 G.13.** 若 $G/F$ 为 split connected reductive group，则局部 L 群为
$$
{}^LG=\widehat G\rtimes W_F
$$
且 $W_F$ 对 $\widehat G$ 的作用平凡。因此
$$
{}^LG=\widehat G\times W_F
$$
作为拓扑群直积。

**命题 G.14.** 对 split `GL(n)`，
$$
{}^L\operatorname{GL}_n=\operatorname{GL}_n(\mathbb C)\times W_F.
$$

**证明.** 由 G.5，$\widehat{\operatorname{GL}_n}=\operatorname{GL}_n(\mathbb C)$。split 条件使 Galois action on pinned root datum 平凡，故半直积为直积。$\square$

**命题 G.15.** 对 split `SL(n)` 和 `PGL(n)`，
$$
{}^L\operatorname{SL}_n=\operatorname{PGL}_n(\mathbb C)\times W_F,
$$
$$
{}^L\operatorname{PGL}_n=\operatorname{SL}_n(\mathbb C)\times W_F.
$$

**证明.** 由 G.8 和 split L 群定义。$\square$

**定义 G.16.** 设 $E/F$ 为有限 separable 扩张。Restriction of scalars torus
$$
T=\operatorname{Res}_{E/F}\mathbb G_m
$$
满足
$$
T(F)=E^\times.
$$

**命题 G.17.** 若 $E/F$ 为 Galois 扩张，则
$$
X^*(T)\simeq\mathbb Z[\operatorname{Gal}(E/F)]
$$
作为 $\Gamma_F$-module，其中 $\Gamma_F$ 通过商 $\operatorname{Gal}(E/F)$ 左平移作用。

**证明.** 基变换到 $\overline F$ 后，有限可分扩张的所有 $F$-嵌入给出
$$
T_{\overline F}\simeq\prod_{\sigma:E\hookrightarrow\overline F}\mathbb G_m.
$$
乘积 torus 的 character lattice 是各坐标 character 的自由 Abel 群，故
$$
X^*(T_{\overline F})\cong
\mathbb Z[\operatorname{Hom}_F(E,\overline F)].
$$
若 $E/F$ Galois，选定一个嵌入后，所有嵌入由
$\operatorname{Gal}(E/F)$ 唯一标号。$\Gamma_F$ 在嵌入上的复合作用通过
商 $\operatorname{Gal}(E/F)$，并在该基上给出正则置换作用；按本书的
character 方差约定，这写成左平移。故得到所述 $\Gamma_F$-module 同构。
$\square$

**推论 G.18.** 对 $T=\operatorname{Res}_{E/F}\mathbb G_m$，
$$
{}^LT=\left((\mathbb C^\times)^{\operatorname{Hom}_F(E,\overline F)}\right)\rtimes W_F,
$$
其中 $W_F$ 通过其在 $\Gamma_F$ 中的像置换各因子。

**证明.** Torus 的对偶群由 cocharacter/character lattice 对偶给出。Galois 在 character lattice 上置换基，故在对偶 torus 上置换相应 $\mathbb C^\times$ 因子。$\square$

## G.6 L 同态的基本样本

**定义 G.19.** Standard representation 是自然嵌入
$$
\operatorname{Std}:\widehat G\to\operatorname{GL}(V)
$$
在给定对偶群的 defining representation 中的表示。若 $G=\operatorname{GL}_n$，则 $\operatorname{Std}$ 为 $\operatorname{GL}_n(\mathbb C)$ 在 $\mathbb C^n$ 上的自然表示。

**命题 G.20.** 行列式映射
$$
\det:\operatorname{GL}_n(\mathbb C)\to\mathbb C^\times
$$
给出 L 同态
$$
{}^L\operatorname{GL}_n\to{}^L\operatorname{GL}_1
$$
在 split 情形中由
$$
(g,w)\mapsto(\det g,w)
$$
定义。

**证明.** 该映射在对偶群部分为代数群同态，并且对 $W_F$ 投影保持不变。split 情形 $W_F$ 作用平凡，所以半直积乘法相容。$\square$

**命题 G.21.** 中心嵌入
$$
\mathbb C^\times\to\operatorname{GL}_n(\mathbb C),\qquad z\mapsto zI_n
$$
对应一个 L 同态
$$
{}^L\operatorname{GL}_1\to{}^L\operatorname{GL}_n.
$$
若 $\chi$ 为 Hecke 特征，则该同态诱导的自守表示应为 $\chi\circ\det$ 类型的一维自守表示。

**证明.** L 同态验证与 G.20 相同。对自守侧，$\operatorname{GL}_n(\mathbb A_K)$ 的 character 可由 determinant 拉回：
$$
g\mapsto\chi(\det g).
$$
这给出一维自守表示。$\square$

**命题 G.22.** 对 `GL(2)`，symmetric square 表示
$$
\operatorname{Sym}^2:\operatorname{GL}_2(\mathbb C)\to\operatorname{GL}_3(\mathbb C)
$$
给出 L 同态
$$
{}^L\operatorname{GL}_2\to{}^L\operatorname{GL}_3
$$
在 split 情形中由 $(g,w)\mapsto(\operatorname{Sym}^2g,w)$ 定义。

**证明.** $\operatorname{Sym}^2$ 是代数群表示，且 split 情形中 Weil 群作用平凡。乘法相容来自表示函子保持复合。$\square$

**注 G.23.** 第十五章的 symmetric square functoriality 正是 G.22 的自守转移问题。其存在不是形式结果；Gelbart-Jacquet 定理是外部输入。

## G.7 本附录小结

本附录建立如下可计算接口：

1. `GL(n)` 的根、coroots、Weyl group 和自对偶性。
2. `SL(n)` 与 `PGL(n)` 的对偶关系。
3. Symplectic 和 orthogonal duality 的 classical group 表格入口。
4. Split L 群和 restriction of scalars torus 的 L 群。
5. determinant、central embedding 和 symmetric square 的 L 同态样本。

这些计算是第十一章抽象 root datum 语言、第十二章局部参数和第十五章函子性的具体模型。

## 练习

**练习 G.1.** 对 `GL(3)`，写出所有正根、simple roots 和 Weyl group 的 simple reflections。

**练习 G.2.** 证明 `SL(2)` 的唯一正根在 character lattice 中是 twice a generator，而其 coroot 是 cocharacter lattice 的 generator。

**练习 G.3.** 对 $T=\operatorname{Res}_{E/F}\mathbb G_m$，在 $E/F$ 二次 Galois 时写出 ${}^LT$ 中非平凡 Galois 元素对 $(z_1,z_2)\in(\mathbb C^\times)^2$ 的作用。

**练习 G.4.** 验证 G.20 的 determinant L 同态在非分歧参数上把 Satake 参数 $\operatorname{diag}(\alpha_1,\ldots,\alpha_n)$ 送到 $\alpha_1\cdots\alpha_n$。

**练习 G.5.** 对 `GL(2)` Satake 参数 $\operatorname{diag}(\alpha,\beta)$，计算 $\operatorname{Sym}^2$ 推前后的 `GL(3)` Satake 参数。
