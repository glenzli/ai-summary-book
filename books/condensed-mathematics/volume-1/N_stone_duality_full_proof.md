# 附录 N：Stone 对偶的完整证明链

## N.0 目标

附录 D 已给出 Stone 对偶和极不连通空间的使用口径。本附录把 Stone 对偶中实际用于凝聚数学的部分展开为可逐步复核的证明：

1. Boolean 代数的超滤子描述。
2. Stone 空间的紧 Hausdorff 与全不连通性。
3. $B\simeq \operatorname{Clop}(\operatorname{Stone}(B))$。
4. profinite 空间与 Boolean 代数的反等价。
5. profinite 空间作为有限离散空间逆极限的构造。

唯一外部集合论输入是 Boolean prime ideal theorem，也即每个真滤子可扩张为超滤子。本书把它视为选择公理的弱形式输入，不在此证明。

## N.1 滤子与超滤子

设 $B$ 是 Boolean 代数。记偏序为

$$
a\le b
\quad\Longleftrightarrow\quad
a\wedge b=a.
$$

**定义 N.1.** $B$ 的滤子是子集 $F\subset B$，满足：

1. $1\in F$，$0\notin F$；
2. 若 $a,b\in F$，则 $a\wedge b\in F$；
3. 若 $a\in F$ 且 $a\le b$，则 $b\in F$。

若滤子在所有真滤子中极大，则称为超滤子。

**引理 N.2.** 真滤子 $F$ 是超滤子，当且仅当对每个 $b\in B$，恰有一个元素属于 $F$：

$$
b\in F
\qquad\text{或}\qquad
\neg b\in F.
$$

**证明.** 两者不能同时成立，因为 $b\wedge\neg b=0$，而滤子不含 $0$。

设 $F$ 是超滤子。若 $b\notin F$，考虑由 $F\cup\{b\}$ 生成的滤子。若这是一个真滤子，则严格包含 $F$，与极大性矛盾。因此生成滤子含 $0$。这等价于存在 $f\in F$ 使

$$
f\wedge b=0.
$$

于是 $f\le\neg b$，由滤子的向上封闭性得 $\neg b\in F$。

反过来，若 $F$ 满足该判别，设 $G$ 是含 $F$ 的真滤子。取 $g\in G$。若 $g\notin F$，则 $\neg g\in F\subset G$，于是 $0=g\wedge\neg g\in G$，矛盾。因此 $G=F$。证毕。

**输入定理 N.3（Boolean prime ideal theorem 的滤子形式）.** 若 $F$ 是 Boolean 代数 $B$ 的真滤子，则存在超滤子 $U$ 使

$$
F\subset U.
$$

**推论 N.4（元素分离）.** 若 $a\ne b$，则存在超滤子 $U$，使 $a\in U$ 且 $b\notin U$，或反之。

**证明.** 若 $a\ne b$，则

$$
c=(a\wedge\neg b)\vee(b\wedge\neg a)
$$

非零。设 $d$ 是 $a\wedge\neg b$ 或 $b\wedge\neg a$ 中的非零项。由 $d$ 生成的滤子是真滤子，按 N.3 扩张为超滤子 $U$。若 $d=a\wedge\neg b$，则 $a\in U$ 且 $\neg b\in U$，故 $b\notin U$；另一情形相同。证毕。

## N.2 Stone 空间

**定义 N.5.** Boolean 代数 $B$ 的 Stone 空间记为

$$
\operatorname{Stone}(B),
$$

其点为 $B$ 的超滤子。对 $b\in B$ 定义

$$
U_b=\{U\in\operatorname{Stone}(B)\mid b\in U\}.
$$

以所有 $U_b$ 为开基定义拓扑。

**引理 N.6.** 对 $a,b\in B$，有

$$
U_{a\wedge b}=U_a\cap U_b,\qquad
U_{a\vee b}=U_a\cup U_b,\qquad
U_{\neg a}=\operatorname{Stone}(B)\setminus U_a.
$$

**证明.** 第一式来自滤子对有限交封闭。第二式中，若 $a\vee b\in U$ 且 $a\notin U$、$b\notin U$，则由引理 N.2 得 $\neg a,\neg b\in U$，从而

$$
\neg(a\vee b)=\neg a\wedge\neg b\in U,
$$

与 $a\vee b\in U$ 矛盾。反向由 $a\le a\vee b$ 和 $b\le a\vee b$ 得到。第三式正是引理 N.2。证毕。

**命题 N.7.** $\operatorname{Stone}(B)$ 是 Hausdorff 且有开闭基。

**证明.** 引理 N.6 说明每个 $U_b$ 的补集是 $U_{\neg b}$，所以 $U_b$ 开闭。若 $U\ne V$ 是两个超滤子，取 $b\in U\setminus V$。由引理 N.2，$\neg b\in V$。于是 $U_b$ 与 $U_{\neg b}$ 是互不相交开集，分别含 $U,V$。证毕。

**命题 N.8.** $\operatorname{Stone}(B)$ 是紧空间。

**证明.** 设 $\{U_{b_i}\}_{i\in I}$ 是一族基本开集，覆盖 $\operatorname{Stone}(B)$。若不存在有限子覆盖，则对每个有限子集 $J\subset I$，

$$
\bigwedge_{j\in J}\neg b_j\ne0.
$$

这些元素生成一个真滤子 $F$：任意有限交仍非零，所以 $0$ 不在生成滤子中。由输入定理 N.3，$F$ 扩张为超滤子 $U$。于是对所有 $i$，$\neg b_i\in U$，故 $b_i\notin U$。这说明 $U$ 不在任何 $U_{b_i}$ 中，与覆盖性矛盾。因此每个基本开覆盖有有限子覆盖。

任意开覆盖可由基本开细化：对每个点选取包含该点且落在某个覆盖开集中的基本开邻域。基本开细化有有限子覆盖，于是原开覆盖也有有限子覆盖。证毕。

## N.3 Boolean 代数到开闭代数

定义映射

$$
\theta_B:B\to \operatorname{Clop}(\operatorname{Stone}(B)),
\qquad
b\mapsto U_b.
$$

**命题 N.9.** $\theta_B$ 是 Boolean 代数同构。

**证明.** 引理 N.6 说明 $\theta_B$ 保持有限交、有限并、补元、$0$ 和 $1$。

若 $a\ne b$，由推论 N.4 有超滤子分离 $a,b$，所以 $U_a\ne U_b$。因此 $\theta_B$ 单射。

设 $W\subset\operatorname{Stone}(B)$ 是开闭集。因为 $W$ 开，存在一族 $b_i$ 使

$$
W=\bigcup_i U_{b_i}.
$$

因为 $W$ 闭且 Stone 空间紧，$W$ 紧。于是存在有限子集 $i_1,\ldots,i_m$，使

$$
W=U_{b_{i_1}}\cup\cdots\cup U_{b_{i_m}}
=U_{b_{i_1}\vee\cdots\vee b_{i_m}}.
$$

故 $\theta_B$ 满射。证毕。

## N.4 连续映射与反变函子

若 $f:B\to C$ 是 Boolean 代数同态，定义

$$
\operatorname{Stone}(f):\operatorname{Stone}(C)\to\operatorname{Stone}(B),
\qquad
U\mapsto f^{-1}(U).
$$

**引理 N.10.** $f^{-1}(U)$ 是 $B$ 的超滤子，且 $\operatorname{Stone}(f)$ 连续。

**证明.** Boolean 同态保持 $0,1,\wedge,\vee,\neg$，所以超滤子的逆像是滤子。对每个 $b\in B$，在 $U$ 中恰有一个 $f(b)$ 或 $\neg f(b)=f(\neg b)$，故逆像滤子满足引理 N.2 的超滤子判别。

连续性由

$$
\operatorname{Stone}(f)^{-1}(U_b)=U_{f(b)}
$$

给出。证毕。

这说明 $B\mapsto\operatorname{Stone}(B)$ 是从 Boolean 代数到紧 Hausdorff 全不连通空间的反变函子。

## N.5 从 profinite 空间回到 Stone 空间

设 $X$ 是紧 Hausdorff 且开闭集构成拓扑基的空间。定义

$$
\eta_X:X\to \operatorname{Stone}(\operatorname{Clop}(X)),
\qquad
x\mapsto\{V\in\operatorname{Clop}(X)\mid x\in V\}.
$$

**命题 N.11.** $\eta_X$ 是同胚。

**证明.** 对每个 $x$，$\eta_X(x)$ 是超滤子：包含 $1=X$、不含 $\varnothing$、对有限交封闭、向上封闭；且对开闭集 $V$，恰有 $x\in V$ 或 $x\in X\setminus V$。

若 $x\ne y$，由 Hausdorff 和开闭基可取开闭集 $V$ 使 $x\in V$、$y\notin V$。故 $\eta_X$ 单射。

设 $U$ 是 $\operatorname{Clop}(X)$ 的超滤子。因为 $X$ 紧，闭集族 $U$ 有有限交性质，所以

$$
\bigcap_{V\in U}V
$$

非空。若 $x,y$ 都在该交中且 $x\ne y$，取开闭 $W$ 使 $x\in W$、$y\notin W$。由超滤子判别，$W\in U$ 或 $X\setminus W\in U$。两种情形都排除其中一个点属于总交，矛盾。因此交为单点 $\{x\}$。对任意开闭 $V$，若 $x\in V$ 而 $V\notin U$，则 $X\setminus V\in U$，与 $x$ 属于总交矛盾；若 $V\in U$，则 $x\in V$。所以 $U=\eta_X(x)$。这证明满射。

连续性由

$$
\eta_X^{-1}(U_V)=V
$$

给出。紧 Hausdorff 空间之间的连续双射为同胚；目标为 Hausdorff，源为紧。证毕。

**定理 N.12（Stone 对偶）.** Boolean 代数范畴与 profinite 空间范畴反等价：

$$
\mathbf{Bool}^{\mathrm{op}}\simeq \mathbf{ProFin}.
$$

**证明.** 命题 N.9 给出 Boolean 代数侧的单位同构，命题 N.11 给出 profinite 空间侧的余单位同构。自然性来自逆像：

$$
g^{-1}(V\cap W)=g^{-1}(V)\cap g^{-1}(W),\qquad
g^{-1}(X\setminus V)=Y\setminus g^{-1}(V).
$$

这说明连续映射 $g:Y\to X$ 诱导 Boolean 同态 $\operatorname{Clop}(X)\to\operatorname{Clop}(Y)$，且上述两个同构与态射相容。证毕。

## N.6 Profinite 空间的逆极限表示

设 $X$ 是 profinite 空间。令 $\mathcal P_X$ 为 $X$ 的有限开闭划分组成的有向集；若 $\mathcal Q$ 细化 $\mathcal P$，有自然映射

$$
X_{\mathcal Q}\to X_{\mathcal P}
$$

把 $\mathcal Q$ 的每个块送到包含它的 $\mathcal P$ 的块。

**命题 N.13.** 自然映射

$$
X\to\varprojlim_{\mathcal P\in\mathcal P_X}X_{\mathcal P}
$$

是同胚。

**证明.** 映射把点 $x$ 送到它所在的每个划分块。

单射：若 $x\ne y$，取开闭集 $V$ 使 $x\in V$、$y\notin V$。划分 $\{V,X\setminus V\}$ 区分二者。

满射：设 $(P_{\mathcal P})_{\mathcal P}$ 是逆极限中的相容族。每个 $P_{\mathcal P}$ 是 $X$ 的非空闭子集。相容性给出有限交性质：任取有限多个划分，取共同细化 $\mathcal R$，则 $P_{\mathcal R}$ 包含于这些 $P_{\mathcal P_i}$ 的交中。由紧性，总交非空。若交中有两个不同点，则某个二块开闭划分区分它们，与该划分下只选择一个块矛盾。因此总交为单点，并给出原相容族。

连续双射从紧空间到 Hausdorff 空间是同胚。右侧为有限离散空间的逆极限，因此 Hausdorff。证毕。

## N.7 在凝聚数学中的使用位置

本附录允许正文把以下步骤视为书内已证事实：

1. profinite 空间等价于 Boolean 代数的 Stone 空间；
2. profinite 空间有开闭基，且可由有限离散商逆极限恢复；
3. Boolean 代数计算可转化为 profinite 空间的开闭集计算；
4. ED 空间与完备 Boolean 代数之间的关系仍需附录 D/J 的 regular open 构造和 Gleason 输入。

Gleason cover 的投射性不是 Stone 对偶的形式后果；它需要额外拓扑定理。

## 练习

1. 证明若 $B$ 有限，则 $\operatorname{Stone}(B)$ 是有限离散空间。
2. 对集合 $S$，计算 $\operatorname{Stone}(\mathcal P(S))$。说明当 $S$ 无限时它不等于离散空间 $S$，而是其 Stone-Cech 紧化的离散版本。
3. 证明 Boolean 同态 $f:B\to C$ 单射当且仅当 $\operatorname{Stone}(f)$ 的像稠密。
4. 用命题 N.13 证明 profinite 空间的连续映射由所有有限离散商上的相容映射决定。
