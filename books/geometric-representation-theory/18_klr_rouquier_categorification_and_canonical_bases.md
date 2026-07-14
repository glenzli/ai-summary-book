# 第十八章：KLR/Rouquier categorification 与 canonical bases

Quiver variety 用 Hecke correspondences 实现 $e_i,f_i$，但验证所有 Kac--Moody relations 需要反复比较高维 correspondence。Khovanov--Lauda--Rouquier algebra 把同一组合压缩成有颜色的 words、dots 与 crossings：idempotent 记录颜色次序，$y_r$ 记录 dot，$\psi_r$ 交换相邻位置，Cartan datum 决定 crossing 的二次与 braid 修正。Induction 把 words 拼接，所以 Grothendieck group 上的乘法应恢复 $U_q^-(\mathfrak g)$。在单色 $\mathfrak{sl}_2$ 情形，代数退化为 nilHecke algebra；$R(2)$ 可在二元多项式上用 divided difference 明确作用，并算成对称多项式环上的 $2\times2$ 矩阵代数，这将把抽象生成关系落实为可检查的线性算子。

## 18.1 KLR algebra 的数据

**定义 18.1.** 固定 symmetrizable Cartan datum，顶点集为 $I$。对 $\nu\in\mathbb N[I]$，KLR algebra $R(\nu)$ 是由 idempotents $e(\mathbf i)$、多项式生成元 $y_r$ 和 braid 生成元 $\psi_r$ 给出的 graded algebra，关系由 Cartan datum 和多项式 $Q_{ij}(u,v)$ 决定。

**版本边界 18.2.** KLR relations 依赖 Cartan datum 以及多项式 $Q_{ij}(u,v)$；这些多项式是代数定义的一部分，不能在多颜色计算中省略。本章的书内计算只使用单色 specialization，此时 $Q_{ii}=0$，因而不依赖相邻顶点之间的参数选择。一般 $Q_{ij}$-版本由外部输入定理 18.9 所引用的定义承担。

**定义 18.3.** 在 simply-laced 情形中，先固定满足
$$
Q_{ij}(u,v)=Q_{ji}(v,u),
\qquad Q_{ii}=0
$$
的齐次多项式。一个常用选择来自底图的定向：若 $i\to j$，取 $Q_{ij}(u,v)=u-v$、$Q_{ji}(u,v)=v-u$；不相邻时取 $Q_{ij}=1$。$R(\nu)$ 的 idempotents 由序列
$$
\mathbf i=(i_1,\ldots,i_m),\qquad \sum_r\alpha_{i_r}=\nu
$$
标号。生成元包括
$$
e(\mathbf i),\qquad y_1,\ldots,y_m,\qquad \psi_1,\ldots,\psi_{m-1}.
$$
它们满足正交幂等元关系、$y_r$ 交换关系、$\psi_r e(\mathbf i)=e(s_r\mathbf i)\psi_r$，以及二次关系
$$
\psi_r^2 e(\mathbf i)=
Q_{i_r,i_{r+1}}(y_r,y_{r+1})e(\mathbf i).
$$
braid relation 在相邻颜色相互作用时带有由同一组 $Q_{ij}$ 决定的修正项。

**警告 18.4.** 改变底图定向或 $Q_{ij}$ 会改变生成元公式。跨文献比较时必须连同 $Q_{ij}$ 与 grading convention 一起翻译；第 18.2 节只用 $i_r=i_{r+1}$ 的 nilHecke 关系。

**定义 18.5.** induction product 定义为
$$
M\circ N=R(\nu+\nu')e_{\nu,\nu'}
\otimes_{R(\nu)\otimes R(\nu')}(M\otimes N).
$$
这里 $e_{\nu,\nu'}$ 是选择前一段权重为 $\nu$、后一段权重为 $\nu'$ 的 words 的 idempotent，restriction functor 由同一 idempotent 截取给出。

**命题 18.6.** induction product 在 Grothendieck group 上给出结合乘法。

**证明.** 三个权重 $\nu,\nu',\nu''$ 的 induction 通过相应 corner idempotent $e_{\nu,\nu',\nu''}$ 控制。代数上使用嵌入
$$
R(\nu)\otimes R(\nu')\otimes R(\nu'')\subset R(\nu+\nu'+\nu'')
$$
以及选择三段 words 的 idempotent。两种加括号方式都自然同构于
$$
R(\nu+\nu'+\nu'')e_{\nu,\nu',\nu''}
\otimes_{R(\nu)\otimes R(\nu')\otimes R(\nu'')}
(M\otimes N\otimes P).
$$
张量积的结合约束和 idempotent 截取的传递性给出自然同构，因此 Grothendieck group 上乘法结合。$\square$

## 18.2 $\mathfrak{sl}_2$ 情形

**例 18.7.** 对 $\mathfrak{sl}_2$，顶点集只有一个元素 $i$。权重 $\nu=n\alpha_i$ 时，所有序列都等于 $(i,\ldots,i)$，因此 idempotent 只有一个。KLR algebra $R(n)$ 由 $y_1,\ldots,y_n$ 和 $\psi_1,\ldots,\psi_{n-1}$ 生成，并满足 nilHecke algebra 关系：
$$
\psi_r^2=0,\qquad
\psi_r\psi_{r+1}\psi_r=\psi_{r+1}\psi_r\psi_{r+1},
$$
以及
$$
\psi_r y_{r+1}-y_r\psi_r=1,\qquad
y_{r+1}\psi_r-\psi_r y_r=1.
$$

**命题 18.8.** $R(1)\simeq E[y_1]$，其 finitely generated graded projective modules 的 split Grothendieck group 是 rank-one free $\mathbb Z[q,q^{-1}]$-module。

**证明.** 当 $n=1$ 时没有 braid 生成元，只有多项式生成元 $y_1$，故 $R(1)=E[y_1]$。设 $M$ 为有限生成 graded projective module。先在 $M/y_1M$ 中选齐次 $E$-基并作齐次提升，由 graded Nakayama lemma 得到一个满射
$$
F=\bigoplus_a R(1)\langle d_a\rangle\longrightarrow M.
$$
因为 $M$ 在 graded module category 中 projective，该满射分裂，kernel $K$ 仍有限生成且 graded。模 $y_1$ 后的映射按所选基是同构，故 $K/y_1K=0$；再次用 graded Nakayama lemma 得 $K=0$。所以每个此类 $M$ 都是若干 grading shifts of $R(1)$ 的直和。split Grothendieck group 因而由 $[R(1)\langle m\rangle]$ 生成，grading shift 对应乘以 $q^m$，故为 rank-one free $\mathbb Z[q,q^{-1}]$-module。$\square$

**例 18.8.1（$R(2)$ 的 divided-difference 作用）.** 令
$$
P=E[x_1,x_2],
$$
并令 $s$ 交换 $x_1,x_2$。在 $P$ 上定义
$$
y_r f=x_rf,
\qquad
\psi_1f=\frac{s(f)-f}{x_1-x_2}.
$$
分子在 $x_1=x_2$ 上消失，故商仍是多项式。直接计算给出
$$
\psi_1^2=0,
\qquad
\psi_1y_2-y_1\psi_1=1,
\qquad
y_2\psi_1-\psi_1y_1=1,
$$
所以这些算子定义了 $R(2)$ 的 polynomial representation。例如
$$
\psi_1(1)=0,
\qquad
\psi_1(x_2)=1,
\qquad
\psi_1(x_1)=-1.
$$

**命题 18.8.2（最低阶矩阵模型）.** 置
$$
S=P^{S_2}=E[e_1,e_2],
\qquad e_1=x_1+x_2,\qquad e_2=x_1x_2.
$$
忘掉 grading 后，例 18.8.1 的作用给出代数同构
$$
R(2)\xrightarrow{\sim}\operatorname{End}_S(P)
\simeq M_2(S).
$$

**证明.** $P$ 是自由 rank-two $S$-module，基可取 $(1,x_2)$。在这组基下，乘以 $x_2$ 与 $\psi_1$ 的矩阵分别为
$$
Y=\begin{pmatrix}0&-e_2\\1&e_1\end{pmatrix},
\qquad
D=\begin{pmatrix}0&1\\0&0\end{pmatrix}.
$$
于是
$$
DY-e_1D=E_{11},\qquad
YD=E_{22},\qquad
YE_{11}=E_{21},\qquad
D=E_{12}.
$$
因此 $Y,D$ 生成 $M_2(S)$，作用映射满射。另一方面，利用 crossing--dot relations 可把 $R(2)$ 的任意 word 化为
$$
f(y_1,y_2)+g(y_1,y_2)\psi_1.
$$
若该元素在 $P$ 上作用为零，先作用于 $1$ 得 $f=0$，再作用于 $x_2$ 并用 $\psi_1(x_2)=1$ 得 $g=0$。故作用忠实，满射因而是同构。若给基向量加入相应 degree shifts，同一计算可写成 graded endomorphism algebra 的版本。$\square$

这个矩阵模型解释了为何 projective modules 而非代数元素本身承载 divided powers：Morita 等价把 $R(2)$-projectives 化为 $S$-projectives，而 grading shifts 保留量子参数。一般颜色与更长 words 中，indecomposable projectives 的识别正是范畴化定理的深层部分。

## 18.3 Categorification theorem

**外部输入定理 18.9.** Khovanov-Lauda 和 Rouquier 的 categorification theorem：KLR algebras 的 finitely generated graded projective modules 的 Grothendieck group categorifies $U_q^-(\mathfrak g)$ 的 integral form。

**外部输入定理 18.10.** 在 symmetric 情形中，Varagnolo-Vasserot 和 Rouquier 识别 indecomposable projective modules 与 Lusztig canonical basis。cyclotomic KLR algebras categorify integrable highest weight modules。

定理 18.9 只识别 Grothendieck group 与 integral form；定理 18.10 进一步指出哪一组不可分 projectives 对应 canonical basis。把 projective 换成 simple modules 时得到对偶基，不能在不说明 pairing 的情况下混用。

**资料入口 18.11.** Kang-Kashiwara, arXiv:1102.4677 证明 cyclotomic KLR algebras 对所有 symmetrizable Kac-Moody algebras categorify highest weight modules。

## 18.4 与 quiver varieties 的关系

**外部输入定理 18.12.** KLR algebras 可通过 quiver varieties、perverse sheaves 或 Ext algebras 几何实现，在这些模型中 canonical basis 对应 simple perverse sheaves 或 indecomposable projectives。

**边界说明 18.13.** “canonical basis = simple objects”需要说明使用 lower global basis、upper global basis、projective modules、simple modules 还是 IC sheaves。不同 convention 下会出现 duality。

KLR induction 通过分段 idempotent 拼接 words，其结合性因此来自三段 words 的同一个 corner。单色 $R(2)$ 的 divided-difference 表示则把 dots 与 crossing 具体化，并给出 $R(2)\simeq M_2(E[e_1,e_2])$ 的最低阶 Morita 模型。一般 Grothendieck group 与 canonical basis 的识别仍由 KLR--Rouquier 定理承担。下一章回到几何侧，研究这些范畴常出现其上的 conical symplectic resolutions 与量子化 category $\mathcal O$。

## 练习

**练习 18.1.** 在 $\mathfrak{sl}_2$ 情形中描述 $R(n)$ 的生成元类型。

**练习 18.2.** 证明 restriction functor 与 induction functor 在有限维代数情形中形成伴随对的标准条件。

**练习 18.3.** 解释 projective Grothendieck group 和 finite-dimensional module Grothendieck group 为什么通常互为配对。

**练习 18.4.** 在基 $(1,x_2)$ 下重新计算 $Y,D$，并仅用四个矩阵单位验证 $R(2)\simeq M_2(S)$。
