# 第二章：Lie 代数表示、category O 与中心 character

旗簇上的 Schubert cells 由 Weyl group 标号，但 localization 所要几何化的并不是 Weyl group 本身，而是带固定中心 character 的 $U(\mathfrak g)$-模。为了看清几何侧将接收什么，先从第一章的正根约定得到 triangular decomposition，再考察由一个最高权向量自由生成的 Verma module。它既足够具体，可以在 $\mathfrak{sl}_2$ 中逐项写出基和作用，又足够普遍，能组织 category $\mathcal O$ 的简单对象与 block。中心的 Harish--Chandra 参数最终把这里的 dot action 和旗簇上线丛的 $\rho$-shift 接到一起。

## 2.1 Universal enveloping algebra 和 triangular decomposition

**约定 2.1.** 本章取 $k=\mathbb C$，$G$ 为连通复 reductive group，$\mathfrak g=\operatorname{Lie}(G)$。由 $B$ 和 $T$ 得到
$$
\mathfrak b=\mathfrak t\oplus\mathfrak n,\qquad
\mathfrak g=\mathfrak n^-\oplus\mathfrak t\oplus\mathfrak n.
$$

**定义 2.2.** $U(\mathfrak g)$ 是 tensor algebra $T(\mathfrak g)$ 对双边理想
$$
x\otimes y-y\otimes x-[x,y]
$$
取商得到的 associative algebra。

这个商把 Lie bracket 编码为非交换乘法，却还没有给出可计算的正规形。PBW 定理所提供的有序乘法分解，是从最高权条件构造诱导模的关键外部输入。

**外部输入定理 2.3.** PBW theorem 给出向量空间同构
$$
U(\mathfrak n^-)\otimes U(\mathfrak t)\otimes U(\mathfrak n)\xrightarrow{\sim}U(\mathfrak g)
$$
由乘法诱导。  
用途：Verma module 的向量空间结构和 category $\mathcal O$ 基础。

**定义 2.4.** 一个 $\mathfrak g$-module $M$ 称为 weight module，若
$$
M=\bigoplus_{\mu\in\mathfrak t^\ast}M_\mu,\qquad
M_\mu=\{m\in M\mid h m=\mu(h)m,\ \forall h\in\mathfrak t\}.
$$
若 $M_\mu\ne0$，称 $\mu$ 为 $M$ 的 weight。

## 2.2 Verma modules

**定义 2.5.** 对 $\lambda\in\mathfrak t^\ast$，令 $\mathbb C_\lambda$ 为一维 $\mathfrak b$-module，其中 $\mathfrak n$ 平凡作用，$\mathfrak t$ 按 $\lambda$ 作用。Verma module 定义为
$$
M(\lambda)=U(\mathfrak g)\otimes_{U(\mathfrak b)}\mathbb C_\lambda.
$$
其生成元 $v_\lambda=1\otimes1$ 称为 highest weight vector。

**命题 2.6.** $M(\lambda)$ 具有如下泛性质：若 $N$ 是 $\mathfrak g$-module，$v\in N$ 满足
$$
\mathfrak n v=0,\qquad h v=\lambda(h)v\quad(\forall h\in\mathfrak t),
$$
则存在唯一 $\mathfrak g$-module morphism
$$
f:M(\lambda)\to N
$$
使得 $f(v_\lambda)=v$。

**证明.** 给定 $v$，定义 $U(\mathfrak g)$-linear map
$$
U(\mathfrak g)\to N,\qquad u\mapsto u v.
$$
若 $x\in\mathfrak n$，则 $xv=0$；若 $h\in\mathfrak t$，则 $(h-\lambda(h))v=0$。因此该映射杀掉诱导模张量关系中的核，下降为
$$
U(\mathfrak g)\otimes_{U(\mathfrak b)}\mathbb C_\lambda\to N.
$$
构造出的映射把 $v_\lambda$ 送到 $v$。唯一性来自 $M(\lambda)$ 由 $v_\lambda$ 作为 $U(\mathfrak g)$-module 生成。$\square$

**推论 2.7.** 若 $N$ 是 highest weight $\lambda$ 的 $\mathfrak g$-module，且由 highest weight vector $v$ 生成，则 $N$ 是 $M(\lambda)$ 的商。

**证明.** 由命题 2.6 得到 $M(\lambda)\to N$。由于 $N=U(\mathfrak g)v$，该映射满。$\square$

**外部输入定理 2.8.** 每个 Verma module $M(\lambda)$ 有唯一 maximal proper submodule，因此有唯一 simple quotient，记为 $L(\lambda)$。  
用途：category $\mathcal O$ 的简单对象标号。来源需定位到 Humphreys 或 Jantzen。

**命题 2.9.** 作为向量空间，
$$
M(\lambda)\simeq U(\mathfrak n^-)
$$
并且 $M(\lambda)$ 的 weights 属于
$$
\lambda-\mathbb Z_{\ge0}\Phi^+.
$$

**证明.** 由 PBW theorem，乘法给出
$$
U(\mathfrak n^-)\otimes U(\mathfrak b)\simeq U(\mathfrak g)
$$
的向量空间分解。因此
$$
U(\mathfrak g)\otimes_{U(\mathfrak b)}\mathbb C_\lambda
\simeq U(\mathfrak n^-)\otimes\mathbb C_\lambda
\simeq U(\mathfrak n^-).
$$
$\mathfrak n^-$ 由负根空间生成，每个负根向量将 weight 减去相应正根。有限乘积给出 weight 集包含于 $\lambda-\mathbb Z_{\ge0}\Phi^+$。$\square$

因此 Verma module 的所有权都从最高权向负根方向移动。Category $\mathcal O$ 正是把有限生成、有限维权空间和正幂零方向的局部有限性同时保留下来的范畴；这三项条件使 Verma modules 留在其中，也使后面的 block 分解有意义。

## 2.3 Category $\mathcal O$

**定义 2.10.** BGG category $\mathcal O$ 是满足下列条件的 $\mathfrak g$-modules 构成的 full subcategory：

1. $M$ 是 finitely generated $U(\mathfrak g)$-module；
2. $M$ 是 $\mathfrak t$-semisimple weight module，且每个 weight space 有限维；
3. $\mathfrak n$ 在 $M$ 上 locally finite，即对每个 $m\in M$，$U(\mathfrak n)m$ 有限维。

**命题 2.11.** 对任意 $\lambda$，Verma module $M(\lambda)$ 属于 $\mathcal O$。

**证明.** $M(\lambda)$ 由 $v_\lambda$ 生成，因此有限生成。由命题 2.9，$M(\lambda)$ 由 $U(\mathfrak n^-)$ 作用在 $v_\lambda$ 上生成，PBW monomials 给出 weight 分解。固定 weight $\lambda-\beta$ 时，只涉及把 $\beta$ 写成正根非负整数组合的有限种方式和有限个 PBW monomials，因此 weight space 有限维。最后，$\mathfrak n$ 对 PBW degree 有降低作用；对任一固定 PBW monomial 生成的向量，反复施加 $\mathfrak n$ 只能落在有限个较低 degree 的 weight space 中，所以 $U(\mathfrak n)m$ 有限维。$\square$

**外部输入定理 2.12.** $\mathcal O$ 是 abelian category，具有有限长度；其简单对象由 $L(\lambda)$ 标号。  
用途：后续 block decomposition 和 KL character formula。当前只作为外部输入。

## 2.4 中心 character 和 dot action

**定义 2.13.** 令
$$
Z(\mathfrak g)=Z(U(\mathfrak g)).
$$
若 $M$ 是 $\mathfrak g$-module，且每个 $z\in Z(\mathfrak g)$ 在 $M$ 上以标量 $\chi(z)$ 作用，则称 $M$ 有中心 character $\chi:Z(\mathfrak g)\to\mathbb C$。

中心 character 只给出 $Z(\mathfrak g)$ 上的代数同态；要把它改写成权的 Weyl orbit，需要 Harish--Chandra 同构。这里的 $\rho$-shift 不是记号修饰，而是线性 Weyl action 与最高权表示实际中心作用之间的校正。

**外部输入定理 2.14.** Harish-Chandra isomorphism 给出
$$
Z(\mathfrak g)\simeq S(\mathfrak t)^W
$$
的 $\rho$-shifted 版本。由此中心 characters 可由 $W$-dot-orbits in $\mathfrak t^\ast$ 参数化，其中
$$
w\cdot\lambda=w(\lambda+\rho)-\rho.
$$

**定义 2.15.** 对中心 character $\chi$，记 $\mathcal O_\chi$ 为 $\mathcal O$ 中所有被 $\ker\chi$ 的某个幂零次方杀掉的对象构成的 full subcategory。

**外部输入定理 2.16.** category $\mathcal O$ 分解为 blocks
$$
\mathcal O=\bigoplus_\chi \mathcal O_\chi
$$
在适当有限生成语境中成立；regular integral block 与 Weyl group combinatorics 紧密相关。

## 2.5 Harish-Chandra modules 的接口

**定义 2.17.** 令 $G_\mathbb R$ 为实 reductive Lie group，$K_\mathbb R$ 为 maximal compact subgroup，$K$ 为其复化。一个 $(\mathfrak g,K)$-module 是 $\mathfrak g$-module $M$ 和 algebraic $K$-action，满足两者在 $\mathfrak k$ 上的微分作用相容，并且 $\mathfrak g$-作用映射为 $K$-equivariant。

本书不会在第二章展开 real groups。这个定义只为后续 microlocal geometry、character sheaves 和 representation of real reductive groups 留接口。

Verma module 的泛性质把最高权向量变成一个通用代数对象，PBW 分解则控制它的权和有限性；中心 character 再把这些对象分入由 dot-orbit 控制的 blocks。几何化这些 blocks 还缺少一套能在奇异分层上稳定工作的层论语言，下一章从 constructible complexes 与 perverse t-structure 建立这套语言。

## 练习

**练习 2.1.** 对 $\mathfrak{sl}_2$ 写出 $M(\lambda)$ 的基、$e,f,h$ 作用公式和 weight 集。

**练习 2.2.** 证明命题 2.6 中的泛性质自然等价于 induction functor $U(\mathfrak g)\otimes_{U(\mathfrak b)}-$ 是 restriction functor 的左伴随。

**练习 2.3.** 在 $\mathfrak{sl}_2$ 情形中判断 $M(\lambda)$ 何时有非零 proper submodule，并与 $L(\lambda)$ 比较。
