# 第十二章：Affine Grassmannian、loop groups 与 convolution

有限旗簇上的 Schubert cells 只记录一次有限维相对位置；把两个 $G$-torsor 在 punctured formal disk 上比较，会得到由 coweight 标号的无限族相对位置，这正是 affine Grassmannian。符号 $LG/L^+G$ 容易掩盖两个关键问题：商必须作 fpqc sheafification，而卷积的 twisted external product 必须沿 $L^+G$-torsor 下降，不能从 contracted product 虚构一个到 Cartesian product 的映射。Betti 层还只能在有限个 Schubert closures 的支撑上使用通常六函子。以下先固定这些模型，再用 $GL_1$ 的离散格点和 $GL_2$ 的二重 minuscule lattice chain 计算卷积；后者会实际检验 properness、semismall inequality 与 IC 分解重数。

**约定 12.0.** 本章固定 $k=\mathbb C$、代数闭特征 $0$ 系数域 $E$，以及连通 reductive complex algebraic group $G$。记
$$
\mathscr O=\mathbb C[[z]],\qquad \mathscr K=\mathbb C((z)).
$$
所有 sheaves 都是第三章的 Betti constructible sheaves。Representability theorem 给出 ind-scheme 后，本章在形成 Betti category 时一律取其 reduction，并仍简记为 $\operatorname{Gr}_G$；Betti sheaves 看不见 nilpotent thickening。Affine Grassmannian 上的对象必须支撑在有限个 reduced finite-dimensional Schubert varieties 的并上；本章不定义无界支撑的六函子范畴。

## 12.1 Loop group、arc group 和 fpqc 商

**定义 12.1.** 对任意复代数 $R$，定义 group-valued functors
$$
LG(R)=G(R((z))),\qquad L^+G(R)=G(R[[z]]).
$$
$LG$ 称为 loop group，$L^+G$ 称为 positive loop group 或 arc group。作用一律为左作用，除非 diagram 中显式写出右作用。

**定义 12.2.** Presheaf $R\mapsto LG(R)/L^+G(R)$ 的 fpqc sheafification 记为
$$
\operatorname{Gr}_G:=LG/L^+G.
$$
商号不是逐点商的简写：对一般 $R$，一个 fpqc-local coset 未必由单个 $LG(R)$ 元素表示。等价的 moduli 描述是：$\operatorname{Gr}_G(R)$ 参数化形式圆盘 $\operatorname{Spec}R[[z]]$ 上的 $G$-torsor，连同其在 punctured disk $\operatorname{Spec}R((z))$ 上的 trivialization；isomorphisms 必须保持 trivialization。

**外部输入定理 12.3（representability）.** 对约定 12.0 中的 $G$，fpqc sheaf $\operatorname{Gr}_G$ 可表示为 ind-projective ind-scheme
$$
\operatorname{Gr}_G=\varinjlim_n Z_n,
$$
其中 transition maps 是 closed immersions，$Z_n$ 可取为 $L^+G$-stable projective schemes，且 $L^+G$ 在每个 $Z_n$ 上的作用 factor through 某个 jet quotient $G(\mathscr O/z^N)$。这一表示性结果采用 Mirkovic--Vilonen §2 与 Zhu 的 affine-Grassmannian 讲义；functor 的 sheafification、moduli 描述和 ind-representability 均不在本书重证。

**例 12.4.** 对 $G=GL_n$，$\operatorname{Gr}_G(\mathbb C)$ 的点可解释为 $\mathscr K^n$ 中的 $\mathscr O$-lattices，即 free rank-$n$ $\mathscr O$-submodules $L$，满足
$$
z^N\mathscr O^n\subset L\subset z^{-N}\mathscr O^n
$$
对某个 $N\ge0$ 成立。Coset $gL^+G$ 对应 lattice $g\mathscr O^n$；右乘 $L^+G$ 只改变 $\mathscr O^n$ 的基，故不改变 lattice。

**命题 12.5.** 对 $G=GL_1$，约定 12.0 的 reduced ind-scheme 有离散分解
$$
(\operatorname{Gr}_{GL_1})_{\mathrm{red}}
\simeq\coprod_{m\in\mathbb Z}\operatorname{Spec}\mathbb C.
$$

**证明.** 对 $R=\mathbb C$，任一 $f\in\mathscr K^\times$ 唯一写成
$$
f=z^m u,\qquad m\in\mathbb Z,\quad u\in\mathscr O^\times.
$$
故 geometric points 由 valuation $m\in\mathbb Z$ 参数化。又因 $GL_1$ abelian，$L^+GL_1$ 固定每个 coset $z^mL^+GL_1$；相应 orbit 是一个 reduced point，且不同 valuations 位于不同 components。`AFFGR-1` 的 orbit exhaustion 因而在 reduction 上给出所述离散并。$\square$

**反例边界 12.5.1（不能升级为 fpqc functor 等式）.** 取
$R=\mathbb C[\varepsilon]/(\varepsilon^2)$。Loop
$$
1+\varepsilon z^{-1}\in R((z))^\times
$$
不属于 $R[[z]]^\times$，却在 $R/\varepsilon$ 上退化为单位 coset。它给出中性点处的 infinitesimal family。因此未取 reduction 的 fpqc quotient 不能由 geometric-point valuation 计算成 constant sheaf $\mathbb Z$；命题 12.5 只陈述本章 Betti 模型实际使用的 reduced ind-scheme。

## 12.2 Orbit stratification 与 finite-support category

**定义 12.6.** 由第一章固定的 $B\supset T$ 决定正根和 dominant coweights $X_\ast(T)^+$。对 $\lambda\in X_\ast(T)^+$，令 $z^\lambda\in T(\mathscr K)\subset G(\mathscr K)$ 为 coweight 在 $z$ 处的值，并定义
$$
\operatorname{Gr}^\lambda=L^+G\cdot z^\lambda L^+G/L^+G,
\qquad
\overline{\operatorname{Gr}}^\lambda
=\overline{\operatorname{Gr}^\lambda}.
$$
若 $\mu,\lambda$ 位于同一 $\pi_1(G)$-component，写 $\mu\le\lambda$ 表示 $\lambda-\mu$ 是非负整数系数的 positive coroots 之和；不同 components 的 coweights 不可比。

**外部输入定理 12.7（Cartan 分层）.** $L^+G$-orbits on $\operatorname{Gr}_G$ 由 $X_\ast(T)^+$ 参数化，并且
$$
\operatorname{Gr}_G=\coprod_{\lambda\in X_\ast(T)^+}\operatorname{Gr}^\lambda,
\qquad
\overline{\operatorname{Gr}}^\lambda
=\coprod_{\mu\le\lambda}\operatorname{Gr}^\mu.
$$
每个 $\overline{\operatorname{Gr}}^\lambda$ 是 projective variety，且
$$
\dim_{\mathbb C}\operatorname{Gr}^\lambda
=\langle2\rho,\lambda\rangle.
$$
这些 orbit 与 closure 公式见 Mirkovic--Vilonen §2。

**定义 12.8.** 记 $D^b_{L^+G,\mathrm{fs}}(\operatorname{Gr}_G,E)$ 为以下 equivariant constructible category：对象 $\mathcal F$ 支撑在某个有限并
$$
\bigcup_{j=1}^r\overline{\operatorname{Gr}}^{\lambda_j}
$$
上，并在包含该支撑的 finite-dimensional $L^+G$-stable stage 上，按第三章的 forgetful convention 对某个 jet quotient $G(\mathscr O/z^N)$ equivariant。不同 stage 与更大 $N$ 之间通过 closed pull--push 和 quotient compatibility 识别。定义 spherical Satake heart
$$
\operatorname{Sat}_G
:=\operatorname{Perv}_{L^+G,\mathrm{fs}}(\operatorname{Gr}_G,E).
$$
对 $\lambda\in X_\ast(T)^+$，记
$$
\operatorname{IC}_\lambda
:=\operatorname{IC}(\overline{\operatorname{Gr}}^\lambda,
E_{\operatorname{Gr}^\lambda}).
$$

**例 12.9.** 对 $G=GL_n$，采用
$$
\lambda=(\lambda_1\ge\cdots\ge\lambda_n)
$$
作为 dominant convention，点 $z^\lambda$ 对应 lattice
$$
z^{\lambda_1}\mathscr Oe_1\oplus\cdots\oplus
z^{\lambda_n}\mathscr Oe_n.
$$
Smith normal form 表明每个 lattice 都落在唯一一个这样的 orbit。这个式子也固定了本章的正负号：正的 $\lambda_i$ 使相应 lattice direction 缩小，而不是放大。

## 12.3 Convolution Grassmannian、descent 与 properness

**定义 12.10.** 在 $LG\times\operatorname{Gr}_G$ 上定义右 $L^+G$-作用
$$
(g,x)\cdot h=(gh^{-1},hx).
$$
convolution Grassmannian 是 fpqc contracted product
$$
\operatorname{Gr}_G\widetilde\times\operatorname{Gr}_G
:=LG\times^{L^+G}\operatorname{Gr}_G.
$$
它进入 diagram
$$
\operatorname{Gr}_G\times\operatorname{Gr}_G
\xleftarrow{\ p\ }
LG\times\operatorname{Gr}_G
\xrightarrow{\ q\ }
\operatorname{Gr}_G\widetilde\times\operatorname{Gr}_G
\xrightarrow{\ m\ }
\operatorname{Gr}_G,
$$
其中
$$
p(g,x)=(gL^+G,x),\qquad q(g,x)=[g,x],\qquad m([g,x])=gx.
$$
$m$ 良定义，因为 $(gh^{-1})(hx)=gx$。注意 $p$ 不沿 $q$ factor；因此 contracted product 上没有由这个公式诱导的 map 到普通 Cartesian product。

**定义 12.11（twisted external product）.** 对
$\mathcal F,\mathcal G\in D^b_{L^+G,\mathrm{fs}}(\operatorname{Gr}_G,E)$，$\mathcal G$ 的 $L^+G$-equivariance 使
$$
p^\ast(\mathcal F\boxtimes\mathcal G)
$$
具有关于 torsor $q$ 的 coherent descent datum。定义
$\mathcal F\widetilde\boxtimes\mathcal G$ 为唯一满足
$$
q^\ast(\mathcal F\widetilde\boxtimes\mathcal G)
\simeq p^\ast(\mathcal F\boxtimes\mathcal G)
$$
的 constructible complex。这个等式是 descent characterization，不是把 $q^\ast$ 形式消去后得到的普通 pullback 公式。

**外部输入定理 12.12（finite-support properness）.** 对 dominant coweights $\lambda,\mu$，令
$$
\overline{\operatorname{Gr}}^\lambda
\widetilde\times
\overline{\operatorname{Gr}}^\mu
\subset
\operatorname{Gr}_G\widetilde\times\operatorname{Gr}_G
$$
为第一和第二个 relative positions 分别不超过 $\lambda,\mu$ 的 finite-type closed subvariety。则 restriction
$$
m_{\lambda,\mu}:
\overline{\operatorname{Gr}}^\lambda
\widetilde\times
\overline{\operatorname{Gr}}^\mu
\longrightarrow
\overline{\operatorname{Gr}}^{\lambda+\mu}
$$
是 proper，且相对于 $L^+G$-orbit stratifications 是 stratified semismall。这里采用 Mirkovic--Vilonen §4，尤其 Proposition 4.2 与 Lemma 4.4；一般 ind-map $m$ 的符号本身不替代这一 finite-support statement。

Properness 只在有限支撑上成立，正好解释了定义 12.8 的限制。得到 proper pushforward 后，卷积仍需检查 stage 独立性、结合约束与单位；这些都来自同一个 torsor descent，而不是另行选择同构。

**定义 12.13（derived convolution）.** 取有限集合
$\Lambda,M\subset X_\ast(T)^+$，使
$$
\operatorname{supp}\mathcal F\subset
X_\Lambda:=\bigcup_{\lambda\in\Lambda}
\overline{\operatorname{Gr}}^\lambda,
\qquad
\operatorname{supp}\mathcal G\subset
X_M:=\bigcup_{\mu\in M}
\overline{\operatorname{Gr}}^\mu.
$$
这样允许输入同时落在不同 $\pi_1(G)$-components。令
$$
m_{\Lambda,M}:X_\Lambda\widetilde\times X_M
\longrightarrow
X_{\Lambda+M}:=
\bigcup_{\substack{\lambda\in\Lambda\\\mu\in M}}
\overline{\operatorname{Gr}}^{\lambda+\mu}
$$
为定理 12.12 的有限个 proper restrictions 的并，定义
$$
\mathcal F\star\mathcal G
:=Rm_{\Lambda,M\ast}
(\mathcal F\widetilde\boxtimes\mathcal G)
\in D^b_{L^+G,\mathrm{fs}}(\operatorname{Gr}_G,E).
$$
定理 12.12 给出 $Rm_!\simeq Rm_\ast$。扩大 $\Lambda,M$ 时，closed-embedding base change 给出同一对象，所以定义独立于 finite stages。Perverse normalization 已包含在输入和 twisted external product 中；本书不再加入“按需要”的未声明 shift。

**命题 12.14（结合性）.** Derived convolution 带有满足 pentagon 的自然 associator
$$
(\mathcal F\star\mathcal G)\star\mathcal H
\xrightarrow{\sim}
\mathcal F\star(\mathcal G\star\mathcal H).
$$

**证明.** 三重 contracted product 为
$$
\operatorname{Gr}_G\widetilde\times
\operatorname{Gr}_G\widetilde\times
\operatorname{Gr}_G
=LG\times^{L^+G}LG\times^{L^+G}\operatorname{Gr}_G.
$$
在包含三个输入支撑的 finite-type closed subvariety 上，两种迭代卷积的目标 map 都是
$$
m_3([g_1,g_2,x])=g_1g_2x,
$$
且 properness 由定理 12.12 对迭代支撑给出。Torsor descent 的 transitivity 把两种 iterated twisted external products 都识别为同一个
$\mathcal F\widetilde\boxtimes\mathcal G\widetilde\boxtimes\mathcal H$。附录 F 命题 F.3 的两个 base-change calculations 因而都化为
$$
Rm_{3\ast}
(\mathcal F\widetilde\boxtimes\mathcal G
\widetilde\boxtimes\mathcal H).
$$
这给出 associator。四重 contracted product 上各条比较都由同一个 quotient 和严格相等的 group products
$(g_1g_2)g_3=g_1(g_2g_3)$ 诱导，故 F.3 的 coherence 假设成立，associator 满足 pentagon。$\square$

**命题 12.15（单位）.** 令 $e=L^+G/L^+G\in\operatorname{Gr}_G$，并令 $\mathbf1=E_e$ 为其 skyscraper perverse sheaf。则
$$
\mathbf1\star\mathcal F\simeq\mathcal F
\simeq\mathcal F\star\mathbf1.
$$

**证明.** 若第一个 modification 是中性 modification，则相应 convolution subspace 由 $x\mapsto[e,x]$ 与 $\operatorname{supp}\mathcal F$ 同构，且 $m([e,x])=x$。若第二个 modification 中性，则 $[g,e]\mapsto gL^+G$ 给出同样的同构。Twisted external product 在这两个 subspaces 上分别限制为 $\mathcal F$，目标 map 是 identity；命题 F.5 给出两侧自然同构。$\square$

## 12.4 两个可完全计算的低秩模型

抽象的 descent 图并不显示 pushforward 会分解成哪些 IC 层。$GL_1$ 先把卷积化为 coweight 加法；$GL_2$ 随后出现第一条非平凡 fiber，闭 stratum 上的 $\mathbb P^1$ 恰好贡献额外的 relevant summand。

**命题 12.16（$GL_1$ convolution）.** 对整数 $a,b$，令 $E_a,E_b$ 为相应 points 上的 skyscraper sheaves，则
$$
E_a\star E_b\simeq E_{a+b}.
$$

**证明.** 命题 12.5 把两个输入支撑识别为 points $z^a\mathscr O$ 与 $z^b\mathscr O$。其 contracted product 仍为一点，且
$$
m([z^a,z^b\mathscr O])=z^{a+b}\mathscr O.
$$
一点到一点的 proper pushforward 保持 $E$，故所得对象正是 $E_{a+b}$。$\square$

**命题 12.17（$GL_2$ 的二重 minuscule modification）.** 令 $G=GL_2$、$L_0=\mathscr O^2$、$\lambda=(1,0)$，并置
$$
Z=\overline{\operatorname{Gr}}^\lambda
\widetilde\times
\overline{\operatorname{Gr}}^\lambda,
\qquad
Y=\overline{\operatorname{Gr}}^{(2,0)}.
$$
则：

1. $\overline{\operatorname{Gr}}^\lambda\simeq\mathbb P^1$；
2. $Z$ 是 smooth projective surface，并且是 $\mathbb P^1$ over $\mathbb P^1$ 的 bundle；
3. convolution map $m:Z\to Y$ 在 $\operatorname{Gr}^{(2,0)}$ 上 fiber 为一点，在唯一 closed stratum $zL_0=\operatorname{Gr}^{(1,1)}$ 上 fiber 为 $\mathbb P^1$；
4. $m$ 是 proper semismall birational morphism。

**证明.** Lattice $L\in\overline{\operatorname{Gr}}^{(1,0)}$ 满足
$$
zL_0\subset L\subset L_0,
\qquad \dim_{\mathbb C}(L_0/L)=1.
$$
映射 $L\mapsto L/zL_0$ 把它识别为二维空间 $L_0/zL_0$ 中的 lines，故得到 $\mathbb P^1$，证明 1。

$Z$ 的点是 lattice chains
$$
L_0\supset L_1\supset L_2,
\qquad
zL_{i-1}\subset L_i,\quad
\dim_{\mathbb C}(L_{i-1}/L_i)=1
\quad(i=1,2).
$$
先选 $L_1$ 给出 base $\mathbb P^1$；固定 $L_1$ 后，$L_2/zL_1$ 是二维空间 $L_1/zL_1$ 中的一条 line，所以 fiber 也是 $\mathbb P^1$。这些 quotients 在 base 上组成 rank-$2$ vector bundle，故 $Z$ 是其 projectivization，因而 smooth projective、纯维数 $2$。这证明 2。

$m$ 忘掉 $L_1$。对 $L_2\in Y$，length-$2$ $\mathscr O$-module
$$
Q=L_0/L_2
$$
由 Smith normal form 只有两种类型。若 $L_2\in\operatorname{Gr}^{(2,0)}$，则 $Q\simeq\mathscr O/(z^2)$；intermediate lattice 对应 $Q$ 的 length-$1$ submodule，而唯一这样的 submodule 是 $zQ$，故 fiber 是一点。此时逆映射把 $L_2$ 送到 $zQ\subset Q$ 在 $L_0$ 中的 inverse image，因此在 open stratum 上 algebraic。若 $L_2$ 类型为 $(1,1)$，则 $Q\simeq(\mathscr O/(z))^2$，并且 $L_2=zL_0$；intermediate lattices 对应 $Q$ 中的 lines，故 fiber 是 $\mathbb P^1$。所以 $m$ 在 dense open stratum 上为同构，因而 birational，并证明 3。

定理 12.12 给出 properness。由定理 12.7，
$$
\dim\operatorname{Gr}^{(2,0)}=2,
\qquad
\dim\operatorname{Gr}^{(1,1)}=0.
$$
在 open stratum 上 $2\dim m^{-1}(y)=0$，在 closed stratum 上
$$
2\dim m^{-1}(zL_0)=2=\dim Z-0.
$$
两条 semismall inequalities 均成立，证明 4。$\square$

**推论 12.18（一个几何证明的 tensor-square 分解）.** 在约定 12.0 的 characteristic-zero Betti category 中，
$$
\operatorname{IC}_{(1,0)}\star\operatorname{IC}_{(1,0)}
\simeq
\operatorname{IC}_{(2,0)}\oplus
\operatorname{IC}_{(1,1)}.
$$

**证明.** 命题 12.17(1) 和命题 3.17 给出
$$
\operatorname{IC}_{(1,0)}\simeq E_{\mathbb P^1}[1].
$$
Twisted external product 在 smooth surface $Z$ 上下降为 $E_Z[2]$，所以左侧等于 $Rm_\ast E_Z[2]$。命题 12.17 证明 $m$ proper semismall；调用外部输入定理 C.7。两个 strata 都 relevant。Dense open stratum 上 fiber 为一点，故其 local system 为 rank one；closed point 上
$$
H^{BM}_2(m^{-1}(zL_0),E)
=H_2(\mathbb P^1,E)\simeq E,
$$
也给出 multiplicity one。因此 semismall decomposition 正是
$$
Rm_\ast E_Z[2]
\simeq\operatorname{IC}_{(2,0)}\oplus E_{zL_0}.
$$
而 $E_{zL_0}=\operatorname{IC}_{(1,1)}$，得到结论。这里 splitting 的存在和 relevant-stratum intersection form 的非退化性属于 `BBD-SS-1`，本书内部完成的是 lattice fibers、properness 引用和 semismall dimension check。$\square$

$GL_1$ 表明 spherical convolution 延续了 coweight 加法，$GL_2$ 则展示闭 fiber 的 top homology 如何增加一个 IC summand。一般情形中，finite-support properness 和 torsor descent 已给出可结合的卷积，但还没有证明 perverse heart 在卷积下封闭，也没有解释所得 tensor category 对应哪个代数群。第十三章用 fusion、weight functors 与 Tannakian reconstruction 回答这两个问题。

## 练习

**练习 12.1.** 证明 $GL_1$ 的 valuation 在任意 field extension $\mathbb C\subset E'$ 后仍参数化 geometric points，并说明为什么这不足以证明 fpqc quotient 的 representability。

**练习 12.2.** 对 $GL_n$，证明 $g\mathscr O^n$ 只依赖 coset $gL^+G$，并用 Smith normal form 证明 geometric-point orbit classification。

**练习 12.3.** 写出三重 contracted product 的两个 $L^+G\times L^+G$ actions，验证 $m_3([g_1,g_2,x])=g_1g_2x$ 在商上良定义。

**练习 12.4.** 在命题 12.17 中证明 $Q\simeq\mathscr O/(z^2)$ 只有一个 length-$1$ submodule，并据此直接验证 $m$ 在 open stratum 上是同构。
