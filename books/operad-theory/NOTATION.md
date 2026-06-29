# 符号约定

本文档记录《Operad Theory》的固定符号。后续章节不得随意更改。

## 集合论与大小

- 固定 Grothendieck universes
  $$
  \mathcal U\in\mathcal V\in\mathcal W.
  $$
  若不特别说明，“集合”指 $\mathcal U$-小集合。
- $\mathbf{Set}_{\mathcal U}$：$\mathcal U$-小集合范畴。
- $\mathbf{Fin}_{\mathcal U}$：$\mathcal U$-小有限集和函数构成的范畴。
- $\mathbf{B}_{\mathcal U}$：$\mathcal U$-小有限集和双射构成的群胚。
- $[n]=\{1,\ldots,n\}$；特别地 $[0]=\varnothing$。
- $\Sigma_n=\operatorname{Aut}_{\mathbf{B}_{\mathcal U}}([n])$。

## 对称序列与有限分块

- 有限集 $S$ 的一个分块记为 $\pi$，其块集合记为 $\operatorname{Bl}(\pi)$。
- 若 $\pi$ 是 $S$ 的分块，则 $S=\coprod_{B\in\operatorname{Bl}(\pi)}B$。
- 对称序列（symmetric sequence）默认指函子
  $$
  X:\mathbf{B}_{\mathcal U}\to\mathbf{Set}_{\mathcal U}.
  $$
  等价地，它是带 $\Sigma_n$ 作用的集合族 $X([n])$。
- 为避免左右作用歧义，有限集口径是默认定义；使用 $X(n)$ 时表示 $X([n])$。

## 代入乘积与 operad

- 对称序列的代入乘积写作 $X\circ Y$。
- 单位对称序列写作 $I$，满足 $I(S)=\{*\}$ 当 $|S|=1$，否则 $I(S)=\varnothing$。
- operad 默认指 $\big(\operatorname{SymSeq}_{\mathcal U},\circ,I\big)$ 中的幺半对象。
- operad 的乘法写作
  $$
  \mu:\mathcal O\circ\mathcal O\to\mathcal O,
  $$
  单位写作
  $$
  \eta:I\to\mathcal O.
  $$
- 用 arity 写法时，单位元素写作 $\mathbf 1\in\mathcal O(1)$。

## Endomorphism operad 与代数

- 对集合 $X$，endomorphism operad 写作
  $$
  \operatorname{End}_X(S)=\mathbf{Set}_{\mathcal U}(X^S,X).
  $$
- $\mathcal O$-代数默认指集合 $X$ 与 operad morphism
  $$
  \alpha:\mathcal O\to\operatorname{End}_X.
  $$
- 对 $o\in\mathcal O(S)$，由 $\alpha$ 给出的具体运算写作
  $$
  \alpha(o):X^S\to X.
  $$
- $\operatorname{Alg}_{\mathcal O}(\mathbf{Set}_{\mathcal U})$：集合值 $\mathcal O$-代数范畴。
- $U_{\mathcal O}$：从 $\mathcal O$-代数到集合的遗忘函子。
- $F_{\mathcal O}$：自由 $\mathcal O$-代数函子。
- $\mathbb T_{\mathcal O}=U_{\mathcal O}F_{\mathcal O}$：由 $\mathcal O$ 确定的集合 monad。

## 非对称 operad 与树

- 非对称序列写作 $X=\{X(n)\}_{n\ge0}$。
- 非对称代入乘积写作 $\circ_{\mathrm{ns}}$。
- 非对称 operad 的偏复合写作
  $$
  p\circ_i q.
  $$
- 平面有根树通常记为 $T$，内部顶点集合记为 $V(T)$，叶集合记为 $\operatorname{Leaf}(T)$。
- 顶点 $v$ 的输入边集合或输入数分别写作 $\operatorname{In}(v)$、$\operatorname{in}(v)$，上下文会区分集合和基数。
- 自由非对称 operad 写作 $\mathbb F_{\mathrm{ns}}(E)$。
- 自由对称 operad 写作 $\mathbb F(E)$。
- 生成元与关系给出的 operad 写作 $\langle E\mid R\rangle$。
- $\mathbf{Tree}_S$：$S$-叶标号有根非平面树及其同构构成的群胚。
- $\operatorname{Dec}_E(T)=\prod_{v\in V(T)}E(\operatorname{In}(v))$：$E$-装饰树 $T$ 的装饰集合。

## Colored 与线性 operad

- $C$：颜色集合。
- $\mathbf B_C$：$C$-轮廓及其同构构成的群胚。
- $\operatorname{SymSeq}_C$：$C$-colored symmetric sequences 范畴。
- Colored 轮廓写作 $(S,\kappa;c)$，其中 $\kappa:S\to C$ 是输入颜色函数，$c$ 是输出颜色。
- Colored substitution product 写作 $\circ_C$。
- $I_C$：$C$-colored symmetric sequence 的单位。
- $\operatorname{End}_A$：$C$-indexed 集合族 $A=(A_c)_{c\in C}$ 的 colored endomorphism operad。
- $R$：线性 operad 章节中的固定交换环。
- $\mathbf{Mod}_R$：交换环 $R$ 上的模范畴。
- $S_M$：$R$-模值对称序列 $M$ 的 Schur functor。
- $(V^{\otimes n})_{\Sigma_n}$：$\Sigma_n$ 作用下的 coinvariants。
- $\operatorname{Pois}_R$：Poisson operad。

## PROP、properad 与 Koszul 对偶

- PROP 记为 $\mathsf P$，并采用
  $$
  \mathsf P(m,n)=\operatorname{Hom}_{\mathsf P}(n,m)
  $$
  表示 $n$ 输入、$m$ 输出运算。
- $\mathbb S$-双模表示带左 $\Sigma_m$ 和右 $\Sigma_n$ 作用的族 $M(m,n)$。
- $\operatorname{End}_V(m,n)=\operatorname{Hom}(V^{\otimes n},V^{\otimes m})$。
- Properad 通常记为 $\mathcal P$，其图复合沿连通 directed graphs。
- Wheeled contraction 写作 $\operatorname{tr}_i^j$。
- 二次数据写作 $(E,R)$，二次 operad 写作 $\mathcal P(E,R)$。
- 自由 operad 的权重 $r$ 部分写作 $\mathbb F^{(r)}(E)$。
- Ginzburg-Kapranov 二次对偶写作 $\mathcal P^!$。
- 二次对偶 cooperad 写作 $\mathcal P^¡$。

## Bar-cobar 与 twisting

- $\mathbf{Ch}_k$：域 $k$ 上链复形范畴，使用同调次数。
- $sC$、$s^{-1}C$：悬挂和去悬挂。
- $\overline{\mathcal P}$：augmented operad 的增广理想。
- $\overline{\mathcal C}$：coaugmented cooperad 的 coaugmentation coideal。
- $\mathbb T^c(M)$：由 dg 对称序列 $M$ 生成的 cofree conilpotent cooperad。
- $\circ_{(1)}$：infinitesimal composition product。
- $\Delta_{(1)}$：cooperad 的 infinitesimal decomposition。
- $\operatorname{Tw}(\mathcal C,\mathcal P)$：从 $\mathcal C$ 到 $\mathcal P$ 的 twisting morphism 集合。
- $\Omega\mathcal C$：cobar construction。
- $B\mathcal P$：bar construction。
- $\operatorname{Conv}(\mathcal C,\mathcal P)$：convolution dg Lie algebra $\operatorname{Hom}_{\mathbb S}(\overline{\mathcal C},\overline{\mathcal P})$。
- $\star$：convolution pre-Lie product。
- $\mathcal C\circ_\alpha\mathcal P$、$\mathcal P\circ_\alpha\mathcal C$：由 twisting morphism $\alpha$ 定义的 twisted composite products。
- $K_r(\mathcal P)=\mathcal P^¡\circ_\kappa\mathcal P$：右 Koszul complex。
- $K_l(\mathcal P)=\mathcal P\circ_\kappa\mathcal P^¡$：左 Koszul complex。

## 同伦代数 operad

- $\mathcal P_\infty=\Omega\mathcal P^¡$：Koszul operad $\mathcal P$ 的同伦版本。
- $A_\infty=\Omega\operatorname{Ass}^¡$。
- $L_\infty=\Omega\operatorname{Com}^¡$。
- $C_\infty=\Omega\operatorname{Lie}^¡$。
- $m_n:A^{\otimes n}\to A$：$A_\infty$-代数结构映射，使用同调约定时 $|m_n|=n-2$。
- $\ell_n:V^{\otimes n}\to V$：$L_\infty$-代数高阶括号，使用同调约定时 $|\ell_n|=n-2$。
- $\mathcal C_n$：little $n$-cubes operad。
- $E_n$-operad：与 $\mathcal C_n$ 弱等价的拓扑 operad，或与 $C_\*(\mathcal C_n;k)$ 弱等价的 dg-operad。
- $\operatorname{Pois}_n$：$n$-Poisson operad。

## Hochschild、Gerstenhaber 与 brace

- $C^n(A,A)=\operatorname{Hom}_k(A^{\otimes n},A)$：Hochschild $n$-cochains。
- $HH^\*(A,A)$：Hochschild cohomology。
- $\smile$：Hochschild cup product。
- $f\circ_i g$：第 $i$ 个 Hochschild insertion。
- $[f,g]$：Gerstenhaber bracket。
- $\operatorname{Ger}$：Gerstenhaber operad。
- $\operatorname{BV}$：BV operad。
- $\Delta$：BV operator。
- $\operatorname{Br}$：brace operad。
- $f\{g_1,\ldots,g_r\}$：brace operation。

## 同伦转移与最小模型

- Contraction 记作
  $$
  H\xrightarrow{i}A\xrightarrow{p}H,\qquad h:A\to A[1].
  $$
- $pi=\operatorname{id}_H$，$ip-\operatorname{id}_A=dh+hd$ 是 contraction 的基本恒等式。
- $m_n^H$：转移到 $H$ 上的 $A_\infty$ 运算。
- $\ell_n^H$：转移到 $H$ 上的 $L_\infty$ 运算。
- $\operatorname{PBT}_n$：有 $n$ 个叶的平面二叉树集合。
- $\Phi_T$、$M_T$：同伦转移附录中由平面二叉树 $T$ 定义的 $A$-值和 $H$-值映射。
- $I_\infty:H\rightsquigarrow A$：转移定理给出的 $A_\infty$ quasi-isomorphism。
- Minimal $\mathcal P_\infty$-algebra：底层微分为零的 $\mathcal P_\infty$-代数。
- Formal dg algebra：与其同调代数通过 dg algebra quasi-isomorphism zigzag 相连的 dg algebra。

## 模型范畴中的 operad

- $\mathsf W$：模型范畴中的弱等价类。
- $\mathsf{Cof}$：cofibration 类。
- $\mathsf{Fib}$：fibration 类。
- $i\square j$：pushout-product。
- $\operatorname{Op}(\mathcal M)$：$\mathcal M$-值 operad 范畴。
- $U:\operatorname{Op}(\mathcal M)\to\operatorname{SymSeq}(\mathcal M)$：底层对称序列遗忘函子。
- $\mathbb F$：自由 operad 函子。
- $\operatorname{Alg}_{\mathcal O}(\mathcal M)$：$\mathcal O$-代数范畴。
- $F_{\mathcal O}\dashv U_{\mathcal O}$：自由 $\mathcal O$-代数与遗忘函子的伴随。
- $\Sigma$-cofibrant：逐 arity 在 $\mathcal M^{\Sigma_n}$ 中 cofibrant。
- $Q\mathcal O\to\mathcal O$：operad 的 cofibrant resolution。
- $W\mathcal O$：Boardman-Vogt 型 resolution。
- $\mathbf R\operatorname{Map}_{\operatorname{Op}(\mathcal M)}(\mathcal P,\mathcal Q)$：operads 的 derived mapping space。
- Admissible operad：其代数范畴具有从底层对象转移来的模型结构。
- Symmetric h-monoidality / symmetric flatness：控制对称幂和 colored operad 代数 admissibility/rectification 的技术条件；具体定义依赖引用文献。

## Simplicial 与 topological operad

- $\mathbf{sSet}$：simplicial sets 范畴，默认采用 Kan-Quillen 模型结构。
- $\mathbf{Top}$：compactly generated weak Hausdorff spaces 范畴。
- $|-|:\mathbf{sSet}\to\mathbf{Top}$：几何实现函子。
- $\operatorname{Sing}:\mathbf{Top}\to\mathbf{sSet}$：奇异复形函子。
- $\operatorname{Op}(\mathbf{sSet})$：simplicial operads 范畴。
- $\operatorname{Op}(\mathbf{Top})$：topological operads 范畴。
- $\mathcal C_d$：little $d$-cubes operad。
- $C_\*(-;k)$：系数在 $k$ 中的奇异链函子。

## Dendroidal sets

- $\Omega$：Moerdijk-Weiss 树范畴。
- $E(T)$：树 $T$ 的边集。
- $V(T)$：树 $T$ 的顶点集。
- $r_T$：树 $T$ 的根边。
- $\Omega(T)$：由树 $T$ 自由生成的 colored operad。
- $\eta$：只有一条边且无顶点的单位树。
- $C_n$：$n$-corolla。
- $L_n$：有 $n$ 个 unary 顶点的线性树。
- $\mathbf{dSet}$：dendroidal sets，即 $\operatorname{Fun}(\Omega^{\operatorname{op}},\mathbf{Set})$。
- $\Omega[T]$：由树 $T$ 表示的 representable dendroidal set。
- $N_d(\mathcal P)$：colored operad $\mathcal P$ 的 dendroidal nerve。
- $\operatorname{Sc}[T]$：树 $T$ 的 Segal core。
- $\partial\Omega[T]$：representable $\Omega[T]$ 的边界。
- $\Lambda^e[T]$：inner edge $e$ 对应的 dendroidal inner horn。
- $i:\Delta\to\Omega$：把 ordinal 送到线性树的嵌入函子。
- Inner Kan dendroidal set：每个 $\Lambda^e[T]\to X$ 均可延拓到 $\Omega[T]\to X$ 的 dendroidal set。
- Normal monomorphism：新增 nondegenerate dendrexes 上树自同构群自由作用的 monomorphism。
- Inner anodyne map：由 inner horn inclusions 生成的弱饱和类中的态射。
- Operadic weak equivalence：Cisinski-Moerdijk operadic model structure 中的弱等价。

## Lurie-style infinity-operad

- $\mathbf{Fin}_*$：有限有基点集合范畴。
- $\langle n\rangle=\{*,1,\ldots,n\}$：标准有限有基点集合。
- Inert morphism：每个非基点目标元素有唯一原像的基点保持映射。
- Active morphism：只把基点映到基点的基点保持映射。
- $\rho^i:\langle n\rangle\to\langle1\rangle$：第 $i$ 个 inert projection。
- $\mu_n:\langle n\rangle\to\langle1\rangle$：active multiplication map。
- $\mathcal O^\otimes\to N(\mathbf{Fin}_*)$：Lurie-style infinity-operad。
- $\mathcal C^\otimes\to N(\mathbf{Fin}_*)$：symmetric monoidal infinity-category。
- $\operatorname{Alg}_{\mathcal O}(\mathcal C)$：$\mathcal C$ 中的 $\mathcal O$-algebras infinity-category。
- $\mathcal P^\otimes$：ordinary colored operad $\mathcal P$ 的 category of operators。

## Localization 与 straightening

- $(\mathcal C,\mathcal W)$：relative category。
- $L(\mathcal C,\mathcal W)$：Dwyer-Kan localization。
- $L^H(\mathcal C,\mathcal W)$：hammock localization。
- $\mathcal M_\infty$：模型范畴 $\mathcal M$ 的 underlying infinity-category。
- $N_\Delta$：simplicial category 的 homotopy coherent nerve。
- $\mathbf L F$、$\mathbf R G$：Quillen adjunction 的左/右导出 infinity-functors。
- $\operatorname{St}_S$：straightening functor over $S$。
- $\operatorname{Un}_S$：unstraightening functor over $S$。
- $X\otimes^{\mathbf L}Y$：derived tensor product。
- $\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty$：$\mathcal O$-代数模型范畴的 localization。
- $\mathcal O^{\operatorname{loc}}$：严格 operad $\mathcal O$ 经 localization/category-of-operators 得到的 infinity-operad；不得与 Koszul resolution $\mathcal O_\infty$ 混用。

## Factorization 与 Fukaya

- $\mathbf{Disk}_n$：有限个 $\mathbb R^n$ 不交并及 embeddings 构成的 disk category。
- $\mathbf{Disk}_{n/M}$：嵌入 $n$-manifold $M$ 的 disk category over $M$。
- $\operatorname{Fact}^{lc}_M(\mathcal C)$：$M$ 上取值于 $\mathcal C$ 的 locally constant factorization algebras。
- $\int_M A$：$E_n$-algebra $A$ 在 $n$-manifold $M$ 上的 factorization homology。
- $E_n$：little disks/cubes 型 $n$-fold monoidal infinity-operad。
- $\mathcal F(X)$：symplectic manifold $X$ 的 Fukaya category，具体模型需额外说明。
- $m_r$：$A_\infty$-category 的 $r$ 元 composition operation。

## 后续章节保留符号

- $\mathbf{Ch}_R$：交换环 $R$ 上链复形范畴，除非另说使用同调次数。
- $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$：结合、交换、Lie operad。

## 经典例子附录

- $\operatorname{Lin}(S)$：有限集 $S$ 上全序的集合。
- $T(V)=\bigoplus_{n\ge0}V^{\otimes n}$：$R$-模 $V$ 上的张量代数。
- $\operatorname{Sym}_R(V)=\bigoplus_{n\ge0}(V^{\otimes n})_{\Sigma_n}$：$R$-模 $V$ 上的对称代数。
- $R[\mathcal O]$：集合值 operad $\mathcal O$ 的自由 $R$-模线性化。

## 符号附录约定

- 全书默认同调分次，微分次数为 $-1$。
- Koszul braiding：$\tau(x\otimes y)=(-1)^{|x||y|}y\otimes x$。
- 张量微分：$d(x\otimes y)=d x\otimes y+(-1)^{|x|}x\otimes d y$。
- 悬挂：$|sx|=|x|+1$，且 $d(sx)=-s(dx)$。
- Hom differential：$d(f)=d\circ f-(-1)^{|f|}f\circ d$。
- Operadic suspension：$(\Lambda M)(n)=s^{1-n}M(n)\otimes\operatorname{sgn}_n$。

## 证明用语

- “同构”用于普通范畴中的可逆态射或严格同构。
- “弱等价”只在指定模型范畴中使用，并必须说明模型结构。
- “等价”在 infinity-语境中使用时必须说明采用的模型。
- “自然”必须指明相对于哪些变量自然。
- “唯一”若在同伦或 infinity 语境中使用，必须说明是严格唯一、合同唯一或在可缩选择空间中唯一。
