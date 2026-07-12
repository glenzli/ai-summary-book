# 符号约定

本文件记录《同伦类型论与单值基础》的全书符号。后续章节新增核心符号时必须同步更新。

## 判断与语境

- $\Gamma\ \mathsf{ctx}$：$\Gamma$ 是合法语境。
- $\Gamma\vdash A:\mathcal U_i$：在语境 $\Gamma$ 中，$A$ 是第 $i$ 层宇宙中的类型。
- $\Gamma\vdash a:A$：在语境 $\Gamma$ 中，$a$ 是类型 $A$ 的项。
- $\Gamma\vdash a\equiv b:A$：$a$ 与 $b$ 在类型 $A$ 中 judgmentally equal，也称 definitional equality。
- $\Gamma\vdash A\equiv B:\mathcal U_i$：类型 $A,B$ 作为宇宙中的项 judgmentally equal；类型转换可把 $a:A$ 视为 $a:B$，不产生路径证明项。
- $a\equiv_s b$：two-level type theory 外部层中的 strict equality；不同于 HoTT identity path。
- $\Gamma,x:A$：语境扩张。若 $A$ 依赖于 $\Gamma$，则 $x$ 可在后续类型和项中出现。
- $B[a/x]$：把 $a$ 替换到 $B$ 中的自由变量 $x$。

## 基础类型构造

- $\prod_{x:A}B(x)$ 或 $\Pi_{x:A}B(x)$：依赖函数类型。
- $\sum_{x:A}B(x)$ 或 $\Sigma_{x:A}B(x)$：依赖对类型。
- $A\to B$：非依赖函数类型，即 $\prod_{x:A}B$，其中 $B$ 不依赖于 $x$。
- $A\times B$：非依赖积类型，即 $\sum_{x:A}B$，其中 $B$ 不依赖于 $x$。
- $\mathbf 0$：空类型。
- $\mathbf 1$：单位类型，其规范元素记为 $\star$。
- $A+B$：和类型。
- $\mathbb N$：自然数类型。
- $\mathbb Z$：整数类型；第十一章默认指附录 M 的归纳整数 $\mathbb Z_{\mathsf{ind}}$。
- $\mathbb Q$：有理数域。
- $\mathbb R_C$：Cauchy 实数高阶归纳-归纳构造。
- $0_{\mathbb Z}$、$\mathsf{succ}_{\mathbb Z}$、$\mathsf{pred}_{\mathbb Z}$：整数的零、后继和前驱。
- $\mathsf{Fin}(n)$：含 $n$ 个元素的标准有限集。

## 恒等类型与路径

- $\mathsf{Id}_A(a,b)$ 或 $a=_A b$：$a$ 与 $b$ 的恒等类型，也称路径类型。
- $A=_{\mathcal U_i}B$：$A,B:\mathcal U_i$ 作为宇宙元素的恒等类型；该类型位于 $\mathcal U_{i+1}$，不同于 $A\equiv B:\mathcal U_i$。
- $\mathsf{refl}_a:a=_A a$：反身路径。
- 若 $p:a=b$，则 $p^{-1}:b=a$ 表示逆路径。
- 若 $p:a=b$ 且 $q:b=c$，则 $p\cdot q:a=c$ 表示路径复合。
- 若 $f:A\to B$ 且 $p:x=y$，则 $\mathsf{ap}_f(p):f(x)=f(y)$。
- 若 $P:A\to\mathcal U_j$、$f:\prod_{x:A}P(x)$ 且 $p:x=y$，则 $\mathsf{apd}_f(p):\mathsf{transport}^{P}(p,f(x))=f(y)$。
- 若 $P:A\to\mathcal U_j$ 且 $p:x=y$，则 $\mathsf{transport}^P(p):P(x)\to P(y)$。
- $\kappa_{p,u}:\mathsf{transport}^{\lambda\_.\,B}(p,u)=u$：常值族 transport 的逐点路径；函数级等式仍需函数外延性。

## 等价与同伦层级

- $\mathsf{fib}_f(y)\coloneqq \sum_{x:A}(f(x)=y)$：函数 $f:A\to B$ 在 $y:B$ 处的 fiber。
- $\mathsf{isContr}(A)\coloneqq \sum_{c:A}\prod_{x:A}(c=x)$：$A$ 可收缩。
- $\mathsf{isProp}(A)\coloneqq \prod_{x,y:A}(x=y)$：$A$ 是命题。
- $\mathsf{isSet}(A)\coloneqq \prod_{x,y:A}\mathsf{isProp}(x=y)$：$A$ 是集合。
- $A\simeq B$：$A$ 与 $B$ 等价；具体定义以后续等价章节为准。
- $\mathsf{isEquiv}(f)$：函数 $f$ 是等价；本书以 fiber 可收缩定义为基准。
- $\mathsf{idtoequiv}_{i,A,B}:(A=_{\mathcal U_i}B)\to(A\simeq B)$：从第 $i$ 层 universe path 得到小类型等价的映射；其定义域位于 $\mathcal U_{i+1}$，值域位于 $\mathcal U_i$。
- $\mathsf{idtofun}(p)\coloneqq\mathsf{transport}^{\lambda X:\mathcal U_i.\,X}(p)$：宇宙路径 $p:A=_{\mathcal U_i}B$ 的底层运输函数 $A\to B$。
- $\mathsf{idEquiv}_A:A\simeq A$：$A$ 的恒等等价。
- $\mathsf{UA}_i$：第 $i$ 层 universe univalence，即 $\mathsf{idtoequiv}_{i,A,B}$ 对所有 $A,B:\mathcal U_i$ 为等价。
- $\mathsf{ua}_i$：$\mathsf{UA}_i$ 选出的 $(A\simeq B)\to(A=_{\mathcal U_i}B)$ 方向；其两个逆律在公理化 HoTT 中是路径，不是 judgmental equality。
- $\beta_e$、$\eta_p$：公理化单值性中 $\mathsf{idtoequiv}$ 与 $\mathsf{ua}_i$ 的两个 propositional 逆律。
- $\|A\|_n$：$n$-截断；$\|A\|$ 表示命题截断。
- $\tau_{n,m}:\|A\|_n\to\|A\|_m$：Postnikov 截断塔中的规范比较映射。
- $\mathsf{isOfHLevel}_n(A)$：$A$ 具有同伦层级 $n$。本书采用 HoTT Book 常见编号：$0$ 表示可收缩，$1$ 表示命题，$2$ 表示集合。
- $L A$、$\eta_A:A\to L A$：反射子宇宙或模态 $L$ 的反射对象和单位映射。
- $\mathsf{isLocal}(A)$：$A$ 对指定模态或局部化为 local。
- $L_S A$：关于映射族 $S$ 的局部化。
- $n$-连通映射：fiber 的 $n$-截断可收缩；$n$-截断映射：fiber 是 $n$-型。

## 高阶归纳类型与合成同伦

- $\mathbb S^1_i:\mathcal U_i$：第 $i$ 层的 universe-polymorphic 圆实例；固定层级后写 $\mathbb S^1$，点与路径构造子写 $\mathsf{base}$、$\mathsf{loop}:\mathsf{base}=\mathsf{base}$。
- $\beta_{\mathsf{loop}}$、$\beta^P_{\mathsf{loop}}$：本书公理化圆中递归与依赖消去在 $\mathsf{loop}$ 上的 propositional computation 路径；点计算为 judgmental。
- $\mathsf{susp}(A)$：$A$ 的悬挂；$\mathsf{suspRec}$、$\mathsf{suspInd}$ 分别为递归子和依赖消去子。
- $\beta^{\mathsf{suspRec}}_{\mathsf{merid}}$、$\beta^{\mathsf{suspInd}}_{\mathsf{merid}}$：悬挂在 meridian 上的 propositional $\mathsf{ap}/\mathsf{apd}$-$\beta$ 路径；north/south 点计算为 judgmental。
- $\mathsf{pushout}(f,g)$：两个映射 $f:A\to B$、$g:A\to C$ 的 pushout 高阶归纳类型；$\mathsf{pushRec}$、$\mathsf{pushInd}$ 分别为递归子和依赖消去子。
- $\beta^{\mathsf{pushRec}}_{\mathsf{glue}}$、$\beta^{\mathsf{pushInd}}_{\mathsf{glue}}$：pushout 在 glue 上的 propositional $\mathsf{ap}/\mathsf{apd}$-$\beta$ 路径；inl/inr 点计算为 judgmental。
- $\mathsf{cofib}(f)$：映射 $f:A\to B$ 的 cofiber，即 $B\leftarrow A\to\mathbf 1$ 的 pushout。
- $\mathsf{Code}_{LR}(b,c)$：pushout 中从左侧点 $\mathsf{inl}(b)$ 到右侧点 $\mathsf{inr}(c)$ 的路径 code family。
- $\mathsf{gap}$：pushout 方块的 gap map，通常从输入交点类型到路径空间或其 code fiber。
- $\mathsf{HIIT}$：higher inductive-inductive type，同时生成相互依赖的类型或类型族。
- $\mathsf{QIT}$、$\mathsf{QIIT}$：quotient inductive type 与 quotient inductive-inductive type。
- $\mathsf{Alg}_\Sigma$：签名 $\Sigma$ 的代数范畴或高阶代数对象。
- $X\vee Y$：pointed 类型的 wedge。
- $X\wedge Y$：pointed 类型的 smash product。
- $X\to_\ast Y$：pointed 映射类型。
- $\Omega X$：基点处 loop space；$\Omega^nX$ 表示迭代 loop space。
- $\pi_1(X,x_0)$：基点类型 $(X,x_0)$ 的基本群；严格定义见第十一章。
- $BG$：群 $G$ 的 delooping 或 classifying type。
- $\mathsf{Aut}(A)$：类型 $A$ 的自等价群，通常为 $A\simeq A$。
- $\mathsf{Tors}_G$：$G$-torsor 类型。
- $\mathsf{Sp}$：谱的类型或范畴，具体定义见附录 AZ。
- $\pi_k^s(E)$：谱 $E$ 的第 $k$ 个稳定同伦群。
- $\Delta$：simplex category；$\Delta^{op}$ 用于 simplicial object。
- $X_n$：simplicial object $X:\Delta^{op}\to\mathcal U$ 的 $n$-simplex 类型。
- $\mathsf{seg}_n$：Segal map，从 $n$-simplex 到 composable 1-simplex 串。
- $\mathsf{Eq}_X(x,y)$：Rezk/Segal object 中从 $x$ 到 $y$ 的 equivalence edge 类型。
- $K(G,n)$：Eilenberg-Mac Lane 型；本书只在研究边界章节作为外部输入和高级构造目标使用。
- $H^n(X;G)$：以 $K(G,n)$ 表示的第 $n$ 阶合成上同调群。
- $H^n(X;L)$：以局部系数系统 $L:X\to\mathsf{AbGroup}$ 表示的扭曲上同调。
- $\widetilde H^n(X;G)$：带基点类型的第 $n$ 阶约化上同调群。
- $\smile$：上同调 cup product。
- $\mathsf{Sq}^i$：mod 2 Steenrod square。
- $\mathcal P^i$、$\beta$：奇素数 Steenrod reduced power 与 Bockstein。
- $\mathcal A_p$：素数 $p$ 处的 Steenrod algebra。
- $\mathsf{Ext}^{s,t}_{\mathcal A_p}(M,N)$：Steenrod algebra 模范畴中的双次数 Ext。
- $E_n\to\Omega E_{n+1}$：谱或 Omega 谱的结构映射口径。
- $\mathsf{gr}_p X$：filtered spectrum 或 filtered object 的第 $p$ 个 associated graded piece。
- $F_f$：pointed map $f:E\to_\ast B$ 在基点处的 homotopy fiber。
- $\partial_n$：homotopy fiber sequence 长正合列中的 connecting map。
- $k_n$：Postnikov tower 中第 $n$ 个 $k$-invariant 或 obstruction map。
- $C_f$：映射 $f$ 的 cofiber，常用于 Puppe sequence。
- $L:X\to\mathsf{AbGroup}$：$X$ 上的阿贝尔群局部系数系统。
- $\underline A$：常系数系统。
- $(D,E,i,j,k)$：exact couple。
- $E_r,d_r$：谱序列第 $r$ 页及其微分。
- $E_2^{p,q}$、$E^2_{p,q}$：上同调/同调谱序列常用双次数页。
- $F^pG$、$\mathsf{gr}^pG$：过滤群及其 associated graded。

## 实数与序

- $d(x,y)<\varepsilon$：Cauchy 实数或预度量空间中的有理误差距离关系。
- $x<y$、$x\le y$：构造性实数序关系；具体口径见附录 AR。
- $x\#0$：$x$ 远离零，即 $x<0$ 或 $0<x$ 的构造性强非零性。
- $\mathbb R_D$：Dedekind 实数。
- $L_x,U_x:\mathbb Q\to\mathsf{Prop}$：Dedekind cut 的下切和上切。
- $[a,b]$：构造性实数中的闭区间，默认含 $a\le x\le b$ 的证据。
- $\mathsf{TotBdd}(X)$：预度量空间或子空间 $X$ totally bounded。
- $\mathsf{Cpt}_C(X)$：Cauchy compactness，即每个序列有 Cauchy 子列或等价的 Cauchy 紧致口径。
- $\mathsf{isCauchy}(a)$：序列 $a:\mathbb N\to X$ 是 Cauchy。
- $S_N$：级数的第 $N$ 个部分和。
- $S(f,P)$：函数 $f$ 对 tagged partition $P$ 的 Riemann sum。
- $\|x\|$：normed Abelian group 或 normed vector space 中的范数。
- $\mathbb R_{\mathsf{sdg}}$：合成微分几何中的线对象，不等同于 $\mathbb R_C$ 或 $\mathbb R_D$。
- $D\coloneqq\sum_{d:\mathbb R_{\mathsf{sdg}}}(d^2=0)$：一阶无穷小对象。
- $T X\coloneqq X^D$：SDG/cohesive 口径下的切丛。

## 范畴论

- $\mathcal C$：预范畴或单值范畴。
- $\mathcal C^{\mathsf{op}}$：$\mathcal C$ 的反范畴。
- $\mathcal C(x,y)$：对象 $x,y$ 之间的 Hom 类型。
- $x\cong y$：范畴中的同构。
- $A\cong_{\mathrm w}B$：universe 的 wild category 中 $A$ 与 $B$ 的同构；逆律位于函数类型的 identity type，不预设函数外延性。
- $\mathsf{idtoiso}_{x,y}:(x=y)\to(x\cong y)$：对象相等诱导同构。
- $\mathsf{isUnivalentCat}(\mathcal C)$：范畴 $\mathcal C$ 是单值范畴，即 $\mathsf{idtoiso}$ 是等价。
- $F:\mathcal C\to\mathcal D$：预范畴之间的函子。
- $F\Rightarrow G$：函子 $F,G:\mathcal C\to\mathcal D$ 之间的自然变换类型。
- $F\cong_{\mathsf{nat}}G$：函子之间的自然同构类型。
- $[\mathcal C,\mathcal D]$：从 $\mathcal C$ 到 $\mathcal D$ 的函子范畴。
- $\mathcal D$ over $\mathcal C$：$\mathcal C$ 上的 displayed category。
- $x\xrightarrow{f}_{\mathcal D}y$：displayed category 中位于基态射 $f$ 上的 displayed morphism。
- $\int_{\mathcal C}\mathcal D$：displayed category 的 total category。
- $\mathcal B(a,b)$：bicategory $\mathcal B$ 中对象 $a,b$ 之间的 Hom category。
- $a\simeq_{\mathcal B}b$：bicategory 对象之间的 adjoint equivalence。
- $y(c)$：Yoneda 嵌入中的可表预层 $\mathcal C(-,c)$。
- $\mathsf{Nat}(P,Q)$：预层或函子之间的自然变换类型。
- $\widehat{\mathcal C}$：预范畴 $\mathcal C$ 的 Rezk 完备化。

## Directed / Simplicial 符号

- $\mathsf{hom}_A(a,b)$：directed 或 simplicial type theory 中从 $a$ 到 $b$ 的有向 Hom 类型；它不是恒等类型。
- $\mathcal S$：directed/simplicial 语境中的离散类型宇宙。
- Segal 类型：带有 directed composition 和 inner horn filler 相干的类型；严格规则见附录 AN。
- Rezk object / Rezk type：满足 Segal 条件和 completeness 条件的高阶范畴对象；接口见附录 BB。
- $p:E\to B$：directed 语境中的 fibration 或 cocartesian fibration。
- $\bar u$：底箭头 $u$ 的 cocartesian lift。
- $\mathbb N_s$：two-level type theory 外部层自然数，用于索引 strict/元理论递归。
- $\mathcal U_{\mathsf{fib}}$：2LTT 中 fibrant types 的 universe 口径。

## Cubical 与元理论符号

- $\mathbb I$、$\mathbb F$：CCHM cubical type theory 的区间与面格。
- $\mathsf{Path}_A(a,b)$：CCHM 对象语言中带端点约束的区间路径；不是第 2 章归纳 $\mathsf{Id}_A(a,b)$ 的同一个语法构造子。
- $\mathsf{pathToEq}:\mathsf{Path}_{\mathcal U}(A,B)\to(A\simeq B)$：CCHM 中从 universe path 到等价的规范映射。
- $\mathsf{Con}(T)$：元语言中“理论 $T$ 一致”的断言；不是本书对象语言中的类型缩写。

## Cohesive 符号

- $\Pi\dashv\mathsf{Disc}\dashv\Gamma\dashv\mathsf{Codisc}$：cohesive HoTT 中的 shape、discrete、global sections、codiscrete 伴随串。
- $D(f)$：Zariski 或 synthetic algebraic geometry 口径中的基本开集。

## 集合层代数与大小

- $\mathsf{Str}(A)$：集合 $A$ 上的代数结构类型。
- $\mathsf{Group}_i:\mathcal U_{i+1}$：carrier 位于 $\mathcal U_i$ 的小群之类型；对小类型宇宙的量化使结构类型高一层。
- $G/N$：群 $G$ 关于正规子群 $N$ 的商群。
- $R/I$：环 $R$ 关于 ideal $I$ 的商环。
- $S^{-1}R$：交换环 $R$ 关于乘法闭子集 $S$ 的局部化。
- $\mathsf{Card}_{\mathcal U}$：宇宙 $\mathcal U$ 中集合的基数类型。
- $|A|$：集合 $A$ 的基数，通常表示其在 $\mathsf{Card}_{\mathcal U}$ 中的等价类。
- $\mathsf{Acc}(a)$：良基关系下 $a$ 可访问。

## 逻辑原则

- $\neg A$：否定，定义为 $A\to\mathbf 0$。
- $\mathsf{LEM}$：命题排中律 $\prod_{P:\mathsf{Prop}}(P+\neg P)$。
- $\mathsf{DNE}$：双重否定消去 $\prod_{P:\mathsf{Prop}}(\neg\neg P\to P)$。
- $\mathsf{AC}_\omega$：可数选择原则。
- $\mathsf{DC}$：依赖选择原则。
- propositional resizing：把高宇宙命题等价替换到低宇宙命题的原则。

## 宇宙约定

- 本书默认使用层级宇宙 $\mathcal U_0,\mathcal U_1,\ldots$。
- 除非章节明确声明，不假设 resizing。
- 默认不假设 cumulativity；$A:\mathcal U_i$ 不自动给出 $A:\mathcal U_j$（$i<j$），也没有隐式 universe lift。
- 若 $A:\mathcal U_i$ 且 fibers $B(x):\mathcal U_j$，则 $B$ 称为 $j$-小族，$\Pi_xB(x)$、$\Sigma_xB(x):\mathcal U_{\max(i,j)}$；但宇宙值函数项 $B:A\to\mathcal U_j$ 自身位于 $\mathcal U_{\max(i,j+1)}$。
- 多层参数的 $\mathcal U_{\max(i,j,\ldots)}$ 是相关形成规则直接声明的层级，不暗示先使用累积性。
