# 范畴论答案手册

本文件给出 `books/category-theory/` 当前版本全部练习的参考答案。答案以“检查定义、写出泛性质、标明自然性变量”为原则；对需要大型外部理论的题目，给出标准结论和应核对的来源。

## 序章

**答案 0.1.** 若不指定 universe，“小范畴”的对象集合可在任意大集合中变化；所有小范畴的对象类不再是同一层级中的集合。固定 $\mathcal U\in\mathcal V$ 后，$\mathcal U$-小范畴组成 $\mathcal V$-层级中的范畴 $\mathbf{Cat}_{\mathcal U}$。

**答案 0.2.** 例：集合 $A,B$ 的积 $A\times B$。输入为二元组 $(A,B)$，输出为集合 $P$ 和投影 $P\to A,P\to B$。候选自然双射为
$$
\mathbf{Set}(X,P)\cong \mathbf{Set}(X,A)\times\mathbf{Set}(X,B).
$$

**答案 0.3.** 同构要求存在严格可逆函子，等价只要求完全忠实且本质满。例：一个含两个同构对象和一条唯一同构的群胚等价于终范畴，但对象数不同，所以不严格同构。

## 第一章

**答案 1.1.** 若采用含幺环和保持单位的环同态，恒等函数保持加法、乘法和单位；两个保持单位的环同态复合仍保持这些结构。函数复合的结合律和单位律给出范畴公理。

**答案 1.2.** 单对象范畴中态射复合就是幺半群乘法。态射 $m$ 同构当且仅当存在 $n$ 使 $nm=e=mn$，即 $m$ 是幺半群中的可逆元素。

**答案 1.3.** 薄范畴中 $x\cong y$ 意味着 $x\le y$ 且 $y\le x$；偏序的反对称性给出 $x=y$。反过来 $x=y$ 时恒等态射给出同构。

**答案 1.4.** 对 $f:X\to Y$ 定义 $\mathcal C(A,f)(u)=fu$。恒等性来自 $\operatorname{id}_X u=u$；复合性来自 $(gf)u=g(fu)$。

**答案 1.5.** 恒等自然变换为 $(\operatorname{id}_F)_X=\operatorname{id}_{F X}$。纵向复合结合律逐点化为 $\mathcal D$ 中态射复合结合律。

**答案 1.6.** 包含函子 $\mathbf{Ab}\hookrightarrow\mathbf{Grp}$ 完全忠实但非本质满，因为非阿贝尔群不与阿贝尔群同构。

**答案 1.7.** 对任意 $X,Y$，Hom 映射
$$
\mathcal C(X,Y)\to\mathcal E(GFX,GFY)
$$
是两个双射的复合，故为双射。

**答案 1.8.** 令 $\mathcal C$ 为有两个同构对象 $a,b$ 的连通群胚，且任意 Hom 集单点；令 $\mathcal D=*$。唯一函子 $\mathcal C\to\mathcal D$ 完全忠实且本质满，是等价；但对象数不同，不可能是严格同构。

## 第二章

**答案 2.1.** 在 $\mathcal C^{\operatorname{op}}$ 中，始对象变为终对象。由终对象唯一性，始对象在唯一同构意义下唯一。

**答案 2.2.** $A\times B$ 带投影 $p_A,p_B$，对任意 $X$ 有自然双射
$$
\mathbf{Set}(X,A\times B)\cong\mathbf{Set}(X,A)\times\mathbf{Set}(X,B).
$$
它表示函子 $X\mapsto\mathbf{Set}(X,A)\times\mathbf{Set}(X,B)$。

**答案 2.3.** 单对象范畴 $\mathcal C_M$ 上的预层是函子 $\mathcal C_M^{op}\to\mathbf{Set}$，等价于带右 $M$-作用的集合。

**答案 2.4.** 对 $a\in F(A)$ 定义 $\alpha^a_X(f:A\to X)=F(f)(a)$。自然性由 $F(gf)=F(g)F(f)$。任意自然变换由 $\alpha_A(\operatorname{id}_A)$ 唯一决定。

**答案 2.5.** 若 $y(f):yA\to yB$ 是同构，则其逆为某个 $y(g)$，因为 $y$ 完全忠实。于是 $y(gf)=\operatorname{id}_{yA}$ 且 $y(fg)=\operatorname{id}_{yB}$；忠实性给出 $gf=\operatorname{id}_A$、$fg=\operatorname{id}_B$。

## 第三章

**答案 3.1.** 图形为 $A\xrightarrow{f}C\xleftarrow{g}B$。拉回 $P$ 带 $p_A:P\to A,p_B:P\to B$，满足 $fp_A=gp_B$；任意 $X$ 上相容二元组唯一分解经 $P$。

**答案 3.2.** 集合 $P=\{(a,b)\mid f(a)=g(b)\}$ 带投影。相容映射 $x\mapsto a_x,b_x$ 唯一给出 $x\mapsto(a_x,b_x)$。

**答案 3.3.** 余等化子为 $q:B\to Q$，满足 $qf=qg$；任意 $h:B\to X$ 若 $hf=hg$，存在唯一 $\bar h:Q\to X$ 使 $\bar hq=h$。

**答案 3.4.** 对小图形 $D:\mathcal J\to\mathcal C$，令
$$
P=\prod_jD(j),\qquad Q=\prod_{\alpha:j\to k}D(k).
$$
两箭头 $P\rightrightarrows Q$ 由 $D(\alpha)p_j$ 与 $p_k$ 给出。其等化子即极限。

**答案 3.5.** 极限逐点由第三章命题 3.12。余极限逐点同理，因自然变换到逐点余极限的自然性逐对象验证，且目标范畴 $\mathbf{Set}$ 余极限逐点存在。

## 第四章

**答案 4.1.** 给集合映射 $S\to U(A)$，由自由阿贝尔群泛性质唯一延拓为群同态 $\mathbb Z[S]\to A$。自然双射为
$$
\mathbf{Ab}(\mathbb Z[S],A)\cong\mathbf{Set}(S,U A).
$$

**答案 4.2.** 右伴随为 $(-)^A$。双射
$$
\mathbf{Set}(A\times X,Y)\cong\mathbf{Set}(X,Y^A)
$$
由 currying 给出，$f(a,x)$ 对应 $x\mapsto(a\mapsto f(a,x))$。

**答案 4.3.** 若 $G$ 右伴随，且 $1_{\mathcal D}$ 为终对象，则
$$
\mathcal C(X,G1)\cong\mathcal D(FX,1)
$$
为单点集，所以 $G1$ 终。

**答案 4.4.** 若 $F$ 完全忠实，则 $\mathcal C(X,Y)\to\mathcal D(FX,FY)$ 是双射，单位是 $\operatorname{id}_{FX}$ 的转置，因而为同构。反过来若 $\eta$ 是同构，则任意 $f:FX\to FY$ 唯一转置为 $\eta_Y^{-1}G(f)\eta_X:X\to Y$，故 $F$ 完全忠实。

**答案 4.5.** 对图形 $D$ 的极限 $L$ 和任意 $X$：
$$
\mathcal C(X,G L)\cong\mathcal D(FX,L)
\cong\lim_j\mathcal D(FX,Dj)
\cong\lim_j\mathcal C(X,G Dj).
$$
由表示性，$G L$ 是 $GD$ 的极限。

## 第五章

**答案 5.1.** 投影函子 $\int_{\mathcal C}P\to\mathcal C$ 送 $(C,x)$ 到 $C$，送态射 $f:(C,x)\to(D,y)$ 到 $f:C\to D$。

**答案 5.2.** $yA(C)=\mathcal C(C,A)$，故元素范畴对象为 $(C,u:C\to A)$，态射为使三角形交换的 $C\to D$；这正是 slice 范畴 $\mathcal C/A$。

**答案 5.3.** 若 $(C,x,f)$ 与 $(D,y,g)$ 在 $P(A)$ 中有同一像 $a$，则 $f:(A,a)\to(C,x)$ 与 $g:(A,a)\to(D,y)$ 在元素范畴中把二者同时连到 $(A,a,\operatorname{id})$，故在余极限商中相等。

**答案 5.4.** 若 $f,g:X\rightrightarrows Y$ 不同，取 $x\in X$ 使 $f(x)\ne g(x)$。单点集 $1\to X$ 选出 $x$，于是 $f\circ x\ne g\circ x$。

**答案 5.5.** 群同态 $\mathbb Z\to G$ 等价于选择 $G$ 的一个元素。若 $f,g:G\rightrightarrows H$ 不同，取 $x$ 使 $f(x)\ne g(x)$，对应 $\mathbb Z\to G$ 检测二者不同。

## 第六章

**答案 6.1.** 若 $u:(c,\alpha)\to(c',\alpha')$、$v:(c',\alpha')\to(c'',\alpha'')$，则 $\alpha''K(vu)=\alpha'K(u)=\alpha$。恒等态射满足 $\alpha K(\operatorname{id})=\alpha$。

**答案 6.2.** $\operatorname{Lan}_iF(c)$ 由所有 $a\in\mathcal A$ 及箭头 $a\to c$ 上的 $F(a)$ 的余极限给出，是把 $F$ 从子范畴按所有进入 $c$ 的方式自由延拓。

**答案 6.3.** 对 $K=\operatorname{id}$，逗号范畴 $\operatorname{id}/c$ 有终对象 $(c,\operatorname{id})$，故左 Kan 延拓点态为 $F(c)$；右 Kan 延拓同理。

**答案 6.4.** 对 $\beta:F\to HK$ 构造 $\bar\beta_d:Ld\to Hd$。若 $v:d\to d'$，两条 $Ld\to Hd'$ 在每个 $F(c)\to Ld$ 上都为 $H(v\alpha)\beta_c$，故相等。

**答案 6.5.** 预层密度可写为
$$
P\cong \operatorname{Lan}_{y_P} y
$$
沿元素范畴投影到 $\mathcal C$ 后由 Yoneda 嵌入作左 Kan 延拓；点态公式就是 co-Yoneda 公式。

## 第七章

**答案 7.1.** 单位把字 $w$ 看成长为一的“字的字”，乘法拼接；左右单位说明拼接单层括号不改变字，结合律说明去括号顺序不影响最终字。

**答案 7.2.** 单子 $T(S)=U\mathbb Z[S]$。$T$-代数 $T(A)\to A$ 等价于给集合 $A$ 一个阿贝尔群结构，使自由阿贝尔群上的线性组合在 $A$ 中求值；代数同态正是群同态。

**答案 7.3.** $\eta_X(x)=\{x\}$。乘法 $\mu_X:\mathcal P\mathcal P(X)\to\mathcal P(X)$ 为并集：$\mathcal A\mapsto\bigcup_{A\in\mathcal A}A$。

**答案 7.4.** 设 $f:X\to TY,g:Y\to TZ,h:Z\to TW$。两种复合分别化为
$$
\mu_W\mu_{TW}T^2h\,Tg\,f
$$
和
$$
\mu_WT\mu_WT^2h\,Tg\,f,
$$
由 $\mu\circ\mu T=\mu\circ T\mu$ 相等。

**答案 7.5.** 余单子为 $(G,\epsilon,\delta)$，其中 $\epsilon:G\to\operatorname{id}$、$\delta:G\to G^2$，满足对偶单位律和结合律。余代数为 $a:A\to G A$，满足 $\epsilon_Aa=\operatorname{id}$ 与 $\delta_Aa=G(a)a$。

## 第八章

**答案 8.1.** $(\mathbf{Set},\sqcup,\varnothing)$ 是幺半范畴，结合和单位由集合余积同构给出。它不是笛卡尔幺半结构，因为单位应为终对象而非始对象。

**答案 8.2.** 单对象范畴的幺半结构等价于该 Hom 幺半群上再给一个与复合相容的幺半运算；可视为严格幺半范畴的一对象情形，即两个幺半结构满足交换律型相容。

**答案 8.3.** 若 $A$ 有乘法 $m$ 和单位 $u$，则 $FA$ 的乘法为
$$
FA\otimes FA\xrightarrow{\phi}F(A\otimes A)\xrightarrow{F m}FA
$$
单位为 $\mathbb1\to F\mathbb1\xrightarrow{F u}FA$。相干图由 $F$ 的幺半相干性和 $A$ 的代数公理给出。

**答案 8.4.** 余代数对象为 $C$ 加态射 $\Delta:C\to C\otimes C$ 与 $\epsilon:C\to\mathbb1$，满足对偶的余结合律和余单位律。

**答案 8.5.** 辫子满足两个六边形，表达 $\beta_{X,Y\otimes Z}$ 与 $\beta_{X,Y},\beta_{X,Z}$ 的相容，以及 $\beta_{X\otimes Y,Z}$ 与 $\beta_{X,Z},\beta_{Y,Z}$ 的相容。对称还要求 $\beta_{Y,X}\beta_{X,Y}=\operatorname{id}$。

## 第九章

**答案 9.1.** 评价映射 $\operatorname{ev}:Z^X\times X\to Z$ 为 $(f,x)\mapsto f(x)$。给 $g:Y\times X\to Z$，其 curry 化为 $\bar g(y)(x)=g(y,x)$，直接代入得 $\operatorname{ev}(\bar g(y),x)=g(y,x)$。

**答案 9.2.** $-\otimes X$ 是左伴随，因此保持余极限，特别保持空余极限即初对象。

**答案 9.3.** 若 $X$ 有限维，线性映射空间 $\operatorname{Hom}_k(X,Z)$ 自然同构于 $X^*\otimes Z$；同构依赖有限维性。

**答案 9.4.** 由 Day 公式：
$$
(ya\star yb)(c)=\int^{u,v}\mathcal C(u,a)\times\mathcal C(v,b)\times\mathcal C(c,u\otimes v)
\cong\mathcal C(c,a\otimes b)
$$
两次 co-Yoneda 给出同构。

**答案 9.5.** coend 公式为
$$
\coprod_{f:c\to c'}H(c',c)\rightrightarrows\coprod_cH(c,c)\to\int^cH(c,c).
$$

## 第十章

**答案 10.1.** $\mathbf{Ab}$-富范畴的复合是阿贝尔群同态
$$
\mathcal A(B,C)\otimes\mathcal A(A,B)\to\mathcal A(A,C),
$$
等价于双线性映射 $\mathcal A(B,C)\times\mathcal A(A,B)\to\mathcal A(A,C)$。

**答案 10.2.** 令 $\mathbf2$ 为幺半偏序 $0<1$ 且张量为 $\wedge$。Hom 对象取 $1$ 当 $x\le y$，否则 $0$。复合公理即传递性。

**答案 10.3.** $\mathbf{Cat}$-富范畴有对象、Hom 范畴、复合函子和单位对象；这正是严格 2-范畴的数据。

**答案 10.4.** 取 $\mathcal V=\mathbf{Set}$、权重常值单点集，则
$$
\mathcal A(A,\{1,D\})\cong\operatorname{Nat}(1,\mathcal A(A,D-))
$$
即从 $A$ 到 $D$ 的锥集合，故恢复普通极限。

**答案 10.5.** 富自然变换对象通常写作 end：
$$
\operatorname{Nat}_{\mathcal V}(F,G)=\int_A [F A,G A].
$$

**答案 10.6.** 若 $\mathcal A$ 是预加性范畴，$F:\mathcal A^{op}\to\mathbf{Ab}$ 是加性函子，则
$$
\operatorname{Nat}_{\mathbf{Ab}}(\mathcal A(-,A),F)\cong F(A)
$$
是阿贝尔群同构。自然变换由 $\operatorname{id}_A$ 的像决定，任意 $x\in F(A)$ 给出 $\alpha_B(f:B\to A)=F(f)(x)$。

**答案 10.7.** 证明需要把态射 $X\to[\mathcal A(B,A),F(B)]$ 与评价型态射 $X\otimes\mathcal A(B,A)\to F(B)$ 互相转换；这正是内部 Hom 与张量的伴随，即闭结构。

## 第十一章

**答案 11.1.** 对 $f:C\to C'$，第一箭头把 $H(C',C)$ 经 $H(f,C)$ 送入 $H(C,C)$ 分量；第二箭头经 $H(C',f)$ 送入 $H(C',C')$ 分量。

**答案 11.2.** 若 $\mathcal C$ 离散，则无非恒等态射约束，end 为 $\prod_C H(C,C)$，coend 为 $\coprod_C H(C,C)$。

**答案 11.3.** 自然变换是 end 中满足自然性等式的族。纵向复合逐点定义；自然性由目标范畴复合结合律和两个族的自然性推出。

**答案 11.4.** co-Yoneda 中 $[C,x,f]$ 映到 $P(f)(x)$。逆把 $a$ 送到 $[A,a,\operatorname{id}]$。关系 $(C,P(u)y,f)\sim(D,y,uf)$ 下二者同映到 $P(f)P(u)y=P(uf)y$，故良定义。

**答案 11.5.** 取 $P$，co-Yoneda 给
$$
P\cong\int^C P(C)\times yC.
$$
右边是按元素 $x\in P(C)$ 对可表 $yC$ 作的余商，即元素范畴上的可表预层余极限。

## 第十二章

**答案 12.1.** 若 $A$ 有限，函数 $A\to\operatorname{colim}X_j$ 的有限多个像可在滤过图形某一共同阶段表示，等式也可在后续共同阶段验证。

**答案 12.2.** 对无限集合 $A$，恒等映射 $A\to\operatorname{colim}_{B\subset A,\ B finite}B=A$ 不经过任一有限阶段，故 $A$ 不是 $\omega$-紧。

**答案 12.3.** 由预层密度定理，任意预层是其元素范畴上的可表预层余极限。

**答案 12.4.** 局部有限可表现范畴是余完备且由有限可表现对象经滤过余极限生成的范畴。$\mathbf{Grp}$ 中有限表现群如 $\langle x\mid x^n=1\rangle$。

**答案 12.5.** 生成族检测态射是否相等；紧生成还要求每个对象由紧对象经滤过余极限构造，并涉及 Hom 保持滤过余极限。

**答案 12.6.** 由 Yoneda，
$$
\widehat{\mathcal C}(yC,P)\cong P(C).
$$
滤过余极限逐点计算，因此
$$
\widehat{\mathcal C}(yC,\operatorname{colim}_jP_j)
\cong(\operatorname{colim}_jP_j)(C)
\cong\operatorname{colim}_jP_j(C)
\cong\operatorname{colim}_j\widehat{\mathcal C}(yC,P_j).
$$

**答案 12.7.** 预层密度定理给出
$$
P\cong\operatorname{colim}_{(C,x)\in\int P}yC.
$$
因此任意预层都由可表预层经小余极限生成；若两个态射在所有可表预层上相同，则由该余极限表达式推出它们处处相同。

**答案 12.8.** 强生成子不仅检测平行态射是否相等，还检测态射是否为同构：$f$ 是同构当且仅当所有 $\mathcal C(G,f)$ 都是双射。这把大范畴中的同构问题化为小生成对象上的集合映射问题。

**答案 12.9.** 伴随函子定理需要解集/可达性控制大小。若只保持小余极限但不可达，可能无法由某个小的紧对象子范畴控制其值，从而右伴随的候选值不满足集合大小的表示性条件。

## 第十三章

**答案 13.1.** 零态射 $X\to Y$ 是复合 $X\to0\to Y$。由于 $0$ 终，$X\to0$ 唯一；由于 $0$ 始，$0\to Y$ 唯一，故复合唯一。

**答案 13.2.** $\ker f=\{a\mid f(a)=0\}$。$\operatorname{coker}f=B/\operatorname{im}f$。

**答案 13.3.** biproduct 带 $i_A,i_B,p_A,p_B$，满足 $p_Ai_A=1$、$p_Bi_B=1$、$i_Ap_A+i_Bp_B=1$。这些等式给出积和余积泛性质。

**答案 13.4.** 有限生成自由 $R$-模及矩阵构成的范畴是加性的，但通常不含所有核余核，因此非阿贝尔。

**答案 13.5.** AB3：有小余积；AB4：小余积正合；AB5：滤过余极限正合。Grothendieck 范畴要求 AB5 和生成元。

**答案 13.6.** 在 $\mathbf{Ab}$ 中，
$$
\operatorname{coim}(f)=A/\ker(f),\qquad
\operatorname{im}(f)=f(A)\subseteq B.
$$
第一同构定理给出 $A/\ker(f)\cong f(A)$，这正是 coimage 到 image 的典范同构。

**答案 13.7.** 若 $k:K\to A$ 是 $f:A\to B$ 的核且 $ku=kv$，则 $k(u-v)=0$。核态射本身作为等化子是单态射；等价地，等化子的泛性质直接给 $u=v$。

**答案 13.8.** 短正合列由核和余核刻画。正合函子保持核、余核以及有限 biproduct 中的零对象，因此把 $0\to A\to B\to C\to0$ 送到仍在中间项正合、左端单且右端满的短正合列。

**答案 13.9.** 若 $f,g:M\rightrightarrows N$ 不同，取 $m\in M$ 使 $f(m)\ne g(m)$。态射 $R\to M$ 由 $1\mapsto m$ 决定，于是 $f$ 与 $g$ 预复合该态射后不同。因此 $R$ 检测平行态射。

**答案 13.10.** Gabriel-Popescu 把 Grothendieck 范畴表示为某个模范畴的正合反射局部化。模范畴提供自由代数模型，局部化则把由生成元看不见或应当变为等价的部分商掉。

## 第十四章

**答案 14.1.** 覆盖 $\{U_i\subset U\}$ 生成的筛由所有开嵌入 $V\subset U$ 组成，其中 $V$ 的映射局部因子化经过某个 $U_i$；等价地 $V\subset\bigcup_iU_i$。

**答案 14.2.** 在连通空间上取常值预层 $A$，对不连通覆盖，匹配族可在不同连通分支取不同值，未必来自一个全局常值元素。

**答案 14.3.** subcanonical 意味每个 $yU$ 是 sheaf，因此 Yoneda 嵌入 $\mathcal C\to\widehat{\mathcal C}$ 的像落入 $\operatorname{Sh}(\mathcal C,J)$。

**答案 14.4.** 预层有限极限逐点计算；sheaf 化 $a$ 左正合，保持有限极限，所以 sheaf 范畴有限极限可由预层极限后 sheaf 化得到。若预层极限已是 sheaf，则无需再 sheaf 化。

**答案 14.5.** Giraud 公理包括有限极限、小余极限、余极限普遍性、等价关系有效、小生成族等。定义 14.7 是外在站点表示；Giraud 给内在刻画。

**答案 14.6.** Separated 只要求局部相等推出全局相等，即唯一性；sheaf 还要求任意相容局部截面存在全局粘合，即存在性加唯一性。

**答案 14.7.** $F^+(U)$ 的元素由某个覆盖筛 $S$ 和自然变换 $S\to F$ 表示，即 $S$ 上的匹配族。若两个匹配族在共同覆盖细化上限制后相等，则它们代表同一个元素。

**答案 14.8.** Sheaf 条件要求 $F(U)\to\operatorname{Nat}(S,F)$ 是双射。双射特别是单射，因此 sheaf 必 separated。

**答案 14.9.** 几何态射 $f:\mathcal E\to\mathcal F$ 中，$f^*:\mathcal F\to\mathcal E$ 左伴随于 $f_*:\mathcal E\to\mathcal F$。要求 $f^*$ 左正合是为了让 inverse image 保持有限极限，从而保留逻辑连接词、交和终对象等几何结构。

**答案 14.10.** 平凡拓扑中每个对象只有最大筛作为覆盖。于是
$$
F^+(U)=\operatorname{Nat}(yU,F)\cong F(U)
$$
由 Yoneda 引理得到，所以 plus 构造不改变预层。

## 第十五章

**答案 15.1.** 若 $\alpha:F\Rightarrow G$、$\beta:H\Rightarrow K$，横向复合在对象 $A$ 上为
$$
(\beta*\alpha)_A=\beta_{G A}\circ H(\alpha_A)=K(\alpha_A)\circ\beta_{F A}.
$$

**答案 15.2.** 交换律逐对象化为自然变换分量的结合律与自然性；两边均为同一复合矩形的外边。

**答案 15.3.** 双模复合为 $M\otimes_S N$。张量积只在典范同构意义下结合，不是严格相等，所以得到双范畴。

**答案 15.4.** 2-函子严格保持对象、1-态射、2-态射、复合和单位。伪函子只到指定可逆 2-态射为止保持复合和单位，并满足相干公理。

**答案 15.5.** bicategory coherence theorem 说明任意双范畴双等价于严格 2-范畴在适当意义下的表示，或至少典范相干图交换；它是 Mac Lane 幺半相干性的高阶推广。

## 第十六章

**答案 16.1.** 若 $f,g,gf$ 中任意两个是弱等价，则第三个也是弱等价。

**答案 16.2.** 平凡纤维化是既为纤维化又为弱等价的态射；平凡余纤维化是既为余纤维化又为弱等价的态射。

**答案 16.3.** 链复形中的 quasi-isomorphism 在同调群上诱导同构，因此应在导出范畴中被视为同伦等价。

**答案 16.4.** 同伦范畴只保留 $\pi_0$ 层面的态射类，丢失映射空间的高阶同伦群和相干复合信息。

**答案 16.5.** Kan fibration 要求对所有 horn $\Lambda_i^n\to\Delta^n$ 有提升；quasi-category 只要求对象本身对内 horn 填充，且不要求外 horn。

## 第十七章

**答案 17.1.** 保序映射 $[1]\to[2]$ 有六个：$(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)$。非退化边对应 $(0,1),(0,2),(1,2)$，其余为退化边。

**答案 17.2.** $\delta^i$ 是漏掉 $i$ 的严格递增函数，故保序；$\sigma^i$ 只把 $i,i+1$ 合并，其余保持顺序，故保序。

**答案 17.3.** Yoneda 给
$$
\mathbf{sSet}(\Delta^n,X)=\operatorname{Nat}(\Delta(-,[n]),X)\cong X([n])=X_n
$$
对 $X$ 和 $[n]$ 自然。

**答案 17.4.** $\Lambda_1^2$ 有三个顶点 $0,1,2$，非退化边为 $0\to1$ 和 $1\to2$，缺少面 $0\to2$ 所在边作为复合边，因此表示两条可复合边。

**答案 17.5.** $N(\mathcal C)_3$ 是函子 $[3]\to\mathcal C$，即对象 $X_0,\dots,X_3$ 和相容态射 $X_i\to X_j$。长边 $X_0\to X_2$、$X_1\to X_3$、$X_0\to X_3$ 由相邻边复合决定。

**答案 17.6.** 内 horn 编码复合存在性；外 horn 编码可逆性或解方程性质。Kan 复形要求所有 horn 填充，对应 $\infty$-群胚；quasi-category 只要求内 horn，因此允许非可逆态射。

**答案 17.7.** 若 $N(\mathcal C)$ 是 Kan 复形，则由命题 17.18，其同伦范畴中每条边都是同构。但 $hN(\mathcal C)\cong\mathcal C$，所以 $\mathcal C$ 中每个态射都可逆。若 $\mathcal C$ 有非可逆态射，矛盾。

**答案 17.8.** Kan-Quillen 模型结构的 fibrant objects 是 Kan 复形，用来建模 spaces；Joyal 模型结构的 fibrant objects 是 quasi-categories，用来建模 $\infty$-categories。

**答案 17.9.** $\infty$-群胚应只有可逆态射及高阶可逆同伦。命题 17.18 说明 Kan 复形作为 quasi-category 时，所有 $1$-态射在同伦范畴中可逆，符合 $\infty$-群胚直觉。

**答案 17.10.** 对任意 $m$，
$$
N([n])_m=\operatorname{Fun}([m],[n])=\Delta([m],[n])=(\Delta^n)_m.
$$
对 $\Delta$ 中态射的作用都是预复合，所以这些逐级双射组成单纯集同构。

**答案 17.11.** $\Lambda_2^3$ 缺少第 $2$ 个面，即漏掉顶点 $2$ 的面 $(0,1,3)$。给定 $X_0\to X_1\to X_2\to X_3$ 后，所有长边由复合确定：$X_0\to X_2$、$X_1\to X_3$、$X_0\to X_3$。缺失面 $(0,1,3)$ 的三角关系要求 $X_0\to X_3$ 等于 $(X_1\to X_3)(X_0\to X_1)$，这由结合律唯一成立。

## 第十八章

**答案 18.1.** $hC$ 只记录 $\pi_0\operatorname{Map}_C(x,y)$，不记录映射空间的高阶同伦群、路径之间的路径或相干复合。

**答案 18.2.** $hN(\mathcal C)\cong\mathcal C$。普通范畴 nerve 的复合严格唯一，1-单纯形的同伦关系不再额外识别不同态射。

**答案 18.3.** 对象 $t$ 终，当且仅当对所有 $x$，映射空间 $\operatorname{Map}_C(x,t)$ 可缩。

**答案 18.4.** 若 $R=\lim p$ 且 $G$ 是右伴随，则对任意 $x$：
$$
\operatorname{Map}_C(x,GR)\simeq\operatorname{Map}_D(Fx,R)
\simeq\lim_k\operatorname{Map}_D(Fx,p(k))
\simeq\lim_k\operatorname{Map}_C(x,Gp(k)).
$$
故 $GR$ 是 $Gp$ 的极限。

**答案 18.5.** HTT 中伴随可由 correspondence 或 adjunction data 定义。本章定义 18.13 是压缩表述；精确定义要求单位、余单位和全部高阶相干三角数据。

**答案 18.6.** 由 $\Delta^m\star\Delta^n\cong\Delta^{m+n+1}$，取 $m=0,n=1$ 得 $\Delta^0\star\Delta^1\cong\Delta^2$。顶点顺序为新锥顶在 $\Delta^1$ 两个顶点之前。

**答案 18.7.** 由 slice 泛性质，$C_{/x}$ 的对象是映射 $\Delta^0\star\Delta^0\to C$，其在右端顶点限制为 $x$。因 $\Delta^0\star\Delta^0\cong\Delta^1$，这就是所有指向 $x$ 的边 $y\to x$。

**答案 18.8.** 对象 $s$ 始，当且仅当对所有 $x$，映射空间 $\operatorname{Map}_C(s,x)$ 可缩。证明与终对象情形对偶，使用 cocone/slice 的对偶定义。

**答案 18.9.** 两个锥为 $(X,u:X\to A,v:X\to B)$ 与 $(X',u':X'\to A,v':X'\to B)$。一个 $1$-单纯形对应态射 $h:X\to X'$，满足 $u'h=u$ 且 $v'h=v$；这些等式自动与 $A\to C\leftarrow B$ 的相容条件兼容。

**答案 18.10.** 始对象对偶地满足对所有 $x$，$\operatorname{Map}_{N(\mathcal C)}(s,x)$ 可缩。该映射空间等价于离散集合 $\mathcal C(s,x)$，故可缩当且仅当 $\mathcal C(s,x)$ 为单点，即 $s$ 是 ordinary initial object。

**答案 18.11.** $0$-单纯形是映射 $\Delta^1\to C$，其第 $0$ 个顶点为 $x$、第 $1$ 个顶点为 $y$，即一条边 $x\to y$。$1$-单纯形是映射 $\Delta^2\to C$，其第 $0$ 个顶点为 $x$，而由顶点 $1,2$ 张成的边退化为 $y$ 的恒等边；它可理解为两条边 $x\to y$ 之间的一条右同伦。

**答案 18.12.** 左映射空间的 $n$-单纯形是函子 $[n+1]\to\mathcal A$，其中由 $0,\dots,n$ 张成的子范畴常值为 $x$，末顶点为 $y$。该函子唯一由边 $n\to n+1$ 的像 $f:x\to y$ 决定；反过来每个 $f$ 给出唯一这样的单纯形。面和退化不改变 $f$，所以得到离散单纯集 $\mathcal A(x,y)$。

**答案 18.13.** 普通范畴情形下，correspondence 是双函子
$$
H:\mathcal C^{op}\times\mathcal D\to\mathbf{Set}.
$$
左可表示意味着存在 $F:\mathcal C\to\mathcal D$，使得对所有 $x,y$ 有自然同构 $H(x,y)\cong\mathcal D(Fx,y)$。

**答案 18.14.** 若同一 $H$ 又右可表示，即 $H(x,y)\cong\mathcal C(x,Gy)$，则复合两个自然同构得到
$$
\mathcal D(Fx,y)\cong H(x,y)\cong\mathcal C(x,Gy),
$$
这正是第四章 Hom 自然同构定义的伴随 $F\dashv G$。

**答案 18.15.** coCartesian 边把源纤维中的对象沿基底箭头 $0\to1$ 推到目标纤维，所以给出 $C=M_0\to M_1=D$。Cartesian 边以目标纤维对象为终点，并把它沿 $0\to1$ 拉回到源纤维，所以给出 $D=M_1\to M_0=C$。

**答案 18.16.** 对 $x$，coCartesian 边 $x\to Fx$ 和以 $Fx$ 为终点的 Cartesian 边 $GFx\to Fx$ 覆盖同一基底箭头。Cartesian 泛性质要求 $x\to Fx$ 唯一分解为纤维内箭头 $x\to GFx$ 后接 $GFx\to Fx$；这个纤维内箭头就是单位 $\eta_x$。

**答案 18.17.** 第一条三角恒等式 $(\varepsilon f)\circ(f\eta)=\operatorname{id}_f$ 作用在左伴随 $f$ 上；第二条 $(g\varepsilon)\circ(\eta g)=\operatorname{id}_g$ 作用在右伴随 $g$ 上。

**答案 18.18.** 传到同伦范畴时，同伦类被当作相等，因此三角恒等式变成严格等式。但原 $\infty$-范畴中仍保留这些同伦本身以及同伦之间的更高相干；这些数据没有消失，只是 $hC$ 不记录它们。

**答案 18.19.** marked simplicial set 标记 $1$-单纯形，常用于记录等价边或 Cartesian edges。scaled simplicial set 标记 $2$-单纯形，常用于记录哪些 $2$-态射应视为 thin 或相干等式。

**答案 18.20.** 严格 $2$-函子把 $+$ 送到 $\mathcal C$，把 $-$ 送到 $\mathcal D$；把 $f:+\to-$ 送到 $F:\mathcal C\to\mathcal D$，把 $g:-\to+$ 送到 $G:\mathcal D\to\mathcal C$。

**答案 18.21.** 只有 $F,G,\eta,\varepsilon$ 还不足以表达伴随；还必须要求复合
$$
F\xrightarrow{F\eta}FGF\xrightarrow{\varepsilon F}F
$$
和
$$
G\xrightarrow{\eta G}GFG\xrightarrow{G\varepsilon}G
$$
等于恒等。高阶口径中这些等式提升为指定相干 $2$-维数据。

**答案 18.22.** correspondence 定义强调 Hom 或 mapping space 的表示性等价；walking adjunction 定义强调由单位、余单位和三角相干生成的代数型结构。二者等价，但突出的是伴随的不同面向。

## 第十九章

**答案 19.1.** 普通 fibration 中，覆盖 $\alpha:b\to p(e)$ 的箭头 $\tilde\alpha:e'\to e$ Cartesian，若任意 $g:x\to e$ 及分解 $p(g)=\alpha\beta$ 唯一提升为 $x\to e'$。

**答案 19.2.** coCartesian lift 沿边 $\alpha:s\to t$ 把纤维 $X_s$ 中对象向前推到 $X_t$，因此给出协变传输。

**答案 19.3.** 对伪函子 $F:B^{op}\to\mathbf{Cat}$，Grothendieck construction 对象为 $(b,x\in F(b))$；态射 $(b,x)\to(c,y)$ 为 $\alpha:b\to c$ 和 $x\to F(\alpha)(y)$。

**答案 19.4.** 第六章公式用普通逗号范畴 $K/d$ 上的余极限；定义 19.9 用 $\infty$-categorical slice $C\times_DD_{/d}$ 上的同伦余极限。

**答案 19.5.** HTT 的 horn 定义用 marked/inner fibration lifting 表达 Cartesian 边的泛性质；映射空间定义是其等价的同伦不变表述。

**答案 19.6.** $C^\natural$ 标记所有等价边；$C^\sharp$ 标记所有边；$C^\flat$ 只标记退化边。通常有包含关系 $C^\flat\subseteq C^\natural\subseteq C^\sharp$。

**答案 19.7.** Cartesian fibration 不只需要总空间和基空间，还需要区分哪些边实现拉回传输。marked simplicial sets 允许把这些 Cartesian edges 作为结构的一部分记录下来。

**答案 19.8.** 普通 Cartesian arrow 要求任意箭头按基底分解唯一提升；映射空间同伦拉回条件把“唯一”替换为“提升空间可缩”，因此是同伦化的唯一分解性质。

**答案 19.9.** 对 $(\alpha,\varphi):(b,x)\to(c,y)$，右乘 $(\operatorname{id}_b,\operatorname{id}_x)$ 给
$$
(\alpha\operatorname{id}_b,\ F(\operatorname{id}_b)(\varphi)\operatorname{id}_x)=(\alpha,\varphi).
$$
左乘 $(\operatorname{id}_c,\operatorname{id}_y)$ 同理给 $(\alpha,\operatorname{id}_{F(\alpha)y}\varphi)=(\alpha,\varphi)$。

**答案 19.10.** 对象为 $(0,a)$ 与 $(1,b)$，其中 $a\in\mathcal A,b\in\mathcal B$。纤维分别是 $\mathcal A$ 与 $\mathcal B$。跨纤维态射只能从 $(0,a)$ 到 $(1,b)$，由 $\alpha:0\to1$ 和态射 $a\to u(b)$ 组成；没有从 $1$ 到 $0$ 的跨纤维态射。

**答案 19.11.** 设有跨纤维态射 $(0,a)\to(1,b)$，它由 $\mathcal A$ 中态射 $\phi:a\to u(b)$ 给出。若它覆盖 $0\to1$，则经 Cartesian lift
$$
(0,u b)\to(1,b)
$$
分解时，唯一候选为纤维 $0$ 中的态射 $(0,a)\to(0,u b)$，即 $\phi:a\to u(b)$。复合公式给回原态射，唯一性由 $\phi$ 被原跨纤维态射唯一决定。

**答案 19.12.** Cartesian lift 的泛性质只给出可缩的选择空间，而不是指定一个严格唯一对象。两个选择都是同一泛性质的解，因此由唯一到同伦唯一的原则给出等价。

**答案 19.13.** 设 $e:x\to y$ 覆盖 $\alpha$、$f:y\to z$ 覆盖 $\beta$ 且二者 Cartesian。任意箭头 $w\to z$ 若基底分解经过 $\beta\alpha$，先由 $f$ 的 Cartesian 性唯一分解到 $y$，再由 $e$ 的 Cartesian 性唯一分解到 $x$。两次唯一性合成给出 $fe$ 的 Cartesian 性。

**答案 19.14.** 对 $x\in\mathcal C_2$，等式 $u_{02}=u_{01}u_{12}$ 表示直接把 $x$ 从纤维 $2$ 限制到纤维 $0$，等于先限制到纤维 $1$ 再限制到纤维 $0$：
$$
u_{02}(x)=u_{01}(u_{12}(x)).
$$

**答案 19.15.** 当 $S=[1]$ 且限制函子为 $u:\mathcal B\to\mathcal A$ 时，Cartesian section 由 $a\in\mathcal A$、$b\in\mathcal B$ 和等价 $a\simeq u(b)$ 组成；在普通严格模型中就是选择 $b$ 并令 $a=u(b)$。

**答案 19.16.** Descent data 要求在各局部对象上选择数据，并在交叠和高重交叠上给出相容等价。把这些局部范畴组织为覆盖单纯形上的 Cartesian fibration 后，Cartesian sections 正是这种同伦相干的相容选择；定理 19.H 把它识别为相应图形的极限。

## 第二十章

**答案 20.1.** 若 $0$ 是零对象，$X\to0$ 与 $0\to Y$ 唯一，故复合 $X\to0\to Y$ 给出唯一零态射。

**答案 20.2.** 链映射 $f:A\to B$ 的映射锥为 $\operatorname{Cone}(f)^n=B^n\oplus A^{n+1}$，微分 $d(b,a)=(d_Bb+f(a),-d_Aa)$。在导出 $\infty$-范畴中它表示余纤维。

**答案 20.3.** 三角范畴只记录同伦范畴和 distinguished triangles，不记录映射谱或高阶相干，因此不能唯一恢复稳定 $\infty$-范畴。

**答案 20.4.** sequential spectrum 为 pointed spaces 序列 $E_n$ 和结构映射 $\Sigma E_n\to E_{n+1}$；$\Omega$-谱要求伴随映射 $E_n\to\Omega E_{n+1}$ 为弱等价。

**答案 20.5.** heart 为阿贝尔范畴需要稳定性提供纤维/余纤维和正合三角，t-结构提供截断和正负正交，从而定义核、余核并证明阿贝尔公理。

**答案 20.6.** $\Sigma X$ 由推出方块 $X\to0\leftarrow0$ 定义。稳定性使该推出方块同时为拉回，因此 $X$ 表示 $0\to\Sigma X$ 的纤维，即 $\Omega\Sigma X\simeq X$。

**答案 20.7.** $\Sigma X=\operatorname{cofib}(X\to0)$。正合函子保持零对象和余纤维，故
$$
F(\Sigma X)\simeq F\operatorname{cofib}(X\to0)\simeq\operatorname{cofib}(FX\to0)=\Sigma F(X).
$$

**答案 20.8.** 结构映射为 $\sigma_n:\Sigma E_n\to E_{n+1}$。由 $\Sigma\dashv\Omega$，它对应伴随映射
$$
\tilde\sigma_n:E_n\to\Omega E_{n+1}.
$$
$\Omega$-谱条件要求每个 $\tilde\sigma_n$ 都是等价。

**答案 20.9.** 球面满足 $\Sigma S^n\simeq S^{n+1}$，结构映射取该标准识别。它与悬挂坐标选择相容，并把 sequential spectrum 的第 $n$ 项自然推进到第 $n+1$ 项。

**答案 20.10.** 三角范畴只记录 $\pi_0$ 层面的 Hom 群和 distinguished triangles；映射谱还记录所有 $\pi_n$，即所有悬挂度数上的态射群及其高阶同伦相干。因此不同稳定 $\infty$-范畴可能有相同三角同伦范畴但不同映射谱。

**答案 20.11.** 正合函子与悬挂相容，所以给出
$$
hC(\Sigma^nX,Y)\to hD(F\Sigma^nX,FY)\cong hD(\Sigma^nFX,FY).
$$
这就是映射谱同伦群上的诱导映射。

**答案 20.12.** $\mathbf{Sp}$ 是幺半 $\infty$-范畴，乘法为 smash product。一个 $E_1$-代数正是带同伦相干结合乘法和单位的对象，因此 ring spectrum 应定义为 $\mathbf{Sp}$ 中的 $E_1$-代数。

**答案 20.13.** 按定义 $H^0(Y)=\tau_{\le0}\tau_{\ge0}Y$。先截到 $C_{\ge0}$，再截到 $C_{\le0}$，所得对象同时在 $C_{\ge0}$ 和 $C_{\le0}$ 中，因此属于 heart。

**答案 20.14.** 余核对象 $Q$ 应满足：对任意 heart 对象 $T$，从 $Q$ 到 $T$ 的映射等价于从 $B$ 到 $T$ 且复合 $A\to B\to T$ 为零的映射。$\operatorname{cofib}(f)$ 表示在稳定范畴中杀掉 $A$ 后的对象；取 $H^0$ 把它返回 heart，因此给出 heart 中余核。

**答案 20.15.** 两步滤过为 $0=F_{-1}X\to F_0X\to F_1X=X$。因此
$$
\operatorname{gr}_0X=\operatorname{cofib}(0\to F_0X)\simeq F_0X,
$$
$$
\operatorname{gr}_1X=\operatorname{cofib}(F_0X\to X).
$$

**答案 20.16.** 在 exact couple 中 $d=jk$。于是
$$
d^2=jkjk=j(kj)k.
$$
正合性给出 $\operatorname{im}(j)=\ker(k)$，因此 $kj=0$，故 $d^2=0$。

**答案 20.17.** 若滤过有限，例如 $F_pX=0$ 对 $p<a$ 且 $F_pX=X$ 对 $p>b$，则 $\operatorname{gr}_pX=0$ 除有限多个 $p$ 外皆为零。固定总次数 $n=p+q$ 时，只有这些有限个 $p$ 可能贡献 $E_1^{p,q}$。

**答案 20.18.** $E_\infty$ 页描述的是目标 $H^*(X)$ 上某个滤过的 associated graded，即各层商 $F_p/F_{p-1}$。从层商恢复对象还需要扩张数据；不同扩张可能有相同 associated graded。

**答案 20.19.** 若 $A,B$ 有 biproduct，则
$$
f+g=
A\xrightarrow{\Delta}A\oplus A\xrightarrow{f\oplus g}B\oplus B\xrightarrow{\nabla}B.
$$
这里 $\Delta$ 是对角态射，$\nabla$ 是余对角态射。

**答案 20.20.** 若 $fu=fv$，则 $f(u-v)=0$。由核的泛性质，$u-v$ 唯一经 $\ker(f)$ 分解。若 $\ker(f)=0$，则 $u-v=0$，故 $u=v$。因此 $f$ 是 monomorphism。

**答案 20.21.** 短正合列给出三角 $A\to B\to C\to\Sigma A$，长正合列为
$$
\cdots\to H^n(A)\to H^n(B)\to H^n(C)\xrightarrow{\partial}
H^{n+1}(A)\to H^{n+1}(B)\to\cdots.
$$
若 $A,B,C$ 本身位于 heart，则只有 $n=0$ 附近非零，并恢复通常短正合列的 exactness。

**答案 20.22.** 三步滤过为 $0=F_{-1}X\to F_0X\to F_1X\to F_2X=X$。因此
$$
\operatorname{gr}_0X\simeq F_0X,\qquad
\operatorname{gr}_1X=\operatorname{cofib}(F_0X\to F_1X),
$$
$$
\operatorname{gr}_2X=\operatorname{cofib}(F_1X\to X).
$$

**答案 20.23.** 有限滤过使每个总次数只含有限多个 filtration degree。导出 exact couple 时，微分不可能从任意远处持续进入或离开固定项，因此没有无限链导致的 $\lim^1$ 或无限扩张障碍。

**答案 20.24.** 标准 t-结构中，heart 由只在次数 $0$ 有上同调的复形组成，等价于原阿贝尔范畴。通常上同调 $H^n(X)$ 被视为集中在次数 $0$ 的对象，因此属于 heart。

**答案 20.25.** Exhaustive filtration 意味滤过的所有层合起来恢复原对象，即 $\operatorname{colim}_pF_pX\to X$ 是等价。它排除“滤过只覆盖了对象一部分”的情况。

**答案 20.26.** Separated 要求无限向下交集为零，避免同一元素在所有滤过层中不可分辨。Complete 要求对象等于由滤过商得到的极限，保证可从所有有限阶段恢复整体。

**答案 20.27.** t-结构 left complete，若每个对象 $X$ 都满足
$$
X\simeq\lim_n\tau_{\le n}X.
$$
也就是说，对象由其 Postnikov tower 的向下截断极限恢复。

**答案 20.28.** 有限滤过只有有限多个 graded pieces，因此不存在无限下降链、无限上升链或 $\lim^1$ 型完成障碍。谱序列在每个总次数经过有限步即稳定。

**答案 20.29.** Postnikov tower 的相邻层通常由 cohomology object 控制：
$$
\operatorname{fib}(\tau_{\le n}X\to\tau_{\le n-1}X)
$$
等价于 $H^n(X)$ 的相应平移。因此 graded pieces 是 cohomology objects 的移位。

**答案 20.30.** 固定总次数后，只有有限多个 $p$ 上的 $E_1^{p,n-p}$ 非零。微分若要进入或离开这些项，只能连接有限集合中的位置；当页数足够大时已无可能的源或靶，因此不存在无限微分链。

## 第二十一章

**答案 21.1.** 集合值 sheaf 的下降是等化子条件；space 值 sheaf 的下降是同伦极限条件，包含高阶相容同伦。

**答案 21.2.** 平凡站点 $*$ 上的 $\infty$-sheaf 范畴就是 $\operatorname{Fun}(*,\mathcal S)\simeq\mathcal S$。

**答案 21.3.** Čech nerve 前三层为 $\coprod_iU_i$、$\coprod_{i,j}U_i\times_UU_j$、$\coprod_{i,j,k}U_i\times_UU_j\times_UU_k$。

**答案 21.4.** left exact localization 是保持有限极限的反射 $L:\mathcal P\to\mathcal X$。sheaf 化应保持有限极限，因为 sheaf topos 的有限极限应与局部粘合相容。

**答案 21.5.** 普通 Giraud 用集合值范畴的余极限普遍性、等价关系有效等；高阶 Giraud 把等价关系替换为群胚对象有效，把集合级条件提升为同伦相容条件。

**答案 21.6.** 离散 space 值 $\infty$-sheaf 对覆盖族的 Čech nerve 给出同伦极限。对离散对象取 $\pi_0$ 后，同伦极限条件化为普通等化子条件，因此得到 ordinary sheaf。

**答案 21.7.** space $X$ 为 $0$-截断意味着对任意基点，其高阶同伦群消失，且路径空间离散；于是 $X$ 等价于离散 space $\pi_0X$。离散 spaces 正是集合。

**答案 21.8.** 覆盖 $U_0\to U$ 的 Čech nerve 满足 $U_n=U_0\times_U\cdots\times_UU_0$。每个 matching map 由相应投影和重复交叉给出，并由覆盖的拉回稳定性仍为覆盖，因此 Čech nerve 是超覆盖。

**答案 21.9.** Čech descent 只检查由单个覆盖族反复取交叉得到的单纯对象。Hyperdescent 还允许每一维用新的覆盖去覆盖 matching object，因此检查高维粘合数据及其相容同伦，比 Čech descent 更强。

**答案 21.10.** 对 $f:U\to X$，Čech nerve 为
$$
U,\quad U\times_XU,\quad U\times_XU\times_XU,\quad\ldots.
$$
其几何实现把 $U$ 中映到同一 $X$ 点的元素按全部高阶相容关系粘合。若 $f$ 是 effective epimorphism，该粘合结果正是 $X$。

**答案 21.11.** 在 $\mathbf{Set}$ 中，$U\to X$ 的 Čech nerve 的几何实现是按关系“两个元素有相同像”取商。该商等于 $X$ 当且仅当 $U\to X$ 满射。

**答案 21.12.** Groupoid object 是内部对象、箭头、复合、单位和逆的同伦相干版本。它描述对象之间的等价关系及其高阶相容同伦，因此是内部同伦等价关系。

**答案 21.13.** 在 $\mathcal S$ 中，任意集合看成离散 space 是 $0$-截断对象。任意普通群胚的 nerve，或 $BG$，是 $1$-截断对象；它可能有非平凡 $\pi_1$，但高于 $1$ 的同伦群消失。

**答案 21.14.** Postnikov tower 把对象逐层截断为 $0$-型、$1$-型、$2$-型等近似。它把高阶 sheaf 的信息拆成可逐级理解的同伦层，并为 hypercompletion 和同伦 sheaf cohomology 提供基础。

**答案 21.15.** Hyperdescent 是对所有 hypercovers 的下降条件。Hypercompletion 是把 $\infty$-topos 局部化到满足这种更强下降或由 Postnikov tower 可恢复的对象。标准条件下，hypercomplete sheaves 正是满足 hyperdescent 的 sheaves。

**答案 21.16.** 几何态射 $f:\mathcal X\to\mathcal Y$ 给出
$$
f^*:\mathcal Y\rightleftarrows\mathcal X:f_*,
$$
其中 $f^*$ 是左伴随且保持有限极限，$f_*$ 是右伴随。

**答案 21.17.** 点 $x$ 的 stalk $F_x$ 把 sheaf $F$ 送到其在 $x$ 附近的局部截面余极限。这个函子从 $\operatorname{Sh}_\infty(X)$ 到 $\mathcal S$，保持有限极限，并作为几何态射 $\mathcal S\to\operatorname{Sh}_\infty(X)$ 的 inverse image。

**答案 21.18.** 条件形式相同：都有伴随 $f^*\dashv f_*$，并要求 inverse image $f^*$ 保持有限极限。差别在于 $\infty$-topos 中这些范畴、函子和极限都在 $\infty$-范畴意义下理解，保留高阶同伦相干。

## 第二十二章

**答案 22.1.** 普通代数对象用乘法和单位加严格交换图定义；定义 22.9 中的 $\infty$-operad 代数是 operad 映射，所有结合律、单位律和交换律由高阶单纯形相干编码。

**答案 22.2.** $E_\infty$-代数的交换律不是等式 $ab=ba$，而是给出一族可兼容的同伦及更高同伦，形成可缩的交换选择空间。

**答案 22.3.** $\mathbf{Fin}_*$ 对象为有限带基点集合 $\langle n\rangle=\{*,1,\dots,n\}$。inert morphism 是每个非基点的原像恰有一个非基点的映射。

**答案 22.4.** 环谱是谱的幺半 $\infty$-范畴中的 $E_1$-代数：乘法 $A\wedge A\to A$ 和单位 $\mathbb S\to A$ 满足同伦相干结合律。

**答案 22.5.** 张量积分别保持余极限用于构造自由代数、证明代数范畴 presentable、建立模范畴的余极限和相对张量积。

**答案 22.6.** 非基点中不被送到 $*$ 的集合为 $T=\{1,3\}$。令 $\rho:\langle3\rangle\to\langle2\rangle$ 把 $1\mapsto1,3\mapsto2,2\mapsto*,*\mapsto*$；这是 inert。令 $\alpha:\langle2\rangle\to\langle2\rangle$ 把 $1\mapsto1,2\mapsto1,*\mapsto*$；这是 active，且 $f=\alpha\rho$。

**答案 22.7.** inert 态射要求目标每个非基点有唯一原像，因此只是在源中选出若干输入并按槽位投影。active 态射不把非基点送到基点，因此保留全部输入并把它们合成为目标输出，正对应多输入运算。

**答案 22.8.** $\rho_2:\langle3\rangle\to\langle1\rangle$ 满足 $\rho_2(2)=1$，并把 $*,1,3$ 都送到 $*$。目标唯一非基点 $1$ 的原像恰为 $\{2\}$，所以 $\rho_2$ 是 inert。

**答案 22.9.** Segal 条件说由 inert 投影诱导的函子
$$
\mathcal O^\otimes_{\langle n\rangle}\to\prod_{i=1}^n\mathcal O^\otimes_{\langle1\rangle}
$$
是等价。因此给定 $\langle n\rangle$ 上对象等价于给定它沿每个输入槽的投影，也就是给定 $n$ 个颜色的列表。

**答案 22.10.** 若 $C$ 是普通集合或阿贝尔群上的幺半范畴，左 $A$-模作用就是映射 $A\otimes M\to M$。在普通环情形中，底层集合上写作 $A\times M\to M$，满足结合律 $(ab)m=a(bm)$ 和单位律 $1m=m$。

**答案 22.11.** 由定义：
$$
\operatorname{Bar}_0(M,A,N)=M\otimes N,
$$
$$
\operatorname{Bar}_1(M,A,N)=M\otimes A\otimes N,
$$
$$
\operatorname{Bar}_2(M,A,N)=M\otimes A\otimes A\otimes N.
$$
面映射分别使用右作用、乘法和左作用；退化映射插入单位。

**答案 22.12.** 普通平衡张量积把 $(ma)\otimes n$ 与 $m\otimes(an)$ 识别。Bar 构造用一整个单纯对象系统地加入这些识别及其高阶相干，因此其几何实现给出同伦意义下的平衡张量积。

**答案 22.13.** 若 $M$ 是 $(A,B)$-双模，$N$ 是 $(B,C)$-双模，则复合应消去中间代数 $B$ 的左右作用。相对张量积 $M\otimes_BN$ 正是把 $mb\otimes n$ 与 $m\otimes bn$ 以同伦相干方式识别，因此得到 $(A,C)$-双模。

**答案 22.14.** 普通代数中 $Z(A)=\{a\in A\mid ax=xa\ \forall x\in A\}$。它也可看作 $A$ 作为 $(A,A)$-双模的双模自同态。定义 22.I 正是把这个 endomorphism 描述提升到 $\infty$-范畴。

**答案 22.15.** 圆周可由区间端点粘合得到；沿圆周积分一个 $E_1$-代数时，局部乘法数据按循环顺序粘合，形成 cyclic bar construction。其几何实现就是 Hochschild homology $HH(A)$。

**答案 22.16.** 在 Morita $\infty$-范畴中，$M:{}_A\to{}_B$ 和 $N:{}_B\to{}_A$ 的复合分别是 $M\otimes_BN$ 与 $N\otimes_AM$。若它们等价于单位双模 $A$ 和 $B$，就正是说 $M$ 与 $N$ 互为逆 1-态射。

**答案 22.17.** Dualizable 只要求对象有对偶以及评价/余评价。Fully dualizable 还要求这些评价/余评价态射本身继续有左右 adjoints，并递归到所有高阶层级，因此是更强的有限性条件。

**答案 22.18.** Cobordism hypothesis 说 framed fully extended TFT 构成的 $\infty$-群胚等价于目标中 fully dualizable objects 的 $\infty$-群胚。等价由“取点的值”给出，因此一旦知道点的取值及其 fully dualizable 结构，整个场论由定理唯一延拓。

**答案 22.19.** 单位双模 ${}_AA_A$ 的底层对象是 $A$。左作用和右作用都是乘法 $A\otimes A\to A$，分别作用在左因子和右因子上；结合律保证左右作用相容，单位律保证模单位公理。

**答案 22.20.** 映射 $A\otimes_AM\to M$ 为 $a\otimes m\mapsto am$。其逆为 $m\mapsto1\otimes m$。平衡关系给出 $a\otimes m=1\otimes am$，所以两个复合都是恒等。

**答案 22.21.** 对右 $A$-模 $P$，
$$
(P\otimes_AM)\otimes_BN
\cong P\otimes_A(M\otimes_BN)
\cong P\otimes_AA
\cong P.
$$
对右 $B$-模 $Q$ 同理得到
$$
(Q\otimes_BN)\otimes_AM\cong Q.
$$

**答案 22.22.** 取标准基 $e_i$ 与对偶基 $e^j$。映射
$$
k^n\otimes(k^n)^*\to M_n(k)
$$
把 $e_i\otimes e^j$ 送到矩阵单位 $E_{ij}$，即只在第 $(i,j)$ 位为 $1$ 的矩阵。

**答案 22.23.** Proper 条件要求 Hom 对象在基环上 perfect 或有限型。它说明任意两个对象之间的态射复形没有无限维不可控部分，是 Morita 理论中保证评价/余评价存在并有伴随的有限性条件之一。

**答案 22.24.** 非退化配对给出 $A\cong A^*$，因此任意双线性型都唯一对应一个线性映射到对偶。要求
$$
\langle ab,c\rangle=\langle a\otimes b,\Delta(c)\rangle
$$
对所有 $a,b,c$ 成立，因配对非退化，$\Delta(c)$ 被唯一确定。

**答案 22.25.** 圆柱是圆周到自身的恒等 bordism。代数上它可分解为单位后接乘法或余乘法后接余单位；Frobenius 代数的单位律和余单位律保证所得线性映射为 $\operatorname{id}_A$。

**答案 22.26.** 普通二维 TFT 只给闭一维流形赋值，因此基本对象是圆周，其值为 Frobenius 代数。Fully extended TFT 还给点、区间和带角 bordism 赋值；cobordism hypothesis 说最高层数据由点上的 fully dualizable object 控制。

## 第二十三章

**答案 23.1.** 由 $\infty$-Yoneda，
$$
\operatorname{Map}_{\mathcal P(C)}(j(x),j(y))\simeq j(y)(x)=\operatorname{Map}_C(x,y).
$$
该等价对 $x,y$ 自然，所以 $j$ 全忠实。

**答案 23.2.** 终对象 $1_{\mathcal P(C)}$ 应满足对任意 $F$，映射空间 $\operatorname{Map}(F,1)$ 可缩。逐点取 $1_{\mathcal S}$ 得到预层 $c\mapsto1_{\mathcal S}$，自然变换到它逐点唯一到可缩，因此它是终对象。

**答案 23.3.** 对滤过图形 $F_i$，用 Yoneda 和逐点余极限：
$$
\operatorname{Map}(j(c),\operatorname{colim}_iF_i)
\simeq
(\operatorname{colim}_iF_i)(c)
\simeq
\operatorname{colim}_iF_i(c)
\simeq
\operatorname{colim}_i\operatorname{Map}(j(c),F_i).
$$

**答案 23.4.** 第十二章用 Hom 集保持滤过余极限定义紧对象；定义 23.4 把 Hom 集替换为映射空间，并要求 $\operatorname{Map}_C(x,-)$ 保持滤过余极限。

**答案 23.5.** $\operatorname{Ind}_\kappa(C)$ 是 $\mathcal P(C)$ 中包含所有可表预层 $j(c)$ 且对 $\kappa$-滤过余极限封闭的最小全子 $\infty$-范畴。

**答案 23.6.** $\mathcal S\simeq\mathcal P(*)$，其中 $*$ 是终小 $\infty$-范畴。预层 $\infty$-范畴 presentable，因此 $\mathcal S$ presentable。

**答案 23.7.** 若 $C,D$ presentable 且 $F:C\to D$ 保持所有小余极限，则 presentable $\infty$-范畴伴随函子定理说明 $F$ 是左伴随，因此有右伴随。

**答案 23.8.** 若 $X$ 局部，则 $\eta_X:X\to LX$ 是等价。对 $LX$，局部化公理给出 $\eta_{LX}:LX\to L^2X$ 与 $L\eta_X$ 等价，而 $L\eta_X$ 是等价，所以 $LX$ 局部。

**答案 23.9.** $Z$ 是 $S$-局部对象，若对每个 $f:A\to B$ 属于 $S$，
$$
\operatorname{Map}(B,Z)\to\operatorname{Map}(A,Z)
$$
是等价。这说明 $Z$ 无法区分 $f$ 的源和靶，即 $f$ 在映入 $Z$ 时被视为等价。

**答案 23.10.** Sheaf 化把任意预层 $F$ 送到 sheaf $aF$，并且对任意 sheaf $G$ 有
$$
\operatorname{Map}(aF,G)\simeq\operatorname{Map}(F,G).
$$
局部对象是 sheaves，局部等价由覆盖下降检测，因此它是 Bousfield localization 的一个几何例子。

**答案 23.11.** $\infty$-topos 可定义为预层 $\infty$-范畴的 left exact accessible localization。因此 left exact localization 是从自由预层世界构造高阶 sheaf 世界的机制。

**答案 23.12.** 稳定 $\infty$-范畴中推出方块等价于拉回方块，纤维可由余纤维和环路表达，环路又是悬挂的逆。因此保持有限余极限和零对象就控制有限极限。

**答案 23.13.** $\operatorname{Pr}^L$ 的对象是 presentable $\infty$-categories，态射是保持小余极限的函子，即左伴随。$\operatorname{Pr}^R$ 对象相同，态射是右伴随。

**答案 23.14.** 等价 $(\operatorname{Pr}^L)^{op}\simeq\operatorname{Pr}^R$ 表示给左伴随 $F:C\to D$ 取其右伴随 $G:D\to C$，可反向得到 $\operatorname{Pr}^R$ 中的态射，并且这个过程在高阶相干意义下可逆。

**答案 23.15.** 张量积 $C\otimes D$ 由泛性质刻画：
$$
\operatorname{Fun}^L(C\otimes D,E)
\simeq
\operatorname{Fun}^{L,L}(C\times D,E),
$$
右侧是分别保持余极限的双变量函子。

**答案 23.16.** 若张量积分别保持余极限，则自由代数、模对象和 bar 几何实现能在 presentable 环境中构造，并且相对张量积与余极限相容。这是第二十二章模 $\infty$-范畴和相对张量积存在性的基本假设。

## 第二十四章

**答案 24.1.** 对 $F:\mathcal C\to\mathcal D$，
$$
F_*(c,d)=\mathcal D(Fc,d),\qquad
F^*(d,c)=\mathcal D(d,Fc).
$$

**答案 24.2.** 若 $P:\mathcal C\nrightarrow\mathcal D$、$Q:\mathcal D\nrightarrow\mathcal E$，则
$$
(Q\circ P)(c,e)=\int^{d\in\mathcal D}P(c,d)\times Q(d,e).
$$

**答案 24.3.** 在 $(c,d)$ 处，
$$
(\operatorname{id}_{\mathcal D}\circ P)(c,d)
=
\int^{d'}P(c,d')\times\mathcal D(d',d)
\cong P(c,d)
$$
由 co-Yoneda 得到。

**答案 24.4.** 余单位
$$
\int^{c}\mathcal D(d,Fc)\times\mathcal D(Fc,d')\to\mathcal D(d,d')
$$
把一对态射 $d\to Fc$ 与 $Fc\to d'$ 送到它们在 $\mathcal D$ 中的复合。

**答案 24.5.** 幂等 $e:X\to X$ 分裂，若存在 $r:X\to Y$ 与 $s:Y\to X$，使 $sr=e$ 且 $rs=\operatorname{id}_Y$。

**答案 24.6.** 若 $u:(X,e)\to(Y,f)$、$v:(Y,f)\to(Z,g)$，则
$$
vu=gvu=vfu=vue,
$$
所以复合满足 Karoubi 态射条件。

**答案 24.7.** 若 $\mathcal C$ 已幂等完备，则每个 $(X,e)$ 中的 $e$ 在 $\mathcal C$ 中分裂为某个对象 $Y$。于是 $(X,e)$ 同构于嵌入像中的 $(Y,\operatorname{id}_Y)$，故 $\mathcal C\to\operatorname{Kar}(\mathcal C)$ 本质满且完全忠实。

**答案 24.8.** 普通余极限是常值单点权重 $1:\mathcal J^{op}\to\mathbf{Set}$ 的加权余极限：
$$
\operatorname{colim}_{j}D(j)\cong\int^{j}1\times D(j).
$$

**答案 24.9.** 普通 profunctor 取集合值，只记录广义态射集合；$\infty$-correspondence 取 space 值，记录广义态射空间及其高阶同伦。

**答案 24.10.** Profunctor 复合用 coend 平衡中间范畴变量；Morita 理论中双模复合用相对张量积 $M\otimes_BN$ 平衡中间代数 $B$ 的作用。后者是前者在线性、导出或谱值环境中的高阶代数版本。

## 第二十五章

**答案 25.1.** $\mathcal V$-profunctor $M:\mathcal A\nrightarrow\mathcal B$ 是富函子
$$
M:\mathcal A^{op}\otimes\mathcal B\to\mathcal V.
$$

**答案 25.2.** 当 $\mathcal V=\mathbf{Ab}$ 时，$\mathcal A,\mathcal B$ 是预加性范畴，profunctor 给每对对象一个阿贝尔群 $M(a,b)$，并且左右态射作用是双线性的。

**答案 25.3.** 若 $M:\mathcal A\nrightarrow\mathcal B$、$N:\mathcal B\nrightarrow\mathcal C$，则
$$
(N\circ M)(a,c)=\int^{b\in\mathcal B}M(a,b)\otimes N(b,c).
$$

**答案 25.4.** 右单位为
$$
\int^{b'}M(a,b')\otimes\mathcal B(b',b)\cong M(a,b),
$$
这是富 co-Yoneda 公式。

**答案 25.5.** 在 equipment 中，companion 是垂直态射 $f:A\to B$ 对应的同向水平态射 $f_*:A\nrightarrow B$；conjoint 是反向水平态射 $f^*:B\nrightarrow A$，二者满足单位/余单位二重胞腔和三角恒等式。

**答案 25.6.** 对富函子 $F:\mathcal A\to\mathcal B$，
$$
F_*(a,b)=\mathcal B(Fa,b),\qquad
F^*(b,a)=\mathcal B(b,Fa).
$$

**答案 25.7.** 在 $\mathbf{Prof}$ 中，水平态射是双变量函子。带有垂直函子的二重胞腔正是使相应四边形相容的自然变换 $M(a,b)\to N(fa,gb)$。

**答案 25.8.** Beck-Chevalley 条件说“先沿一个方向推/求和再拉回”与“先拉回再推/求和”得到同构。它是 base change 合理性的抽象表达。

**答案 25.9.** 若 $X'\cong Y'\times_YX$，则对 $E\to X$，
$$
Y'\times_YE\cong X'\times_XE.
$$
两边分别对应先推后拉和先拉后推，因此给出 Beck-Chevalley 同构。

**答案 25.10.** 对伪函子 $\mathcal F:B^{op}\to\mathbf{Cat}$，Grothendieck construction 的对象是 $(b,x)$，其中 $x\in\mathcal F(b)$；态射 $(b,x)\to(c,y)$ 是 $\alpha:b\to c$ 与 $x\to\alpha^*y$。

**答案 25.11.** 给定 fibration $p:E\to B$ 和 $\alpha:b\to c$，对 $y\in E_c$ 取 Cartesian lift $\alpha^*y\to y$。该选择对对象和态射给出重索引函子 $\alpha^*:E_c\to E_b$。

**答案 25.12.** Indexed category 是普通伪函子 $B^{op}\to\mathbf{Cat}$；Cartesian fibration 是 $\infty$-范畴版本，编码 $S^{op}\to\mathcal{Cat}_\infty$。二者由 straightening/Grothendieck construction 相连。

**答案 25.13.** Ordinary bicategory 只有对象、1-态射和 2-态射。Equipment 额外区分垂直态射与水平态射，并要求垂直态射有 companion/conjoint，从而能表达 base change 方块。

**答案 25.14.** Morita 理论既有代数之间的结构保持映射，也有双模作为广义态射；复合由相对张量积给出。Equipment 或 $(\infty,2)$-equipment 能同时记录这些方向、二重胞腔和 Beck-Chevalley 型相干。

## 第二十六章

**答案 26.1.** 对有滤过余极限的 $\infty$-范畴 $C$，对象 $K$ compact，若 $\operatorname{Map}_C(K,-)$ 保持滤过余极限。稳定情形可等价地用映射谱函子表述。

**答案 26.2.** 对谱 $X$，若所有 $\pi_nX\cong\pi_0\operatorname{Map}_{\mathbf{Sp}}(\Sigma^n\mathbb S,X)$ 为零，则 $X\simeq0$。因此 $\mathbb S$ 及其悬挂检测零对象并生成 $\mathbf{Sp}$。

**答案 26.3.** 在 $D(R)$ 中，
$$
\operatorname{Map}_{D(R)}(R,\Sigma^nX)
$$
计算 $X$ 的第 $n$ 个同调或上同调对象。若这些全为零，则复形为零对象，因此 $R$ 生成 $D(R)$。

**答案 26.4.** Localizing subcategory 是稳定全子范畴，且对所有小余积封闭。

**答案 26.5.** Verdier quotient $q:C\to C/L$ 把 $L$ 中对象送为零，并且任意把 $L$ 送为零的正合函子 $C\to D$ 唯一因子化经 $C/L$。

**答案 26.6.** 三角范畴中的 Verdier quotient 只控制同伦范畴和三角；稳定 $\infty$-范畴商保留映射空间/映射谱，并在传到同伦范畴后给出三角 Verdier quotient。

**答案 26.7.** 一个形式是：若 $C$ compactly generated stable，且 $H:C^{op}\to\mathcal S$ cohomological 并把小余积送到小积，则 $H$ 由某个对象表示。

**答案 26.8.** 保持小余积的正合函子在稳定 presentable 范畴中保持所有小余极限，因为小余极限由小余积和有限余极限生成。由 presentable 伴随函子定理，它有右伴随。

**答案 26.9.** $X$ 是 $L$-acyclic，若 $LX\simeq0$。$Y$ 是 $L$-local，若单位 $Y\to LY$ 是等价。

**答案 26.10.** 若每个 $X_i$ acyclic，则
$$
L(\coprod_iX_i)\simeq\coprod_iLX_i\simeq0,
$$
因为 $L$ 保持小余积。

**答案 26.11.** 令 $A_X=\operatorname{fib}(X\to LX)$。稳定范畴中纤维序列也是余纤维序列，因此有 $A_X\to X\to LX$。应用 $L$ 后 $LX\to L^2X$ 为等价，所以 $LA_X=0$。

**答案 26.12.** Smashing localization 是形如 $LX\simeq E\otimes X$ 的 Bousfield localization。谱中写作 $LX\simeq E\wedge X$。

**答案 26.13.** 若 $L\simeq E\otimes-$，而 $E\otimes-$ 是左伴随，则保持所有小余极限。

**答案 26.14.** Verdier quotient 后 compact objects 的像未必已经幂等完备；某些 retract 只在商中出现。为了得到全部 compact objects，需要对小商 $C^\omega/L^\omega$ 作 Karoubi 完备化。

## 第二十七章

**答案 27.1.** Small dg category 是 $\operatorname{Ch}(k)$-富范畴：有对象集、Hom 复形 $\mathcal A(x,y)$、链映射形式的复合
$$
\mathcal A(y,z)\otimes\mathcal A(x,y)\to\mathcal A(x,z)
$$
和单位 $k\to\mathcal A(x,x)$，满足结合律和单位律。

**答案 27.2.** Hom 复形复合是链映射，所以 cycle 的复合仍为 cycle，boundary 与任意 cycle 复合后仍为 boundary。故复合下降到 $H^0$。链级结合律和单位律取 $H^0$ 后给出普通范畴公理。

**答案 27.3.** dg 函子是对象函数加 Hom 复形链映射 $\mathcal A(x,y)\to\mathcal B(Fx,Fy)$，相容于单位和复合。Quasi-equivalence 要求这些 Hom 链映射都是 quasi-isomorphism，且 $H^0(F)$ 本质满。

**答案 27.4.** Hom quasi-isomorphism 在 $H^0$ 上给出 Hom 集同构，因此 $H^0(F)$ 完全忠实。本质满是 quasi-equivalence 的第二条件，所以 $H^0(F)$ 是范畴等价。

**答案 27.5.** 右 dg 模是 dg 函子 $M:\mathcal A^{op}\to\operatorname{Ch}(k)$。可表模为
$$
h_a=\mathcal A(-,a).
$$

**答案 27.6.** 富自然变换 $h_a\to M$ 由单位 $k\to\mathcal A(a,a)$ 上的像决定，即由 $M(a)$ 的元素决定；反向由右模作用
$$
M(a)\otimes\mathcal A(x,a)\to M(x)
$$
给出。两构造互逆并与微分相容。

**答案 27.7.** 对滤过图形 $M_i$，
$$
\operatorname{Map}(h_a,\operatorname{colim}_iM_i)
\simeq
(\operatorname{colim}_iM_i)(a)
\simeq
\operatorname{colim}_iM_i(a)
\simeq
\operatorname{colim}_i\operatorname{Map}(h_a,M_i).
$$
所以 $h_a$ compact。

**答案 27.8.** Perfect modules 是 $D(\mathcal A)$ 中由可表模 $h_a$ 生成的最小稳定、幂等完备全子范畴。等价地，它们是有限锥、悬挂、有限直和和 retract 后得到的紧型对象。

**答案 27.9.** Pretriangulated dg category 是可表模在 $D(\mathcal A)$ 中对有限稳定操作封闭的 dg category；于是 $H^0(\mathcal A)$ 从 dg 模导出范畴继承三角结构。

**答案 27.10.** 三角范畴的 dg enhancement 是 $H^0(\mathcal A)\simeq T$。稳定 $\infty$-范畴的 dg enhancement 是 $N_{\operatorname{dg}}(\mathcal A)\simeq C$。后者保留映射空间或映射谱的高阶信息，比仅给出 $H^0$ 更强。

**答案 27.11.** dg 函子 $F:\mathcal A\to\mathcal B$ 是 Morita equivalence，若限制函子
$$
F^*:D(\mathcal B)\to D(\mathcal A)
$$
是稳定 $\infty$-范畴等价。

**答案 27.12.** Quasi-equivalence 在可表模之间诱导 quasi-isomorphism，并且 $H^0$ 本质满保证 $\mathcal B$ 的可表模由 $F$ 的像生成。导出模范畴由可表模紧生成，因此 $F^*$ 在紧生成子上是等价，进而在整个导出模范畴上是等价。

**答案 27.13.** $\mathcal A$-$\mathcal B$ dg bimodule 是 dg 函子
$$
M:\mathcal A^{op}\otimes\mathcal B\to\operatorname{Ch}(k).
$$
它也可看成从 $\mathcal A$ 到 $\mathcal B$ 的链复形值 profunctor。

**答案 27.14.** 恒等 bimodule 为 $\mathcal A(-,-)$。对右模 $M$，
$$
M\otimes_{\mathcal A}^{\mathbb L}\mathcal A(-,a)\simeq M(a),
$$
这是富 co-Yoneda 公式的导出版；左单位同理。

**答案 27.15.** 等价 $D(\mathcal A)^\omega\simeq\operatorname{Perf}(\mathcal A)$ 说明大导出模范畴的紧对象完全由可表模经有限稳定操作和 retract 生成。因此 Morita 理论可用小的 perfect 子范畴控制整个 $D(\mathcal A)$。

**答案 27.16.** 把普通代数 $A$ 看成单对象 dg category，则 enveloping 代数为 $A^{op}\otimes A$，恒等 bimodule 是 $A$。因此
$$
HH(A)\simeq A\otimes^{\mathbb L}_{A^{op}\otimes A}A.
$$

## 第二十八章

**答案 28.1.** 稳定系数系统是在有有限拉回的基 $\mathcal B$ 上的反变函子
$$
\mathcal D:\mathcal B^{op}\to\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}}).
$$
它把 $X$ 送到稳定闭幺半 presentable $\infty$-范畴 $\mathcal D(X)$，把 $f:X\to Y$ 送到拉回 $f^*:\mathcal D(Y)\to\mathcal D(X)$。

**答案 28.2.** Presentable $\infty$-范畴伴随函子定理说，presentable $\infty$-范畴之间保持小余极限的函子是左伴随。因此保持小余极限的 $f^*$ 有右伴随 $f_*$。

**答案 28.3.** 六个操作是
$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-).
$$

**答案 28.4.** 强对称幺半函子定义中包含结构等价
$$
f^*\mathbb 1_Y\simeq\mathbb 1_X,\qquad
f^*(A\otimes_YB)\simeq f^*A\otimes_Xf^*B,
$$
并要求它们满足单位、结合和对称相干。

**答案 28.5.** 对 Cartesian 方块，先用单位
$$
g^*f_*\to f'_*f'^*g^*f_*,
$$
再用 $f'^*g^*\simeq g'^*f^*$，最后用余单位 $f^*f_*\to\operatorname{id}$，得到
$$
g^*f_*\to f'_*g'^*.
$$

**答案 28.6.** 非常基变换可取为右伴随比较
$$
g'^*f^!\to f'^!g^*
$$
的左伴随转置。即在映射空间等价
$$
\operatorname{Map}(g^*f_!A,B)\simeq\operatorname{Map}(A,f^!g_*B)
$$
与相应方块相干下，得到 $g^*f_!A\to f'_!g'^*A$。

**答案 28.7.** 投影公式是自然等价
$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB.
$$
普通推前版本把 $f_!$ 换为 $f_*$。

**答案 28.8.** $\mathcal D(Y)$-线性正是要求 $f_!$ 与 $\mathcal D(Y)$-作用相容：
$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB.
$$
这就是投影公式。

**答案 28.9.** 对 $f:X\to Y$、$g:Y\to Z$，
$$
(gf)_!(A\otimes f^*g^*C)
\simeq g_!f_!(A\otimes f^*g^*C)
\simeq g_!(f_!A\otimes g^*C)
\simeq g_!f_!A\otimes C.
$$
最后一项等于 $(gf)_!A\otimes C$。

**答案 28.10.** 若 $f$ proper，则 $f_!\simeq f_*$；拉回 $f'$ 仍 proper 时 $f'_!\simeq f'_*$。非常基变换
$$
g^*f_!\simeq f'_!g'^*
$$
因此识别为普通基变换
$$
g^*f_*\simeq f'_*g'^*.
$$

**答案 28.11.** 对 $j:U\hookrightarrow X$ 和闭补 $i:Z\hookrightarrow X$，recollement 给出
$$
j_!j^*K\to K\to i_*i^*K
$$
和
$$
i_*i^!K\to K\to j_*j^*K.
$$

**答案 28.12.** 若 $j^*K=0$ 且 $i^*K=0$，则第一条 recollement 余纤维序列两端为零：
$$
0\to K\to0.
$$
稳定范畴中这推出 $K=0$，所以 $j^*$ 与 $i^*$ 联合保守。

**答案 28.13.** 若 $p_X:X\to *$，dualizing object 为
$$
\omega_X=p_X^!\mathbb 1_*.
$$
Verdier duality functor 为
$$
\mathbb D_X(K)=\underline{\operatorname{Hom}}_X(K,\omega_X).
$$

**答案 28.14.** 若 $K$ dualizable，则内部 Hom 满足
$$
\underline{\operatorname{Hom}}_X(K,M)\simeq K^\vee\otimes_XM.
$$
取 $M=\omega_X$ 得
$$
\mathbb D_X(K)\simeq K^\vee\otimes_X\omega_X.
$$

**答案 28.15.** Equipment 中 Beck-Chevalley 条件比较“沿方块两条路径先拉后推或先推后拉”的结果。六操作中的
$$
g^*f_*\simeq f'_*g'^*,\qquad g^*f_!\simeq f'_!g'^*
$$
就是这一比较在稳定 presentable $\infty$-范畴值 sheaf 理论中的版本。

## 第二十九章

**答案 29.1.** Relative category 是一对 $(\mathcal C,W)$，其中 $\mathcal C$ 是普通范畴，$W$ 是含所有对象和恒等态射的宽子范畴；$W$ 中态射称为 weak equivalences。

**答案 29.2.** $\infty$-categorical localization 是函子 $N\mathcal C\to\mathcal C[W^{-1}]$，把 $W$ 送为等价，并满足：对任意 $\infty$-范畴 $D$，从 $\mathcal C[W^{-1}]$ 到 $D$ 的函子等价于从 $N\mathcal C$ 到 $D$ 且把 $W$ 送为等价的函子。

**答案 29.3.** 若 $L$ 与 $L'$ 都满足泛性质，则由 $L$ 的泛性质得 $L\to L'$，由 $L'$ 的泛性质得 $L'\to L$。两个复合预合成到 $N\mathcal C$ 后等于恒等；由全忠实性，复合等价于恒等，所以 $L\simeq L'$。

**答案 29.4.** $W$ saturated，若 $f\in W$ 当且仅当 $f$ 在 ordinary localization $\mathcal C[W^{-1}]_{\operatorname{ord}}$ 中成为同构。等价地，$W$ 已包含所有被局部化强制为等价的态射。

**答案 29.5.** Simplicial category 是 $\mathbf{sSet}$-富范畴：对象成集合，Hom 为单纯集 $\operatorname{Map}_{\mathcal A}(x,y)$，复合为单纯集映射并满足富范畴公理。

**答案 29.6.** Simplicial functor $F:\mathcal A\to\mathcal B$ 是 Dwyer-Kan equivalence，若每个映射单纯集映射是弱同伦等价，且 $\pi_0F:\pi_0\mathcal A\to\pi_0\mathcal B$ 本质满。

**答案 29.7.** 映射单纯集弱等价给出 $\pi_0$ 上 Hom 集同构，因此 $\pi_0F$ 完全忠实；再加本质满，$\pi_0F$ 是范畴等价。

**答案 29.8.** 模型范畴 $\mathcal M$ 的 underlying $\infty$-category 是相对范畴 $(\mathcal M,W_\mathcal M)$ 的 $\infty$-局部化：
$$
\mathcal M_\infty=\mathcal M[W_\mathcal M^{-1}].
$$

**答案 29.9.** 同伦范畴只记录 $\pi_0$ 级 Hom。$\infty$-范畴等价还必须比较高阶映射空间。Quillen 等价通过 cofibrant-fibrant derived mapping spaces 或 hammock localization 比较这些映射空间。

**答案 29.10.** Coherent nerve
$$
N_{\operatorname{hc}}:\mathbf{sCat}\to\mathbf{sSet}
$$
把 simplicial category 转为 quasi-category；当 Hom 为 Kan 复形时，它保留同伦相干复合信息。

**答案 29.11.** 若 Hom 单纯集离散，则同伦相干 $n$-单纯形没有非平凡高维选择，只是普通可复合箭头串。因此 coherent nerve 与普通 nerve 一致。

**答案 29.12.** Simplicial space $X:\Delta^{op}\to\mathcal S$ 满足 Segal 条件，若对 $n\ge2$，
$$
X_n\simeq X_1\times_{X_0}\cdots\times_{X_0}X_1.
$$
这表示 $n$-单纯形由 $n$ 条可复合 $1$-态射控制。

**答案 29.13.** Complete Segal space 是满足 Segal 条件的 simplicial space，且退化映射 $X_0\to X_{\operatorname{eq}}$ 为等价，其中 $X_{\operatorname{eq}}$ 是在同伦范畴中可逆的 $1$-态射空间。

**答案 29.14.** Rezk nerve 把 relative category 送到 complete Segal space 模型，使只给定 weak equivalences 的数据能表示完整的同伦理论。

**答案 29.15.** 若构造由 localization 泛性质刻画，则任何模型中得到的对象都满足同一泛性质。由 localization 的唯一性，不同模型给出等价结果。

## 附录 A

**答案 A.1.** 若 $\mathbf{Set}_{\mathcal U}$ 是 $\mathcal U$-小，则其对象集合属于 $\mathcal U$，从而包含 $\mathcal U$ 中所有集合；这与 universe 的大小闭包和 Cantor 型论证冲突。

**答案 A.2.** 函子由对象函数和态射函数组成。小范畴的对象集与态射集都是集合，因此所有可能函数的集合也是集合，满足函子性等式的子集仍是集合。

**答案 A.3.** 选择原则用于对每个 $D\in\mathcal D$ 选择 $G(D)\in\mathcal C$ 和同构 $F(GD)\cong D$。

## 附录 B

**答案 B.1.** 例 $\alpha(0)=0,\alpha(1)=2,\alpha(2)=2$。像为 $\{0,2\}$，先满射 $[2]\to[1]$ 给 $0\mapsto0,1,2\mapsto1$，再单射 $[1]\to[4]$ 给 $0\mapsto0,1\mapsto2$。

**答案 B.2.** 对 $k\in[n-2]$，两边都把 $k$ 送到 $[n]$ 中漏掉 $i$ 和 $j$ 后的第 $k$ 个元素；逐点相等。

**答案 B.3.** $\Lambda_1^3$ 是 $\Delta^3$ 的四个二维面中去掉第 1 个面的并。第 $i$ 个面为漏掉顶点 $i$ 的面，所以缺失面是 $(0,2,3)$；保留的三个面是 $(1,2,3)$、$(0,1,3)$、$(0,1,2)$。

## 附录 C

**答案 C.1.** 若 $P,Q$ 都是 $A,B$ 的积，由 $P$ 泛性质得 $Q\to P$，由 $Q$ 泛性质得 $P\to Q$；复合保持投影，故由唯一性等于恒等。

**答案 C.2.** 自由群 $F(S)$ 表示函子 $G\mapsto\mathbf{Set}(S,U G)$。表示映射把群同态 $F(S)\to G$ 限制到生成元集合 $S$。

**答案 C.3.** 双射由 currying 给出：
$$
\mathbf{Set}(X\times A,Y)\cong\mathbf{Set}(X,Y^A).
$$
单位为 $x\mapsto(a\mapsto(x,a))$ 的相应形式，余单位为评价映射 $Y^A\times A\to Y$。

## 附录 E

**答案 E.1.** $\Delta^0$ 各有一个顶点。join 把第一个顶点放在第二个顶点之前，并添加一条从前者到后者的边，因此得到有两个有序顶点的 $\Delta^1$。

**答案 E.2.** $C_{/x}$ 的对象是映射 $\Delta^0\star\Delta^0\cong\Delta^1\to C$，其在右端顶点为 $x$；因此对象是 $C$ 中所有指向 $x$ 的边 $y\to x$。

**答案 E.3.** Kan complex 对所有 horn $\Lambda_i^n\to\Delta^n$ 有填充；quasi-category 只要求内 horn 填充。因此 Kan complex 自动满足 quasi-category 条件。

**答案 E.4.** $C^\natural$ 标记所有等价边；$C^\sharp$ 标记全部 $1$-单纯形；$C^\flat$ 只标记退化边。三者满足 $C^\flat\subseteq C^\natural\subseteq C^\sharp$。

**答案 E.5.** 映射空间定义说某条 lift 对所有测试对象诱导同伦拉回，即具有同伦泛性质；horn lifting 定义把同一泛性质展开为对所有有限单纯形边界数据的填充条件。模型范畴理论证明二者等价。

**答案 E.6.** marked simplicial set 标记边，即 $1$-单纯形；scaled simplicial set 标记 $2$-单纯形。前者适合记录等价或 Cartesian 边，后者适合记录 $2$-态射层面的可逆性和相干关系。

**答案 E.7.** 它应记录三个对象、两条可复合 $1$-态射、第三条比较用的 $1$-态射，以及一个从某个复合到该比较态射的 $2$-态射；thin 标记说明该 $2$-态射被视为相干等式或可逆比较。

**答案 E.8.** 普通有向图只能记录 $f$ 和 $g$ 两条箭头。伴随还需要单位 $\eta:\operatorname{id}\Rightarrow gf$、余单位 $\varepsilon:fg\Rightarrow\operatorname{id}$ 以及三角恒等式，这些都是 $2$-态射及其关系，因此必须使用 $2$-维数据。
