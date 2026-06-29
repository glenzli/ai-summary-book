# 凝聚数学讲义练习答案与提示

作者：Dr. Stochastic Parrot

## 使用说明

本文件给出四卷显式练习的答案要点。短题给出直接答案；证明题给出关键步骤；查阅题给出应当核对的结论。若题目依赖 Gleason、Nöbeling、Cartan、Clausen-Scholze 或 Scholze 的深层定理，答案会明确标为“输入定理”。

这不是完整教师手册。完整教师手册应把每个“证明提示”扩成逐行证明。

## 卷一答案要点

### 第 0 章

- 0.1：Scholze 讲义前部属于凝聚集合、凝聚阿贝尔群和测试站点基础；solid、analytic、liquid、复几何和对偶属于后续结构与应用。
- 0.2：若 $g:S'\to S$ 连续，则 $f:S\to T$ 被送到 $f\circ g:S'\to T$，故 $S\mapsto\operatorname{Cont}(S,T)$ 反变。
- 0.3：典型现象是拓扑阿贝尔群范畴中 cokernel 后的商拓扑、kernel 后的子空间拓扑与严格正合不总相容；这使它不像良好阿贝尔范畴。

### 第 1 章

- 1.1：sheaf 的唯一粘合性推出 separated：若两个截面在覆盖上限制相同，则它们同为同一个匹配族的粘合。
- 1.2：三元覆盖的等化子为 $F(U)\to\prod_iF(U_i)\rightrightarrows\prod_{i,j}F(U_i\times_UU_j)$；三重交进入 Čech 微分下一阶。
- 1.3：开集 $U_i,U_j\subset U$ 的纤维积在开集偏序范畴中就是交集 $U_i\cap U_j$。
- 1.4：拓扑空间上预层 $U\mapsto$ 局部常值函数但只允许全局常值粘合，可满足唯一性而缺少存在性；也可用非 sheaf 化预层的标准例子。

### 第 2 章

- 2.1：有限不交并的开覆盖逐块验证紧性；Hausdorff 性由不同分量开闭分离、同分量内原 Hausdorff 分离给出。
- 2.2：若 $X\to Z\leftarrow Y$，纤维积是 $X\times Y$ 中满足 $f(x)=g(y)$ 的等化子；Hausdorff 使对角线闭，故它闭。
- 2.3：相容条件保证若 $q(x)=q(y)$，则 $x,y$ 在某个交对象上对应同一点，两个局部映射取值相等。
- 2.4：无限离散集合不是紧空间，因此不是 compact Hausdorff，也不是 profinite；profinite 必须紧 Hausdorff 全不连通。
- 2.5：对 Stone 空间，布尔代数元素对应 clopen 子集；这些 clopen 集构成基。

### 第 3 章

- 3.1：商拓扑论证是：若 $q:\coprod S_i\to S$ 是商映射且 $\varphi q$ 连续，则 $\varphi$ 连续。
- 3.2：离散 $A$ 上，$\operatorname{Cont}(S_1\sqcup S_2,A)\cong\operatorname{Cont}(S_1,A)\times\operatorname{Cont}(S_2,A)$。
- 3.3：由 Yoneda，$\operatorname{Hom}(\underline K,\underline L)\cong\underline L(K)=\operatorname{Hom}_{\mathbf{CHaus}}(K,L)$。
- 3.4：同一集合取离散拓扑和非离散拓扑时，从紧空间进入它们的连续映射集合不同，因此凝聚化可不同。

### 第 4 章

- 4.1：群对象给每个 $S$ 一个阿贝尔群，态射反向诱导群同态，函子性来自预层函子性。
- 4.2：逐点加法连续，因为 $G\times G\to G$ 连续；限制映射保持逐点加法。
- 4.3：kernel 是 fiber product $A\times_B0$，sheaf 范畴有限极限逐点计算，故仍为 sheaf。
- 4.4：开集 sheaf 上的满射通常是 stalkwise 满，不必对每个开集截面满；例如指数序列中的指数映射在有非零绕数开集上不逐截面满。

### 第 5 章

- 5.1：$P\times_KQ$ 是 $(P\times Q)\to K\times K$ 拉回对角线；$K$ Hausdorff 使对角线闭。
- 5.2：由站点比较，sheaf 在基子站点上的限制决定原 sheaf；可表对象也由 profinite 测试值恢复。
- 5.3：证明分三步：限制全忠实、由基覆盖扩张 sheaf、本质满；关键是共同细化。
- 5.4：还需拉回覆盖能被子范畴对象覆盖，否则交叠相容条件不能在子站点上检测。

### 第 6 章

- 6.1：离散空间中任意开集闭包等于自身，开集闭包开，故极不连通。
- 6.2：截面选择每点所在覆盖块；连续性迫使选择在 clopen 块上局部常值，得到有限 clopen 分解。
- 6.3：自然性计算是检查提升后沿态射取值与先取值再映射一致。
- 6.4：Gleason 定理说明 compact Hausdorff 范畴中的投射对象正由极不连通紧 Hausdorff 空间控制。

### 第 7 章

- 7.1：双射为 $\operatorname{Hom}(\mathbb Z[\underline S],A)\cong A(S)$；一个方向由生成元给截面，反方向由截面给自由对象态射。
- 7.2：满射 $B\to C$ 在 ED 空间 $E$ 上取值满，故任意 $\mathbb Z[\underline E]\to C$ 可提升到 $B$。
- 7.3：sheaf 满射意为局部可提升；在覆盖上构造提升，再用 sheaf 条件检查相容。
- 7.4：$\operatorname{Hom}(\bigoplus_iP_i,-)\cong\prod_i\operatorname{Hom}(P_i,-)$；乘积的满射仍满，故直和投射。

### 第 8 章

- 8.1：若截面在覆盖上为零，则零截面和该截面限制相同，由 separated 性唯一性推出相等。
- 8.2：ED 上取值保持满射，故 $\operatorname{Hom}(\mathbb Z[\underline E],-)\cong(-)(E)$ 保持满射。
- 8.3：直和投射见 7.4，因此由 ED 自由对象组成的直和投射。
- 8.4：$\mathbf{CondAb}$ 的 Ext 由凝聚范畴投射分解决定，不能与普通阿贝尔群 Ext 混同。

### 第 9 章

- 9.1：自然双射来自张量积的双线性泛性质；与复合相容由泛性质唯一性给出。
- 9.2：有限离散 $S$ 上凝聚化取值退化为有限份普通对象，因此公式为普通自由模公式。
- 9.3：结合律、单位律是乘法 $R\otimes R\to R$ 与单位 $\mathbb Z\to R$ 满足的交换图。
- 9.4：逐点张量未必满足 sheaf 粘合，需先在预层中构造再 sheaf 化。

### 第 10 章

- 10.1：模作用 $R\otimes M\to M$ 与乘法和单位组成结合律、单位律交换图。
- 10.2：有限离散 $S$ 上 $\underline M(S)\cong M^S$，$R$-作用逐坐标计算。
- 10.3：命题同构由泛性质给出；对 $M\to N$ 的自然性由交换图检查。
- 10.4：对目标截面局部选取生成元表达，再用 sheaf 满射定义粘合。
- 10.5：$M\otimes_RN$ 表示 $R$-平衡双线性映射 $M\times N\to P$。

### 第 11 章

- 11.1：取投射分解 $P_\bullet\to M$，$H_0(P_\bullet\otimes_RN)$ 由右正合性等于 $M\otimes_RN$。
- 11.2：$R[\underline E]$ 的 Hom 为取值 $M(E)$，ED 保持满射，故投射；高阶 Tor 消失。
- 11.3：长正合列前六项为 $\operatorname{Tor}_1(M,N')\to\operatorname{Tor}_1(M,N)\to\operatorname{Tor}_1(M,N'')\to M\otimes N'\to M\otimes N\to M\otimes N''$。
- 11.4：horseshoe lemma 只用逐阶投射提升、核和拉回推出相容分解。

### 第 12 章

- 12.1：有限 $S$ 上 solid 自由对象无完备化差异，等于普通自由凝聚阿贝尔群。
- 12.2：连续 $S\to\mathbb Z$ 到离散空间，紧性给有限像，每个纤维 clopen；有限 clopen 分解来自某个有限商。
- 12.3：Nöbeling 定理给 $C(S,\mathbb Z)$ 自由，代入固体对象 Hom 判别推出相应推论。
- 12.4：固体性给测度对象上的线性泛性质，$f$ 与 $\mu$ 配对得到积分。

### 第 13 章

- 13.1：普通张量积不自动满足 solid Hom 判别，需 solidification。
- 13.2：有限 $S,T$ 时 solid 自由对象等于普通自由对象，故张量公式退化为 $\mathbb Z[S]\otimes\mathbb Z[T]\cong\mathbb Z[S\times T]$。
- 13.3：固体环是在 solid 阿贝尔群对称幺半范畴中的交换代数对象，图同普通交换代数对象。

### 第 14 章

- 14.1：有限不交并对应函数/测度的乘积，因为在每个分量独立给数据。
- 14.2：solid 判别使用 $\mathbb Z^\square[S]$；analytic 模判别使用 $\mathcal M[S]$ 和 Dirac cone。
- 14.3：若 $S=\varprojlim S_i$，$A^\square[S]=\varprojlim A[S_i]$ 可理解为有限商上 $A$-值测度的相容族。
- 14.4：Radon 测度满足的拓扑分析条件不同，不能自动满足 analytic ring 公理。

### 第 15 章

- 15.1：valuation 满足 $|0|=0,|1|=1$、乘法性、非阿基米德三角不等式；$|A^+|\le1$ 表示有界元素。
- 15.2：非 proper 映射允许逃向无穷远，$A=\mathbb Z[T]$ 的无穷远边界记录这种行为。
- 15.3：应查 Scholze 第八讲中 $A_\infty$ 对无穷远贡献的定义。
- 15.4：proper 情形中 $f_!=f_*$；非 proper 时 $f_!$ 带紧支/边界修正。

### 附录 A-G

- A.1：所有小集合的集合通常超过同一 universe，形成大范畴。
- A.2：第七章直和指标集必须落在选定 universe 内，否则对象不小。
- A.3：$\kappa$-condensed sets 固定测试对象大小；本书用固定 universe 简化。
- B.1：共同细化上的匹配族限制相同，由 sheaf 唯一性得到覆盖选择无关。
- B.2：开集基决定 sheaf 是基子站点比较的经典例子。
- B.3：profinite 覆盖拉回后仍可由 profinite 空间细化，关键是共同细化。
- C.1：极限逐点计算，匹配条件与极限交换，故 sheaf 条件对小极限稳定。
- C.2：同 4.4。
- C.3：由 ED 上取值满推出定理 6.11 的满射检测。
- D.1：regular open 集的补、交、并在正则化后满足布尔代数公理。
- D.2：$\operatorname{Stone}(\mathcal P(S))$ 的 ultrafilter 都是主 ultrafilter，故与 $S$ 同胚。
- D.3：Cantor 集 profinite；但存在开集闭包非开，故非 ED。
- D.4：retract 中开集闭包可由母空间的 ED 性和 retract 映射推出仍开。
- E.1：预层张量由自由阿贝尔群商掉平衡关系，满足双线性泛性质。
- E.2：sheaf 化是左伴随，保持表示的泛性质到 sheaf 范畴。
- E.3：由相对张量积的单位律或泛性质得 $R\otimes_RM\simeq M$。
- E.4：平坦定义即 $F\otimes-$ 保持短正合列。
- F.1：有限满射可按纤维选择代表，得到拉回映射的分裂。
- F.2：把 $\mathbb Z_p$ 写成有限商逆极限，用命题 F.2 得自由性。
- F.3：前三级商为 $\{0,1\}^n$，分解对应逐级新增 locally constant 函数。
- F.4：同构依赖自由基选择；不同基给不同同构。
- G.1：$\operatorname{Ext}^0$ 是 Hom 复形第零同调，即 $\operatorname{Hom}$。
- G.2：同 7.2。
- G.3：两项分解给 $\operatorname{Ext}^1(M,A)=\operatorname{coker}(\operatorname{Hom}(P_0,A)\to\operatorname{Hom}(P_1,A))$。
- G.4：长度为 $n$ 的投射分解使 Hom 复形在 $n$ 阶后为零，故高阶 Ext 消失。
- H.1：若 $F$ 是 sheaf，则覆盖上的匹配族唯一粘合到 $F(U)$；因此 $F^+(U)$ 中每个元素来自唯一的 $F(U)$。
- H.2：对表示 $t$ 的覆盖 $\{U_i\}$ 和使 $\beta(t_i)$ 为零的覆盖 $\{V_{ij}\to U_i\}$，取复合覆盖 $\{V_{ij}\to U\}$；匹配条件由原匹配条件限制得到。
- H.3：生成元只需一组覆盖所有对象的代表；若不取代表，可取所有对象的直和，仍是集合，因为站点小。
- H.4：$(P[n])\otimes A\simeq(P\otimes A)[n]$；若 $A$ acyclic，则右侧 acyclic，故 $P[n]$ K-flat。
- I.1：次数 0 先提升 $P_0\to A\to B$ 到 $Q_0$；次数 1 使用 $Q_1\to\ker(Q_0\to B)$ 的满射和 $P_1$ 投射性。同伦构造同理。
- I.2：把 $0\to A'\to A\to A''\to0$ 与 $P'_0\oplus P''_0\to A$ 的 kernel 图组成短正合列，snake lemma 给出 $0\to K'\to K\to K''\to0$。
- I.3：若 $0\to K\to P_0\to A\to0$ 且 $0\to K_1\to P_1\to K\to0$，则 $\operatorname{Ext}^2(A,B)\cong\operatorname{Ext}^1(K,B)\cong\operatorname{coker}(\operatorname{Hom}(P_1,B)\to\operatorname{Hom}(K_1,B))$。
- I.4：张量积只右正合，不一定保持单射；所以短正合复形逐项张量后左端可能不正合，必须用导出张量或 K-flat 分解。
- J.1：$\neg V=\operatorname{int}(X\setminus V)$，故 $\overline{\neg V}\subset X\setminus V$；反向由 $X\setminus V$ 是 regular closed，即等于 $\overline{\operatorname{int}(X\setminus V)}$。
- J.2：紧 Hausdorff 空间正规。取 $x\in O$，先选开集 $O_1$ 使 $x\in O_1\subset\overline{O_1}\subset O$，再选 $W$ 使 $x\in W\subset\overline W\subset O_1$，则 $\overline{\operatorname{int}\overline W}\subset\overline{O_1}\subset O$。
- J.3：有限离散 $X$ 中所有子集都是 regular open，$\operatorname{RO}(X)=\mathcal P(X)$，$E_X=\operatorname{Stone}(\mathcal P(X))\cong X$，$p$ 为恒等同胚。
- J.4：本附录只构造 $E_X\to X$ 的连续满射；投射性要求对任意满射 $Y\to Z$ 和任意 $E\to Z$ 构造提升 $E\to Y$，这是 Gleason 定理的另一个方向。
- K.1：separated 性正是说覆盖上限制相同的两个截面相等；单个覆盖 $p:E\to X$ 是覆盖族的一种。
- K.2：有限不交并的紧性来自有限并；Hausdorff 性由不同分量开闭分离、同分量内原 Hausdorff 分离给出。
- K.3：任取 $a\in A(X)$，在 ED 覆盖 $E\to X$ 上限制为零，由 K.4 得 $a=0$；因此 $A=0$，任意 $A\to B$ 只能是零态射。
- K.4：短正合列取 ED 值后短正合由推论 K.13；因此 $(-)(E)$ 保持 kernel、cokernel 和短正合列，是正合函子。
- L.1：若 $g:S^1\to S^1$ 有 winding number $m\ne0$ 且有连续 argument $f:S^1\to\mathbb R$，则沿一圈 lift 的端点差应为 $m$；但同一基点的函数值必须相同，矛盾。
- L.2：例如令 $F(W)$ 为所有局部常值整数函数再商掉全局符号，使不同全局截面可有相同局部限制；也可取非 separated presheaf 的标准构造：给每个非空开集两份同一 sheaf 的截面，限制遗忘份标记。
- L.3：拓扑空间的开集基若对有限交封闭，则覆盖交叠仍在基中；sheaf 条件的等化子目标 $F(U_i\cap U_j)$ 可在基上表达。
- L.4：对 $\mathbb Z[1/p]$，像中坐标分母只含有限个 $p$ 的幂且有统一上界；序列 $(1,p^{-1},p^{-2},\ldots)$ 不在像中，因此不满。

### 附录 M

- M.1：$\mathbb Z[\underline{\{0,1\}}]\cong\mathbb Z[\underline *]\oplus\mathbb Z[\underline *]$，故 Hom 为 $A(*)\times A(*)$。
- M.2：由推论 M.5，$\operatorname{Ext}^1(\mathbb Z_{\operatorname{cond}}/2,A)\cong A(*)/2A(*)$。
- M.3：若 $N$ 没有 $n$-torsion，则 $\ker(n:N\to N)=0$，故 $\operatorname{Tor}_1(\mathbb Z_{\operatorname{cond}}/n,N)=0$。
- M.4：solid tensor product 是普通派生张量后再 solidification；局部化可能改变普通 Tor 复形的同调对象。

### 附录 N

- N.1：有限 Boolean 代数同构于有限集合的幂集代数；其超滤子均为主超滤子，因此 Stone 空间为有限离散集合。
- N.2：$\operatorname{Stone}(\mathcal P(S))$ 是 $S$ 上所有超滤子空间；当 $S$ 无限时含非主超滤子，故严格大于离散 $S$。
- N.3：若 $f$ 单射，任意非空基本开 $U_b$ 的原像 $U_{f(b)}$ 非空，故像稠密；反向若 $f(b)=f(c)$ 且 $b\ne c$，由 Stone 分离得到像避开某个非空开集，矛盾。
- N.4：连续映射 $X\to Y$ 诱导每个有限开闭划分商之间的映射；命题 N.13 说明点由所有有限商相容数据唯一恢复。

## 卷二答案要点

### 第 0 章

- 0.1：solid 判别式是 $R\operatorname{Hom}(K_S,C)=0$，其中 $K_S$ 是 Dirac/free 到 solid 测度对象的 cone。
- 0.2：$\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}$ 依赖 Nöbeling 基选择，不是典范同构。
- 0.3：普通张量在 $\mathbf{CondAb}$；solid 张量先普通张量再 solidification；派生 solid 张量还需投射/平坦分解。

### 第 1 章

- 1.1：命题 1.4 的等价来自 Bousfield 局部对象定义：$C$ local 当且仅当对所有被倒置对象 $K_S$ 有 $R\operatorname{Hom}(K_S,C)=0$。
- 1.2：有限 $S$ 时 Dirac 映射已经同构，故 cone $K_S\simeq0$。
- 1.3：stable category 中局部对象对 shift 封闭，故 $C[n]$ solid。
- 1.4：$M\to L^\square M$ 对任意 solid $C$ 诱导 $\operatorname{Hom}(L^\square M,C)\cong\operatorname{Hom}(M,C)$。

### 第 2 章

- 2.1：有限离散情形归结为 $\mathbb Z[S]\otimes\mathbb Z[T]\cong\mathbb Z[S\times T]$。
- 2.2：ordinary solid 环是 solid 阿贝尔群对称幺半范畴中的代数对象，结合律图与普通环相同。
- 2.3：普通张量不保持无限乘积；统一分母反例见卷四附录 B。
- 2.4：$p$-进完备化相对于 $p$-进拓扑；solidification 相对于 solid 测试对象的反射泛性质。

### 第 3 章

- 3.1：定义 3.3 与 $R\operatorname{Hom}(K_S^\mathcal M,C)=0$ 等价，因为 $K_S^\mathcal M$ 是 analytic Dirac 映射的 cone。
- 3.2：solid 例子中有限 $S$ 上 $\mathcal M[S]$ 等于普通有限自由对象。
- 3.3：Dirac 映射比较自由生成元与测度对象，analytic condition 要求这种比较对模块无障碍。
- 3.4：solid 只针对 $\mathbb Z^\square$ 测度；analytic 允许一般 $(A,\mathcal M)$。

### 第 4 章

- 4.1：命题 4.5 由 localization 泛性质：先局部化再映射到 local 对象等于直接映射。
- 4.2：solid 例子中 $K_S^\mathcal M=K_S$。
- 4.3：普通张量后可能不 analytic，需 analyticization 才落回解析模范畴。
- 4.4：Bousfield localization 是稳定/派生范畴中的局部化；反射子范畴是 1-范畴层面的左伴随包含。

### 第 5 章

- 5.1：Banach completion 补 Cauchy 列；solidification 是范畴反射，修正 Hom/张量行为。
- 5.2：$p$-liquid 判别用 $\operatorname{Hom}(\mathcal M_{<p}[S],-)$ 或相应导出 Hom 对 profinite $S$ 测试。
- 5.3：不同 $p$ 或 $<p$ 给不同增长条件和测试对象，命题不写清就无类型。
- 5.4：Banach 空间范畴中 cokernel、tensor、projective resolution 对同调代数不够稳定。

### 第 6 章

- 6.1：valuation 乘法性 $|ab|=|a||b|$，三角不等式 $|a+b|\le\max(|a|,|b|)$。
- 6.2：$|g(x)|\le|f(x)|\ne0$ 表示在该 rational domain 上 $g/f$ 有界。
- 6.3：有限素数点对应 $p$-进方向；无穷远方向对应阿基米德/边界行为的直观成分。
- 6.4：rational localization 是构造结构层和局部几何的基本覆盖。

### 第 7 章

- 7.1：proper 时 $f_*$ 保持紧支；非 proper 时 $f_!$ 需要排除无穷远逃逸。
- 7.2：两边均在对应 solid/analytic 派生范畴，$f^*M$ 是派生 analytic tensor。
- 7.3：$\mathbb Z[T]$ 对应仿射线，非 proper 推前有无穷远贡献。
- 7.4：投影公式涉及导出张量和导出推前，普通范畴陈述会丢失 Tor 信息。

### 第 8 章

- 8.1：Serre duality：$H^i(X,\mathcal F)^\vee\cong\operatorname{Ext}^{n-i}(\mathcal F,\omega_X)$；$\omega_X$ 对应 $f^!$ 的相对 dualizing 对象。
- 8.2：有限维上同调意味着导出全局截面是有限/紧对象。
- 8.3：GAGA 比较 algebraic coherent sheaves 与 analytic coherent sheaves，并比较上同调。
- 8.4：Riemann-Roch 需要 Chern character、Todd class、trace 与 pushforward 相容，超出本章目标。

### 附录 B

- B.1：$[s]\otimes[t]\mapsto[(s,t)]$ 给基双射。
- B.2：普通张量积不保无限乘积，不能逐坐标化简。
- B.3：检查对象范畴、张量类型、localization 和目标范畴。
- B.4：投影公式两边都在目标的 analytic/solid 派生范畴。

### 附录 E

- E.1：若 $e$ 是等价的 retract，存在 $r,s$ 使目标态射为 $r\circ e\circ s$ 且相应幂等关系成立；映射空间中等价的 retract 仍是等价。
- E.2：稳定左伴随保持有限余极限，cofiber 是有限余极限，因此 $L$ 把 cofiber sequence 送到 cofiber sequence。
- E.3：把 $A$ 看作 solid 交换环，应用附录 E 命题 E.9 于幺半局部化 $L^\square$，得到 $L^\square(M\otimes_A^LN)$。
- E.4：可在任一对称幺半稳定范畴中取一个反射局部化 $L$，若存在 $N\in\ker L$ 与 $X$ 使 $N\otimes X\notin\ker L$，则该反射不能是幺半局部化；这正是附录 E 命题 E.12 的形式反例。

### 附录 F

- F.1：单位为 $X\to f^!f_!X$，对应 $\operatorname{id}_{f_!X}$；余单位为 $f_!f^!Y\to Y$，对应 $\operatorname{id}_{f^!Y}$。
- F.2：由生成集生成的最小 localizing subcategory 是全范畴；若某全子范畴含生成集且对所有小余极限和等价封闭，则包含该最小 localizing subcategory。
- F.3：第一行用 $f_!\dashv f^!$；第二行用 $\otimes\dashv\mathcal Hom$；第三行用投影公式；第四行再用 $f_!\dashv f^!$；第五行用 $\otimes\dashv\mathcal Hom$。
- F.4：proper 情形 $f_!=Rf_*$，定理 F.9 给出 $f^!R\mathcal Hom(M,Y)\simeq R\mathcal Hom(Lf^*M,f^!Y)$，这是 Grothendieck duality 的内部 Hom 相容公式。

### 附录 G

- G.1：$U^0_X=U_1\coprod U_2$；$U^1_X=(U_1\times_XU_1)\coprod(U_1\times_XU_2)\coprod(U_2\times_XU_1)\coprod(U_2\times_XU_2)$。
- G.2：totalization 的 $0$ 阶数据是覆盖上的截面族；$1$ 阶相容给交叠上相等；更高阶相容由这些限制等式沿三重及更高纤维积拉回得到。
- G.3：稳定范畴中 fiber 和 cofiber 只差 shift。若 cofiber 为零，则 $X\to Y\to0$ 是 cofiber sequence，等价地 fiber 为零，故 $X\to Y$ 是等价。
- G.4：投影公式和 Grothendieck duality 常需要控制 $\operatorname{Map}(F(X),T)$ 方向；这个方向不能由 ordinary descent 自动推出，需额外的伴随、紧性或 dualizability 假设。

### 附录 H

- H.1：$\operatorname{Hom}(K_1\oplus K_2,\bigoplus_iX_i)\cong\operatorname{Hom}(K_1,\bigoplus_iX_i)\oplus\operatorname{Hom}(K_2,\bigoplus_iX_i)$，再用 $K_1,K_2$ 紧性把它化为 $\bigoplus_i\operatorname{Hom}(K_1\oplus K_2,X_i)$。
- H.2：若 $X\to Y\to Z$ 是 cofiber sequence，$F,G$ 精确使 $F(X)\to F(Y)\to F(Z)$ 和 $G(X)\to G(Y)\to G(Z)$ 都是 cofiber sequence；若其中两项自然变换为等价，第三项由三角范畴二出三得到。
- H.3：把 $\eta_{M,N}$ 视为双变量自然变换；投影公式两边关于每个变量保持小余积和 cofiber，故只需在两组紧生成元上检查。
- H.4：若目标范畴还有不在生成元本质像生成的直和、cone 或 retract 中的对象，则生成元上本质满不能覆盖这些对象；命题 H.10 需要本质像生成整个目标范畴。

### 附录 I

- I.1：由有限不交并相容，$\mathcal M[\{0,1\}]\simeq\mathcal M[\{0\}]\times\mathcal M[\{1\}]$。
- I.2：若 $K_S^\mathcal M\simeq0$，则 $R\operatorname{Hom}(K_S^\mathcal M,C)=0$ 对所有 $C$ 成立，因此该 $S$ 的局部条件自动满足。
- I.3：解析张量积需要检查表 I.6 的张量理想性；没有它不能保证普通张量后仍在解析对象中。
- I.4：Bousfield localization 只给局部对象和反射；rational Cech descent 涉及几何覆盖及交叠上的范畴 totalization，需要额外几何输入。

### 附录 J

- J.1：紧 Hausdorff 覆盖上的连续映射可由匹配族唯一粘合；向量空间运算逐点连续。
- J.2：算子 $T$ 是 compact，因为它是有限秩截断算子的范数极限；截断后的尾部算子范数为 $1/(N+1)$。
- J.3：闭像涉及拓扑闭包；纯代数向量空间只记录子空间，不记录闭包。
- J.4：需说明 analytic ring、凝聚化对象、微分连续性、Hom/tensor 所在范畴，以及忘记结构后的经典 Fréchet 空间。

### 附录 K

- K.1：若 $X$ 局部，则 $\operatorname{Map}(A,\mathcal Hom(Y,X))\simeq\operatorname{Map}(A\otimes Y,X)$；用 $s\otimes Y$ 为局部等价得到局部性。
- K.2：有理化的核为 torsion 复形；torsion 复形与任意复形张量后仍在有理化后为零，故核为张量理想。
- K.3：取一个反射局部化，其核含 $N$ 但不含 $N\otimes X$；则 $0\to N$ 是局部等价，而 $0\to N\otimes X$ 不是。
- K.4：solid 条件由所有 profinite 测试对象控制；无限 profinite 对象携带极限和测度信息，不能由有限离散对象的逐点检查推出。

### 附录 L

- L.1：局部对象由生成态射的映射空间判别定义；若对生成态射成立，则对其生成的局部等价由 cofiber、shift 和余极限封闭性推出。
- L.2：在有理化局部化中，$\mathcal Hom_{\mathbb Q}(M\otimes\mathbb Q,N\otimes\mathbb Q)$ 可写为 $R\operatorname{Hom}_{\mathbb Z}(M,N\otimes\mathbb Q)$ 的 $\mathbb Q$-对象；若 $M$ perfect，则进一步等于 $R\operatorname{Hom}_{\mathbb Z}(M,N)\otimes\mathbb Q$。
- L.3：无限直和 $\bigoplus_i\mathbb Z$ 不是 compact dualizable；$\operatorname{Hom}(\bigoplus_i\mathbb Z,-)\cong\prod_i(-)$ 不保持任意直和。
- L.4：perfect complex 局部为有限秩向量丛的有限复形，因此有对偶复形，evaluation/coevaluation 满足 dualizable 条件。

## 卷三答案要点

### 第 0 章

- 0.1：需要 analytic 派生范畴、liquid 函数空间、$f_!$、$f^!$/trace 与相干对象紧性。
- 0.2：Serre duality 给紧复流形上相干层上同调与对偶izing 层 Ext 的完美配对。
- 0.3：复几何函数空间带 Fréchet、nuclear 或分布拓扑；忽略拓扑会破坏连续算子和导出结构。

### 第 1 章

- 1.1：复解析空间局部同构于 $\mathbb C^n$ 开集的解析子集，由全纯函数零点定义，并带结构层。
- 1.2：$\mathcal O(U)$ 带 compact-open 或一致收敛于紧集的 Fréchet 拓扑，限制映射连续。
- 1.3：普通凝聚化只记录连续族；liquid 结构还记录测度测试对象和同调良性。

### 第 2 章

- 2.1：相干层局部有有限表示 $\mathcal O_U^m\to\mathcal O_U^n\to\mathcal F|_U\to0$。
- 2.2：$\operatorname{Coh}(X)$ 是 abelian/精确层面；$D_{\operatorname{coh}}(X)$ 是同调集中且同调层相干的导出范畴。
- 2.3：紧对象与有限生成/有限维上同调相关；有限维全局截面是紧性在向量空间层面的影子。

### 第 3 章

- 3.1：Dolbeault 微分为 $\bar\partial:\mathcal A^{p,q}\to\mathcal A^{p,q+1}$，只对反全纯方向求导。
- 3.2：局部坐标下 $\bar\partial^2$ 的二阶偏导反对称相消。
- 3.3：Fréchet 拓扑保证 $\bar\partial$ 连续、闭图/Fredholm 等性质可谈；普通向量空间会丢失分析信息。

### 第 4 章

- 4.1：有界复形 $C^i$ 各项有限维，则 $\ker d^i$ 与 $\operatorname{im}d^{i-1}$ 有限维，商有限维。
- 4.2：非紧空间上全局全纯函数可无限维，例如 $\mathbb C$ 上多项式/全纯函数空间。
- 4.3：Dolbeault resolution $\mathcal O_X\to\mathcal A^{0,\bullet}$ 给 $H^i(X,\mathcal O_X)\cong H^i(\Gamma(X,\mathcal A^{0,\bullet}),\bar\partial)$。

### 第 5 章

- 5.1：Riemann surface 上 $H^i(X,L)^\vee\cong H^{1-i}(X,K_X\otimes L^{-1})$。
- 5.2：$\omega_X$ 是 holomorphic top forms，作为 dualizing sheaf 进入配对。
- 5.3：$f_!\dashv f^!$ 的 counit 是 $f_!f^!N\to N$；对 $N$ 为单位对象给 trace。

### 第 6 章

- 6.1：经典 GAGA：代数相干层与解析相干层范畴等价；相干上同调自然同构。
- 6.2：properness 控制无穷远，保证解析对象代数化和上同调有限。
- 6.3：导出版本一次比较 $R\Gamma$ 和派生范畴对象，自动包含所有上同调与长正合相容。

### 第 7 章

- 7.1：$\chi(\mathbb P^1,\mathcal O)=h^0-h^1=1-0=1$。
- 7.2：$\operatorname{ch}(E)=\operatorname{rk}(E)+c_1(E)+\frac12(c_1^2-2c_2)+\cdots$。
- 7.3：Euler characteristic 是 trace of identity 的范畴化影子，trace map 把局部/上同调数据推到基域。

### 第 8 章

- 8.1：典型伴随：$f^*\dashv f_*$、$f_!\dashv f^!$、$\otimes\dashv R\mathcal Hom$。
- 8.2：proper 情形紧支与普通推前一致，所以 $f_!\simeq f_*$。
- 8.3：示例：若选 GAGA，需补 analyticification、相干层比较、上同调比较。

### 附录 C

- C.1：Čech 微分 $d=\sum(-1)^ir_i$，$d^2$ 中每对删除指标以相反符号出现而相消。
- C.2：Cartan B 给 Stein 开集上相干层高阶上同调消失，故 Stein 覆盖 acyclic。
- C.3：若性质对短正合列 two-out-of-three，且对自由层成立，则用有限表示和归纳推广到相干层。

### 附录 D

- D.1：在 $\mathbb C$ 上 $\bar\partial f=(\partial f/\partial\bar z)d\bar z$，再次作用得到 $d\bar z\wedge d\bar z=0$。
- D.2：紧 Riemann surface 上 $H^0(X,\mathcal O_X)^\vee\cong H^1(X,K_X)$，$H^1(X,\mathcal O_X)^\vee\cong H^0(X,K_X)$。
- D.3：用 Leibniz 规则展开 $\bar\partial(\alpha\wedge\beta)$，符号由 $\alpha$ 的次数决定；积分后由 Stokes 定理得到相容。

### 附录 E

- E.1：用覆盖 $U_0,U_1$，转移函数为 $z^d$；全局截面对应次数不超过 $d$ 的多项式，$d\ge0$ 时维数 $d+1$，$d<0$ 时为 0。
- E.2：Serre duality 给 $H^1(\mathbb P^1,\mathcal O(d))^\vee\cong H^0(\mathbb P^1,\mathcal O(-d-2))$。
- E.3：在 $\mathbb P^1$ 上 $\operatorname{ch}(\mathcal O(d))=1+dH$，$\operatorname{td}(T)=1+H$，积分得 $d+1$。

### 附录 H

- H.1：在 $U_0$ 上用 $X_0^d$ 平凡化，在 $U_\infty$ 上用 $X_1^d$ 平凡化；交集上 $X_1^d=z^dX_0^d$，故转移函数为 $z^d$。
- H.2：$d=1$ 时 $H^0$ 基为 $1,z$，$H^1=0$；$d=0$ 时 $H^0$ 基为 $1$，$H^1=0$；$d=-1$ 时两者均为 0；$d=-2$ 时 $H^1$ 基为 $z^{-1}$；$d=-3$ 时 $H^1$ 基为 $z^{-2},z^{-1}$。
- H.3：若 $d\ge0$，$\chi=d+1$；若 $d=-1$，$\chi=0$；若 $d\le-2$，$\chi=0-(-d-1)=d+1$。
- H.4：$d=-3$ 时 $H^1(\mathcal O(-3))$ 基为 $z^{-2},z^{-1}$，$H^0(\mathcal O(1))$ 基为 $1,z$；留数配对只在指数和为 $-1$ 时非零，矩阵按这两个基为反对角单位矩阵。

### 附录 I

- I.1：若 $D=d+(-1)^q\delta$，总次数仍为 $p+q$，符号改为竖直优先约定；验证 $d\delta=\delta d$ 后中间项相消。
- I.2：flasque sheaf 的限制映射满；用逐阶选择构造 Čech 同伦，把每个 cocycle 写成上一阶 cochain 的 coboundary。
- I.3：$\mathbb P^1=U_0\cup U_\infty$ 的两个开集及交集均仿射/Stein，对线丛相干层 acyclic；定理 I.5 说明两项 Čech 复形计算上同调，附录 H 做了逐项计算。
- I.4：Stein 开集上的 $\mathcal F(U)$ 往往无限维；有限覆盖只给有限个无限维空间组成的复形，不能推出上同调有限维。有限性需要 Grauert 或 Hodge-Fredholm 输入。

### 附录 J

- J.1：对偶复形微分中的 $(-1)^{k+1}$ 使 $d_{C^\vee}^2=0$，并保证链级配对对应真正的复形态射。
- J.2：若 $C,D$ 只在次数 0 非零，命题 J.3 说 $C\to D^\vee$ 是同构当且仅当 $C\times D\to\mathbb C$ 非退化。
- J.3：对 Riemann surface，得到 $H^0(X,L)^\vee\cong H^1(X,L^{-1}\otimes\omega_X)$ 和 $H^1(X,L)^\vee\cong H^0(X,L^{-1}\otimes\omega_X)$。
- J.4：投影公式用于把 $f_!(f^*A\otimes F)$ 识别为 $A\otimes f_!F$，从而把 $\operatorname{Map}_{\mathcal C}(f_!(f^*A\otimes F),B)$ 改写为 $\operatorname{Map}_{\mathcal C}(A\otimes f_!F,B)$。

### 附录 K

- K.1：exact equivalence 保持 kernel、cokernel 和短正合列；quasi-inverse 与它互为等价，因此也保持这些结构。
- K.2：长正合列中相邻 image/kernel 维数相消；有限维情形下交错维数和为 0，故 $\chi(B)=\chi(A)+\chi(C)$。
- K.3：$d=-2$ 给 $-1$，$d=-1$ 给 $0$，$d=0$ 给 $1$，$d=1$ 给 $2$；与附录 H 的 $h^0-h^1=d+1$ 一致。
- K.4：命题 K.8 只说明若 RR 已在生成元上成立且可加，则推广到 $K$-群；它没有证明 Chern character、Todd class、trace 相容或 RR 在生成元上成立。

### 附录 L

- L.1：由 $\langle\Delta x,x\rangle=\|dx\|^2+\|d^\ast x\|^2$，$\Delta x=0$ 等价于两个范数均为 0，即 $dx=0$ 且 $d^\ast x=0$。
- L.2：取内积 $\langle d d^\ast b,b\rangle=\|d^\ast b\|^2$；若左侧为 0，则范数为 0。
- L.3：长正合列中 $H^q(E_1)\to H^q(E_0)\to H^q(\mathcal F)\to H^{q+1}(E_1)$，相邻三项有限维推出中间商和子空间有限维。
- L.4：非紧流形上椭圆算子可能不 Fredholm，kernel 或 cokernel 可无限维；例如非紧复平面上全纯函数空间无限维。

### 附录 M

- M.1：若已知 $\mathcal F$ 与 $\mathcal F''$ 有限，则用 $H^{n-1}(\mathcal F'')\to H^n(\mathcal F')\to H^n(\mathcal F)$；若已知 $\mathcal F'$ 与 $\mathcal F$ 有限，则用 $H^n(\mathcal F')\to H^n(\mathcal F)\to H^n(\mathcal F'')\to H^{n+1}(\mathcal F')$。
- M.2：取任意无限维向量空间 $V$，复形 $0\to V\xrightarrow{\operatorname{id}}V\to0$ 有界且每项无限维，但同调为零。
- M.3：函数 $1,z,z^2,\ldots$ 都属于 $\mathcal O(\Delta)$ 且线性无关，因此 $\mathcal O(\Delta)$ 无限维。
- M.4：$H^2$ 的过滤分级片可能为 $E_\infty^{0,2}$、$E_\infty^{1,1}$、$E_\infty^{2,0}$；在给定矩形 $0\le q\le1$ 中实际只有 $E_\infty^{1,1}$ 与 $E_\infty^{2,0}$。

### 附录 N

- N.1：取光滑函数 $\phi_1,\phi_2\ge0$，使 $\phi_1$ 支撑在 $U_1$、$\phi_2$ 支撑在 $U_2$ 且 $\phi_1+\phi_2>0$；令 $\rho_i=\phi_i/(\phi_1+\phi_2)$。
- N.2：对 $1$-cocycle $c_{ij}$，$(Kc)_i=\sum_j\theta_j(c_{ji})$；展开 $(\delta Kc)_{ij}=(Kc)_j-(Kc)_i$，用 cocycle 条件 $c_{ji}+c_{ij}=0$ 和 $c_{ki}-c_{kj}=c_{ji}$ 得到 $c_{ij}$。
- N.3：直和用同一组 $\theta_i$ 分量作用；直和因子由投影和包含把 endomorphism 限制到因子上，支撑和求和条件保持。
- N.4：由定理 N.8，$R\Gamma(\mathcal F)$ 由两项复形 $\Gamma(\mathcal G^0)\to\Gamma(\mathcal G^1)$ 计算；于是 $H^0$ 是 kernel，$H^1$ 是 cokernel，高阶为 0。

### 附录 O

- O.1：局部平凡化下 $E\simeq\mathcal O^r$，于是 $\mathcal Hom(E,\mathcal G)\simeq\mathcal G^r\simeq E^\vee\otimes\mathcal G$；转移函数相容给全局同构。
- O.2：对 $E^{-1}\to E^0$，对偶复形为 $\mathcal Hom(E^0,\omega_X)\to\mathcal Hom(E^{-1},\omega_X)$，微分由原微分预合成并带复形对偶符号。
- O.3：若 $C$ 位于 cohomological degree，$C[-n]^i=C^{i-n}$，故 $H^i(C[-n])=H^{i-n}(C)$；代入 $C=R\operatorname{Hom}(D,\mathbb C)$ 给 $H^{n-i}(D)^\vee$。
- O.4：非 proper 映射下普通推前不控制无穷远逃逸；Serre/Verdier 型对偶需要紧支推前 $f_!$ 与右伴随 $f^!$。

### 附录 P

- P.1：若 $c_1(L)=x$、$c_1(M)=y$，则 $c_1(L\otimes M)=x+y$，所以 $\operatorname{ch}(L\otimes M)=e^{x+y}=e^xe^y$。
- P.2：在 splitting space 上，$E\oplus F$ 的 Chern roots 是两组 roots 的并集；Todd class 是所有 roots 的 Todd 因子乘积。
- P.3：对曲线，$\operatorname{ch}(L)=1+c_1(L)$，$\operatorname{td}(T_X)=1+\frac12c_1(T_X)$。顶次项积分为 $\deg L+\frac12\deg T_X=\deg L+1-g$。
- P.4：附录 P 假设 Chern 类、splitting principle 和 HRR 输入定理；它只证明这些输入接受后，公式两边的形式代数和低维计算。

### 附录 Q

- Q.1：$\sin z$ 是整函数；若是多项式，则因它有无限多个零点而矛盾。
- Q.2：范畴等价自带 quasi-inverse；若等价函子 exact，则 quasi-inverse 也 exact，因为短正合列可由等价反映 kernel 和 cokernel。
- Q.3：stupid filtration 的 successive quotients 是各项 sheaf 的 shift，有限次 cone 重构原 bounded complex。
- Q.4：Euler characteristic 是有限维上同调的交错维数和；没有 coherent finiteness 时该和可能不收敛或维数无限。

### 附录 R

- R.1：令 $u=Tf$。Cauchy-Green 公式给 $\partial u/\partial\bar z=f$，所以 $\bar\partial u=f\,d\bar z$。
- R.2：$H_1(a\,d\bar z_1+b\,d\bar z_2)=T_1(a)$，$H_2(a\,d\bar z_1+b\,d\bar z_2)=T_2(b)$，符号由收缩算子把 $d\bar z_j$ 移到首位时的交错号决定。
- R.3：Cauchy-Green 积分在边界附近会产生边界项；缩小 polydisc 并取 cutoff 可使局部计算只依赖内部基本解。
- R.4：对 $E=\mathcal O^r$，$\bar\partial$ 逐分量作用；局部解算子是对每个分量应用同一个标量同伦算子。

### 附录 S

- S.1：在 $\mathbb P^2$ 上，$H^2(\mathcal O(-3))$ 由单项式 $X_0^{-1}X_1^{-1}X_2^{-1}$ 表示，为一维。
- S.2：取 $j\ne0$，定义锥同伦 $h(e_I)=\pm e_{I\cup\{j\}}$；直接计算 $\delta h+h\delta=\operatorname{id}$。
- S.3：$\chi(\mathbb P^3,\mathcal O(d))=\binom{d+3}{3}=\frac{(d+1)(d+2)(d+3)}6$，负整数处按多项式取值。
- S.4：$\mathcal O(1)$ 的基为 $X_0,X_1,X_2$；$\mathcal O(-4)$ 的 $H^2$ 基由 $X_i^{-2}X_j^{-1}X_k^{-1}$ 型单项式给出，配对抽取乘积中 $(X_0X_1X_2)^{-1}$ 的系数。

### 附录 T

- T.1：$H^0(\mathcal O(1))$ 基为 $X_0,X_1,X_2$；$H^2(\mathcal O(-4))$ 对偶基为 $X_0^{-2}X_1^{-1}X_2^{-1}$、$X_0^{-1}X_1^{-2}X_2^{-1}$、$X_0^{-1}X_1^{-1}X_2^{-2}$，配对矩阵为单位矩阵。
- T.2：$\mathbb P^1$ 中 $\omega\simeq\mathcal O(-2)$；$H^1(\mathcal O(-2))$ 由 $(X_0X_1)^{-1}$ 生成，trace 给出其对偶与 $H^0(\mathcal O)$ 同构。
- T.3：Euler sequence 给 $\det T_{\mathbb P^1}=\mathcal O(2)$、$\det T_{\mathbb P^2}=\mathcal O(3)$，故 canonical bundle 分别为 $\mathcal O(-2)$、$\mathcal O(-3)$。
- T.4：本附录只处理线丛 $\mathcal O(d)$；一般相干层需要 resolution、Ext 形式、有限性和 Serre perfectness。

### 附录 U

- U.1：在 $\mathbb P^2$ 中，$\operatorname{td}(T)=1+\frac32H+H^2$，$e^{dH}=1+dH+\frac{d^2}{2}H^2$；$H^2$ 系数为 $\frac{d^2+3d+2}{2}$。
- U.2：对 $n=3$，residue 为 $\operatorname{Res}_{u=1}u^{d+3}(u-1)^{-4}du=\binom{d+3}{3}$。
- U.3：Euler sequence 在 $K$-理论中给 $[T]=(n+1)[\mathcal O(1)]-[\mathcal O]$；Todd class 乘法性把它化为 $\operatorname{td}(\mathcal O(1))^{n+1}$。
- U.4：一般 HRR 需要对任意 proper smooth variety 的 pushforward、Todd class、Chern character 与 trace 相容；本附录只证明 $\mathbb P^n$ 的线丛模型。

## 卷四答案要点

### 第 0 章

- 0.1：典型难计算对象是 $\mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]$；需要 solidification、profinite 极限和导出张量输入。
- 0.2：有限覆盖的 sheaf 等化子为 $F(U)\to\prod_iF(U_i)\rightrightarrows\prod_{i,j}F(U_i\times_UU_j)$。

### 第 1 章

- 1.1：数据为覆盖、匹配族、粘合函数、唯一性证明；Lean 中应拆为结构体字段和等式 lemma。
- 1.2：需要基覆盖存在、拉回稳定、共同细化、限制拓扑相容和小性控制。

### 第 2 章

- 2.1：$C^0=\prod_iF(U_i)$，$C^1=\prod_{i,j}F(U_{ij})$，$C^2=\prod_{i,j,k}F(U_{ijk})$；微分见第二章正文。
- 2.2：紧性保证有限余并紧；Hausdorff 性保证紧到 Hausdorff 的满射为闭映射，从而商映射。
- 2.3：共同细化使两个覆盖上的匹配族限制到同一覆盖后相同，由 sheaf 唯一粘合推出同一元素。

### 第 3 章

- 3.1：有限 $S$ 给有限直和 $\mathbb Z[\underline *]$，点是 ED，故投射，高阶 Ext 消失。
- 3.2：类 $\phi:P_1\to A$ 通过推出图 $A\leftarrow P_1\to P_0$ 给扩张 $0\to A\to E\to M\to0$。
- 3.3：Hom 复形在 $n$ 阶后为零，因此 $H^i=0$ 对 $i>n$。

### 第 4 章

- 4.1：$[s]\otimes[t]\mapsto[(s,t)]$ 给自由基之间的双射。
- 4.2：profinite 情形中元素是有限商上测度的相容族；外积逐有限层定义并与极限相容。
- 4.3：普通反例为 $(\prod_n\mathbb Z)\otimes\mathbb Q\to\prod_n\mathbb Q$ 非满。
- 4.4：$(1,1/2,1/3,\ldots)$ 若来自统一分母 $m$，则所有 $n$ 整除 $m$，矛盾。

### 第 5 章

- 5.1：$\mathbb Z^\square[S]$ 是 solid 整系数测度/自由对象；有限 $S$ 时为 $\mathbb Z[S]$。
- 5.2：有限 $S$ 上 $A^\square[S]\cong A[S]\cong\bigoplus_{s\in S}A$。
- 5.3：$A^+$ 记录有界元素，控制 rational localization 和解析几何边界。
- 5.4：两个有限商系统有共同细化；极限在共同细化上给出同构。

### 第 6 章

- 6.1：Banach completion 是对象内部完备化；liquid localization 是相对于测度测试对象的范畴局部化。
- 6.2：判别式涉及 $\operatorname{Hom}(\mathcal M_{<p}[S],V)$，测试对象为 profinite $S$ 上的受控测度。
- 6.3：Dolbeault 复形的项是拓扑向量空间，$\bar\partial$ 连续；这些信息用于同调和对偶。
- 6.4：逆极限拓扑定义保证 $S\to V$ 连续当且仅当所有坐标 $S\to V_n$ 连续。

### 第 7 章

- 7.1：pro-etale morphism 是 etale morphism 的 cofiltered limit 型推广，满足相应局部有限展示/弱形式条件；精确定义见 Bhatt-Scholze。
- 7.2：pro-etale site 对象是 $U\to X$；compact Hausdorff site 对象是紧 Hausdorff 空间。
- 7.3：共同点是使用投射型局部对象简化覆盖提升和 sheaf cohomology。
- 7.4：陈述“$S\in\mathbf{CHaus}$ 是 $X_{\operatorname{proet}}$ 的对象”通常无意义，除非给出到 $X$ 的几何结构。

### 第 8 章

- 8.1：示例：Ext 模板输入投射分解 $P_\bullet\to M$ 和目标 $A$，输出 $H^\bullet\operatorname{Hom}(P_\bullet,A)$。
- 8.2：形式化 sheaf 等化子需要小范畴、覆盖族、有限纤维积、等化子和匹配族定义。
- 8.3：第五卷会重复主线；更合理是专题小册，如计算习题集或 Lean 形式化。

### 附录 E

- E.1：0-截断 anima 等价于集合；空间值 sheaf 条件在 0-截断对象中退化为集合值 sheaf 条件。
- E.2：前四项为 $F(U)$、$\prod_iF(U_i)$、$\prod_{i,j}F(U_i\times_UU_j)$、$\prod_{i,j,k}F(U_i\times_UU_j\times_UU_k)$，并带有交替面映射。
- E.3：谱值预层范畴逐点稳定；sheaf 条件由极限定义，对有限极限、有限余极限和 suspension 封闭。
- E.4：凝聚阿贝尔群是阿贝尔群值 sheaf；0-截断 pyknotic abelian group 是离散空间值的阿贝尔群对象，等价于同一数据。
