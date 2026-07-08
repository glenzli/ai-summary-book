# 习题解答要点

本文件给出全书练习的一版解答要点。编号与正文、附录中的练习编号一致。这里的目标是教学可用：给出关键构造、引用位置和证明路线；若作为正式出版详解，可在每题下继续展开。

## 序章

**解答 0.1.** `\operatorname{Sm}_S` 若不取小骨架，其对象类可能是 proper class。Presheaf category、sheafification 和 accessible localization 都需要在小范畴上形成函数范畴和生成集合，因此取小骨架保证 presentability 论证有集合论基础。

**解答 0.2.** 先取 `\operatorname{Sm}_S` 上的 space-valued presheaves，再做 Nisnevich sheafification，得到 `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)`；然后反演所有投影 `X\times\mathbb A^1\to X`，得到 `\mathbf H(S)`。

**解答 0.3.** `f_*` 是 `f^*` 的右伴随，可由 presentability 和保持余极限得到。`f_!` 与 `f^!` 是非常推前和非常拉回，需要 proper、localization、purity、base change 等额外相干结构，不能从 `f^*` 的定义推出。

**解答 0.4.** 例如 “accessible localization 存在”在本书中作为一般范畴论内部命题使用；Morel-Voevodsky homotopy purity 或 motivic 六操作存在性则是外部输入定理，因为其证明依赖专门文献。

## 第一章

**解答 1.1.** Smooth morphisms 对 base change 稳定。若 `X,Y\in\operatorname{Sm}_S`，则 `X\times_SY\to S` 可看作 `X\times_SY\to Y\to S`，第一箭头是 `X\to S` 的 base change，第二箭头 smooth，复合仍 smooth。

**解答 1.2.** Nisnevich covering 要求对每个点 `x\in X`，存在覆盖中的点 `u\in U_i` 映到 `x` 且残差域扩张 `k(x)\simeq k(u)`。Zariski 覆盖是开覆盖，因此点提升可取同一点；Nisnevich 覆盖还允许 etale 局部替换。

**解答 1.3.** 对 Zariski 开覆盖 `X=U\cup V`，sheaf condition 给出 `F(X)\simeq F(U)\times_{F(U\cap V)}F(V)`。这是命题 1.11 在开覆盖 Cech nerve 上截断后的通常粘合条件。

**解答 1.4.** Elementary Nisnevich square 中 `j:U\hookrightarrow X` 是开嵌入，`p:V\to X` 是 etale，且 `V\times_X(X\setminus U)\to X\setminus U` 为同构。因此任意点要么在 `U` 中被 `U` 提升，要么在闭补中被 `V` 以同残差域提升。

**解答 1.5.** Yoneda 只说明 representable presheaf 由映射对象表示；它本身不保证 sheaf 条件。Representable presheaf 是 Nisnevich sheaf 使用的是 Nisnevich topology 在 schemes 上 subcanonical，即覆盖下降对表示对象有效。

## 第二章

**解答 2.1.** `\operatorname{Sm}_S` 已取小骨架，其对象构成集合。每个对象 `X` 给出一个投影 `X\times\mathbb A^1\to X`，所以这些投影的集合由小对象集索引，不是 proper class。

**解答 2.2.** 取 `X=S`，命题 2.9 说明投影 `S\times_S\mathbb A^1_S\simeq\mathbb A^1_S\to S` 在 `\mathbf H(S)` 中为等价，因此 `\mathbb A^1_S` 是 motivically contractible。

**解答 2.3.** `\mathbf H_*(S)` 是终对象下的 under-category。对象 `*_S\to *_S` 同时是初对象和终对象：到任意带基点对象有唯一保持基点的态射，任意带基点对象到它也唯一。

**解答 2.4.** 取一个 presheaf 满足对象值上的 `\mathbb A^1`-不变性但不满足 Nisnevich gluing，例如人为在某个 Nisnevich square 上破坏拉回条件的 presheaf。它不能直接成为 `\mathbf H(S)` 的对象，因为进入 `\mathbf H(S)` 前必须先满足 sheaf 条件。

**解答 2.5.** 由局部化泛性质，realization 要从 `\mathbf H(S)` 因子化，必须先是 Nisnevich descent 的函子，并且把每个 `X\times\mathbb A^1\to X` 送为等价；若还要从稳定范畴因子化，还需与 `T`-稳定化相容。

## 第三章

**解答 3.1.** 在 pointed category 中，悬挂 `\Sigma Y` 定义为 `*\amalg_Y*`。态射 `Y\to *` 的 cofiber 是推出 `*\amalg_Y*`，因此等于 `\Sigma Y`。

**解答 3.2.** `\mathbb G_m` 是乘法群概形，单位截面 `1:S\to\mathbb G_m` 是自然的全局点。以它为基点使群结构、smash product 和 Tate sphere 的定义与乘法单位相容。

**解答 3.3.** 命题 3.5 给出 `T\simeq S^{1,0}\wedge\mathbb G_m=S^{1,1}`。按双指标约定，`S^{2,1}=S^{1,0}\wedge S^{1,1}\simeq S^{1,0}\wedge T`。

**解答 3.4.** 反演张量积 `T=S^{1,0}\wedge\mathbb G_m` 只强制这个整体可逆。一般幺半范畴中张量积可逆不推出两个因子分别可逆；若要分别反演，需要额外稳定化或证明。

**解答 3.5.** 若 `\mathcal L` 是被一族生成子检测为零的对象构成的 full subcategory，给定 cofiber sequence `A\to B\to C`，对任一生成子取 mapping spectrum 得 fiber sequence。若其中两个映射谱为零，则第三个也为零，所以 `\mathcal L` 对 cofiber 封闭。

## 第四章

**解答 4.1.** 强对称幺半函子 `f^*` 带有结构等价 `\mathbb 1_X\simeq f^*\mathbb 1_Y` 和 `f^*(A\otimes B)\simeq f^*A\otimes f^*B`。第一条就是单位保持性。

**解答 4.2.** 对 Cartesian 方块，ordinary base-change transformation 在 derived category 中写为 `g^*Rf_* \to Rf'_*g'^*`，由单位 `\operatorname{id}\to Rf'_*f'^*` 和交换同构 `f'^*g^*\simeq g'^*f^*` 组合得到。

**解答 4.3.** 普通推前版本的投影公式是 `f_*(A\otimes f^*B)\simeq f_*A\otimes B`。证明用 `f_*` 的 `\mathcal D(Y)`-模线性：推前把源上的张量外部标量移到目标上。

**解答 4.4.** 对非 proper 态射，ordinary pushforward 和 compact-support pushforward 表示不同几何行为。开嵌入时 `j_!` 是 extension by zero，而 `j_*` 允许边界附近的截面延拓，二者通常不同。

**解答 4.5.** 在 recollement 中，`j^*j_!\simeq\operatorname{id}` 且 `i^*j_!=0`。对 `j_!j^*E\to E` 的 cofiber `C` 作用 `j^*` 得零，故 `C` 位于 `i_*` 本质像；再用 `i^*` 识别为 `i_*i^*E`。

## 第五章

**解答 5.1.** Proper morphism 定义为 separated、finite type、universally closed；这三条对复合稳定。命题 5.7 中需要用到复合仍 proper，才能同时把 `(gf)_!`、`g_!f_!` 与相应 `*`-推前识别。

**解答 5.2.** `j_!` fully faithful 等价于伴随单位 `\operatorname{id}\to j^*j_!` 为等价。对开嵌入有 `j^!\simeq j^*`，所以由 fully faithful 得 `j^*j_!\simeq\operatorname{id}`。

**解答 5.3.** 命题 5.16 使用 localization cofiber sequence `j_!j^*E\to E\to i_*i^*E`。若 `E` 支撑在 `Z` 上，则 `j^*E\simeq0`，第一项消失，故 `E\simeq i_*i^*E`。

**解答 5.4.** `j^*E\simeq0` 表示 `E` 在开补 `U` 上消失，因此支撑在闭子集 `Z`。`i^*E\simeq0` 表示 `E` 限制到闭子集消失，描述的是远离 `Z` 的条件，二者不是同一支撑。

**解答 5.5.** 对 `E=\Sigma_T^\infty X_+`，命题 5.15 给出 `j_!\mathbb 1_U\to\mathbb 1_X\to i_*\mathbb 1_Z` 的形式。若加入 `X` 的结构态射，可把它理解为开部分、全体和闭补在稳定 motivic homotopy 中的分解。

## 第六章

**解答 6.1.** 零向量丛 `0_X` 的总空间是 `X`，去掉零截面后为空集。因此 Thom space `0_X/(0_X-X)` 为 `X/\varnothing\simeq X_+`。

**解答 6.2.** 平凡秩 `r` 向量丛为 `\mathbb A^r_X`，其 Thom space 是 `\mathbb A^r/(\mathbb A^r\setminus0)\wedge X_+`。由 direct sum formula 得 `Th(\mathcal O^r)\simeq T^{\wedge r}\wedge X_+`。

**解答 6.3.** Homotopy purity 比较 `X/(X-Z)` 与 `Z` 的法方向。法丛记录 `Z` 在 `X` 中一阶邻域的横向几何；没有法丛就无法描述闭嵌入附近被压缩后的 Thom twist。

**解答 6.4.** Etale morphism 的相对切丛为零，所以 smooth purity `f^!\simeq\Sigma^{T_f}f^*` 化为 `f^!\simeq f^*`。

**解答 6.5.** Thom space 是向量丛的几何商对象，不需要 orientation 才能定义。Orientation 是对 Thom class 或 Thom isomorphism 的选择，使 cohomology 中可把 Thom twist 解开。

## 第七章

**解答 7.1.** Etale morphism smooth 且 `T_f=0`。代入 smooth ambidexterity `f_!\simeq f_\sharp\Sigma^{-T_f}`，得到 `f_!\simeq f_\sharp`。

**解答 7.2.** 若 `f` smooth proper，则 proper compatibility 给出 `f_!\simeq f_*`，smooth ambidexterity 给出 `f_!\simeq f_\sharp\Sigma^{-T_f}`。合并即得到命题 7.5 的 `f_*\simeq f_\sharp\Sigma^{-T_f}` 型识别。

**解答 7.3.** Dualizable object `A` 有对偶 `A^\vee`、coevaluation `\mathbb 1\to A\otimes A^\vee` 和 evaluation `A^\vee\otimes A\to\mathbb 1`。两个三角恒等式要求 `A` 和 `A^\vee` 经 coevaluation 再 evaluation 的合成分别为恒等。

**解答 7.4.** `u=\operatorname{id}_A` 时，Euler characteristic 是 `\mathbb 1\xrightarrow{coev} A\otimes A^\vee\xrightarrow{\tau}A^\vee\otimes A\xrightarrow{ev}\mathbb 1` 的 trace。

**解答 7.5.** Additive transfer 来自稳定加性结构中的推前/迹，作用于加法或群对象。Norm 是对称幺半乘法转移，保留乘法结构；二者需要 Tambara-like 分配律才能混合。

## 第八章

**解答 8.1.** 从 `g^*f_*` 出发，插入单位 `\operatorname{id}\to f'_*f'^*`，得到 `g^*f_*\to f'_*f'^*g^*f_*`；用方块交换同构把 `f'^*g^*` 换为 `g'^*f^*`；再用余单位 `f^*f_*\to\operatorname{id}` 得 `f'_*g'^*`。

**解答 8.2.** Proper morphism 对 base change 稳定，因为 separated、finite type、universally closed 分别对 base change 稳定。三者合取仍成立。

**解答 8.3.** 设两个 Cartesian 方块横向复合。先对右方块使用 exchange equivalence，再对左方块使用 exchange equivalence；复合相干说明该合成等于外矩形的 exchange transformation。

**解答 8.4.** 对 closed-open pair 的 cofiber sequence 作用 `g^*`。由开嵌入和闭嵌入的 base change 等价，把 `g^*j_!` 和 `g^*i_*` 分别识别为拉回方块中的 `j_{Y!}g_U^*` 和 `i_{Y*}g_Z^*`，得到拉回后的 localization sequence。

**解答 8.5.** Dualizable 假设用于把 internal Hom 改写为张量：`\underline{Hom}(f^*A,B)\simeq f^*(A^\vee)\otimes B`。没有该识别，projection formula 不能直接推出 internal Hom 公式。

## 第九章

**解答 9.1.** 由定义，`H^{0,0}(S,\mathbb Z)=\pi_0\operatorname{Map}_{\mathbf{SH}(S)}(\mathbb 1_S,H\mathbb Z_S)`，即 `H\mathbb Z` 的全局 0 次 motivic cohomology。

**解答 9.2.** Commutative ring spectrum `E` 有乘法 `E\otimes E\to E`。两个 cohomology class 由映射到 `E` 表示，先 smash 再乘法得到 cup product；交换性和结合性来自 `E` 的 `E_\infty` 结构。

**解答 9.3.** 对 cofiber sequence `A\to B\to C` 应用 `\operatorname{Map}(-,\Sigma^{p,q}E)` 得 fiber sequence。取同伦群得到长正合列；带紧支撑版本同理作用于 `p_!` 表达式。

**解答 9.4.** `CH^n(X)` 是由代数循环定义的经典对象，`H^{2n,n}` 是由 `H\mathbb Z` 在 `\mathbf{SH}` 中表示的 cohomology。二者相等需要 Voevodsky/Bloch 比较定理，不是定义。

**解答 9.5.** 环同态 `\mathbb Z\to\mathbb Z/m` 诱导谱映射 `H\mathbb Z\to H\mathbb Z/m`，再由后合成得到 `H^{a,b}(X,\mathbb Z)\to H^{a,b}(X,\mathbb Z/m)`。

## 第十章

**解答 10.1.** 对 commutative algebra object `A`，自由函子为 `A\otimes -`，遗忘函子忘掉 module action。映射空间等价 `Map_A(A\otimes X,M)\simeq Map(X,U(M))` 给出自由-遗忘伴随。

**解答 10.2.** 命题 10.4 给出 `H\mathbb Z`-module category 的稳定、presentable 和张量性质。命题 10.6 的 `H\mathbb Z`-linearization 由自由 module 函子 `H\mathbb Z\otimes -` 给出，并继承这些性质。

**解答 10.3.** Tate motive `\mathbb Z(1)` 对应 motivic 双次数中的 weight shift。一般 `\mathbb Z(q)` 表示第 `q` 个 Tate twist，并与 `S^{2q,q}` 或 `T^q` 的稳定悬挂坐标相连。

**解答 10.4.** Sphere spectrum 中可能存在不被 `H\mathbb Z`-homology 检测的稳定同伦信息，例如 torsion 或更高 chromatic 信息。自由 `H\mathbb Z`-module 化会只保留 motivic cohomology 线性部分，因此不应保守。

**解答 10.5.** 需检查基是否为 perfect field 或更一般允许基、系数环、有无有理化或反演特征指数、使用 triangulated 还是 infinity-categorical enhancement、six operations 是否相容。

**解答 10.6.** Action 为 `H\mathbb Z\otimes M\to M`。结合律图比较 `(H\mathbb Z\otimes H\mathbb Z)\otimes M\to H\mathbb Z\otimes M\to M` 与 `H\mathbb Z\otimes(H\mathbb Z\otimes M)\to H\mathbb Z\otimes M\to M`，通过乘法和结合约束相等。

**解答 10.7.** Effective motives 中 Tate object 未必可逆。Stable motives 通过反演 Tate object 或加入所有 Tate desuspensions，使 `\mathbb Z(1)` 成为可逆坐标。

## 第十一章

**解答 11.1.** `KGL` 是 ring spectrum，乘法 `KGL\otimes KGL\to KGL` 使两个由 `\Sigma_T^\infty X_+` 到悬挂 `KGL` 的类可 cup。结合律、单位和交换性由 ring spectrum 结构给出。

**解答 11.2.** 奇异 scheme 的 Quillen K-theory 不一定 `\mathbb A^1`-invariant。Homotopy K-theory `KH` 通过强制 `\mathbb A^1`-不变或几何 realization 修正该缺陷，因此会改变奇异情形的信息。

**解答 11.3.** 把第九章 localization 长正合列中的 `E` 取为 `KGL`。由 `KGL` 表示 cohomology，closed-open cofiber sequence 直接给出 `KGL`-cohomology 长正合列。

**解答 11.4.** Nisnevich descent 使用 etale 局部且残差域不变的覆盖。Cdh descent 还包括 proper blow-up 型覆盖，能处理奇异和抽象 blow-up square，因此更强。

**解答 11.5.** 在 Chern character 等价假设下，`KGL_\mathbb Q` 分解为 Tate shifts 的 motivic cohomology。形式上 `KGL_\mathbb Q^{0,0}(X)` 分解为若干 `H^{2i,i}(X,\mathbb Q)` 的乘积或直和，具体取决于完备化和有界性约定。

**解答 11.6.** Bott element 给出 `KGL\simeq\Sigma^{2,1}KGL`。迭代得 `\Sigma^{p,q}KGL\simeq\Sigma^{p-2q,0}KGL`，从而 `KGL^{p,q}(X)\simeq KGL^{p-2q,0}(X)`。

**解答 11.7.** 抽象 blow-up square 由 closed immersion `Z\hookrightarrow X` 和 proper morphism `X'\to X` 组成，并要求 `X'\setminus Z'\to X\setminus Z` 同构。它强于 closed-open localization，因为包含 proper 修改而不只是开闭分解。

## 第十二章

**解答 12.1.** Orientation 给出 `\mathbb P^\infty` 上的通用类。线丛 `L` 由分类映射 `X\to\mathbb P^\infty` 表示，拉回通用类得到 `c_1^E(L)\in E^{2,1}(X)`。

**解答 12.2.** `L\otimes(M\otimes N)\simeq(L\otimes M)\otimes N`。对 first Chern class 写成 `F_E(c_1(L),c_1(M))`，张量积的结合性强制 formal group law 满足结合律。

**解答 12.3.** Universality 表示给定 oriented spectrum `E`，orientation 等价于唯一的 ring spectrum map `MGL\to E`。因此 `MGL` 是 orientations 的初始对象，而不只是一个例子。

**解答 12.4.** Additive formal group law 为 `F(x,y)=x+y`。Multiplicative formal group law 的典型形式为 `F(x,y)=x+y-\beta xy` 或符号变体；首个非线性项是 `xy` 项。

**解答 12.5.** Thom isomorphism 需要选择 Thom class，使 `E^*(X)` 与 `E^*(Th(V))` 通过 cup product 与该类相连。无 orientation 时 Thom space 存在，但 cohomology 中未必有自然同构。

**解答 12.6.** Projective bundle formula 断言 `E^{*,*}(\mathbb P(V))` 是 `E^{*,*}(X)` 上由 `1,\xi,\ldots,\xi^{r-1}` 生成的自由模，`\xi=c_1(\mathcal O(1))` 或对偶约定下的 tautological line bundle Chern class。

**解答 12.7.** Splitting principle 允许在合适拉回后把向量丛分解为线丛和。线丛上 Chern class 的乘法公式已知，因拉回足够保守，可把公式下降回原空间。

## 第十三章

**解答 13.1.** Localizing subcategory 定义为 full stable subcategory，且对小余极限封闭。Stable 性给出对 cofiber/fiber 封闭；localizing 条件给出对任意小余极限封闭。

**解答 13.2.** `f_q(E)` 是 `E` 在 `q`-effective localizing subcategory 中的最佳近似，通常由 inclusion 的右伴随构造。右伴随把 `E` 投影到该 effective 层。

**解答 13.3.** `s_q(E)` 是 `f_q(E)\to f_{q-1}(E)` 或约定中的相邻 effective 近似之间的 cofiber。它表示 filtration 在第 `q` 层新增的信息。

**解答 13.4.** 若 cellular subcategory 由 spheres 生成，态射在所有 spheres 映射谱上为等价，则 fiber 对所有生成子映射为零。生成性推出 fiber 为零，因此态射为等价。

**解答 13.5.** Slice tower 是无限 tower。由 tower 产生的 spectral sequence 是否收敛取决于完备性、connectivity、lim^1 项和有界性条件，不能只由 tower 存在推出。

**解答 13.6.** 对 tower 的相邻 fiber/cofiber 取同伦群，形成 exact couple；exact couple 的 derived couple 给出 spectral sequence，其 `E_1` 或 `E_2` 页由 slices 的 cohomology 表达。

**解答 13.7.** Slice tower 的输入是 effective Tate filtration 和 motivic weight。Adams tower 的输入是某个 ring spectrum 或 homology theory 的 Adams resolution。二者都是 filtration，但生成机制不同。

## 第十四章

**解答 14.1.** Finite correspondence from `X` to `Y` 是 `X\times Y` 中的代数循环，其支撑对 `X` 的每个连通分支 finite and surjective。该条件保证 correspondence 能作为从 `X` 到 `Y` 的多值有限态射并可复合。

**解答 14.2.** 普通 morphism `f:X\to Y` 的图给出 correspondence。两个 morphisms 的图在 fiber product 中相交后再推前，得到复合 morphism 的图，因此图 correspondence 与复合相容。

**解答 14.3.** Presheaf with transfers 是加性反变函子 `Cor_k^{op}\to Ab`，其中 `Cor_k` 的对象是光滑概形，态射是 finite correspondences。

**解答 14.4.** Sheafification 通常只在 underlying presheaf 上定义。要让 finite correspondences 的作用延到 sheafification，需要证明 transfer action 与 Nisnevich 局部化兼容，这是 Voevodsky/MVW 的非平凡结果。

**解答 14.5.** Additive transfers 来自 correspondences 的加性结构，适合 abelian groups 或 motives。Multiplicative norms 来自 finite etale/finite locally free 的乘法性转移，作用于 ring-like objects。

**解答 14.6.** 对 `\alpha:X\rightsquigarrow Y` 与 `\beta:Y\rightsquigarrow Z`，先在 `X\times Y\times Z` 中拉回两者，取交积，再沿 `X\times Y\times Z\to X\times Z` proper pushforward，得到 `\beta\circ\alpha`。

**解答 14.7.** Suslin complex 把 presheaf with transfers `F` 送到链复形 `C_nF(X)=F(X\times\Delta^n)`。`\Delta^1` 提供同伦参数，使 complex 强制 `\mathbb A^1`-同伦不变。

## 第十五章

**解答 15.1.** 核心数据包括 finite syntomic morphism、到目标的 morphism，以及 cotangent complex 或 normal data 的 trivialization/framing。不同模型中表现为 equational、normal 或 tangential framing。

**解答 15.2.** Cotangent complex 控制 morphism 的虚切/法方向。Framing 是对这类 twist 的平凡化，因此可把 Gysin/Thom twist 转化为无扭转的 transfer 数据。

**解答 15.3.** Recognition theorem 说明适当 grouplike framed motivic spaces 等价于 very effective motivic spectra。换言之，framed transfers 提供 motivic infinite loop space 的几何模型。

**解答 15.4.** Finite correspondences 使用有限循环并主要给 additive transfers。Framed transfers 使用 finite syntomic maps 加 framing，能捕捉稳定 motivic homotopy 中的 Thom twist 和 infinite loop structure。

**解答 15.5.** Fundamental classes 定义 Gysin maps，framed transfers 定义几何转移。若二者不相容，同一 finite syntomic correspondence 会给出不同的 cohomological operation，理论无法粘合。

**解答 15.6.** Finite etale morphism finite、flat、locally of finite presentation，且 cotangent complex 为零。因此它是 finite syntomic，虚相对维数为零。

**解答 15.7.** Infinite loop recognition 需要可逆化加法 monoid 结构。Grouplike 条件保证 `\pi_0` 上的 monoid 已是 group，从而能对应稳定谱而不仅是不稳定 `E_\infty`-space。

**解答 15.8.** 三步为：加入 framed correspondences 的 presheaf；做 Nisnevich sheafification 和 `\mathbb A^1`-localization；再做 group completion/very effective stabilization，得到相应 motivic spectrum。

## 第十六章

**解答 16.1.** 对 lci morphism `f:X\to Y`，fundamental class 可看作从 Thom-twisted 单位到 extraordinary pullback 的类，例如 `\Sigma^{T_f}\mathbb 1_X\to f^!\mathbb 1_Y`。这里 `T_f=-L_f`，与 smooth purity `f^!\simeq\Sigma^{T_f}f^*` 的符号约定一致。

**解答 16.2.** `L_f` 是 cotangent complex，`T_f=-L_f` 是其对偶或虚切丛类。Smooth 情形 `L_f\simeq\Omega_{X/Y}`，`T_f` 为相对切丛；lci 情形给出 K-theory 中的虚向量丛。

**解答 16.3.** 恒等态射 `id_X` 的 bivariant group 按定义由 `id_X^!` 或等价的单位 twist 表示。因为 `id^!\simeq id^*`，该 group 退化为 `X` 上的普通 cohomology。

**解答 16.4.** 普通 base change 只比较拉回和推前。Excess formula 还要记录方块非横截时产生的 excess bundle，并以 Euler/Thom class 修正 Gysin map。

**解答 16.5.** Todd class 衡量两个 orientation 或两个 cohomology theories 下 Gysin/Riemann-Roch 变换之间的差异。它是从一种 orientation 转换到另一种 orientation 的校正因子。

**解答 16.6.** Deformation to the normal cone 是把闭嵌入 `Z\hookrightarrow X` 的几何从 `X` 连续退化到 normal cone `C_ZX` 的构造，通常通过 blow-up `X\times\mathbb A^1` 沿 `Z\times0` 实现。

**解答 16.7.** Excess bundle 为零表示方块横截。此时 excess Euler class 是单位，excess formula 退化为普通 transverse base change compatibility。

## 第十七章

**解答 17.1.** 对 finite etale `f:T\to S`，stable norm functor 是 symmetric monoidal functor `f_\otimes:\mathbf{SH}(T)\to\mathbf{SH}(S)`，与 base change 和复合相容。

**解答 17.2.** `f_*` 是 `f^*` 的右伴随，属于加性稳定范畴结构。`f_\otimes` 是乘法性 norm，保持对称幺半结构。二者目标相同但编码不同操作。

**解答 17.3.** Commutative ring spectrum 只有每个基上的乘法。Normed spectrum 还指定所有 finite etale maps 上的 norm operations，并要求这些 operations 与复合、base change 和乘法相容。

**解答 17.4.** Finite separable field extension `L/k` 对应 finite etale morphism `\operatorname{Spec}L\to\operatorname{Spec}k`。Norm operation 是沿该 finite etale map 的乘法转移，几何上来自点的有限纤维乘积。

**解答 17.5.** 只有加法 transfer 和乘法 norm 还不足以控制 `N(a+b)`。Distributivity 相干说明 norm 如何与加法转移交互，是 Tambara-like 结构的核心。

**解答 17.6.** 若 `U\xrightarrow g T\xrightarrow f S` finite etale，则 norm 相干公式为 `(fg)_\otimes\simeq f_\otimes g_\otimes`，并与单位态射的 norm 为恒等函子相容。

**解答 17.7.** Commutative ring spectrum 的乘法只在同一基上定义。Normed spectrum 需要跨 finite etale morphisms 的额外乘法转移；这不是由单个基上的 commutative algebra 结构自动给出的。

## 第十八章

**解答 18.1.** `GW(k)` 是非退化对称双线性型的 Grothendieck 群。Rank map 把一个二次型送到其底层向量空间维数。

**解答 18.2.** Morel 定理把 motivic sphere 的自同态环识别为 `GW(k)`。因此 trace/Euler characteristic 作为 `End(\mathbb 1_k)` 的元素，自然取值于 Grothendieck-Witt 群。

**解答 18.3.** Milnor K-theory 记录符号和乘法群信息。Milnor-Witt K-theory 还记录二次型数据和 `\eta`，因此能分辨 rank 之外的 quadratic 信息。

**解答 18.4.** Rank map `GW(k)\to\mathbb Z` 忘记二次型的判别式、signature、Witt class 等信息。两个不同二次型可有同一 rank。

**解答 18.5.** Orientation 决定 Thom class 和 Euler class 的归一化。Quadratic refinements 中，不同 orientation 会改变局部 degree 或 Euler class 的二次型值。

**解答 18.6.** 在 `\mathbb R` 上，`\langle1\rangle` 与 `\langle-1\rangle` 都 rank 为 1。Signature 分别为 `1` 和 `-1`，所以 rank 不能区分它们。

**解答 18.7.** 非退化零点的 motivic local degree 通常由 Jacobian determinant 的一维二次型给出：`\deg_x(f)=\langle \det J_f(x)\rangle`，再按残差域扩张做 trace 到基域。

**解答 18.8.** Ordinary Euler class 是 quadratic Euler class 经 rank 或 forgetful map 的影像。影像为零只说明 rank 层面抵消，Witt 部分仍可能非零。

## 第十九章

**解答 19.1.** 平凡群 torsor 唯一且无非平凡自同构，`[X/1](T)` 就是 `Map(T,X)`。因此 quotient stack `[X/1]` 与 `X` 表示的 stack 等价。

**解答 19.2.** Borel theory 用 `EG\times_GX` 把 equivariant 问题转为非 equivariant quotient。Genuine theory 保留 stabilizers、representation spheres、fixed points 和 genuine transfers，因此信息更多。

**解答 19.3.** 对 invariant closed immersion `i:Z\hookrightarrow X` 和开补 `j:U\hookrightarrow X`，在 quotient stack 口径下有 `j_!j^*E\to E\to i_*i^*E`。

**解答 19.4.** Equivariant purity 中闭嵌入的 normal bundle 带 `G`-linearization。Thom twist 必须使用该 equivariant normal bundle，否则会忘记 stabilizer 对法方向的表示。

**解答 19.5.** Linearly reductive 假设保证 quotient stacks 和 equivariant categories 有较好的 descent、compact generation 和 exactness 性质，使六操作和 purity 可控。

**解答 19.6.** 若 `X\to X/G` 为 `G`-torsor，则给出 `T\to X/G` 等价于给出其拉回 torsor `T\times_{X/G}X\to T` 及 equivariant map 到 `X`。这正是 `[X/G](T)` 的对象。

**解答 19.7.** `G_m` 作用 `t\cdot x=tx`。若点 `x` 被所有 `t` 固定，则 `tx=x` 对所有 `t` 成立，只能有 `x=0`。故 fixed locus 为原点。

**解答 19.8.** 群同态 `H\to G` 使任何 `H`-equivariant datum 可诱导到或映到 `G`-quotient 口径；restriction 对应自然态射 `[X/H]\to[X/G]`。

**解答 19.9.** Equivariant cells 必须检测 stabilizer 表示方向。Representation spheres 记录群作用在向量空间上的非平凡权重，普通 motivic spheres 无法捕捉这些信息。

## 第二十章

**解答 20.1.** 不同 algebraic stacks 类别在 atlas、diagonal、stabilizer、局部商性质和有限性上差别很大。六操作是否存在依赖这些假设，因此必须声明允许类别。

**解答 20.2.** Lisse-extended motivic homotopy type 通过所有 smooth maps `U\to\mathcal X` from schemes 形成 diagram，再用 descent/limit 把 scheme-level motivic categories 粘合成 stack 上的 theory。

**解答 20.3.** 限制到 schemes 时，应恢复前文的 `\mathbf{SH}(X)`、`f^*,f_*,f_!,f^!`、tensor 和 internal Hom，并保持 base change、projection formula、localization、purity 等相干。

**解答 20.4.** Quotient stack `[X/G]` 同时记录底层 quotient 和 stabilizer。Equivariant motivic homotopy 正需要这些 stabilizer、fixed point 和 representation sphere 信息。

**解答 20.5.** Fixed point localization 需要 equivariant purity、normal bundle 的权重分解、可逆化某些 Euler classes，以及与 stacky/equivariant six operations 的相容。

**解答 20.6.** Atlas `p:U\to\mathcal X` 的 Cech nerve 为 `U_n=U\times_{\mathcal X}\cdots\times_{\mathcal X}U`，含 `n+1` 个 `U`，face maps 省略一个因子，degeneracy maps 重复一个因子。

**解答 20.7.** `BG=[S/G]` 的粗空间可能是 `S`，但 `BG` 的点有 automorphism group `G`。Equivariant vector bundles 和 representation spheres 依赖该 automorphism group，因此不能由粗空间决定。

**解答 20.8.** 若 stack 局部为 `[U/G]`，该局部片段上的 motivic theory 可用 `G`-equivariant motivic theory on `U` 描述。全局 theory 由这些局部模型和交叠 descent data 粘合。

**解答 20.9.** Normal directions 在 stack 上带 stabilizer 表示。Stacky purity 的 Thom twist 必须记录该表示，否则 Gysin map 会退化为粗空间上的错误 twist。

## 第二十一章

**解答 21.1.** Trivial log structure 是 `\mathcal O_X^\times\hookrightarrow\mathcal O_X`。带此 log structure 的 log scheme 不含额外边界数据，因此应恢复普通 scheme 的 motivic theory。

**解答 21.2.** Perfect scheme 通常指特征 `p` 下 Frobenius morphism 为同构的 scheme。等价地，其仿射环上的 Frobenius 是同构。

**解答 21.3.** Universal homeomorphism 是 integral、surjective、universally injective 的 morphism。它在任意 base change 后仍为 topological homeomorphism。

**解答 21.4.** `[1/p]` 局部化会把 `p`-primary torsion 信息杀掉或可逆化。许多 integral phenomena 正存在于 `p`-torsion 或 Frobenius 效应中，因此会丢失。

**解答 21.5.** Open complement localization 只记录闭开分解。Log boundary 把边界 divisor 作为结构的一部分保留，能记录交叉、monoid 和退化方向等更细信息。

**解答 21.6.** Perfect scheme 的定义即 Frobenius 为同构；在仿射情形，环 `R` perfect 表示 `x\mapsto x^p` 为双射，因此对应 scheme Frobenius 为同构。一般情形由仿射局部粘合。

**解答 21.7.** Perfectization 是沿 Frobenius 迭代取极限或 colimit 的过程，可能把非 perfect 的 nilpotent、inseparable 或 `p`-primary integral 信息改变，因此 integral motivic theory 不必保持。

**解答 21.8.** 对 divisor `D\subset X`，log structure 可由允许在 `D` 有零/极点的函数或 complement `X\setminus D` 的单位诱导。它把 “接近边界” 的 monoid 信息写入结构层之外的数据。

**解答 21.9.** 普通 `\mathbb A^1` interval 强制 affine line homotopy。Log interval 允许带边界或 log 结构的 homotopy，能适配 semistable degeneration 或 log smooth geometry。

**解答 21.10.** 闭嵌入 `\operatorname{Spec}(A/I)\to\operatorname{Spec}A`，其中 `I` nilpotent，是 integral、surjective 且诱导同一底层拓扑空间，因此是 universal homeomorphism。

## 第二十二章

**解答 22.1.** 若 realization `R` 从 `\mathbf H(S)` 因子化，则由 `\mathbb A^1`-局部化泛性质，`R(X\times\mathbb A^1\to X)` 必须是等价。

**解答 22.2.** 在 complex topology 中，`\mathbb A^1(\mathbb C)\simeq\mathbb C` 可缩。因此 `X(\mathbb C)\times\mathbb C\to X(\mathbb C)` 是 homotopy equivalence，Betti realization 反演 `\mathbb A^1`。

**解答 22.3.** Etale homotopy type 通常只能以 pro-space 捕捉所有 finite etale covers 和 cohomological approximations。单个 space 往往不能保留这些 inverse system 信息。

**解答 22.4.** Complex Betti realization 使用 `X(\mathbb C)` 的拓扑空间。Real realization 使用 `X(\mathbb R)` 或带 `C_2`-作用的 complex points，能反映实闭域和共轭作用的信息。

**解答 22.5.** Realization functor 可能忘记 motivic weight、torsion 或 arithmetic 信息。它保守需要额外假设；不能由 functor 存在推出。

**解答 22.6.** 需要自然等价 `R_Y(f_!E)\simeq f^{top}_!R_X(E)`，并与复合、base change、projection formula 和单位对象相容。

**解答 22.7.** 命题 22.13 的证明路线是：若 realization 与六操作相容，则把 motivic localization sequence 逐项送到目标理论；目标中的 exactness 保持 cofiber sequence，因此得到 realized localization sequence。

## 第二十三章

**解答 23.1.** Coefficient system 是把每个基对象 `S` 赋予一个稳定 presentable symmetric monoidal category，并对态射给出 pullback/pushforward 等结构且满足相干公理的系统。

**解答 23.2.** Universal property 表示从 `\mathbf{SH}` 到任意满足相同 motivic 公理的 coefficient system 的 morphism 由唯一方式存在。它描述的是初性，而不是某个具体模型。

**解答 23.3.** Pullback formalism 主要控制 `f^*` 和其相干。完整六操作还需要 `f_*`、`f_!`、`f^!`、projection formula、base change、localization、purity 等额外结构。

**解答 23.4.** 一个 functor 可与 pullback commute，即 `R_Xf^*\simeq f^*R_Y`；但它未必与 `f_!` 或 `f^!` commute。后者是 six-operation compatibility，强得多。

**解答 23.5.** 2025-2026 pullback formalism 结果较新，假设、适用几何类别和与既有六操作的关系仍需逐项核查。因此本书把它们列为研究边界，而非基础外部输入。

**解答 23.6.** 若 category 由 smooth generators 在 colimits 下生成，则保持 colimits 的 functor 由其在这些 generators 上的值决定。验证 morphism 等价也可在 generators 上检测。

**解答 23.7.** Universal property 描述对象在某个结构范畴中的唯一性。具体模型构造给出一个实现该性质的对象；不同模型若满足同一初性，则等价。

**解答 23.8.** 可列为：Nisnevich descent、`\mathbb A^1`-invariance、stability、symmetric monoidal structure、base change/projection formula、localization/purity 或相应六操作相干。

**解答 23.9.** `\mathbf{SH}` 已经反演 `X\times\mathbb A^1\to X`。若某理论不满足 `\mathbb A^1`-invariance，则它不能接收来自局部化后的 universal functor，除非先做局部化修正。

**解答 23.10.** 范畴等价只比较两个给定对象。Universal property 还控制从该对象到所有目标的映射空间，因此包含唯一性、自然性和与结构相容的信息。

**解答 23.11.** 若 `A` 和 `B` 都在某范畴中初始，则由初性有唯一态射 `A\to B` 和 `B\to A`。复合 `A\to A` 与恒等同为从初始对象到自身的唯一态射，因此为恒等；同理另一复合为恒等。

## 第二十四章

**解答 24.1.** P0 外部输入是正文主链依赖的标准结果，缺少它会影响理论构造。研究边界是近期或扩展方向，只作说明或开放问题，不支撑主链证明。

**解答 24.2.** Theorem locator 应包含作者、题名、版本或出版信息、章节、定理/命题编号、页码、精确假设、与本书使用语句的差异。

**解答 24.3.** Conceptual closure 表示理论范围和内部逻辑已成形。Publication closure 还要求编号、页码、排版、完整 locator、索引、习题详解和校对全部完成。

**解答 24.4.** 例如选择 realization 保守性问题，其前三章基础包括 Nisnevich descent、`\mathbb A^1`-localization、`T`-stabilization 和 `\mathbf{SH}(S)` 的定义。

**解答 24.5.** 例：Betti realization 在何种基域、完备化或 cellular 子范畴上保守？该问题依赖 realization 与稳定 motivic homotopy、weight 和 torsion 信息的相互作用。

**解答 24.6.** 用定义 24.12 检查任一章：是否有定义、核心命题、证明或外部输入、例子/边界/失败模式、练习和与主线的连接。若只列主题而无这些结构，则仍像大纲。

## 附录 A

**解答 A.1.** Presentability 要求由一集合小对象生成。若生成类是 proper class，functor category、局部化集合和 small object argument 都可能超出固定宇宙，破坏可控性。

**解答 A.2.** 若 `F:C\to D` 是小范畴等价，则预合成给出 `Fun(D^{op},\mathcal S)\to Fun(C^{op},\mathcal S)` 的等价，逆由 quasi-inverse 预合成给出。

**解答 A.3.** 对每个 covering sieve 或 Cech nerve，sheaf condition 要求 `F(X)` 等于相应 diagram 的 limit。把这些 canonical maps 收集为集合 `W`，sheaves 正是 `W`-local objects。

**解答 A.4.** 由于 `\operatorname{Sm}_S` 是小骨架，其对象是集合。每个对象给出一个 `X\times\mathbb A^1\to X`，所以 `W_{\mathbb A^1}` 是集合。

**解答 A.5.** 命题 A.11 说保持 colimits 的 functor 通过局部化因子化当且仅当反演被局部化的态射。代入 `W_{\mathbb A^1}`，realization 必须把所有 `X\times\mathbb A^1\to X` 送为等价。

## 附录 B

**解答 B.1.** Zariski 开覆盖中每个点已在某个开集中出现，残差域不变。开嵌入是 etale，因此 Zariski 覆盖满足 Nisnevich 点提升条件。

**解答 B.2.** 对非平凡 finite separable field extension `L/k`，`Spec L\to Spec k` 是 etale surjective，但唯一点的残差域为 `L`，不等于 `k`，所以不是 Nisnevich 覆盖。

**解答 B.3.** Elementary distinguished square 中 `V'\to V` 是开补对应的拉回，定义上 `V'=U\times_XV`。可直接由 fiber product 的 universal property 验证。

**解答 B.4.** Cech descent 对覆盖 `U\amalg V\to X` 给出 `F(X)` 为 Cech nerve 的 limit。在 elementary square 情形，该 Cech 数据化简为 `F(U)\times_{F(U\times_XV)}F(V)`。

**解答 B.5.** Henselian local schemes 检测 Nisnevich sheaves，因为 Nisnevich topology 的点由 henselian local points 给出。它们把 etale 局部同残差域提升条件转化为 stalk 条件。

## 附录 C

**解答 C.1.** 对 unpointed object `X`，`X_+=X\amalg *`。保持基点的映射 `X_+\to Y` 等价于普通映射 `X\to U(Y)`，故 `(-)_+` 左伴随于遗忘函子。

**解答 C.2.** 对 pointed objects `X,Y`，smash product 是 pushout `X\times * \leftarrow *\times * \to *\times Y` 从 `X\times Y` 中压掉楔和 `X\vee Y`，即 `X\wedge Y=(X\times Y)/(X\vee Y)`。

**解答 C.3.** `\operatorname{cofib}(Y\to *)` 是 pushout `*\amalg_Y*`，这正是 suspension `\Sigma Y` 的定义。

**解答 C.4.** Betti realization 反演 Nisnevich descent、`\mathbb A^1`-equivalences，并把 `T` 送到可逆的拓扑 sphere。由稳定化泛性质，它从 `\mathbf{SH}` 因子化。

**解答 C.5.** Ring spectra 是 symmetric monoidal stable category 中的 algebra objects。若稳定化没有 symmetric monoidal refinement，就无法稳定地谈论乘法、module 和 commutative algebra structures。

## 附录 D

**解答 D.1.** 若 `L\dashv R`，态射 `LX\to Y` 转置为 `X\to RY`，由单位后接 `R`；反向由 `L` 后接余单位。三角恒等式保证双重转置回到原态射。

**解答 D.2.** Ordinary base-change map `g^*f_*\to f'_*g'^*` 由 `g^*f_*\xrightarrow{\eta}f'_*f'^*g^*f_*\simeq f'_*g'^*f^*f_*\xrightarrow{\epsilon}f'_*g'^*` 得到。

**解答 D.3.** Mate construction 由伴随等价的 mapping spaces 给出。若自然变换逐点为 equivalence，则在等价的 mapping spaces 下对应的 mate 也逐点为 equivalence。

**解答 D.4.** 对两个相邻 Cartesian 方块，外矩形的 mate 等于先取右方块 mate 再取左方块 mate 的复合。这由单位、余单位的自然性和三角恒等式化简得到。

**解答 D.5.** 对任意方块通常能构造 exchange natural transformation，但它未必是等价。Beck-Chevalley 条件正是要求该自然变换为等价；存在性和可逆性是不同层级。

## 附录 E

**解答 E.1.** 开嵌入由开子集和结构层限制定义。任意 base change 后，开子集的逆像仍开，结构层仍为限制，因此仍是开嵌入。

**解答 E.2.** 闭嵌入是 affine、finite type、separated，且底层闭映射在任意 base change 后仍闭，因此 universally closed。故闭嵌入 proper。

**解答 E.3.** Etale morphism 局部有限表示且 unramified，故相对微分 `\Omega_{X/Y}=0`。相对切丛是其对偶，因此为零。

**解答 E.4.** Smoothness 可由局部有限表示、flat 和几何纤维光滑或 infinitesimal lifting 表征。这些性质对复合稳定，所以 smooth morphisms 对复合封闭。

**解答 E.5.** Regular immersion 由 ideal sheaf `I` 局部 regular sequence 定义。Conormal sheaf 为 `I/I^2`，normal bundle 为其对偶 `(I/I^2)^\vee`。

## 附录 F

**解答 F.1.** Stable infinity-category 中 cofiber sequence `X\to Y\to Z` 在 homotopy category 中给出 distinguished triangle `X\to Y\to Z\to\Sigma X`。

**解答 F.2.** Exact functor 保持有限极限和有限余极限，特别保持 cofiber sequences。因此它把 distinguished triangle 送到 distinguished triangle。

**解答 F.3.** Mapping spectrum 满足 `\pi_nMap(X,Y)\simeq\pi_0Map(\Sigma^nX,Y)`。右侧就是 homotopy category 中的 `Hom(\Sigma^nX,Y)`。

**解答 F.4.** Homotopy category 只记录 `\pi_0` 层面的 morphisms。Higher coherence 涉及 higher homotopies 和 mapping spaces，仅有同构无法恢复这些数据。

**解答 F.5.** 例如 base change theorem 在 triangulated category 中给出 `g^*f_*\cong f'_*g'^*` 的同构；infinity-categorical 版本还要求这些同构在复合、pasting 和 higher homotopies 下相干。

## 附录 H

**解答 H.1.** `T=\mathbb A^1/(\mathbb A^1\setminus0)`。在 `\mathbf H(S)` 中 `\mathbb A^1\simeq *`，而 `\mathbb A^1\setminus0=\mathbb G_m`，故 `T\simeq */\mathbb G_m\simeq\Sigma\mathbb G_m`。

**解答 H.2.** 把 `\infty:S\hookrightarrow\mathbb P^1` 看作 smooth closed immersion，其 normal bundle 为平凡 line bundle。Homotopy purity 给出 `\mathbb P^1/(\mathbb P^1-\infty)\simeq Th(\mathcal O)\simeq T`。

**解答 H.3.** 闭嵌入 `0\hookrightarrow\mathbb A^2` 的 normal bundle 是平凡秩二向量丛。Homotopy purity 给出 `\mathbb A^2/(\mathbb A^2\setminus0)\simeq Th(\mathcal O^2)\simeq T^{\wedge2}`。

**解答 H.4.** 对 `L/k`，additive transfer 是沿 finite etale map 的加性 pushforward 或 trace。Norm 是乘法性转移，作用于 ring-like data；两者在域上分别对应 trace-like 和 norm-like 操作。

**解答 H.5.** 若 `p:X\to S` smooth proper，则 `E^{a,b}(X)\simeq\pi_0Map_{\mathbf{SH}(S)}(p_\sharp\mathbb 1_X,\Sigma^{a,b}E)`。Proper 和 smooth ambidexterity 可进一步把 `p_\sharp` 与 duality/trace 表达联系起来。
