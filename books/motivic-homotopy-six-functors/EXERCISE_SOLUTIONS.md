# 习题解答要点

本文件给出全书练习的一版解答要点。编号与正文、附录中的练习编号一致。这里的目标是教学可用：给出关键构造、引用位置和证明路线；若作为正式出版详解，可在每题下继续展开。

## 序章

**解答 0.1.** `\operatorname{Sm}_S` 若不取小骨架，其对象类可能是 proper class。Presheaf category、sheafification 和 accessible localization 都需要在小范畴上形成函数范畴和生成集合，因此取小骨架保证 presentability 论证有集合论基础。

**解答 0.2.** 先取 `\operatorname{Sm}_S` 上的 space-valued presheaves，再做 Nisnevich sheafification，得到 `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)`；然后反演所有投影 `X\times\mathbb A^1\to X`，得到 `\mathbf H(S)`。

**解答 0.3.** `f_*` 是 `f^*` 的右伴随，可由 presentability 和保持余极限得到。`f_!` 与 `f^!` 是非常推前和非常拉回，需要 proper、localization、purity、base change 等额外相干结构，不能从 `f^*` 的定义推出。

**解答 0.4.** 例如 “一小集合 maps 生成的 accessible localization 存在”是
外部基础输入（HTT Proposition 5.5.4.15）；命题 2.4 则是把该输入应用到
`W_{\mathbb A^1}` 的书内推论。Morel--Voevodsky homotopy purity 和
motivic 六操作同样是外部输入，但属于专门几何定理。

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

**解答 3.4.** 若 `(A\otimes B)\otimes C\simeq\mathbb 1`，则在 symmetric
monoidal category 中 `B\otimes C` 是 `A` 的双边逆，`C\otimes A` 是 `B`
的双边逆；双边性使用 braiding。故 `T` 可逆确实推出 `S^{1,0}` 与
`\mathbb G_m` 分别可逆。在仅 monoidal、非 braided 的 category 中，
`AB` 可逆一般只给 one-sided inverse 数据；需假设 `A,B` commute 或另证
双边逆。

**解答 3.5.** 若 `\mathcal L` 是被一族生成子检测为零的对象构成的 full subcategory，给定 cofiber sequence `A\to B\to C`，对任一生成子取 mapping spectrum 得 fiber sequence。若其中两个映射谱为零，则第三个也为零，所以 `\mathcal L` 对 cofiber 封闭。

## 第四章

**解答 4.1.** 强对称幺半函子 `f^*` 带有结构等价 `\mathbb 1_X\simeq f^*\mathbb 1_Y` 和 `f^*(A\otimes B)\simeq f^*A\otimes f^*B`。第一条就是单位保持性。

**解答 4.2.** 对 Cartesian 方块，ordinary base-change transformation 在 derived category 中写为 `g^*Rf_* \to Rf'_*g'^*`，由单位 `\operatorname{id}\to Rf'_*f'^*` 和交换同构 `f'^*g^*\simeq g'^*f^*` 组合得到。

**解答 4.3.** Ordinary canonical map 的方向是
`f_*A\otimes B\to f_*(A\otimes f^*B)`：先用强幺半性把其 pullback 写成
`f^*f_*A\otimes f^*B`，再对第一因子用 counit
`f^*f_*A\to A`，最后伴随转置。若 `B` dualizable，命题 4.15 对任意测试
对象连续使用 `B^\vee` 的 duality 和 `f^*\dashv f_*`，证明该 map 为等价。
没有 dualizability 或 properness 时不能无条件断言 ordinary projection
formula。

**解答 4.4.** 对非 proper 态射，ordinary pushforward 和 compact-support pushforward 表示不同几何行为。开嵌入时 `j_!` 是 extension by zero，而 `j_*` 允许边界附近的截面延拓，二者通常不同。

**解答 4.5.** 在 recollement 中，`j^*j_!\simeq\operatorname{id}` 且 `i^*j_!=0`。对 `j_!j^*E\to E` 的 cofiber `C` 作用 `j^*` 得零，故 `C` 位于 `i_*` 本质像；再用 `i^*` 识别为 `i_*i^*E`。

## 第五章

**解答 5.1.** Proper morphism 定义为 separated、finite type、universally closed；这三条对复合稳定。命题 5.7 中需要用到复合仍 proper，才能同时把 `(gf)_!`、`g_!f_!` 与相应 `*`-推前识别。

**解答 5.2.** `j_!` fully faithful 等价于伴随单位 `\operatorname{id}\to j^*j_!` 为等价。对开嵌入有 `j^!\simeq j^*`，所以由 fully faithful 得 `j^*j_!\simeq\operatorname{id}`。

**解答 5.3.** 命题 5.16 使用 localization cofiber sequence
`j_!j^*E\to E\to i_*i^*E`。若 `i^*E\simeq0`，第三项为零；stable
infinity-category 中一条 map 的 cofiber 为零当且仅当该 map 为等价，故
`j_!j^*E\to E` 为等价。

**解答 5.4.** `j^*E\simeq0` 表示 `E` 在开补 `U` 上消失，因此支撑在闭子集 `Z`。`i^*E\simeq0` 表示 `E` 限制到闭子集消失，描述的是远离 `Z` 的条件，二者不是同一支撑。

**解答 5.5.** 对 `E=\Sigma_T^\infty X_+`，命题 5.15 给出 `j_!\mathbb 1_U\to\mathbb 1_X\to i_*\mathbb 1_Z` 的形式。若加入 `X` 的结构态射，可把它理解为开部分、全体和闭补在稳定 motivic homotopy 中的分解。

**解答 5.6.** 定理 5.21 说明 proper `f` 的 `f^!` 保持 filtered colimits，
所以命题 5.20 应用于 `f_!\dashv f^!`，得 `f_!` 保持紧致对象。Properness
只用于 proper comparison `f_*\simeq f_!`；代入后得到 `f_*` 保持紧致对象。

## 第六章

**解答 6.1.** 零向量丛 `0_X` 的总空间是 `X`，去掉零截面后为空集。因此 Thom space `0_X/(0_X-X)` 为 `X/\varnothing\simeq X_+`。

**解答 6.2.** 平凡秩 `r` 向量丛为 `\mathbb A^r_X`，其 Thom space 是 `\mathbb A^r/(\mathbb A^r\setminus0)\wedge X_+`。由 direct sum formula 得 `Th(\mathcal O^r)\simeq T^{\wedge r}\wedge X_+`。

**解答 6.3.** Homotopy purity 比较 `X/(X-Z)` 与 `Z` 的法方向。法丛记录 `Z` 在 `X` 中一阶邻域的横向几何；没有法丛就无法描述闭嵌入附近被压缩后的 Thom twist。

**解答 6.4.** Separated etale morphism 的相对切丛为零，且本书的
exceptional pullback 已定义，所以 smooth purity
`f^!\simeq\Sigma^{T_f}f^*` 化为 `f^!\simeq f^*`。

**解答 6.5.** Thom space 是向量丛的几何商对象，不需要 orientation 才能定义。Orientation 是对 Thom class 或 Thom isomorphism 的选择，使 cohomology 中可把 Thom twist 解开。

## 第七章

**解答 7.1.** Separated etale morphism smooth 且 `T_f=0`，并且 `f_!`
已定义。代入 smooth ambidexterity
`f_!\simeq f_\sharp\Sigma^{-T_f}`，得到 `f_!\simeq f_\sharp`。

**解答 7.2.** 若 `f` smooth proper，则 proper compatibility 给出 `f_!\simeq f_*`，smooth ambidexterity 给出 `f_!\simeq f_\sharp\Sigma^{-T_f}`。合并即得到命题 7.5 的 `f_*\simeq f_\sharp\Sigma^{-T_f}` 型识别。

**解答 7.3.** Dualizable object `A` 有对偶 `A^\vee`、coevaluation `\mathbb 1\to A\otimes A^\vee` 和 evaluation `A^\vee\otimes A\to\mathbb 1`。两个三角恒等式要求 `A` 和 `A^\vee` 经 coevaluation 再 evaluation 的合成分别为恒等。

**解答 7.4.** `u=\operatorname{id}_A` 时，Euler characteristic 是 `\mathbb 1\xrightarrow{coev} A\otimes A^\vee\xrightarrow{\tau}A^\vee\otimes A\xrightarrow{ev}\mathbb 1` 的 trace。

**解答 7.5.** Additive transfer 来自稳定加性结构中的推前/迹，作用于加法或群对象。Norm 是对称幺半乘法转移，保留乘法结构；二者需要 Tambara-like 分配律才能混合。

## 第八章

**解答 8.1.** 从 `g^*f_*` 出发，插入单位 `\operatorname{id}\to f'_*f'^*`，得到 `g^*f_*\to f'_*f'^*g^*f_*`；用方块交换同构把 `f'^*g^*` 换为 `g'^*f^*`；再用余单位 `f^*f_*\to\operatorname{id}` 得 `f'_*g'^*`。

**解答 8.2.** Proper morphism 对 base change 稳定，因为 separated、finite type、universally closed 分别对 base change 稳定。三者合取仍成立。

**解答 8.3.** 设两个 Cartesian 方块横向复合。先对右方块使用 exchange equivalence，再对左方块使用 exchange equivalence；复合相干说明该合成等于外矩形的 exchange transformation。

**解答 8.4.** 对 closed-open pair 的 cofiber sequence 作用 `g^*`。由开嵌入和闭嵌入的 base change 等价，把 `g^*j_!` 和 `g^*i_*` 分别识别为拉回方块中的 `j_{Y!}g_U^*` 和 `i_{Y*}g_Z^*`，得到拉回后的 localization sequence。

**解答 8.5.** 命题 8.16 依次使用 closed adjunction、
`f^*\dashv f_*` 和 `f^*` 强对称幺半，对任意 `A` 直接得到 internal-Hom
等价，所以不需 dualizability。只有进一步写
`\underline{Hom}(A,-)\simeq A^\vee\otimes-` 时，才需要 `A` dualizable；
此后可与命题 8.7 的 ordinary projection formula 比较。

## 第九章

**解答 9.1.** 由定义，`H^{0,0}(S,\mathbb Z)=\pi_0\operatorname{Map}_{\mathbf{SH}(S)}(\mathbb 1_S,H\mathbb Z_S)`，即 `H\mathbb Z` 的全局 0 次 motivic cohomology。

**解答 9.2.** Commutative ring spectrum `E` 有乘法 `E\otimes E\to E`。两个 cohomology class 由映射到 `E` 表示，先 smash 再乘法得到 cup product；交换性和结合性来自 `E` 的 `E_\infty` 结构。

**解答 9.3.** 对 cofiber sequence `A\to B\to C` 应用 `\operatorname{Map}(-,\Sigma^{p,q}E)` 得 fiber sequence。取同伦群得到长正合列；带紧支撑版本同理作用于 `p_!` 表达式。

**解答 9.4.** `CH^n(X)` 是由代数循环定义的经典对象，`H^{2n,n}` 是由 `H\mathbb Z` 在 `\mathbf{SH}` 中表示的 cohomology。二者相等需要 Voevodsky/Bloch 比较定理，不是定义。

**解答 9.5.** 环同态 `\mathbb Z\to\mathbb Z/m` 诱导谱映射 `H\mathbb Z\to H\mathbb Z/m`，再由后合成得到 `H^{a,b}(X,\mathbb Z)\to H^{a,b}(X,\mathbb Z/m)`。

## 第十章

**解答 10.1.** 对 commutative algebra object `A`，自由函子为 `A\otimes -`，遗忘函子忘掉 module action。映射空间等价 `Map_A(A\otimes X,M)\simeq Map(X,U(M))` 给出自由-遗忘伴随。

**解答 10.2.** 取命题 10.4 中 `E=\Sigma_T^\infty X_+`、
`M=\Sigma^{p,q}H\mathbb Z_S`。自由 module
`H\mathbb Z_S\otimes E` 正是 `M_S(X)`，而 `M` 的底层对象仍为
`\Sigma^{p,q}H\mathbb Z_S`。对映射空间取 `\pi_0`，右端就是定义 9.2，故得到
命题 10.6。

**解答 10.3.** Tate motive `\mathbb Z(1)` 对应 motivic 双次数中的 weight shift。一般 `\mathbb Z(q)` 表示第 `q` 个 Tate twist，并与 `S^{2q,q}` 或 `T^q` 的稳定悬挂坐标相连。

**解答 10.4.** 对固定 `E`，若 `A\otimes E` 与 `B\otimes E` 为零，则由张量积
的正合性，任意由 `A,B` 经有限 cofiber、有限直和和 retract 构造的对象与 `E`
张量后仍为零；若 `A\otimes E=0`，则对任意 `C`，
`(C\otimes A)\otimes E\simeq C\otimes(A\otimes E)=0`。因此
`\mathcal I_E` 是 thick tensor ideal。代数对象结构只给出
`\mathbb 1_S\to H\mathbb Z_S` 和乘法，并不给出从有限 cofiber、retract 与张量
构造 `\mathbb 1_S` 的表达；厚生成是额外条件。

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

**解答 16.1.** 对 smoothable lci separated morphism `f:X\to Y`，
fundamental class 在单位对象上的类型是
`\Sigma^{\tau_f}\mathbb 1_X\to f^!\mathbb 1_Y`。其中
`\tau_f=\langle L_f\rangle` 采用 virtual-tangent convention；smooth 时它
等于 `[T_f]`，所以恢复 `f^!\simeq\Sigma^{T_f}f^*`。

**解答 16.2.** `L_f` 是 perfect cotangent complex，但记号
`\langle L_f\rangle` 按资料源表示 associated virtual tangent class。
Smooth 时 `\tau_f=[T_f]`；regular closed immersion 时
`\tau_f=-[N_f]`；若 `f=p\circ i`，则
`\tau_f=i^*[T_p]-[N_i]`。把 dualization 与 K-theory 的 additive inverse
混为一谈会得到错误的 `-[L_f]` 公式。

**解答 16.3.** 恒等态射 `id_X` 的 bivariant group 按定义由 `id_X^!` 或等价的单位 twist 表示。因为 `id^!\simeq id^*`，该 group 退化为 `X` 上的普通 cohomology。

**解答 16.4.** 普通 base change 只比较拉回和推前。DJK 的 excess formula
还要求原 morphism 与拉回 morphism 都是 smoothable lci，并要求
Paragraph 3.3.3 的法丛单射有 locally free cokernel `\xi`；此时才以
`e(\xi)` 修正 Gysin map。任意非 Tor-independent 方块不自动满足这些条件。

**解答 16.5.** Todd class 衡量两个 orientation 或两个 cohomology theories 下 Gysin/Riemann-Roch 变换之间的差异。它是从一种 orientation 转换到另一种 orientation 的校正因子。

**解答 16.6.** 设 `\mathcal I` 是 `Z\hookrightarrow X` 的理想层，并记
`\widetilde{X\times\{0\}}` 为 `X\times\{0\}` 的严格变换。定义

$$
D_ZX=
\operatorname{Bl}_{Z\times\{0\}}(X\times\mathbb A^1)
\setminus\widetilde{X\times\{0\}}
\longrightarrow\mathbb A^1.
$$

它在 `\mathbb G_m` 上是 `X\times\mathbb G_m`，而特殊纤维为
`C_ZX=\operatorname{Spec}_Z(\bigoplus_{n\geq0}\mathcal I^n/
\mathcal I^{n+1})`。若闭嵌入 regular，则 associated graded algebra 规范同构于
`\operatorname{Sym}_{\mathcal O_Z}(\mathcal I/\mathcal I^2)`，故该特殊纤维
规范识别为法丛 `N_{Z/X}` 的总空间。

**解答 16.7.** 在定理 16.13 的适用范围内，`\xi=0` 时 excess Euler class
是单位，公式退化为无修正 base-change compatibility。Tor-independence 保证
`\xi=0`；这里不把反向蕴含当作任意 Cartesian square 上的判据。

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

**解答 20.1.** 第 `i` 个方块是
`\mathcal W_i\to\mathcal V_i`、`\mathcal U_{i-1}\to\mathcal U_i` 组成的
Cartesian Nisnevich 方块。这里
`\mathcal W_i=\mathcal V_i\times_{\mathcal U_i}\mathcal U_{i-1}`，而
`\mathcal V_i\setminus\mathcal W_i\to
\mathcal U_i\setminus\mathcal U_{i-1}` 为同构，所以 `\mathcal V_i` 只在旧开集上
与新片交叠。

**解答 20.2.** 长度为一时，`\mathcal U_0=\varnothing`、
`\mathcal U_1=\mathcal X`。下降给出
`F(\mathcal X)\simeq F(\varnothing)\times_{F(\varnothing)}F(\mathcal V_1)
\simeq F(\mathcal V_1)`，对 `G` 同理；故 `\eta_{\mathcal V_1}` 为等价即推出
`\eta_{\mathcal X}` 为等价。

**解答 20.3.** Khan--Ravi Theorem 7.1 只对可表有限型 `f` 构造
`f_!\dashv f^!`。任意栈态射虽有 `f^*\dashv f_*`，但非可表态射的 exceptional
pair 不在该定理定义域内，必须另引外部结果。

**解答 20.4.** 指标范畴 `\operatorname{Lis}_{\mathcal X}` 的对象是 smooth
`u:U\to\mathcal X`，其中 `U` 为 qcqs algebraic space；态射是与结构映射相容的
smooth-chart 态射。过渡函子是 inverse image，故
`\mathbf{SH}^{\triangleleft}(\mathcal X)=\lim_{(U,u)}\mathbf{SH}(U)`。

**解答 20.5.** Genuine `\mathbf{SH}(BG)` 保留 representation spheres、稳定子与
genuine transfers。Lisse extension 由所有 smooth charts 的 inverse-image 相容族构成，
在 quotient stack 上表现为 Borel 型理论；其 `K`-理论甚至可给出完备化，所以自然比较
函子一般不是等价。

**解答 20.6.** 对 `[Z/G]\hookrightarrow[X/G]`，法丛是带 `G`-线性化的
`N_{Z/X}`。Thom twist 要使用这个 equivariant vector bundle；忘掉线性化会同时忘掉
固定点处的权重和相应 Euler 类。

**解答 20.7.** 基 `S` 必须连通、Noetherian、仿射；
`T=\mathbb G_{m,S}^{\times l}` 为 split torus；`i:Z\hookrightarrow X` 是有限型
`T`-equivariant derived algebraic spaces 的闭浸入；`T` 在 `X\setminus Z` 上无固定点；
系数 `F\in\mathbf{SH}(BT)`。结论还必须在指定 Euler 类局部化以后读取。

**解答 20.8.** 对正权 `r>0`，作用为 `t\cdot x=t^rx`，固定点只有 `x=0`。
Concentration 反演与该 character 对应的 Euler 类 `e(L^{\otimes r})`。

**解答 20.9.** 自然函子只给出比较方向。它可能是 localization、completion 或遗忘
函子；若没有 fully faithfulness 与 essential surjectivity 的额外定理，就不能推出等价。

## 第二十一章

**解答 21.1.** 取 `M_X=\mathcal O_X^\times`，结构映射为包含。则
`\alpha^{-1}(\mathcal O_X^\times)=M_X`，所需映射为恒等；特征幺半群为零。

**解答 21.2.** 在 `X=\operatorname{Spec}k[x,y]` 上，图表
`\mathbb N^2\to k[x,y]` 把 `(1,0)` 送到 `x`、`(0,1)` 送到 `y`。它记录两条分支
`x=0`、`y=0` 及其交点。

**解答 21.3.** 仿射局部看 Frobenius `A\to A`。每个 `a\in A` 满足首一方程
`T^p-a^p=0`，故态射 integral；素理想在 Frobenius 逆像下不变，故谱映射为同胚，
且任意基变换后仍 universally injective。于是它是 universal homeomorphism。

**解答 21.4.** 对非 perfect 域 `k`，Frobenius 在 `K_1(k)=k^\times` 上是
`x\mapsto x^p`，并非满射；所以 integral motivic 等价一般失败。反演 `p` 正是
Elmanto--Khan 定理消除该障碍所需的局部化。

**解答 21.5.** 若 `I^N=0`，每个素理想都含 `I`，所以
`\operatorname{Spec}(A/I)\to\operatorname{Spec}A` 在点集上双射；闭集由根理想决定，
故该双射是同胚。

**解答 21.6.** 对 `X=\operatorname{Spec}A`，概形的 inverse limit 对偶于环的 direct
limit，因此
`X^{\mathrm{perf}}=\operatorname{Spec}(\operatorname{colim}_F A)`。

**解答 21.7.** 若 `X` perfect，则 tower 中每个 `F_X` 都是同构；由同构组成的逆系统
的极限典范同构于任一项，故 `X^{\mathrm{perf}}\simeq X`。

**解答 21.8.** 闭开 localization 记录 `D\hookrightarrow X\hookleftarrow U` 所产生的
recollement。Divisorial log structure 还在对象上记录边界分支的 monoid 与交叉方式；
相同开补可以来自不同边界模型。

**解答 21.9.** 若 integral Frobenius pullback 为等价，则它在由 `KGL` 表示的
`K_1(k)` 上也应为同构；但该映射是 `p` 次幂，非 perfect 时不满，矛盾。

**解答 21.10.** Perfect motivic category 从定义域与同伦关系开始就令 Frobenius
可逆，且其中乘 `p` 已可逆。普通 integral `\mathbf{SH}(X)` 仍可检测非 perfect
Frobenius；二者只有经论文给出的 localization/comparison 才能比较。

## 第二十二章

**解答 22.1.** 先由 Nisnevich descent 通过层化因子化，再由
`X\times\mathbb A^1\to X` 的像为等价通过 `\mathbb A^1`-局部化因子化；最后由
`T` 的像可逆通过 symmetric monoidal `T`-stabilization 因子化。三步泛性质分别给出
唯一性。

**解答 22.2.** Betti realization 把 `T` 送到 pointed homotopy cofiber
`\operatorname{hocofib}(\mathbb C^*\to\mathbb C)`；它是复直线在原点处的 Thom
space，故同伦等价于 `S^2`。而 `\mathbb P^1(\mathbb C)\cong S^2`，所以
`T\simeq(\mathbb P^1,\infty)` 的两个描述在 realization 后一致。

**解答 22.3.** Etale homotopy type 通常只能以 pro-space 捕捉所有 finite etale covers 和 cohomological approximations。单个 space 往往不能保留这些 inverse system 信息。

**解答 22.4.** 对实概形 `X=\operatorname{Spec}\mathbb C`，`X(\mathbb R)=\varnothing`；
但 `X(\mathbb C)` 有两个点，复共轭交换它们。因此 real-points realization 与
`C_2`-equivariant Betti realization 不同。

**解答 22.5.** `S` 必须 Noetherian 且有限维；局部化元素是由 `-1` 给出的
`\rho:\mathbb 1\to\mathbb G_m`；结论是
`\mathbf{SH}(S)[\rho^{-1}]\simeq\mathbf{SH}(S_{\mathrm{ret}})`，不能删去
`\rho`-局部化。

**解答 22.6.** Pullback 相容只给出
`R_Xf^*\simeq(f^{\mathrm{an}})^*R_Y`。要对 `f_!` 取 mate，仍需 realization
保持相应伴随并控制开浸入延零、proper compactification 与 base change；仅有两边各自
存在伴随不足以证明 mate 可逆。

**解答 22.7.** 对 motivic localization sequence 施加正合 `R`，再用它与
`j_!,j^*,i_*,i^*` 的四个相容等价逐项替换。正合性保持 cofiber，所得正是目标范畴的
localization sequence。

## 第二十三章

**解答 23.1.** Coefficient system 是把每个基对象 `S` 赋予一个稳定 presentable symmetric monoidal category，并对态射给出 pullback/pushforward 等结构且满足相干公理的系统。

**解答 23.2.** Universal property 表示在 `\operatorname{CoSy}^{c}_B` 中，从
`\mathbf{SH}` 到任意对象的 morphism space 可缩。唯一性只针对该 ambient category
编码的 coefficient-system 结构，不自动包括定义之外的全部六操作相容性。

**解答 23.3.** Pullback formalism 主要控制 `f^*` 和其相干。完整六操作还需要 `f_*`、`f_!`、`f^!`、projection formula、base change、localization、purity 等额外结构。

**解答 23.4.** 一个 functor 可与 pullback commute，即 `R_Xf^*\simeq f^*R_Y`；但它未必与 `f_!` 或 `f^!` commute。后者是 six-operation compatibility，强得多。

**解答 23.5.** 2025-2026 pullback formalism 结果较新，假设、适用几何类别和与既有六操作的关系仍需逐项核查。因此本书把它们列为研究边界，而非基础外部输入。

**解答 23.6.** 对两个保持余极限的正合函子之间的自然变换，令 `\mathcal L` 为其分量
是等价的对象。`\mathcal L` 是 localizing subcategory；若它包含 smooth generators，
生成假设给出 `\mathcal L` 等于整个纤维范畴。这个论证检测自然变换，不凭空构造函子。

**解答 23.7.** Universal property 描述对象在某个结构范畴中的唯一性。具体模型构造给出一个实现该性质的对象；不同模型若满足同一初性，则等价。

**解答 23.8.** 三层分别是：由 Nisnevich descent 与闭开 localization 组成的局部性；
由 `\mathbb A^1`-invariance 与 `T`-stability 组成的 motivic 不变性；以及 monoidal、
smooth pushforward、base change、projection formula 等系数结构。初性只保持 ambient
category 实际编码的部分。

**解答 23.9.** `\mathbf{SH}` 已经反演 `X\times\mathbb A^1\to X`。若某理论不满足 `\mathbb A^1`-invariance，则它不能接收来自局部化后的 universal functor，除非先做局部化修正。

**解答 23.10.** 底层范畴等价不必保持 pullback、monoidal product 或 localization。
结构化 universal property 还控制到所有目标的结构保持映射空间；但其额外内容严格受
ambient category 限制，不能自动扩张到未编码的全部六操作。

**解答 23.11.** 若 `A` 和 `B` 都在某范畴中初始，则由初性有唯一态射 `A\to B` 和 `B\to A`。复合 `A\to A` 与恒等同为从初始对象到自身的唯一态射，因此为恒等；同理另一复合为恒等。

## 第二十四章

**解答 24.1.** 默认五元组为：有限型 `B`-概形组成的几何范畴
`\mathcal C`；光滑站点上的 Nisnevich 拓扑 `\tau`；区间
`I=\mathbb A^1`；悬挂坐标
`T=\mathbb A^1/(\mathbb A^1-0)`；以及 separated finite-type 态射类
`\mathcal E`。最后一项控制 exceptional 伴随 `f_!\dashv f^!` 的定义域。

**解答 24.2.** `I`-局部化按定义反演所有 `X\times I\to X`。若一个预层级
函子经该局部化因子化，则这些态射的像必须为等价。取 `I=\mathbb A^1`，这正要求
目标中 `u(X\times\mathbb A^1)\to u(X)` 为等价；若 `u` 保持乘积，可改写为
`u(X)\times u(\mathbb A^1)\to u(X)`，故 `u(\mathbb A^1)` 的可缩性给出该条件。

**解答 24.3.** 取只有一条非恒等箭头 `f:X\to Y` 的基范畴，并令四个纤维范畴都
等于某个稳定范畴 `\mathcal C`。令第一套 pullback 为恒等函子，第二套 pullback
为一个不与恒等自然等价的自等价 `\Phi`。逐纤维取 `R_X=R_Y=\operatorname{id}`
都是范畴等价，但 pullback 相容要求
`R_Xf^*=\operatorname{id}\simeq\Phi=f'^*R_Y`，这并不存在。因此它不是
coefficient-system morphism，更不是六操作比较。

**解答 24.4.** 对 `p:\coprod_{r=1}^dS\to S`，有
`\mathbf{SH}(\coprod S)\simeq\mathbf{SH}(S)^d`。普通推前为
`p_*(E_1,\ldots,E_d)=\prod_rE_r\simeq\bigoplus_rE_r`，norm 为
`p_\otimes(E_1,\ldots,E_d)=\bigotimes_rE_r`。取 `d=2`、`E_1=0`、
`E_2=\mathbb 1`，前者为 `\mathbb 1`，后者为零，故加性推前不能形式决定 norm。

**解答 24.5.** 正合局部化把 `a` 的 cofiber 送到 `L(a)` 的 cofiber；因此
`L(a)` 为等价恰好说明 `L(\operatorname{cofib}(a))=0`。在 `D(\mathbb Z)` 中，
非零对象 `\mathbb Z/p` 满足
`\mathbb Z/p\otimes_{\mathbb Z}^{\mathbf L}\mathbb Z[1/p]\simeq0`，说明反演
`p` 的局部化有非零核。

**解答 24.6.** 保持余极限只保证 `R` 把定义 slice 的 cofiber 送到 cofiber；它
并不给出 `R(f_qE)\simeq f'_qR(E)`。缺少的是 `R` 与右伴随 `r_q` 的交换，亦即
命题 24.12(2)。只有先识别 cofiber sequence 的前两项，才能识别第三项。

**解答 24.7.** 以问题 24.17 为例：已知输入是某个明确 target 和系数下的
realization `R`；所求对象是 `\mathbf{SH}(S)` 的一个生成子明确的 full
subcategory `\mathcal C`；需要验证的相干图包括 `R` 与悬挂、cofiber、张量和所用
六操作的交换图；预期输出是 `R|_{\mathcal C}` 反射等价并给出其 kernel 的生成族。

**解答 24.8.** Perfectization 只在正特征语境中定义；它与普通 Nisnevich 下降、
`\mathbb A^1` 及 Tate 球的比较需要 universal-homeomorphism localization 和
`p` 可逆性的外部定理。复点函子要求复基；它保持有限纤维积，Nisnevich 覆盖给出
拓扑局部粘合，`\mathbb A^1(\mathbb C)=\mathbb C` 可缩，而 Tate 球实现为稳定可逆
的拓扑球。前一构造的关键障碍是 integral `p`-primary 信息，后一构造的关键障碍是
紧支撑与全部六操作相容，而不是前三项局部化本身。

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

**解答 C.3.** `\operatorname{cofib}(Y\to *)` 是 pushout
`*\amalg_Y*`，这正是 suspension `\Sigma Y` 的定义。又因
`S^1\simeq\operatorname{cofib}(S^0\to*)` 且 `-\wedge Y` 保持余极限，

$$
S^1\wedge Y\simeq
\operatorname{cofib}(S^0\wedge Y\to *\wedge Y)
\simeq\operatorname{cofib}(Y\to*)\simeq\Sigma Y.
$$

**解答 C.4.** 若 `A\otimes B` 的逆为 `C`，则 `B\otimes C` 是 `A` 的逆，
因为 `A\otimes(B\otimes C)\simeq\mathbb 1`，并由 symmetry 得另一侧；
`C\otimes A` 同理是 `B` 的逆。所有重排由 symmetric monoidal coherence
给出。

**解答 C.5.** Betti realization 满足 Nisnevich descent 与
`\mathbb A^1`-invariance，并把 `T` 送到可逆拓扑 sphere；由
`\operatorname{Fun}^{L,\otimes}` 中的反演泛性质，它从 `\mathbf{SH}`
因子化。要搬运 commutative ring spectra，该因子化必须 symmetric
monoidal；裸函子只搬运对象和 maps，不能自动搬运 coherent multiplication。

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

**解答 F.6.** 对 sequence `X_0\to X_1\to\cdots`，令 `s` 为
`\bigoplus_nX_n` 上由结构 maps 移到后一 summand 的 shift，则
`\operatorname*{colim}_nX_n\simeq\operatorname{cofib}(1-s)`。Exact functor
保持该 cofiber；若它还保持 coproducts，就保持公式两端，因而保持 sequential
colimit。Simplicial replacement 的 skeletal filtration 再把一般小 colimit
归约到 coproduct、有限 cofiber 和这种 sequential colimit。

## 附录 H

**解答 H.1.** `T=\mathbb A^1/(\mathbb A^1\setminus0)`。在 `\mathbf H(S)` 中 `\mathbb A^1\simeq *`，而 `\mathbb A^1\setminus0=\mathbb G_m`，故 `T\simeq */\mathbb G_m\simeq\Sigma\mathbb G_m`。

**解答 H.2.** 把 `\infty:S\hookrightarrow\mathbb P^1` 看作 smooth closed immersion，其 normal bundle 为平凡 line bundle。Homotopy purity 给出 `\mathbb P^1/(\mathbb P^1-\infty)\simeq Th(\mathcal O)\simeq T`。

**解答 H.3.** 闭嵌入 `0\hookrightarrow\mathbb A^2` 的 normal bundle 是平凡秩二向量丛。Homotopy purity 给出 `\mathbb A^2/(\mathbb A^2\setminus0)\simeq Th(\mathcal O^2)\simeq T^{\wedge2}`。

**解答 H.4.** 对 `L/k`，additive transfer 是沿 finite etale map 的加性 pushforward 或 trace。Norm 是乘法性转移，作用于 ring-like data；两者在域上分别对应 trace-like 和 norm-like 操作。

**解答 H.5.** 若 `p:X\to S` smooth proper，则 `E^{a,b}(X)\simeq\pi_0Map_{\mathbf{SH}(S)}(p_\sharp\mathbb 1_X,\Sigma^{a,b}E)`。Proper 和 smooth ambidexterity 可进一步把 `p_\sharp` 与 duality/trace 表达联系起来。
