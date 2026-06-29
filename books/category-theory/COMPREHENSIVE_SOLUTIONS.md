# 综合习题答案

本文件给出 [COMPREHENSIVE_EXERCISES.md](COMPREHENSIVE_EXERCISES.md) 的参考答案。

## 综合题 1

1. 对有限图形 $D:\mathcal J\to\mathcal C$，令
   $$
   P=\prod_{j\in\operatorname{Ob}\mathcal J}D(j),\qquad
   Q=\prod_{\alpha:j\to k}D(k).
   $$
   两个态射 $s,t:P\rightrightarrows Q$ 的 $\alpha:j\to k$ 分量分别为 $D(\alpha)p_j$ 与 $p_k$。取等化子 $E\to P$，则 $E$ 是极限。
2. 若 $\theta:D\Rightarrow D'$ 是图形自然变换，由积泛性质得到 $P\to P'$ 与 $Q\to Q'$，并与 $s,t$ 相容；等化子泛性质给出 $E\to E'$。这就是极限对图形的函子性。
3. Pullback 是图形 $A\to C\leftarrow B$ 的极限；按上述构造，它是 $A\times B$ 上两个态射 $f p_A,g p_B:A\times B\rightrightarrows C$ 的等化子。

## 综合题 2

1. 伴随是对 $X,Y$ 自然的双射
   $$
   \mathcal D(FX,Y)\cong\mathcal C(X,GY).
   $$
2. 单位 $\eta_X:X\to GFX$ 是 $\operatorname{id}_{FX}$ 的转置；余单位 $\varepsilon_Y:FGY\to Y$ 是 $\operatorname{id}_{GY}$ 的逆转置。
3. 若 $q:B\to Q$ 是 $f,g:A\rightrightarrows B$ 的余等化子，则对任意 $Y$，
   $$
   \mathcal D(FQ,Y)\cong\mathcal C(Q,GY)
   $$
   是等化 $\mathcal C(B,GY)\rightrightarrows\mathcal C(A,GY)$ 的集合；再用伴随改写为等化 $\mathcal D(FB,Y)\rightrightarrows\mathcal D(FA,Y)$。故 $FQ$ 表示余等化子。
4. 张量积 $M\otimes_R-$ 是左伴随于 $\operatorname{Hom}(M,-)$，故保持余等化子和 cokernel，因此右正合。

## 综合题 3

1. Yoneda 给
   $$
   \operatorname{Nat}(yA,yB)\cong yB(A)=\mathcal C(A,B),
   $$
   所以 $y$ 完全忠实。
2. 预层密度定理：
   $$
   P\cong\operatorname{colim}_{(C,x)\in\int P}yC.
   $$
   在 $A$ 处，元素由 $(C,x,f:A\to C)$ 表示，映到 $P(f)(x)$。
3. co-Yoneda 公式为
   $$
   P\cong\int^{C}P(C)\times yC.
   $$

## 综合题 4

1. 富自然变换对象：
   $$
   \operatorname{Nat}_{\mathcal V}(F,G)=\int_A\mathcal B(F A,G A).
   $$
   当 $\mathcal B=\mathcal V$ 时为 $\int_A[F A,G A]$。
2. 对预层 $F:\mathcal A^{op}\to\mathcal V$，
   $$
   \operatorname{Fun}_{\mathcal V}(\mathcal A^{op},\mathcal V)(\mathcal A(-,A),F)
   =
   \int_B[\mathcal A(B,A),F(B)].
   $$
   由闭结构，映入该 end 等价于族 $X\otimes\mathcal A(B,A)\to F(B)$。dinatural 性使该族唯一由 $X\to F(A)$ 决定，故 end 同构于 $F(A)$。
3. 取 $\mathcal V=\mathbf{Ab}$，得到
   $$
   \operatorname{Nat}_{add}(\mathcal A(-,A),F)\cong F(A),
   $$
   且为阿贝尔群同构。

## 综合题 5

1. Day 卷积：
   $$
   (P\star Q)(c)=\int^{a,b}P(a)\times Q(b)\times\mathcal C(c,a\otimes b).
   $$
2. 代入 $P=ya,Q=yb$：
   $$
   \int^{u,v}\mathcal C(u,a)\times\mathcal C(v,b)\times\mathcal C(c,u\otimes v)
   \cong\mathcal C(c,a\otimes b)
   $$
   两次 co-Yoneda 即得。
3. Day 卷积的结合约束、单位约束和五边形/三角相干涉及 coend 的 Fubini、幺半结构相干和多个自然同构的兼容，通常引用 Day-Kelly 的完整定理。

## 综合题 6

1. 对覆盖筛 $S\subset yU$，sheaf 条件为
   $$
   F(U)\to\operatorname{Nat}(S,F)
   $$
   是双射。
2. 覆盖族 $\{U_i\to U\}$ 生成筛时，匹配族为 $(s_i)$ 且在 $U_i\times_UU_j$ 上相等，所以得到等化子
   $$
   F(U)\to\prod_iF(U_i)\rightrightarrows\prod_{i,j}F(U_i\times_UU_j).
   $$
3. subcanonical 意味所有 $yU$ 是 sheaf，因此 Yoneda 嵌入落入 $\operatorname{Sh}(\mathcal C,J)$。
4. $\infty$-sheaf 条件把等化子升级为 Čech nerve 的同伦极限，记录高阶相容同伦。

## 综合题 7

1. $N(\mathcal C)$ 的内角填充由普通范畴复合给出，且唯一；见附录 B。
2. $hN(\mathcal C)\cong\mathcal C$。
3. 图形 $p:K\to N(\mathcal C)$ 的 slice $N(\mathcal C)_{/p}$ 的对象是锥；终对象即普通极限锥。

## 综合题 8

1. Kan complex 要求所有 horn 填充；quasi-category 要求 inner horn 填充；ordinary category nerve 的 inner horn 填充唯一。
2. 外 horn 填充给每条边构造左右同伦逆，因此每条边在同伦范畴中为同构。
3. 若 $\mathcal C$ 有非可逆态射，则 $N(\mathcal C)$ 中对应边不应有外 horn 逆填充，所以通常不是 Kan complex。

## 综合题 9

1. 边 $e:x\to y$ Cartesian 若
   $$
   \operatorname{Map}_X(z,x)\to
   \operatorname{Map}_X(z,y)\times_{\operatorname{Map}_S(pz,py)}
   \operatorname{Map}_S(pz,px)
   $$
   是同伦拉回。
2. 普通 fibration 中 Cartesian lift 表示任意箭头经它唯一分解；映射空间条件是该唯一分解的同伦版本。
3. Straightening/unstraightening：
   $$
   \operatorname{Fun}(S^{op},\mathcal{Cat}_\infty)\simeq\operatorname{CartFib}_{/S}.
   $$
4. Kan 延拓点态公式在 slice 上取极限/余极限；straightening 把随基点变化的纤维组织成 functor，因此使参数化 Kan 延拓可表述。

## 综合题 10

1. 稳定 $\infty$-范畴 pointed，有限极限和有限余极限存在，且方块推出当且仅当拉回。
2. 在稳定环境中，cofiber square 也是 fiber square，因此纤维和余纤维通过悬挂/环路互相转换。
3. distinguished triangles 来自余纤维序列 $X\to Y\to\operatorname{cofib}(f)\to\Sigma X$；稳定性给出三角公理。
4. 三角范畴只记录 $hC$ 和三角；稳定 $\infty$-范畴还记录映射空间/映射谱和高阶相干。

## 综合题 11

例选：

1. Mac Lane 相干性：用于省略幺半括号并保证典范图交换；若不可用，第八章和 Day 卷积相干性必须全程保留括号路径。
2. Giraud 定理：用于 Grothendieck topos 的内在刻画；若不可用，第十四章只能使用“sheaf 范畴”外在定义。
3. Straightening/unstraightening：用于把 Cartesian fibrations 等价为 $\infty$-函子；若不可用，第十九章只能保留 fibration 语言，不能自由转换为 functor language。

## 综合题 12

1. 普通工具：Yoneda、极限/余极限、伴随、Kan 延拓、单子、幺半与富范畴、可表现范畴。
2. 过渡技术：模型范畴、simplicial localization、quasi-category、Joyal 模型结构、marked simplicial sets、Cartesian fibration。
3. $E_n$-代数涉及多输入运算和全部高阶同伦相干；普通 operad 只能编码严格复合，$\infty$-operad 才能编码相干同伦层级。

## 综合题 13

1. 预层范畴余极限逐点计算。由预层密度定理，
   $$
   P\cong\operatorname{colim}_{(C,x)\in\int P}yC.
   $$
   可表预层满足
   $$
   \widehat{\mathcal C}(yC,\operatorname{colim}_jP_j)
   \cong
   \operatorname{colim}_j\widehat{\mathcal C}(yC,P_j),
   $$
   因为左边等于逐点值 $(\operatorname{colim}_jP_j)(C)$。因此 $\widehat{\mathcal C}$ 由小的紧对象族经余极限生成，是局部可表现范畴。
2. 若态射 $u:P\to Q$ 在所有 $yC$ 上诱导双射，则 $u_C:P(C)\to Q(C)$ 对所有 $C$ 为双射，故 $u$ 是同构。由 Yoneda，映射出 $yC$ 正是取 $C$ 点值。
3. 若 $\mathcal E$ 局部可表现，且 $L$ 保持小余极限并可达，则外部输入定理 12.D 给出 $L$ 有右伴随。

## 综合题 14

1. $R\text{-}\mathbf{Mod}$ 是阿贝尔范畴，所有小余极限存在且逐底层集合构造。滤过余极限与有限极限相容，故短正合列经滤过余极限仍短正合。左正则模 $R$ 是生成元，因为 $m\in M$ 对应态射 $R\to M,\ 1\mapsto m$。
2. 对 $f:M\to N$，
   $$
   \operatorname{coim}(f)=M/\ker(f),\qquad
   \operatorname{im}(f)=f(M)\subseteq N.
   $$
   第一同构定理给出 $M/\ker(f)\cong f(M)$。
3. Gabriel-Popescu 说带生成元的 Grothendieck 范畴可由某个模范畴经正合局部化得到。模范畴提供“自由表示”，局部化施加该 Grothendieck 范畴中的关系。

## 综合题 15

1. Mapping space 口径要求对 $x\in C,y\in D$ 有自然等价
   $$
   \operatorname{Map}_D(Fx,y)\simeq\operatorname{Map}_C(x,Gy).
   $$
2. Correspondence 口径取 $H:C^{op}\times D\to\mathcal S$，要求
   $$
   H(x,y)\simeq\operatorname{Map}_D(Fx,y)
   \simeq\operatorname{Map}_C(x,Gy).
   $$
3. Walking adjunction/scaled nerve 口径包含两个对象、两个 $1$-态射 $F,G$、单位 $\eta:\operatorname{id}\to GF$、余单位 $\varepsilon:FG\to\operatorname{id}$，以及三角恒等式的相干 $2$-维数据。
4. 普通伴随给出 Hom 自然同构，也给出 correspondence 的集合值表示性；单位余单位满足严格三角恒等式，因此诱导 $\operatorname{Adj}\to\mathbf{Cat}$ 的严格 $2$-函子，再嵌入 $\mathcal{Cat}_\infty$。

## 综合题 16

1. 对 $y\in X_t$ 选择覆盖 $\alpha:s\to t$ 的 Cartesian lift
   $$
   \alpha^*y\to y.
   $$
   Cartesian lift 的可缩选择空间给出函子 $\alpha^*:X_t\to X_s$。
2. 对 $z\in X_u$，复合
   $$
   \alpha^*\beta^*z\to\beta^*z\to z
   $$
   覆盖 $\beta\alpha$。Cartesian 边在复合下保持 Cartesian，因此它与 $(\beta\alpha)^*z\to z$ 由同一泛性质唯一到等价地比较，得到自然等价。
3. Straightening 下有
   $$
   \operatorname{Sect}^{Cart}_S(X)\simeq\lim_{s\in S^{op}}F(s).
   $$
4. Descent data 是在各局部对象上选对象，并在交叠、高重交叠上给相容等价。Cartesian fibration 把这些局部范畴组织为随基变化的范畴族，Cartesian sections 正是同伦相干的相容选择。

## 综合题 17

1. 对 heart 中 $f:A\to B$，
   $$
   \ker(f)=H^0(\operatorname{fib}(f)),\qquad
   \operatorname{coker}(f)=H^0(\operatorname{cofib}(f)).
   $$
   纤维/余纤维先在稳定 $\infty$-范畴中构造，再用 $H^0$ 投回 heart。
2. 对有限滤过 $F_\bullet X$，
   $$
   E_1^{p,q}=H^{p+q}(\operatorname{gr}_pX),
   $$
   并强收敛到 $H^{p+q}(X)$ 上诱导滤过的 associated graded：
   $$
   E_\infty^{p,q}\cong\operatorname{gr}_pH^{p+q}(X).
   $$
3. 令 $V=k^n$、$A=\operatorname{End}_k(V)=M_n(k)$。取双模 ${}_AV_k$ 与 ${}_kV^*_A$。映射
   $$
   V\otimes_kV^*\to A,\quad v\otimes\varphi\mapsto(w\mapsto v\varphi(w))
   $$
   是 $(A,A)$-双模同构；评价
   $$
   V^*\otimes_AV\to k,\quad \varphi\otimes v\mapsto\varphi(v)
   $$
   是 $(k,k)$-双模同构。因此 $A$ 与 $k$ Morita 等价。
4. Extended TFT 需要给点、边、曲面及更高 bordism 都赋值。赋给点的对象必须有对偶，而且评价/余评价及其更高层态射也要有伴随，才能对所有带边界和角的 bordism 作相容赋值；这正是 fully dualizable 条件。

## 综合题 18

1. 对滤过图形 $F_i$，
   $$
   \operatorname{Map}_{\mathcal P(C)}(j(c),\operatorname{colim}_iF_i)
   \simeq
   (\operatorname{colim}_iF_i)(c)
   \simeq
   \operatorname{colim}_iF_i(c)
   \simeq
   \operatorname{colim}_i\operatorname{Map}_{\mathcal P(C)}(j(c),F_i).
   $$
   第一和第三步是 $\infty$-Yoneda，中间一步是预层余极限逐点计算。
2. Sheaf 化把预层局部化到满足覆盖下降的局部对象；Bousfield localization 把对象局部化到对指定态射族 $S$ 满足映射空间等价的局部对象。二者都是由“局部对象”刻画的 accessible localization。
3. 若 $C,D$ presentable，外部输入定理 23.11 说 $F:C\to D$ 是左伴随当且仅当它保持所有小余极限。因此保持小余极限已经给出右伴随的存在。
4. $\operatorname{Pr}^L$ 的张量积把“分别保持余极限的双变量函子”表示为单变量左伴随。高阶代数中的 presentable 幺半 $\infty$-category、代数对象和模范畴都要求张量积与余极限相容，这正由 $\operatorname{Pr}^L$ 的幺半结构组织。

## 综合题 19

1. Profunctor $P:\mathcal C\nrightarrow\mathcal D$ 是函子
   $$
   P:\mathcal C^{op}\times\mathcal D\to\mathbf{Set}.
   $$
   若 $Q:\mathcal D\nrightarrow\mathcal E$，则
   $$
   (Q\circ P)(c,e)=\int^{d\in\mathcal D}P(c,d)\times Q(d,e).
   $$
2. 右单位在 $(c,d)$ 处为
   $$
   \int^{d'}P(c,d')\times\mathcal D(d',d)\cong P(c,d),
   $$
   由 co-Yoneda 得到。左单位同理。
3. 函子 $F:\mathcal C\to\mathcal D$ 给出
   $$
   F_*(c,d)=\mathcal D(Fc,d),\qquad
   F^*(d,c)=\mathcal D(d,Fc).
   $$
   单位来自 $\mathcal C(c,c')\to\mathcal D(Fc,Fc')$，余单位来自 $\mathcal D$ 中复合。
4. Profunctor 复合把中间对象 $d$ 通过 coend 商掉；双模复合 $M\otimes_BN$ 把中间代数 $B$ 的左右作用平衡掉。二者都是“沿中间变量作同伦/代数平衡合成”的形式。

## 综合题 20

1. 富 profunctor 为富函子
   $$
   M:\mathcal A^{op}\otimes\mathcal B\to\mathcal V.
   $$
   若 $N:\mathcal B\nrightarrow\mathcal C$，则
   $$
   (N\circ M)(a,c)=\int^{b\in\mathcal B}M(a,b)\otimes N(b,c).
   $$
2. 富函子 $F:\mathcal A\to\mathcal B$ 给出
   $$
   F_*(a,b)=\mathcal B(Fa,b),\qquad
   F^*(b,a)=\mathcal B(b,Fa).
   $$
   它们把垂直态射 $F$ 表示为水平广义态射。
3. 对拉回方块 $X'\cong Y'\times_YX$ 和对象 $E\to X$，先推前再拉回得到 $Y'\times_YE$；先拉回到 $X'$ 再推前得到 $X'\times_XE$。由拉回同构，两者自然同构。
4. Indexed category 是伪函子 $B^{op}\to\mathbf{Cat}$；Grothendieck construction 把它变成 fibration；Cartesian fibration 是该事实的 $\infty$-版本。Equipment 在这些纤维和重索引函子之外，还允许 profunctors 作为水平态射，并用二重胞腔表达 base change。

## 综合题 21

1. 稳定 presentable $\infty$-范畴 $C$ compactly generated，若存在一小集 compact objects $\mathcal G$，使 $X=0$ 当且仅当所有 $\pi_0\operatorname{Map}(\Sigma^nG,X)$ 为零。等价地，$\mathcal G$ 生成的 localizing subcategory 是整个 $C$。
2. Brown 表示性说合适的 cohomological functor 可表示；对函子 $F:C\to D$，固定 $Y\in D$ 后 $X\mapsto\operatorname{Map}_D(FX,Y)$ 若满足 Brown 条件，就由某个 $G(Y)$ 表示，从而构造右伴随 $G$。
3. 取局部化单位 $\eta_X:X\to LX$，令 $A_X=\operatorname{fib}(\eta_X)$。因 $LX\to L^2X$ 是等价，应用 $L$ 后得 $LA_X=0$，故 $A_X$ acyclic，并有余纤维序列 $A_X\to X\to LX$。
4. 普通 Bousfield localization 是 exact accessible localization。Smashing localization 还要求存在 $E$ 使 $LX\simeq E\otimes X$；因此它由张量控制，自动保持小余极限，并与幺半结构强相容。

## 综合题 22

1. Small dg category 是 $\operatorname{Ch}(k)$-富范畴，有对象集、Hom 复形 $\mathcal A(x,y)$、链映射复合和单位。定义
   $$
   H^0(\mathcal A)(x,y)=H^0(\mathcal A(x,y)).
   $$
   因复合是链映射，它下降到 $0$ 次同调，给出普通范畴。
2. 由 dg Yoneda，
   $$
   \operatorname{Map}_{D(\mathcal A)}(h_a,M)\simeq M(a).
   $$
   对滤过图形 $M_i$，导出模范畴余极限逐点计算，所以
   $$
   \operatorname{Map}(h_a,\operatorname{colim}_iM_i)
   \simeq
   \operatorname{colim}_i\operatorname{Map}(h_a,M_i).
   $$
   因此 $h_a$ compact。
3. Quasi-equivalence 要求 Hom 复形 quasi-isomorphism 且 $H^0$ 本质满，因而是链级的强等价。Morita equivalence 要求诱导 $D(\mathcal B)\to D(\mathcal A)$ 等价。前者推出后者；后者允许加入锥、悬挂、retract 和 perfect modules，所以更粗。
4. 把 $A$ 看成单对象 dg category，则
   $$
   HH(A)\simeq A\otimes^{\mathbb L}_{A^{op}\otimes A}A.
   $$
   Morita 不变性说若 $A$ 与 $B$ 的导出模范畴等价，则相应 Hochschild 型不变量等价；它依赖的是模块理论而不是代数的逐元素呈现。

## 综合题 23

1. 稳定系数系统是
   $$
   \mathcal D:\mathcal B^{op}\to\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}}).
   $$
   六操作额外给出 $f^*,f_*,f_!,f^!$、张量、内 Hom、基变换、投影公式、proper compatibility、recollement 和对偶性相干。
2. 对方块
   $$
   \begin{array}{c}
   X'\xrightarrow{g'}X\\
   \downarrow f'\quad\downarrow f\\
   Y'\xrightarrow{g}Y
   \end{array}
   $$
   用单位得
   $$
   g^*f_*\to f'_*f'^*g^*f_*.
   $$
   由方块相干 $f'^*g^*\simeq g'^*f^*$，再用余单位 $f^*f_*\to\operatorname{id}$，得到
   $$
   g^*f_*\to f'_*g'^*.
   $$
3. 若 $f:X\to Y$、$g:Y\to Z$ 都满足投影公式，则
   $$
   (gf)_!(A\otimes f^*g^*C)
   \simeq g_!f_!(A\otimes f^*g^*C)
   \simeq g_!(f_!A\otimes g^*C)
   \simeq g_!f_!A\otimes C
   \simeq (gf)_!A\otimes C.
   $$
4. 若 $j^*K=0$ 且 $i^*K=0$，则 recollement 余纤维序列
   $$
   j_!j^*K\to K\to i_*i^*K
   $$
   两端为零，故 $K=0$。因此 $j^*$ 与 $i^*$ 联合保守。
5. 若 $K$ dualizable，则
   $$
   \underline{\operatorname{Hom}}_X(K,M)\simeq K^\vee\otimes_XM.
   $$
   令 $M=\omega_X$，得到
   $$
   \mathbb D_X(K)=\underline{\operatorname{Hom}}_X(K,\omega_X)\simeq K^\vee\otimes_X\omega_X.
   $$

## 综合题 24

1. Relative category 是 $(\mathcal C,W)$，其中 $W$ 是指定 weak equivalences 的宽子范畴。$\infty$-categorical localization 是
   $$
   N\mathcal C\to\mathcal C[W^{-1}]
   $$
   把 $W$ 送为等价，并且对任意 $D$，从 $\mathcal C[W^{-1}]$ 到 $D$ 的函子等价于从 $N\mathcal C$ 到 $D$ 且把 $W$ 送为等价的函子。
2. 若 $L$ 与 $L'$ 都满足该泛性质，则分别取目标为对方得到 $L\to L'$ 和 $L'\to L$。两个复合预合成到 $N\mathcal C$ 后等于恒等；由全忠实性，复合等价于恒等。因此 $L\simeq L'$。
3. Dwyer-Kan equivalence 是映射单纯集弱等价且同伦范畴本质满的 simplicial functor。映射空间弱等价给出 $\pi_0$ Hom 集同构，所以同伦范畴完全忠实；再加本质满，诱导同伦范畴等价。
4. 若 $\mathcal M$ 是 simplicial model category，$x$ cofibrant、$y$ fibrant，则
   $$
   \operatorname{Map}_{\mathcal M_\infty}(x,y)\simeq\operatorname{Map}_{\mathcal M}(x,y).
   $$
   一般模型范畴先作 cofibrant-fibrant replacement，再用 cosimplicial/simplicial resolutions 或 hammock localization 计算。
5. Quasi-category 适合直接做 $\infty$-范畴内部构造；simplicial category 适合显式映射空间和富化；complete Segal space 适合把对象空间和态射空间分层呈现。模型比较定理说明三者表示同一同伦理论。

## 综合题 25

1. Exact sequence 是 $A\to B\to C$，其中 $A\to B$ 全忠实，复合到 $C$ 为零，且
   $$
   \operatorname{Kar}(B/A)\simeq C.
   $$
2. Drinfeld quotient 给每个 $a\in\mathcal A$ 加入次数 $-1$ 的 $\varepsilon_a$，满足
   $$
   d\varepsilon_a=\operatorname{id}_a.
   $$
   因此 $\operatorname{id}_a$ 在 $H^0$ 中为零，$a$ 成为零对象。
3. Split-exact sequence 是 exact sequence 的特殊情形。Localizing invariant 把 exact sequence 送为纤维序列；split 情况下纤维序列分裂，所以得到直和分解。故 localizing invariant 是 additive。
4. $\operatorname{Mot}_{\operatorname{loc}}$ 配有 $U_{\operatorname{loc}}$，使左伴随 $\operatorname{Mot}_{\operatorname{loc}}\to\mathcal D$ 等价于取值于 $\mathcal D$ 的 localizing invariants。若 $U_{\operatorname{loc}}(A)\simeq U_{\operatorname{loc}}(B)$，任意 $E=\overline E U_{\operatorname{loc}}$ 给出 $E(A)\simeq E(B)$。
5. $M_n(R)$ 与 $R$ 有等价的 perfect module 范畴，因此 derived Morita 等价。Morita 不变的 $K$ 和 $HH$ 只依赖该 perfect module 范畴，所以
   $$
   K(M_n(R))\simeq K(R),\qquad HH(M_n(R))\simeq HH(R).
   $$

## 综合题 26

1. 对 stratum $i_\alpha:S_\alpha\hookrightarrow X$，
   $$
   K\in{}^pD^{\le0}\iff H^i(i_\alpha^*K)=0\quad(i>-\dim_\mathbb C S_\alpha),
   $$
   $$
   K\in{}^pD^{\ge0}\iff H^i(i_\alpha^!K)=0\quad(i<-\dim_\mathbb C S_\alpha).
   $$
2. 对 strata 数归纳。取开 stratum $j:U\hookrightarrow X$ 和闭补 $i:Z\hookrightarrow X$。若所有 stratum restrictions 为零，则 $j^*K=0$ 且归纳给出 $i^*K=0$。由
   $$
   j_!j^*K\to K\to i_*i^*K
   $$
   得 $K=0$。
3. 在开闭分解下，粘合 t-结构由
   $$
   K\in D^{\le0}(X)\iff j^*K\in D^{\le0}(U),\ i^*K\in D^{\le0}(Z)
   $$
   和
   $$
   K\in D^{\ge0}(X)\iff j^*K\in D^{\ge0}(U),\ i^!K\in D^{\ge0}(Z)
   $$
   定义。
4. 中间延拓为
   $$
   j_{!*}P=\operatorname{im}({}^pj_!P\to{}^pj_*P).
   $$
   若 $P$ simple，任意非零 subobject $Q\subseteq j_{!*}P$ 限制到 $U$ 后为 $0$ 或 $P$。第一种给闭支撑 subobject，第二种给闭支撑 quotient；均与中间延拓刻画冲突，故 $Q=j_{!*}P$。
5. Verdier 对偶交换 $j_!$ 与 $j_*$，并给出 perverse heart 的反等价。它把 ${}^pj_!P\to{}^pj_*P$ 的 image 送到 ${}^pj_!\mathbb D_UP\to{}^pj_*\mathbb D_UP$ 的 image，因此
   $$
   \mathbb D_X(j_{!*}P)\simeq j_{!*}(\mathbb D_UP).
   $$

## 综合题 27

1. Bousfield class 为
   $$
   \langle E\rangle=\{X\mid E\wedge X=0\}.
   $$
   偏序定义为 $\langle E\rangle\le\langle F\rangle$，若 $F$-acyclic 蕴含 $E$-acyclic。
2. 因
   $$
   (\bigvee_iE_i)\wedge X\simeq\bigvee_i(E_i\wedge X),
   $$
   左边为零当且仅当所有 $E_i\wedge X$ 为零。因此 acyclics 是交，对应 Bousfield classes 的 join。
3. 固定素数 $p$，
   $$
   K(n)_*\cong\mathbb F_p[v_n^{\pm1}],\qquad |v_n|=2(p^n-1).
   $$
   有限 $p$-local 谱的 type 为第一个使 $K(n)_*F\ne0$ 的高度 $n$。
4. Thick subcategory theorem 说 $p$-local finite spectra 的厚子范畴由 chromatic type 分类。Telescope conjecture 说有限局部化 $L_n^f$ 应与 telescope 谱 $T(0),\dots,T(n)$ 生成的局部化一致。
5. Fracture square
   $$
   \begin{array}{c}
   L_nX\to L_{K(n)}X\\
   \downarrow\quad\downarrow\\
   L_{n-1}X\to L_{n-1}L_{K(n)}X
   \end{array}
   $$
   是拉回。若 $L_{n-1}X=0$ 且 $L_{K(n)}X=0$，则右下角也为零，故拉回 $L_nX=0$。反向显然。

## 综合题 28

1. 左 $D_X$-module 给出 $\mathcal T_X$ 对 $M$ 的作用，满足 Leibniz 公式；这等价于 connection。$D_X$ 中 Lie bracket 关系要求曲率为零，所以 connection flat。反过来，flat connection 延拓为 $D_X$-作用。
2. 对 coherent $D_X$-module $M$，取 good filtration，$\operatorname{Char}(M)$ 是 associated graded 模在 $T^*X$ 中的支撑。Bernstein inequality 给出 $\dim\operatorname{Char}(M)\ge\dim X$；等号时 $M$ holonomic。
3. 平凡 connection 的 de Rham complex 是
   $$
   \mathcal O_X\to\Omega_X^1\to\cdots\to\Omega_X^{\dim X}.
   $$
   Poincaré lemma 给出它 quasi-isomorphic 于 $\mathbb C_X$，按 perverse 约定平移后为 $\mathbb C_X[\dim X]$。
4. Riemann-Hilbert correspondence 给出
   $$
   D^b_{\operatorname{rh}}(D_X)\simeq D^b_c(X,\mathbb C).
   $$
   在 heart 层，regular holonomic $D_X$-modules 对应 perverse sheaves。
5. Kashiwara equivalence 把 $D_Z$-modules 等价于支撑在闭子空间 $Z$ 的 $D_X$-modules。这正是 recollement 中闭嵌入 $i:Z\hookrightarrow X$ 的本质像，即闭支撑部分。

## 综合题 29

1. Derived affine scheme 是 $\operatorname{Spec}A$，其中 $A\in\operatorname{CAlg}^{cn}$。Derived stack 是 prestack $F:\operatorname{dAff}^{op}\to\mathcal S$，满足给定拓扑的 hyperdescent。
2. 因 $\operatorname{dAff}=(\operatorname{CAlg}^{cn})^{op}$，
   $$
   \operatorname{Map}_{\operatorname{dAff}}(\operatorname{Spec}B,\operatorname{Spec}A)
   \simeq
   \operatorname{Map}_{\operatorname{CAlg}^{cn}}(A,B).
   $$
3. 定义
   $$
   \operatorname{QCoh}(X)=\lim_{\operatorname{Spec}A\to X}\operatorname{Mod}_A.
   $$
   若 $X=\operatorname{Spec}A$，overcategory 有终对象 $\operatorname{Spec}A\to X$，所以极限为 $\operatorname{Mod}_A$。
4. 对 $A\to B\to C$ 和 $C$-module $M$，导子给纤维序列
   $$
   \operatorname{Der}_B(C,M)\to\operatorname{Der}_A(C,M)\to\operatorname{Der}_A(B,M).
   $$
   由表示性和 Yoneda，得到余纤维序列
   $$
   C\otimes_BL_{B/A}\to L_{C/A}\to L_{C/B}.
   $$
5. Lurie-Pridham 定理说特征 $0$ 下 formal moduli problems 等价于 dg Lie algebras 的合适 $\infty$-范畴。点 $x$ 处变形由 $\operatorname{Map}(x^*L_X,M)$ 控制，因此切复形是 $x^*L_X$ 的对偶。

## 综合题 30

1. 伴随 $F:C\rightleftarrows D:G$ 给出 monad $T=GF$，单位为伴随单位，乘法为 $GFGF\xrightarrow{G\varepsilon F}GF$。Comparison functor 把 $Y\in D$ 送到 $GY$，作用由 $GFGY\to GY$ 给出。
2. Barr-Beck-Lurie 定理说：若 $G$ 保守并保持 $G$-split simplicial objects 的几何实现，则
   $$
   D\simeq\operatorname{Alg}_{GF}(C).
   $$
3. 若 $D\simeq\operatorname{Alg}_T(C)$，遗忘函子反映等价，因为 $T$-代数态射是否等价由底层 $C$ 中态射是否等价判定。因此 monadic 遗忘函子保守。
4. 若 $f^*$ comonadic，则 $\mathcal D(X)$ 等价于 comonad $f^*f_*$ 的 coalgebras。Cech nerve 的 cosimplicial diagram 是该 comonad 的 cobar construction，其 totalization 正是 descent data。
5. 对 faithfully flat $A\to B$，下降数据是 $B$-模 $M$，在 $B\otimes_AB$ 上的两个拉回之间的同构，并且该同构在 $B\otimes_AB\otimes_AB$ 上满足 cocycle condition。

## 综合题 31

1. Neutral Tannakian category 是刚性 $k$-线性阿贝尔对称幺半范畴 $\mathcal C$，配 faithful exact 对称幺半函子 $\omega:\mathcal C\to\operatorname{Vect}_k^{fd}$。
2. Matrix coefficient coalgebra 为
   $$
   \int^{X\in\mathcal C}\omega(X)^\vee\otimes\omega(X).
   $$
   它把所有对象的矩阵系数按自然性关系合并，恢复坐标 Hopf algebra。
3. 保小余极限的对称幺半函子 $\operatorname{Mod}_R\to\operatorname{Mod}_A$ 由单位和 $R$-作用决定，等价于 $E_\infty$-ring map $R\to A$。给定 $R\to A$，函子为 $-\otimes_RA$。
4. 若 $\operatorname{QCoh}(X)\simeq\operatorname{Tot}\operatorname{QCoh}(U_\bullet)$，则从 $\operatorname{QCoh}(X)$ 出发的张量函子等价于 $U_\bullet$ 上相容张量函子系统；这正是点的 descent data。
5. $\operatorname{QCoh}(BG)\simeq\operatorname{Rep}(G)$。Fiber functor 对应基点，张量自同构群为 $\Omega_*BG\simeq G$，所以带 fiber functor 的范畴恢复 $G$。

## 综合题 32

1. tt-category 是带精确对称幺半结构的小幂等完备三角范畴。Thick tensor ideal 是对直和项、三角和张量任意对象封闭的 thick subcategory。Prime tensor ideal 满足 $x\otimes y\in\mathfrak p$ 蕴含 $x\in\mathfrak p$ 或 $y\in\mathfrak p$。
2. $\operatorname{Spc}(T)$ 是 prime thick tensor ideals 的集合，
   $$
   \operatorname{supp}(x)=\{\mathfrak p\mid x\notin\mathfrak p\}.
   $$
   由 prime 条件，$x\otimes y\notin\mathfrak p$ 当且仅当 $x,y\notin\mathfrak p$，故
   $$
   \operatorname{supp}(x\otimes y)=\operatorname{supp}(x)\cap\operatorname{supp}(y).
   $$
3. Balmer 分类定理说 rigid tt-category 中 radical thick tensor ideals 与 $\operatorname{Spc}(T)$ 的 Thomason subsets 对应。
4. 对交换环 $R$，
   $$
   \operatorname{Spc}(\operatorname{Perf}(R))\cong\operatorname{Spec}R.
   $$
   Perfect complex 的支撑对应局部化后非零的素理想。
5. $p$-local finite spectra 的 thick tensor ideals 按 chromatic type 分类；Morava $K(n)$ 检测高度，因此 chromatic primes 给出 Balmer spectrum 的稳定同伦论例子。

## 综合题 33

1. $THH(C)$ 是小稳定幂等完备 $\infty$-范畴 $C$ 的谱值 Hochschild trace，即恒等 bimodule 在 Morita $(\infty,2)$-范畴中的 trace。
2. Cyclic bar construction 的循环对称性给出 $\mathbb T=S^1$ 作用。Cyclotomic structure 在此基础上加入 Frobenius 映射 $X\to X^{tC_p}$。
3. $p$-complete 形式中
   $$
   TC(X;p)=\operatorname{fib}(X^{h\mathbb T}\xrightarrow{\operatorname{can}-\varphi}X^{t\mathbb T}).
   $$
4. Cyclotomic trace 是自然变换 $K(C)\to TC(C)$。Dundas-Goodwillie-McCarthy 定理说 nilpotent 相对情形中相对 $K$ 与相对 $TC$ 的 $p$-完成等价。
5. $K$、$THH$、$TC$ 都是 Morita 不变量或 localizing invariants 的核心例子；cyclotomic trace 是这些不变量之间的自然变换，因此应在 noncommutative motives 或 localizing invariants 的范畴中理解。

## 综合题 34

1. $F:C\to D$ 为 $n$-excisive，若它把 strongly homotopy cocartesian $(n+1)$-cubes 送到 homotopy cartesian $(n+1)$-cubes。$P_nF$ 是从 $F$ 到 $n$-excisive functor 的 universal approximation。
2. $D_nF=\operatorname{fib}(P_nF\to P_{n-1}F)$。$H$ 为 $n$-homogeneous，若 $H$ 是 $n$-excisive 且 $P_{n-1}H\simeq0$。
3. 若 $F$ reduced 且 $1$-excisive，则 $X\vee Y$ 是 $X\leftarrow0\to Y$ 的 pushout，故
   $$
   F(X\vee Y)\simeq F(X)\times_{F(0)}F(Y)\simeq F(X)\times F(Y).
   $$
   因此 $\operatorname{cr}_2F(X,Y)=0$。
4. 对合适 spaces 到 spectra 的 reduced finitary functor，
   $$
   D_nF(X)\simeq(\partial_nF\wedge X^{\wedge n})_{h\Sigma_n}.
   $$
5. Chain rule 说 $\partial_*(F\circ G)\simeq\partial_*F\circ\partial_*G$。取 $F=G=\operatorname{id}$ 时，恒等函子的 derivatives 在复合积下形成 operad；一般函子的 derivatives 形成相应模。

## 综合题 35

1. 从 $\mathcal P(\operatorname{Sm}_S)=\operatorname{Fun}(\operatorname{Sm}_S^{op},\mathcal S)$ 出发，先作 Nisnevich sheaf 化，再把所有 $X\times\mathbb A^1\to X$ 局部化，得到 $\mathbf H(S)$。
2. 预层范畴 presentable；Nisnevich sheaf 化和 $\mathbb A^1$-局部化都是 accessible localization，所以 $\mathbf H(S)$ presentable。局部对象正是满足 $F(X)\simeq F(X\times\mathbb A^1)$ 的 Nisnevich sheaves。
3. Tate sphere 为 $T=\mathbb A^1/(\mathbb A^1\setminus0)\simeq S^1\wedge\mathbb G_m$。稳定 motivic homotopy category 为 $\mathbf{SH}(S)=\operatorname{Sp}_T(\mathbf H_*(S))$。
4. 对开闭分解 $j:U\hookrightarrow X$、$i:Z\hookrightarrow X$，localization triangle 为
   $$
   j_!j^*E\to E\to i_*i^*E.
   $$
   Homotopy purity 说闭嵌入 $Z\hookrightarrow X$ 满足
   $$
   X/(X\setminus Z)\simeq\operatorname{Th}(N_{Z/X}).
   $$
5. 若 $\mathbf{SH}(S)$ 由 smooth schemes 的悬挂谱和 Tate twists 紧生成，则保持小余极限的正合 realization functor 只需在这些紧生成子上检测零对象；对一般态射，检测其纤维是否为零即可。

## 综合题 36

1. 子对象纤维化是
   $$
   \operatorname{Sub}_{\mathcal C}:\mathcal C^{op}\to\mathbf{Pos},
   $$
   把 $X$ 送到 $\operatorname{Sub}(X)$，把 $f:X\to Y$ 送到 pullback $f^*$；这解释谓词沿替换的重索引。
2. 对 $U\hookrightarrow X$，令 $\exists_f(U)$ 为 $U\to X\xrightarrow fY$ 的 image。则 $\exists_f(U)\le V$ 当且仅当 $U\le f^*V$，故 $\exists_f\dashv f^*$。
3. $\Sigma_f(U\to X)=U\to X\xrightarrow fY$，slice Hom 的 pullback 泛性质给出 $\Sigma_f\dashv f^*$。$\Pi_f$ 是 $f^*$ 的右伴随，解释依赖函数类型。
4. Comprehension category 中 $\Gamma$ 为上下文，纤维 $\mathcal T_\Gamma$ 的对象为 $\Gamma$ 中类型，$\Gamma.A\to\Gamma$ 为上下文扩张，项为其 section。
5. Univalence 说 $\operatorname{Id}_{\mathcal U}(A,B)\simeq\operatorname{Equiv}(A,B)$。因此等价 $A\simeq B$ 给出 universe 中路径，依赖构造可沿该路径运输。

## 综合题 37

1. $\operatorname{Disk}_n$ 由有限个 $\mathbb R^n$ 的不交并和嵌入空间组成；$E_n$-代数是对称幺半函子 $\operatorname{Disk}_n\to C$。
2. 因子化同调为
   $$
   \int_MA\simeq\operatorname*{colim}_{(U\hookrightarrow M)\in\operatorname{Disk}_{n/M}}A(U).
   $$
3. $\operatorname{id}_{\mathbb R^n}$ 是 overcategory 的终对象，故 $\int_{\mathbb R^n}A\simeq A$。对称幺半性给出
   $$
   \int_{M\sqcup N}A\simeq\int_MA\otimes\int_NA.
   $$
4. 若 $M=M_-\cup_{N\times\mathbb R}M_+$，则
   $$
   \int_MA\simeq
   \left(\int_{M_-}A\right)\otimes_{\int_{N\times\mathbb R}A}
   \left(\int_{M_+}A\right).
   $$
   因而流形分解转化为相对张量积计算。
5. 对 $E_1$-代数，$\int_{S^1}A\simeq HH(A)$。对 grouplike $E_n$-空间，非阿贝尔 Poincare 对偶给出 $\int_MA\simeq\operatorname{Map}_c(M,B^nA)$。

## 综合题 38

1. $\operatorname{ProFin}$ 的对象为 profinite sets，覆盖为有限 jointly surjective families。Condensed set 是该站点上的 set-valued sheaf。
2. 离散集合 $A$ 送到 $\underline A(S)=\operatorname{Map}_{cts}(S,A_{disc})$。自然变换 $\underline A\to\underline B$ 由点 $*$ 上的函数 $A\to B$ 唯一决定，故全忠实。
3. Grothendieck abelian category 有生成元、足够小余极限，且 filtered colimits exact；因此 condensed abelian groups 支持标准同调代数和 derived category 构造。
4. Solidification 是反射性对称幺半局部化 $(-)^{\mathrm{solid}}$；solid tensor product 为 $M\otimes^{\mathrm{solid}} N=(M\otimes N)^{\mathrm{solid}}$；solid $A$-module 是 solid 对象范畴中的 $A$-module。
5. Derived solid category 来自反射性局部化和 derived $\infty$-category；在合适假设下它稳定、presentable，且张量积保持小余极限，适合 higher algebra 和解析几何。

## 综合题 39

1. $\mathcal C_T^{\operatorname{syn}}$ 的对象为公式化上下文 $\{\vec x\mid\varphi\}$，态射为可证唯一存在的函数式关系。泛性质为
   $$
   \operatorname{Lex}(\mathcal C_T^{\operatorname{syn}},\mathcal E)\simeq\operatorname{Mod}_T(\mathcal E)
   $$
   对有限极限范畴 $\mathcal E$ 自然成立。
2. 分类 topos $\mathcal E_T$ 表示模型 2-函子：
   $$
   \operatorname{Geom}(\mathcal F,\mathcal E_T)\simeq\operatorname{Mod}_T(\mathcal F).
   $$
3. 若 $\mathcal E_T,\mathcal E'_T$ 都分类 $T$，则它们表示同一 2-函子 $\operatorname{Mod}_T(-)$；由 2-Yoneda，二者等价。
4. 泛模型 $U_T$ 是恒等几何态射对应的模型。任意模型 $M\in\operatorname{Mod}_T(\mathcal F)$ 对应 $f:\mathcal F\to\mathcal E_T$，且 $M\simeq f^*U_T$。
5. Tripos 是带 Heyting 纤维、量词伴随、Beck-Chevalley 和 generic predicate 的谓词纤维化。Generic predicate 分类所有谓词；tripos-to-topos 说明这样的逻辑数据可生成 elementary topos。

## 综合题 40

1. 关系 $R:X\nrightarrow Y$ 是子对象 $R\hookrightarrow X\times Y$。复合为
   $$
   S\circ R=\exists_{\pi_{XZ}}\bigl(\pi_{XY}^*R\wedge\pi_{YZ}^*S\bigr).
   $$
2. 在 $\mathbf{Set}$ 中，该公式选出所有 $(x,z)$，使存在 $y$ 满足 $xRy$ 且 $ySz$，即通常关系复合。
3. $\Gamma_f$ 由 $y=f(x)$ 给出，$\Gamma_g$ 由 $z=g(y)$ 给出；复合存在唯一 $y=f(x)$，故条件等价于 $z=gf(x)$，即 $\Gamma_{gf}$。
4. Exact completion $\mathcal C_{\mathrm{ex}}$ 泛地把 regular category 嵌入 exact category。若 $\mathcal C$ 已 exact，则自身满足该泛性质，所以完成等价于 $\mathcal C$。
5. Allegory 把对象作为类型、1-态射作为关系，并以内置反向、交和复合公理抽象 regular 逻辑中的合取、存在量词和关系反转。

## 综合题 41

1. Cohesive 伴随串为
   $$
   \Pi\dashv\operatorname{Disc}\dashv\Gamma\dashv\operatorname{Codisc}.
   $$
2. 三个模态为
   $$
   \int=\operatorname{Disc}\Pi,\qquad
   \flat=\operatorname{Disc}\Gamma,\qquad
   \sharp=\operatorname{Codisc}\Gamma.
   $$
3. 若 $\operatorname{Disc}$ 全忠实，则 $\Gamma\operatorname{Disc}\simeq\operatorname{id}$，所以
   $$
   \flat^2=\operatorname{Disc}\Gamma\operatorname{Disc}\Gamma\simeq\operatorname{Disc}\Gamma=\flat.
   $$
4. 恒等类型由对角线和 pullback/path object 结构解释；left exact modality 保持有限极限，因此保持这些结构的范畴语义。
5. 对离散系数，
   $$
   \operatorname{Map}_{\mathcal H}(X,\operatorname{Disc}B^nA)
   \simeq
   \operatorname{Map}_{\mathcal S}(\Pi X,B^nA).
   $$
   取 $\pi_0$ 得到 $H^n(X;A)$ 等于 shape $\Pi X$ 上的 cohomology。

## 综合题 42

1. Conically stratified space 局部形如 $\mathbb R^k\times C(L)$。Exit path 是层标号随时间只能沿偏序增大的路径。
2. $\operatorname{Exit}(X)$ 的对象为点，$n$-单纯形为 exit-simplex $\Delta^n\to X$。单层时 exit 条件自动满足，故为 singular complex，即 fundamental $\infty$-groupoid。
3. 对好层化空间，
   $$
   \operatorname{Shv}_{cbl}(X;\mathcal S)\simeq\operatorname{Fun}(\operatorname{Exit}(X),\mathcal S).
   $$
4. 开闭分解中，限制到 $U,Z$ 给两部分函子数据；从闭层流向开层的 exit morphisms 给相容传输，没有反向传输。
5. 层化因子化同调用层化 $\operatorname{Disk}$-范畴和各层系数；单层时局部模型退化为普通 $\operatorname{Disk}_n$，故恢复普通因子化同调。

## 综合题 43

1. $\operatorname{Alg}_n(C)$ 的对象为 $E_n$-代数，1-态射为 $E_{n-1}$-双模，更高态射由低阶双模递归给出。
2. $n=1$ 时，对象为结合代数，1-态射为双模，复合为相对张量积 $M\otimes_BN$。
3. 取 $P=k^n$ 与 $Q=(k^n)^*$，有
   $$
   P\otimes_{M_n(k)}Q\simeq k,\qquad Q\otimes_kP\simeq M_n(k),
   $$
   所以 $M_n(k)$ 与 $k$ Morita 等价。
4. 对合适 $A$，
   $$
   \operatorname{Tr}(\operatorname{id}_A)\simeq HH(A)\simeq\int_{S^1}A.
   $$
5. $E_n$-Koszul dual 为 $A^!=\operatorname{End}_A(\mathbb 1)$。若 $A=\mathbb 1$，则 $\operatorname{End}_{\mathbb 1}(\mathbb 1)\simeq\mathbb 1$，故 $\mathbb 1^!\simeq\mathbb 1$。

## 综合题 44

1. 预 derivator 是严格 2-函子 $\mathbb D:\mathbf{Cat}^{op}\to\mathbf{CAT}$。Derivator 还要求点值联合保守、限制函子有同伦 Kan 延拓伴随、以及点态公式等公理。
2. 对 $u:I\to J$，限制为 $u^*:\mathbb D(J)\to\mathbb D(I)$。其左、右伴随
   $$
   u_!\dashv u^*\dashv u_*
   $$
   分别为同伦左、右 Kan 延拓。
3. 若 $u:I\to *$，则 $u_!$ 是 $I$-形同伦余极限，$u_*$ 是 $I$-形同伦极限，因为它们分别伴随于常值图函子。
4. Stable derivator 是 pointed 且 cocartesian squares 与 cartesian squares 一致的 derivator。因此同伦 pushout square 同时是同伦 pullback square。
5. 定义 $\mathbb D_C(I)=h\operatorname{Fun}(N(I),C)$。因 $N(*)=\Delta^0$ 且 $\operatorname{Fun}(\Delta^0,C)\simeq C$，得 $\mathbb D_C(*)\simeq hC$。

## 综合题 45

1. Groupoid-valued prestack 是伪函子 $F:\mathcal C^{op}\to\mathbf{Grpd}$。Stack 要求 $F(U)\to\operatorname{Desc}(F,U_\bullet)$ 对每个覆盖为等价。
2. Descent datum 是局部对象 $x_i$、重叠同构 $\phi_{ij}$ 和三重交 cocycle。Stack 条件说这种局部数据来自全局对象，并且唯一到唯一同构。
3. $G$-torsor 是局部同构于 $G$ 正则作用的 sheaf；$BG(U)$ 是 $U$ 上 $G$-torsors 的 groupoid。
4. $H^1(U,G)$ 是 $G$-torsors 的同构类。若 $A$ abelian，则 $A$-banded gerbes 的等价类由 $H^2(U,A)$ 分类。
5. Groupoid 逐点取 nerve 得到 1-truncated space-valued presheaf；stack descent 变为 higher stack 的 1-truncated descent。

## 综合题 46

1. $\operatorname{Desc}(p)$ 的对象是 $X\to E$ 配 $\pi_1^*X\simeq\pi_2^*X$ 并满足 cocycle 的数据。
2. $p$ effective descent，若 $p^*:\mathcal C_{/B}\to\operatorname{Desc}(p)$ 是等价。
3. 若 $p^*$ monadic 且 monad 代数范畴等价于 descent category，则
   $$
   \mathcal C_{/B}\simeq\operatorname{Alg}_T(\mathcal C_{/E})\simeq\operatorname{Desc}(p),
   $$
   故 $p$ effective descent。
4. Trivial covering 由反射子范畴对象拉回得到；covering 是经 effective descent morphism 拉回后 trivial 的 extension；normal extension 是 covering $p$，使 $p^*p$ trivial 且 $p$ effective descent。
5. 对有限 Galois 扩张 $L/K$，$L\otimes_KL\cong\prod_{\sigma\in G}L$。Descent 同构等价于每个 $\sigma$ 的半线性自同构，cocycle 等价于群作用律。

## 综合题 47

1. 映射串 $I\xleftarrow{s}E\xrightarrow pB\xrightarrow tJ$ 定义
   $$
   P=\Sigma_t\Pi_ps^*:\mathcal C_{/I}\to\mathcal C_{/J}.
   $$
2. 在 Set 且 $I=J=1$ 时，$\Pi_p$ 对每个 $b$ 给 $X^{E_b}$，$\Sigma_t$ 对 $b$ 求和，得
   $$
   P(X)=\sum_{b\in B}X^{E_b}.
   $$
3. Species 是 $F:\mathbf{FinBij}\to\mathbf{Set}$；解析函子为
   $$
   \widehat F(X)=\sum_{n\ge0}F[n]\times_{\Sigma_n}X^n.
   $$
4. 若 $F[n]=1$，则 $\widehat F(X)=\sum_nX^n/\Sigma_n$，即所有有限无序带重复 $X$-标签集合，所以是有限多重集函子。
5. $1+X$-代数是对象 $A$ 配点 $1\to A$ 和后继 $A\to A$。初这样的代数正是自然数对象。

## 综合题 48

1. $\infty$-cosmos 是带映射 quasi-categories、equivalences、isofibrations、cotensors 和相关 pullbacks 的 simplicially enriched category。
2. $\mathcal K_2$ 与 $\mathcal K$ 同对象，Hom category 为
   $$
   \mathcal K_2(A,B)=h\operatorname{map}_{\mathcal K}(A,B).
   $$
3. Equivalence 是在 $\mathcal K_2$ 中为等价的态射；isofibration 是公理指定的 fibration-like maps；adjunction 是 $\mathcal K_2$ 中的伴随。
4. 若 $f\dashv u$ 且 $c$ 表示 $D$ 的 colimit，则
   $$
   \mathcal K_2(fc,y)\cong\mathcal K_2(c,uy)\cong\lim\mathcal K_2(D-,uy)\cong\lim\mathcal K_2(fD-,y).
   $$
   故 $fc$ 表示 $fD$ 的 colimit。
5. 因为伴随、极限、Kan 延拓、modules 等只用 $\infty$-cosmos 结构表述，等价模型之间可运输这些定理，从而避免依赖单一模型。

## 综合题 49

1. $f\perp g$ 指任意以 $f$ 为左边、$g$ 为右边的交换方块有唯一对角填充。${}^\perp\mathcal S$ 是左正交于 $\mathcal S$ 的态射类，$\mathcal S^\perp$ 是右正交于 $\mathcal S$ 的态射类。
2. 正交因子化系统 $(\mathcal E,\mathcal M)$ 要求每个态射分解为 $me$，且 $\mathcal E={}^\perp\mathcal M$、$\mathcal M=\mathcal E^\perp$。两个分解 $me=m'e'$ 之间由 $e\perp m'$ 与 $e'\perp m$ 得到互逆比较态射，唯一填充保证唯一。
3. 任意函数分解为满射到 image 再单射入陪域。满射-单射方块中，用满射选原像定义填充，单射保证良定义和唯一。
4. $X$ 为 $\mathcal S$-局部当且仅当对每个 $s:A\to B$，$\mathcal C(B,X)\to\mathcal C(A,X)$ 为双射；这等价于 $s\perp(X\to1)$。
5. 弱因子化系统只要求提升存在而不要求唯一。每个正交因子化系统遗忘唯一性后给出弱因子化系统。

## 综合题 50

1. Sketch 是小范畴配指定锥和余锥；模型是把指定锥送到极限锥、指定余锥送到余极限余锥的函子。
2. 若没有指定锥或余锥，模型条件为空，因此模型范畴为 $\operatorname{Fun}(\mathcal S,\mathcal C)$。
3. 有限积理论是带有限积的小范畴，模型为保持有限积的函子。群对象由 $m,e,i$ 和群公理交换图组成，全部只需有限积表达。
4. Doctrine 指定允许结构和保持结构的函子。Doctrine 越强，保持条件越强，模型范畴作为对象类越小。
5. 小范畴有对象、态射、源靶、恒等和复合；复合定义域为 pullback $M\times_OM$，结合与单位是有限极限图上的等式，故由有限极限 sketch 表示。

## 综合题 51

1. 幂等为 $e:X\to X$ 且 $e^2=e$。分裂为 $X\xrightarrow rY\xrightarrow sX$，满足 $rs=\operatorname{id}_Y$、$sr=e$。
2. 两个分裂 $e=sr=s'r'$ 给 $u=r's$、$v=rs'$，计算得 $vu=\operatorname{id}$、$uv=\operatorname{id}$。
3. $\operatorname{Kar}(\mathcal C)$ 对象为 $(X,e)$，态射 $f:(X,e)\to(Y,d)$ 满足 $f=dfe$。当 $e=d=\operatorname{id}$ 时条件为空，故嵌入全忠实。
4. 若 $p$ 是 $(X,e)$ 上幂等，则对象 $(X,p)$ 存在，两个方向都取 $p$，给出 $p$ 的分裂。
5. 绝对余极限是被所有函子保持的余极限。分裂 coequalizer 的泛性质由 splitting data 和等式验证；任意函子保持这些等式，所以保持该 coequalizer。

## 综合题 52

1. $U:\mathcal I\to\mathcal J$ 共尾指每个 $j/U$ 非空且连通。若 $L\dashv U$，单位 $\eta_j:j\to ULj$ 给出 $j/U$ 的始对象；故 $j/U$ 非空连通，$U$ 共尾。
2. 点态公式给
   $$
   (\operatorname{Lan}_KF)(d)\cong\operatorname{colim}_{K/d}F\pi.
   $$
   若 $V_d:\mathcal I_d\to K/d$ 共尾，则第三章共尾性定理给
   $$
   \operatorname{colim}_{K/d}F\pi\cong\operatorname{colim}_{\mathcal I_d}F\pi V_d.
   $$
3. 反射子范畴是全子范畴 $I:\mathcal A\hookrightarrow\mathcal C$ 且 $L\dashv I$。若 $C=\operatorname{colim}ID$，则
   $$
   \mathcal A(LC,A)\cong\mathcal C(C,IA)
   \cong\lim_j\mathcal C(IDj,IA)
   \cong\lim_j\mathcal A(Dj,A),
   $$
   所以 $LC$ 为 $\mathcal A$ 中余极限。
4. Kleisli 范畴 $\mathcal C_T$ 有 Hom $\mathcal C(X,TY)$。函子 $J:\mathcal C\to\mathcal C_T$ 由单位给出，$G_T(Y)=TY$，$G_T(f)=\mu_YTf$。自然等式
   $$
   \mathcal C_T(JX,Y)=\mathcal C(X,TY)=\mathcal C(X,G_TY)
   $$
   给 $J\dashv G_T$，诱导单子为 $G_TJ=T$，乘法为 $\mu$。
5. 自由代数为 $F^TX=(TX,\mu_X)$，遗忘函子为 $U^T$，伴随双射
   $$
   \mathcal C^T(F^TX,(A,a))\cong\mathcal C(X,A)
   $$
   把代数同态 $h$ 送到 $h\eta_X$，逆把 $k$ 送到 $aTk$。由该双射，
   $$
   \mathcal C^T(F^TX,F^TY)\cong\mathcal C(X,TY)=\mathcal C_T(X,Y),
   $$
   所以 $\mathcal C_T\to\mathcal C^T$ 全忠实，且对象落在自由代数上。
