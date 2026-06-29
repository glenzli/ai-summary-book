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
