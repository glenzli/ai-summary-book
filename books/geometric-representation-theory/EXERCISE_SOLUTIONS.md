# 习题答案与提示

本文件集中给出全书练习的答案提示。正文保留练习题干；这里给关键步骤、计算结果或所需外部输入。

## 第 0 章

**0.1.** 复代数簇 $G/B$ 可取 Zariski topology，解析空间 $(G/B)^{an}$ 取 classical topology，quotient stack 口径可写为 $[\mathrm{pt}/B]$ 在 equivariant 场景中的局部模型。三者的 sheaf categories 分别依赖代数 constructible、解析 constructible 和 stack/equivariant descent。

**0.2.** 还需给出 correspondence
$$
G/B\times G/B\xleftarrow{p}G\times^B G/B\xrightarrow{m}G/B
$$
以及卷积公式 $m_!p^\ast(-\boxtimes -)$，并声明 equivariance、properness 或 compact-support convention。

**0.3.** 例：Beilinson-Bernstein localization。需检查底域 $k=\mathbb C$、$G$ reductive、$\lambda$ regular dominant、TDO convention、category $\mathcal O$ block 和 global sections 的 exactness。

## 第 1 章

**1.1.** $gB\mapsto (gV_i^0)$，其中 $V_i^0=\langle e_1,\ldots,e_i\rangle$。稳定标准旗标的子群正是上三角 Borel，因此 $GL_n/B$ 参数化完整旗标。

**1.2.** $G$-equivariant line bundles on $G/B$ 等价于一维 $B$-representations，即 $X^\ast(B)=X^\ast(T)$。本书 convention 为 $\mathcal L_\lambda=G\times^B k_{-\lambda}$。

**1.3.** 六个元素为 $e,s_1,s_2,s_1s_2,s_2s_1,w_0=s_1s_2s_1$，长度为 $0,1,1,2,2,3$。Hasse diagram 分四层，$s_i$ 低于两个长度 $2$ 元素中含有相应 reduced subword 的元素。

## 第 2 章

**2.1.** $M(\lambda)$ 有基 $f^n v_\lambda$，权为 $\lambda-2n$。作用公式为
$$
h f^n v=(\lambda-2n)f^n v,\quad
f f^n v=f^{n+1}v,\quad
e f^n v=n(\lambda-n+1)f^{n-1}v.
$$

**2.2.** 由 tensor-Hom adjunction：
$$
\operatorname{Hom}_{\mathfrak g}(U(\mathfrak g)\otimes_{U(\mathfrak b)}k_\lambda,M)
\simeq
\operatorname{Hom}_{\mathfrak b}(k_\lambda,\operatorname{Res}M).
$$
右侧正是选择 $M$ 中一个 $\mathfrak n$-annihilated 的 $\lambda$-weight vector。

**2.3.** 对 $\mathfrak{sl}_2$，当 $\lambda\in\mathbb Z_{\ge0}$ 时 $f^{\lambda+1}v_\lambda$ 生成 proper submodule，simple quotient $L(\lambda)$ 维数为 $\lambda+1$；否则 Verma module 通常简单。

## 第 3 章

**3.1.** 对 open stratum $\mathbb G_m$ 要求 $H^i(j_x^\ast\mathcal F)=0$ for $i>-\dim\mathbb G_m=-1$，closed stratum $\{0\}$ 要求 $H^i(i_0^\ast\mathcal F)=0$ for $i>0$；cosupport 用 $i^!$ 和 $j^!$ 取对偶不等式。

**3.2.** $\operatorname{IC}_e$ 是闭点上的 skyscraper perverse sheaf；$\operatorname{IC}_s$ 是 $\mathbb P^1$ 上常值 sheaf shift $E_{\mathbb P^1}[1]$。

**3.3.** 同构的 inverse 给出四个函子的逆等价。Verdier duality 满足 $\mathbb D f^\ast\simeq f^!\mathbb D$ 和 $\mathbb D f_!\simeq f_\ast\mathbb D$，同构情形下 $f_!=f_\ast$。

## 第 4 章

**4.1.** $SL_3$ 的轨道由 $S_3$ 标号，维数为 Bruhat length：$0,1,1,2,2,3$。

**4.2.** 在 $G\times^B G/B$ 中 $(g_1b,b^{-1}g_2B)$ 与 $(g_1,g_2B)$ 等价，乘积 $g_1bb^{-1}g_2B=g_1g_2B$ 不变。

**4.3.** 在 Hecke algebra 中 $C_s^2=(v+v^{-1})C_s$。标准基下对应二次关系 $(T_s-v)(T_s+v^{-1})=0$。

**4.4.** $\Delta_s=j_{s!}E_{\mathbb A^1}[1]$，$\nabla_s=j_{s\ast}E_{\mathbb A^1}[1]$，$\operatorname{IC}_s=E_{\mathbb P^1}[1]$；在 open stratum 限制均为 $E[1]$，在 closed stratum 的 stalk/costalk 区分 standard 与 costandard。

## 第 5 章

**5.1.** Jordan normal form 在 conjugation 下不变；nilpotent Jordan blocks 的大小给出 partition。每个 partition 构造相应 Jordan block direct sum，得到一个 orbit。

**5.2.** $\widetilde{\mathcal N}=\{(x,\ell)\mid x\ell=0,\ x(k^2)\subset \ell\}$。$x=0$ 时 fiber 为 $\mathbb P^1$；regular nilpotent 时唯一稳定线，fiber 为点。

**5.3.** $Z=\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N}$ 映到 $\mathcal B\times\mathcal B$。像按相对位置分解为 $G$-orbits，标号为 $W$。

## 第 6 章

**6.1.** 与 5.1 相同：Jordan block 大小在 conjugation 下不变，每个 partition 给出一个 nilpotent conjugacy class。

**6.2.** $G$-equivariant local systems on $\mathcal O_x\simeq G/G_x$ 对应 component group $\pi_0(G_x)$ 的表示。若 $G_x$ 连通，则 component group 平凡。

**6.3.** 在 induction correspondence 中先取 parabolic $P=LU$，再用 $P\to L$ 把 $L$ 上的 sheaf 拉到 $P$ 或相应 nilpotent piece，随后沿 $G\times^P(-)\to\mathcal N_G$ 推前。

## 第 7 章

**7.1.** $\Gamma(\mathbb A^n,\mathcal D)=k\langle x_i,\partial_i\rangle$，关系 $[x_i,x_j]=[\partial_i,\partial_j]=0$，$[\partial_i,x_j]=\delta_{ij}$。

**7.2.** $\delta_0=\mathcal D/\mathcal D x$ 的 associated graded support 由 $x=0$ 定义，所以 characteristic variety 为 $T_0^\ast\mathbb A^1$。

**7.3.** de Rham functor 对 left D-modules 是 covariant；solution functor 是 contravariant，因此有些文献把 RH 写成反等价。

## 第 8 章

**8.1.** 在坐标 $z$ 上，$\mathfrak{sl}_2$ 可取向量场 $\partial_z$、$z\partial_z$、$z^2\partial_z$，差一个符号取决于左/右作用 convention。

**8.2.** sheaf of rings 情形中局部检查即可：extension of scalars $\mathcal D_\lambda\otimes_{U_\lambda}-$ 左伴随于 forgetful/global sections，因为 Hom 由 balanced maps 表示。

**8.3.** 链条：BB localization、regular holonomic D-modules、RH correspondence、Schubert constructibility、IC-KL theorem、BBD decomposition/purity、KL polynomial stalk formula。

## 第 9 章

**9.1.** $H^0(\mathbb P^1,\mathcal O(n))$ 维数 $n+1$ for $n\ge0$；$H^1$ 为 $0$。若 $n\le-2$，$H^1$ 维数 $-n-1$；$n=-1$ 全部为 $0$。

**9.2.** $V$ 有限维，$V\otimes-$ 在 vector spaces 上 exact；$\mathfrak g$-作用不改变 underlying exactness，且有限生成、weight decomposition 和 local $\mathfrak n$-finiteness 保持。

**9.3.** 对 $SL_2$，唯一 simple root 的 parabolic 已为 $G$，所以 $G/B\to G/G$ 是到点的投影，wall crossing 几何退化为全局 cohomology 型操作。

## 第 10 章

**10.1.** 若 $I=\operatorname{Ann}_{U(\mathfrak g)}L$，则对 $u\in U(\mathfrak g)$、$a\in I$，$(ua)L=u(aL)=0$ 且 $(au)L=a(uL)=0$，所以 $I$ 是 two-sided。

**10.2.** 有限维 simple $L(n)$ 的中心 character 由 Casimir 标量决定；在常见归一化中 Casimir 作用为与最高权 $n$ 对应的 Weyl orbit/dot orbit 标量。

**10.3.** associated variety 定义为 $\operatorname{Supp}\operatorname{gr}M$；不同 good filtration 若不给独立性定理，support 可能看似依赖选择。

## 第 11 章

**11.1.** $R=E[\alpha]$，$s(\alpha)=-\alpha$，$R^s=E[\alpha^2]$，$B_s=R\otimes_{R^s}R(1)$。

**11.2.** 对任意 Soergel bimodule $B$，$R\otimes_R B\simeq B\simeq B\otimes_R R$，且 associativity 来自 tensor product。

**11.3.** 一般 Coxeter system 不一定来自代数群或 Schubert variety；Soergel theory 以 reflection representation 和 polynomial ring 为输入。

**11.4.** 用分解 $R=R^s\oplus R^s\alpha$。投影到两个 summands 后张量上左右 $R$，得到到 $B_s(1)$ 与 $B_s(-1)$ 的投影；具体 degree 由 shift convention 决定。

**11.5.** 一种 convention 是 $C_s=T_s+v^{-1}$ 对应 $\underline H_s=H_s+v$，换元可取 $v\mapsto v^{-1}$ 并同时调整标准基生成元。

## 第 12 章

**12.1.** $GL_1(k((z)))/GL_1(k[[z]])=k((z))^\times/k[[z]]^\times\simeq\mathbb Z$，由 valuation 标号。

**12.2.** $g\in GL_n(k((z)))$ 把标准 lattice $k[[z]]^n$ 送到 lattice $L$；右乘 $GL_n(k[[z]])$ 不改变 $L$，每个 lattice 可由某个 $g$ 得到。

**12.3.** 二重 convolution 参数化 $(\mathcal L_0,\mathcal L_1,\mathcal L_2)$ 的 lattice chain；三重参数化四个 lattices。相邻相对位置给出两个或三个 Schubert 条件。

## 第 13 章

**13.1.** $\operatorname{Gr}_{GL_1}\simeq\mathbb Z$；点 $n$ 的 skyscraper 对应 $G^\vee=GL_1$ 的 character $t\mapsto t^n$。

**13.2.** $SL_2$ 的 coweight lattice 对应 $PGL_2$ 的 weight lattice；dominant coweights $n\ge0$ 对应 $PGL_2$ 的 dominant weights，注意根/权格 quotient。

**13.3.** 五项为：perversity、convolution t-exactness、commutativity constraint、global cohomology fiber functor、dual root datum 识别。

**13.4.** 在 $GL_1$ 中 Verdier duality 把点 $n$ 的对象送到 dual character，对应 $n\mapsto -n$。

**13.5.** $SL_2$ 的 $\operatorname{Gr}$ 不只是离散分支；orbit closures 有非平凡几何，表示的 weight spaces 由 MV cycles/weight functors 给出。

## 第 14 章

**14.1.** $I=\{g(z)\in GL_n(k[[z]])\mid g(0)\in B\}$，即模 $z$ 后为上三角矩阵。

**14.2.** $\operatorname{Gr}_G$ 的 $L^+G$-orbits 由 dominant coweights 标号；$\operatorname{Fl}_G$ 的 $I$-orbits 由 extended affine Weyl group 标号。

**14.3.** affine simple reflection 的最小 Schubert closure 是 $P_s/I\simeq\mathbb P^1$，open cell 为 $\mathbb A^1$，另有 closed point。

**14.4.** 对 $SL_2$，$W_{\mathrm{aff}}$ 为 infinite dihedral group $\langle s_0,s_1\mid s_i^2=1\rangle$。前四层长度为 $e$，$s_0,s_1$，$s_0s_1,s_1s_0$，$s_0s_1s_0,s_1s_0s_1$。

**14.5.** 若 $\Delta_s\star\Delta_s$ 是单一对象，则 Grothendieck group 中会给出单一 basis element；但 Hecke 二次关系给出含 lower/shift terms 的组合。

## 第 15 章

**15.1.** $c(xz^m,yz^n)=\operatorname{Res}\langle xz^m,d(yz^n)\rangle=n\langle x,y\rangle\delta_{m+n,0}$。

**15.2.** 若 $m,n\ge0$，则 $m+n\ne -1$ in residue convention for $z^{-1}dz$，无 residue；因此 $\mathfrak g[[z]]$ 上 cocycle 为零。

**15.3.** 多出：topological completion、level、central extension、ind-scheme geometry、critical center/opers、factorization compatibility。

**15.4.** 若 $m+n=0$，则 $c(xz^m,yz^n)=n\langle x,y\rangle$，交换后为 $m\langle y,x\rangle=-n\langle x,y\rangle$，故反对称。

## 第 16 章

**16.1.** $GL_1$-bundle 即 line bundle，故 $\operatorname{Bun}_{GL_1}(C)=\operatorname{Pic}(C)$ 的 stack 版本。

**16.2.** Hecke modification 是在点 $x\in C$ 处给两个 vector bundles $E,E'$ 和同构 $E|_{C\setminus x}\simeq E'|_{C\setminus x}$，相对位置由 coweight 控制。

**16.3.** Hecke functors 对 $\operatorname{Rep}(G^\vee)$ 是 tensor action；若缺少 tensor compatibility，只得到逐个表示的同构，不能保证 eigensheaf 与 spectral parameter 相容。

**16.4.** 在 formal disk 上取 lattices $L'\subset L$，商 $L/L'$ 长度 $1$；fiber 是一维 quotient 或 line choice，等价于 $\mathbb P^1$ 型数据。

## 第 17 章

**17.1.** 无边单顶点时 $\mathbf M=\operatorname{Hom}(W,V)\oplus\operatorname{Hom}(V,W)$，moment map 为 $ij=0$。

**17.2.** 稳定条件令 $i:W\to V$ 满射，quotient 给 $\operatorname{Gr}(v,W)$；cotangent vector 由 $j:V\to W$ 且 $ij=0$ 给出。

**17.3.** Hecke correspondence 参数化 $(x,V'\subset V)$，其中 $V/V'$ 在顶点 $i$ 处一维，其余顶点不变，并与 quiver maps 相容。

**17.4.** Grassmannian 的 tangent 为 $\operatorname{Hom}(\ker i,V)$；条件 $ij=0$ 表示 $j$ annihilates 与 quotient 方向相容的 tangent data，即 cotangent covector 条件。

## 第 18 章

**18.1.** $\mathfrak{sl}_2$ 时 $R(n)$ 由 idempotent $e(i^n)$、dots $y_1,\ldots,y_n$ 和 crossings $\psi_1,\ldots,\psi_{n-1}$ 生成，满足 nilHecke relations。

**18.2.** 若 $A,B$ 为有限维代数且有 $(A,B)$-bimodule $M$，则 $M\otimes_B-$ 与 $\operatorname{Hom}_A(M,-)$ 在有限性条件下伴随。

**18.3.** projectives 与 finite-dimensional modules 通过 Euler pairing $\langle[P],[M]\rangle=\dim\operatorname{Hom}(P,M)$ 配对；在 categorification 中对应 canonical/dual canonical duality。

## 第 19 章

**19.1.** $\mathbb C^\times$ 在 cotangent fiber 上按 $t\cdot \xi=t\xi$ 缩放；moment map 到 $\mathcal N$ 对应 cotangent vector，故带权缩放。

**19.2.** Weyl algebra $A_1=\mathbb C\langle x,\partial\rangle/([\partial,x]=1)$ 的 associated graded 为 $\mathbb C[x,\xi]$，即 $\mathcal O(T^\ast\mathbb A^1)$。

**19.3.** 两者都用 highest weight、standards、simples、order 和 dualities；BGG 来自 Lie algebra，symplectic $\mathcal O$ 来自 quantized symplectic resolution 与 Hamiltonian torus action。

**19.4.** Liouville 1-form $\theta$ 在 fiber scaling $t$ 下变为 $t\theta$，symplectic form $d\theta$ 也按 $t$ 缩放。

**19.5.** 表 19.13 只是经验对应；定义必须给出两侧空间、categories、functors 和 equivalence/Koszul duality statement。

## 第 20 章

**20.1.** pure gauge $N=0$ 时 $\mathcal R=\operatorname{Gr}_G$ 型对象；需记录 $G$、等变 Borel-Moore homology、loop rotation、finite generation 和 BFN theorem。

**20.2.** loop rotation 给 $\mathbb C^\times_\hbar$-equivariance，$H^\ast_{\mathbb C^\times}(\mathrm{pt})=\mathbb C[\hbar]$，从而量子化代数带参数 $\hbar$。

**20.3.** 三者都由 correspondence 上 pull/intersect/push 定义；差别在对象空间分别为 affine Grassmannian、Steinberg fiber product、BFN space $\mathcal R$。

**20.4.** 对 torus pure gauge，$\operatorname{Gr}_T=X_\ast(T)$，卷积为 lattice 加法；识别为 $T^\vee$ 或 $T^\ast T^\vee$ 需 BFN equivariant homology 计算。

**20.5.** 若 $s=\sum a_i z^i$，条件 $z^{-m}s\in\mathbb C[[z]]$ 等价于 $a_i=0$ for $i<m$。$m\le0$ 自动满足；$m>0$ 要求至少 $m$ 阶 vanishing。

**20.6.** $u_0\star u_\lambda=u_{\lambda}$ 因为 $0+\lambda=\lambda$；同理右单位。

## 第 21 章

**21.1.** 对一维简单对象 $S$，$[S]\ast[S]$ 计数短正合列 $0\to S\to V\to S\to0$；未 twisted 系数由子空间数和 extension classes 决定。

**21.2.** correspondence 为
$$
\operatorname{Rep}_{d_1}(Q)\times\operatorname{Rep}_{d_2}(Q)
\xleftarrow{p}
\operatorname{SES}_{d_1,d_2}
\xrightarrow{q}
\operatorname{Rep}_{d_1+d_2}(Q).
$$

**21.3.** ordinary CoHA 用 representation stack 和 cohomology；critical CoHA 还需 potential、critical locus、vanishing cycles 和 orientation data。

**21.4.** $[S]\ast[\mathbb F_q^2]$ 的系数为三维空间中 quotient 或 subobject 类型的计数；未 twisted 情形核心是计算相应一维子空间数，如 $q^2+q+1$。

## 第 22 章

**22.1.** 生成元 $E,F,K^{\pm1}$，关系 $KEK^{-1}=q^2E$，$KFK^{-1}=q^{-2}F$，$[E,F]=(K-K^{-1})/(q-q^{-1})$。

**22.2.** 图为链
$$
b_0\xrightarrow{f}b_1\xrightarrow{f}\cdots\xrightarrow{f}b_n,
$$
权为 $n-2r$。

**22.3.** indecomposable projectives 通常对应 lower/canonical basis，simple modules 通过 duality pairing 对应 dual canonical basis；具体取决于 KLR convention。

**22.4.** $B(2)$ 为 $b_0\to b_1\to b_2$，weights $2,0,-2$；$\varepsilon(b_r)=r$，$\varphi(b_r)=2-r$。

## 第 23 章

**23.1.** 表格应含作者、版本、精确定理、假设翻译、模型、locator、是否进入证明链。缺任一项则只能作为边界。

**23.2.** 因为 proof series 使用 derived stacks、renormalized D-modules、IndCoh、singular support 和 factorization/Kac-Moody machinery，未建立这些模型前不能作为第十六章内部定理。

**23.3.** 需核查 finite type、normality、Poisson/symplectic singularity、是否有 symplectic resolution、quantization、category $\mathcal O$、与 mirror/dual 空间的具体定理。

**23.4.** characteristic zero 的 IC arguments 依赖 semisimplicity/purity；modular coefficients 下 parity sheaves、torsion 和 $p$-canonical basis 会改变 KL 型结论。

**23.5.** Hecke 情形 $\mathcal C=\mathsf H$，$\mathcal M$ 可为 sheaf category on flag variety；KLR 情形 $\mathcal C$ 为 KLR 2-category 或 projective module category，$\mathcal M$ 为 categorified highest weight representation。

## 附录 A

**A.1.** $H$ 左乘自身是 transitive free action，商栈 $[H/H]$ 等价于点。

**A.2.** 向量丛由 fiber over $eK$ 和 $K$-作用决定；这与 $[H/K/H]\simeq BK$ 上 vector bundles 等价于 $K$-representations 一致。

**A.3.** 两种加括号对应两个 iterated fiber products；associativity 需要 fiber product associativity、proper base change 和 projection formula。

## 附录 G

**G.10.** 与 1.3/4.1 相同：六个 cells 按长度 $0,1,1,2,2,3$ 分层，闭包由 Bruhat order 给出。

**G.11.** $H^\ast(\mathbb P^1)=\mathbb C\oplus\mathbb C[-2]$。Springer sheaf shift 使 top cohomology 与 Springer representation 的 sign/trivial convention 对齐，具体取决于 top-degree normalization。
