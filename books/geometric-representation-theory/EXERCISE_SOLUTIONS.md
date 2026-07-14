# 习题答案与提示

本文件集中给出全书练习的答案提示。正文保留练习题干；这里给关键步骤、计算结果或所需外部输入。

## 第 0 章

**0.1.** 复代数簇 $G/B$ 可取 Zariski topology，解析空间 $(G/B)^{an}$ 取 classical topology，quotient stack 口径可写为 $[\mathrm{pt}/B]$ 在 equivariant 场景中的局部模型。三者的 sheaf categories 分别依赖代数 constructible、解析 constructible 和 stack/equivariant descent。

**0.2.** 还需给出 correspondence
$$
G/B\times G/B\xleftarrow{p}G\times^B G/B\xrightarrow{m}G/B
$$
若 source 真有 map $p$ 到 Cartesian product，可写卷积公式 $m_!p^\ast(-\boxtimes-)$；若 source 是 contracted product，则必须另写 torsor atlas $\widetilde C\xrightarrow q C$，先把 $p^\ast(-\boxtimes-)$ 沿 $q$ descent，再作 $m_!$。两种情形都要声明 equivariance、support-properness 或 compact-support convention。

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

**12.1.** 对任意 field extension $E'/\mathbb C$，每个 $f\in E'((z))^\times$ 唯一写成 $z^n u$，其中 $n\in\mathbb Z$、$u\in E'[[z]]^\times$，所以 geometric points 由 valuation 参数化。但 representability 是对全部 $\mathbb C$-algebras 的 functor statement；dual-number loop $1+\varepsilon z^{-1}$ 说明仅算 fields 看不见 nilpotent families。

**12.2.** $g\in GL_n(k((z)))$ 把标准 lattice $k[[z]]^n$ 送到 lattice $L$；右乘 $GL_n(k[[z]])$ 不改变 $L$，每个 lattice 可由某个 $g$ 得到。

**12.3.** 可在 $LG\times LG\times\operatorname{Gr}_G$ 上取
$(h_1,h_2)$ 作用为
$(g_1,g_2,x)\mapsto(g_1h_1^{-1},h_1g_2h_2^{-1},h_2x)$。于是 $g_1g_2x$ 不变；先商 $h_1$ 或 $h_2$ 给出两种加括号的 contracted products，四重版本给出 pentagon coherence。

**12.4.** $\mathscr O/(z^2)$ 的非零 proper submodule 中任取元素 $a+bz$；若 $a\ne0$，它生成全模，故 proper submodule 的元素都落在 $(z)$。而 $(z)$ 是一维 $\mathbb C$-space，故唯一 length-$1$ submodule 是 $zQ$。其 inverse image 给出 open stratum 上唯一的 intermediate lattice。

## 第 13 章

**13.1.** 对 $t\in T^\vee(R)$，在 $F_\mu(\mathcal F)\otimes R$ 上乘 $\mu(t)$；naturality、$(\mu+\eta)(t)=\mu(t)\eta(t)$ 和 unit weight $0$ 分别给 morphism、tensor 和 unit compatibility。仅此构造不能证明 injective：若实际出现的 weights 未生成 $X^\ast(T^\vee)$，其公共 kernel 会作用平凡；排除此情形需要额外几何输入。

**13.2.** 对 $T=\{\operatorname{diag}(t,t^{-1})\}$，$X_\ast(T)\simeq\mathbb Z$，dominant cone 为 $n\ge0$。它成为 $PGL_2$ 的 character lattice；若用 simply connected $SL_2$ 的 fundamental weight 作单位，这些正是 even weights $2n$，所以 odd $SL_2$ highest weights 不下降为 $PGL_2$-representations。

**13.3.** 在 graded-vector-space model 中 rigid dual 为 $(V^\vee)_n=(V_{-n})^\vee$，故 point object $E_n$ 的 dual 是 $E_{-n}$。命题 13.14 下分别对应 $\chi_n$ 与 $\chi_{-n}=\chi_n^\vee$。

**13.4.** $\dim\operatorname{Sym}^2E^2=3$，$\dim\det(E^2)=1$。Semismall decomposition 的 multiplicities 则都为 $1$：open fiber 的 $H_0$ 与 closed fiber $\mathbb P^1$ 的 top $H_2$ 都是一维。表示维数来自各 IC summands 的 total cohomology，不能与 decomposition multiplicity 混同。

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

**17.4.** 令 $K=\ker i$。因为 $i$ 满射，$ij=0$ 当且仅当 $\operatorname{im}j\subset K$，也就是 $j$ 唯一分解为 $V\to K\hookrightarrow W$。而
$$
T_{[i]}\operatorname{Gr}(v,W)\simeq\operatorname{Hom}(K,V),
$$
trace pairing $\operatorname{Hom}(K,V)\times\operatorname{Hom}(V,K)\to\mathbb C$ 是非退化的，所以这样的 $j$ 正好给出一个 cotangent covector。

**17.5.** 在 $a=1$ 图上，$i=(1,x)$、$j=(-xp,p)^T$。到 $b=1$ 图需用 $t=x^{-1}$ 换基，故 $y=x^{-1}$，而 $j$ 变为 $xj=(-x^2p,xp)^T=(q,-yq)^T$，所以 $q=-x^2p$。并且
$$
ji=\begin{pmatrix}-xp&-x^2p\\ p&xp\end{pmatrix},
$$
其迹与行列式都为零；也可直接用 $(ji)^2=j(ij)i=0$。

## 第 18 章

**18.1.** $\mathfrak{sl}_2$ 时 $R(n)$ 由 idempotent $e(i^n)$、dots $y_1,\ldots,y_n$ 和 crossings $\psi_1,\ldots,\psi_{n-1}$ 生成，满足 nilHecke relations。

**18.2.** 若 $A,B$ 为有限维代数且有 $(A,B)$-bimodule $M$，则 $M\otimes_B-$ 与 $\operatorname{Hom}_A(M,-)$ 在有限性条件下伴随。

**18.3.** projectives 与 finite-dimensional modules 通过 Euler pairing $\langle[P],[M]\rangle=\dim\operatorname{Hom}(P,M)$ 配对；在 categorification 中对应 canonical/dual canonical duality。

**18.4.** 对 $S=E[e_1,e_2]$ 的基 $(1,x_2)$，由 $x_2^2=e_1x_2-e_2$ 与 $\psi_1(1)=0,\psi_1(x_2)=1$ 得
$$
Y=\begin{pmatrix}0&-e_2\\1&e_1\end{pmatrix},
\qquad
D=\begin{pmatrix}0&1\\0&0\end{pmatrix}.
$$
于是 $D=E_{12}$、$YD=E_{22}$、$DY-e_1D=E_{11}$、$YE_{11}=E_{21}$。四个矩阵单位都在像中，故作用满到 $M_2(S)$；命题 18.8.2 的 normal form 计算给出忠实性。

## 第 19 章

**19.1.** $\mathbb C^\times$ 在 cotangent fiber 上按 $t\cdot \xi=t\xi$ 缩放；moment map 到 $\mathcal N$ 对应 cotangent vector，故带权缩放。

**19.2.** Weyl algebra $A_1=\mathbb C\langle x,\partial\rangle/([\partial,x]=1)$ 的 associated graded 为 $\mathbb C[x,\xi]$，即 $\mathcal O(T^\ast\mathbb A^1)$。

**19.3.** 两者都用 highest weight、standards、simples、order 和 dualities；BGG 来自 Lie algebra，symplectic $\mathcal O$ 来自 quantized symplectic resolution 与 Hamiltonian torus action。

**19.4.** Liouville 1-form $\theta$ 在 fiber scaling $t$ 下变为 $t\theta$，symplectic form $d\theta$ 也按 $t$ 缩放。

**19.5.** 表 19.13 只是经验对应；定义必须给出两侧空间、categories、functors 和 equivalence/Koszul duality statement。

**19.6.** 由 $x\partial^n\overline1=-n\partial^{n-1}\overline1$ 归纳得
$$
x^r\partial^n\overline1=
(-1)^r\frac{n!}{(n-r)!}\partial^{n-r}\overline1
$$
当 $r\le n$，而 $r>n$ 时为零。正权齐次 monomial $x^a\partial^b$ 满足 $a-b>0$，因而把多项式次数至少降低 $a-b$；正权算子的任意足够长乘积最终杀死给定向量，所以 $A_{>0}$ 局部幂零。

## 第 20 章

**20.1.** pure gauge $N=0$ 时 $\mathcal R=\operatorname{Gr}_G$ 型对象；需记录 $G$、等变 Borel-Moore homology、loop rotation、finite generation 和 BFN theorem。

**20.2.** loop rotation 给 $\mathbb C^\times_\hbar$-equivariance，$H^\ast_{\mathbb C^\times}(\mathrm{pt})=\mathbb C[\hbar]$，从而量子化代数带参数 $\hbar$。

**20.3.** 三者都由 correspondence 上 pull/intersect/push 定义；差别在对象空间分别为 affine Grassmannian、Steinberg fiber product、BFN space $\mathcal R$。

**20.4.** 对 torus pure gauge，$\operatorname{Gr}_T=X_\ast(T)$，卷积为 lattice 加法；识别为 $T^\vee$ 或 $T^\ast T^\vee$ 需 BFN equivariant homology 计算。

**20.5.** 若 $s=\sum a_i z^i$，条件 $z^{-m}s\in\mathbb C[[z]]$ 等价于 $a_i=0$ for $i<m$。$m\le0$ 自动满足；$m>0$ 要求至少 $m$ 阶 vanishing。

**20.6.** $u_0\star u_\lambda=u_{\lambda}$ 因为 $0+\lambda=\lambda$；同理右单位。

**20.7.** 权 $r$ 表示把 $z^m$ 作用成 $z^{rm}$，故 regularity 条件为 $z^{-rm}s\in\mathcal O$。于是 $s\in z^{\max(rm,0)}\mathcal O$。模 $z^d$ 后需要消失的独立系数个数为
$$
\min\bigl(\max(rm,0),d\bigr),
$$
这就是该线性子空间的 codimension。

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

**21.5.** 先在 $V_{a+b+c}$ 中选 $U_c$ 有 $\binom{a+b+c}{c}_q$ 种，再在商 $V/U_c$ 中选维数为 $b$ 的 $U_{b+c}/U_c$ 有 $\binom{a+b}{b}_q$ 种。反过来，先选 $U_{b+c}$ 再在其中选 $U_c$，分别给出 $\binom{a+b+c}{b+c}_q$ 与 $\binom{b+c}{c}_q$。两者计数同一批二步旗标，故得到推论 21.9.2。

## 第 22 章

**22.1.** 生成元 $E,F,K^{\pm1}$，关系 $KEK^{-1}=q^2E$，$KFK^{-1}=q^{-2}F$，$[E,F]=(K-K^{-1})/(q-q^{-1})$。

**22.2.** 图为链
$$
b_0\xrightarrow{f}b_1\xrightarrow{f}\cdots\xrightarrow{f}b_n,
$$
权为 $n-2r$。

**22.3.** indecomposable projectives 通常对应 lower/canonical basis，simple modules 通过 duality pairing 对应 dual canonical basis；具体取决于 KLR convention。

**22.4.** $B(2)$ 为 $b_0\to b_1\to b_2$，weights $2,0,-2$；$\varepsilon(b_r)=r$，$\varphi(b_r)=2-r$。

**22.5.** $v=0,1,2,3$ 时，$\operatorname{Gr}(v,3)$ 依次为点、$\mathbb P^2$、对偶 $\mathbb P^2$、点，而 kernel 维数依次为 $3,2,1,0$。$Z_v$ 选择相邻 kernels $K_{v+1}\subset K_v$，故给出链
$$
c_0\longrightarrow c_1\longrightarrow c_2\longrightarrow c_3,
$$
四个权为 $3,1,-1,-3$，与 $B(3)$ 完全一致。

## 第 23 章

**23.1.** $T_DT_{D'}(L)=L\otimes\mathcal O_C(D')\otimes\mathcal O_C(D)$，线丛张量积给出规范同构 $T_DT_{D'}\simeq T_{D+D'}$，结合性来自 associator。有效除子 monoid 的群化是 divisor group；映射 $D\mapsto\mathcal O_C(D)$ 再经过线性等价得到 $\operatorname{Pic}(C)$。对任意线丛 $M$ 定义 $T_M(L)=L\otimes M$，便得到 Picard group 的平移作用。

**23.2.** 对 $xy=z^m$ 微分并与 $dz$ 外积：$y\,dx\wedge dz+x\,dy\wedge dz=0$。除以 $xy$ 后即得 $dx\wedge dz/x=-dy\wedge dz/y$。$m=1$ 时 $X_1\simeq\mathbb A^2$，无奇点；$m=2,3$ 时 Jacobian 只在原点消失，所以 singular locus 都是 $\{0\}$。

**23.3.** 在开 cell 上，$j^*E_{\mathbb P^1}[1]\simeq j^!E_{\mathbb P^1}[1]\simeq E_{\mathbb A^1}[1]$；在闭点上，$i^*E_{\mathbb P^1}[1]\simeq E[1]$、$i^!E_{\mathbb P^1}[1]\simeq E[-1]$。它们都只在奇次数非零，因此是 odd parity；点支撑对象集中在偶次数。两个 Schubert closures 均光滑，计算与系数特征无关，所以 type $A_1$ 没有 $p$-canonical 修正。

**23.4.** 令 $R=E[\varepsilon]/(\varepsilon^2)$。简单模 $E=R/(\varepsilon)$ 有周期自由分解
$$
\cdots\xrightarrow{\varepsilon}R
\xrightarrow{\varepsilon}R
\longrightarrow E\longrightarrow0.
$$
施加 $\operatorname{Hom}_R(-,E)$ 后所有微分为零，故 $\operatorname{Ext}^1_R(E,E)\simeq E$；它由命题 23.13 的非分裂扩张生成。有限维向量空间范畴是半单的，所以其中的 $\operatorname{Ext}^1(E,E)$ 为零。

**23.5.** 例如比较 Nakajima homology 与 cyclotomic KLR projectives 时，除 $K_0$ 外至少要比较：grading shift，它记录 $q$-参数；induction/convolution，它记录量子群乘法；duality，它区分 canonical 与 dual canonical convention。若再比较完整 2-representation，还必须核对 generators 之间的 2-morphisms 与关系。只比较其中一部分不能推出范畴等价。

## 附录 A

**A.1.** $H$ 左乘自身是 transitive free action，商栈 $[H/H]$ 等价于点。

**A.2.** 向量丛由 fiber over $eK$ 和 $K$-作用决定；这与 $[H/K/H]\simeq BK$ 上 vector bundles 等价于 $K$-representations 一致。

**A.3.** 两种加括号对应两个 iterated fiber products；associativity 需要 fiber product associativity、proper base change 和 projection formula。

## 附录 G

**G.10.** 与 1.3/4.1 相同：六个 cells 按长度 $0,1,1,2,2,3$ 分层，闭包由 Bruhat order 给出。

**G.11.** $H^\ast(\mathbb P^1)=\mathbb C\oplus\mathbb C[-2]$。Springer sheaf shift 使 top cohomology 与 Springer representation 的 sign/trivial convention 对齐，具体取决于 top-degree normalization。
