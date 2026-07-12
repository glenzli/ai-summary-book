# 附录 F：卷积 correspondence、proper base change 与结合性检查

## 本章目标

本附录给出 Hecke category、Steinberg variety、geometric Satake 和 Coulomb branch 中反复使用的卷积类型模板。重点是区分普通 correspondence 与 equivariant descent、区分 $!$-pushforward 与 $\ast$-pushforward，并把结合性所需的三重几何和 coherence 写成可检查的假设。

## F.1 普通 correspondence 与 descent 模板

**定义 F.1（普通 sheaf convolution）.** 在固定的六函子模型中，令 $X,C_2$ 为允许的 spaces 或 stacks，并给定 morphisms
$$
c=(c_1,c_2):C_2\longrightarrow X\times X,
\qquad
m:C_2\longrightarrow X.
$$
对 $\mathcal F,\mathcal G\in D^b_c(X,E)$，若 $c^\ast(\mathcal F\boxtimes\mathcal G)$ constructible，定义
$$
\mathcal F\star_!\mathcal G
:=m_!c^\ast(\mathcal F\boxtimes\mathcal G),
\qquad
\mathcal F\star_\ast\mathcal G
:=m_\ast c^\ast(\mathcal F\boxtimes\mathcal G).
$$
存在自然 morphism $\star_!\to\star_\ast$。若 $m$ 在
$$
\operatorname{supp}c^\ast(\mathcal F\boxtimes\mathcal G)
$$
上 proper，也就是该 support 的闭包到 $X$ 的限制为 proper，则此 morphism 是同构；这时简记共同值为 $\mathcal F\star\mathcal G$。

对 ind-scheme，本书只对 finite-dimensional support 定义上述式子：先取包含两个输入支撑的 finite-type closed stage，再要求 $m$ 在相应卷积支撑上的限制 proper。不同 stage 给出同一对象需要 closed-embedding base change；这是定义良定性的组成部分，不能只写“$m$ ind-proper”。

**定义 F.1.1（contracted product 上的 descent）.** 几何表示论中常出现 diagram
$$
X\times X\xleftarrow{\ p\ }\widetilde C_2
\xrightarrow{\ q\ }C_2\xrightarrow{\ m\ }X,
$$
其中 $q$ 是群 $A$ 的 torsor 或相应 quotient atlas，而 $p$ 通常不 factor through $C_2$。若输入的 $A$-equivariance 给 $p^\ast(\mathcal F\boxtimes\mathcal G)$ 一个 coherent $q$-descent datum，则定义 twisted external product $\mathcal F\widetilde\boxtimes\mathcal G$ 为满足
$$
q^\ast(\mathcal F\widetilde\boxtimes\mathcal G)
\simeq p^\ast(\mathcal F\boxtimes\mathcal G)
$$
的下降对象，并置
$$
\mathcal F\star_!\mathcal G
=m_!(\mathcal F\widetilde\boxtimes\mathcal G).
$$
下降对象的存在与唯一性属于所选 equivariant sheaf formalism；不能用一个并不存在的 $C_2\to X\times X$ 代替它。第十二章的 affine-Grassmannian convolution 使用这一模板。

**检查表 F.2.** 每次定义卷积必须记录：

1. $X,C_2,\widetilde C_2$ 的几何类型和 sheaf theory；
2. $c,p,q,m$ 的定义域、值域及群作用方向；
3. twisted external product 是普通 pullback 还是由 coherent descent 得到；
4. 使用 $m_!$ 还是 $m_\ast$，以及 properness 是全局、分层限制还是只在输入支撑上成立；
5. smooth pullback、relative dimension、perverse shift 和 Tate twist 的 convention；
6. associativity 使用的两个三重 fiber products 及其 coherence；
7. 单位对象的支撑和左右单位 base-change squares；
8. 若声称卷积保持 perverse，所用的是 small/semismall estimate 还是另一个 t-exactness theorem。

## F.2 结合性和单位

**命题 F.3（有 coherence 假设的结合性）.** 在定义 F.1 的普通 correspondence 中，令
$$
C_L=C_2\mathop{\times}_{m,X,c_1}C_2,
\qquad
C_R=C_2\mathop{\times}_{m,X,c_2}C_2.
$$
把 $C_L$ 的点写成 $(u,v)$，其中 $m(u)=c_1(v)$，并定义
$$
c_L(u,v)=(c_1(u),c_2(u),c_2(v)),
\qquad m_L(u,v)=m(v).
$$
把 $C_R$ 的点写成 $(u,v)$，其中 $m(u)=c_2(v)$，并定义
$$
c_R(u,v)=(c_1(v),c_1(u),c_2(u)),
\qquad m_R(u,v)=m(v).
$$
假设：

1. 存在 $\alpha:C_L\xrightarrow{\sim}C_R$，满足 $c_R\alpha=c_L$ 和 $m_R\alpha=m_L$；
2. 六函子 formalism 对构造 $C_L,C_R$ 的 squares 给出所需 Beck--Chevalley 和 external-product isomorphisms；
3. 四重 correspondence 上由 $\alpha$ 诱导的两条复合同构相同，也就是几何 associator 满足 pentagon coherence。

则 $\star_!$ 有自然 associator
$$
(\mathcal F\star_!\mathcal G)\star_!\mathcal H
\xrightarrow{\sim}
\mathcal F\star_!(\mathcal G\star_!\mathcal H),
$$
并满足 pentagon。若两个二重与全部三重支撑上的目标映射 proper，同一结论适用于共同的 $\star_!=\star_\ast$。

**证明.** 对左加括号，依次应用 external-product compatibility、构造 $C_L$ 的 Cartesian square 的 base change，以及 $!$-pushforward 的复合性，得到自然同构
$$
(\mathcal F\star_!\mathcal G)\star_!\mathcal H
\simeq
(m_L)_!c_L^\ast
(\mathcal F\boxtimes\mathcal G\boxtimes\mathcal H).
$$
同样的三步对右加括号给出
$$
\mathcal F\star_!(\mathcal G\star_!\mathcal H)
\simeq
(m_R)_!c_R^\ast
(\mathcal F\boxtimes\mathcal G\boxtimes\mathcal H).
$$
由 $c_R\alpha=c_L$、$m_R\alpha=m_L$ 和 $\alpha$ 为同构，右侧两个 functors 自然同构；与前后两个同构复合便得到 associator。四个输入时，五边形两条路径在前述重写后恰是四重 correspondence 上的两条几何同构，假设 3 断言它们相同，故 pentagon 成立。若使用 $\ast$-pushforward，则支撑上的 properness 先把每一步的 $!$ 与 $\ast$ 自然识别，结论随之成立。$\square$

contracted-product 情形先在 $\widetilde C_2$ 及其三重版本上完成同一计算，再沿 torsors 作 coherent descent。此时“contracted products 结合”只给出假设 1；要得到 monoidal category，仍需检查假设 2 和 3。

**定义 F.4（单位数据）.** 令 $e:\mathrm{pt}\to X$ 为 closed unit stratum。假设存在同构
$$
C_2\mathop{\times}_{c_1,X,e}\mathrm{pt}\simeq X,
\qquad
C_2\mathop{\times}_{c_2,X,e}\mathrm{pt}\simeq X,
$$
使各自剩余的 source map 与 $m$ 都对应 $\operatorname{id}_X$。在第三章的 perverse convention 下，定义
$$
\mathbf 1=e_!E_{\mathrm{pt}}=e_\ast E_{\mathrm{pt}}.
$$
点的复维数为 $0$，所以这里没有额外 shift。若单位是 stacky orbit 或正维 stratum，必须重新计算 normalization，不能沿用此式。

**命题 F.5.** 在定义 F.4 的假设下，存在自然同构
$$
\mathbf 1\star_!\mathcal F\simeq\mathcal F
\simeq\mathcal F\star_!\mathbf 1.
$$
若单位和输入的相关支撑上 $m$ proper，同一式子成立于 $\star$。

**证明.** 对左单位，把 $e_!E\boxtimes\mathcal F$ 拉回到 $C_2$，再对 $e$ 的 Cartesian square 应用 base change，所得对象支撑在 $C_2\times_{c_1,X,e}\mathrm{pt}$ 上。由定义 F.4，该空间、剩余 source map 和目标 map 分别识别为 $X,\operatorname{id}_X,\operatorname{id}_X$，故 pushforward 等于 $\mathcal F$。右单位使用第二个 fiber product，步骤完全相同。两侧同构的自然性来自 base-change transformation 的自然性。$\square$

## F.3 核心实例

| 场景 | 对象 | 二重几何 | properness 口径 |
| --- | --- | --- | --- |
| finite Hecke | $B\backslash G/B$ 或 $G/B$ 的等变模型 | contracted product $G\times^B G$ | Schubert 支撑上的乘法图 proper |
| Springer | $\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N}$ | 三重 Springer fiber product | 投影在相应 Borel--Moore 支撑上 proper |
| affine Satake | $L^+G\backslash LG/L^+G$ | $LG\times^{L^+G}\operatorname{Gr}_G$ | finite Schubert 支撑上的卷积图 proper |
| BFN Coulomb | $\mathcal R$ over affine Grassmannian | Borel--Moore fiber product | finite-type approximation 中验证 |

**例 F.6.** finite Hecke 情形中，单位对象支撑在闭点 $B/B\subset G/B$。若 $j_w:X_w=BwB/B\hookrightarrow G/B$，且 $\Delta_w=j_{w!}E_{X_w}[\ell(w)]$，则单位 fiber product 满足定义 F.4，因而
$$
\mathbf 1\star\Delta_w\simeq\Delta_w
\simeq\Delta_w\star\mathbf 1.
$$
等式使用 Schubert 支撑上的 properness；它不是从 $G/B$ 上一个未定义的群乘法直接推出。

**例 F.7.** affine Grassmannian 情形中，中性点 $e=L^+G/L^+G$ 是闭的零维 orbit。卷积 Grassmannian 在第一或第二个 modification 为中性 modification 时都识别为原 Grassmannian，所以定义 F.4--命题 F.5 给出 tensor unit
$$
\mathbf 1=E_e\in\operatorname{Perv}_{L^+G}(\operatorname{Gr}_G,E).
$$
这里的 pushforward 只在 finite Schubert support 上使用 properness；详细 descent diagram 见第十二章。

## F.4 失败模式和反例

**警告 F.8.** 不能只写
$$
\mathcal F\star\mathcal G=m_\ast(\mathcal F\boxtimes\mathcal G)
$$
而省略 source correspondence 或 descent atlas。多数几何表示论卷积先经过 contracted product 或 fiber product；在 affine Grassmannian 上，$LG\times^{L^+G}\operatorname{Gr}_G$ 并没有卷积定义所需的普通 map 到 $\operatorname{Gr}_G\times\operatorname{Gr}_G$。

**警告 F.9.** $m_!\simeq m_\ast$ 需要 $m$ 在实际 sheaf support 上 proper。对 ind-scheme，只说两个输入各自 finite-dimensional 不够；还要给出包含卷积支撑的 finite-type stage 和该 stage 上的 proper restriction。

**反例 F.10（非 proper 时 $!\ne\ast$）.** 令 $j:\mathbb C^\times\hookrightarrow\mathbb C$ 为 open immersion。则
$$
i_0^\ast Rj_!E_{\mathbb C^\times}=0,
$$
而对以 $0$ 为中心的小圆盘 $\Delta$，有
$$
i_0^\ast Rj_\ast E_{\mathbb C^\times}
\simeq R\Gamma(\Delta^\times,E),
$$
其 $H^0$ 与 $H^1$ 均同构于 $E$。因此比较 morphism $Rj_!\to Rj_\ast$ 不是同构。任何非 proper convolution 若交换 $m_!$ 与 $m_\ast$，都必须提供额外的 support argument。

**反例 F.11（proper 不推出 perverse t-exact）.** 令 $a:\mathbb P^1\to\mathrm{pt}$。对象 $E_{\mathbb P^1}[1]$ 在 $\mathbb P^1$ 上 perverse，但
$$
Ra_\ast E_{\mathbb P^1}[1]
=R\Gamma(\mathbb P^1,E)[1]
$$
在 standard degrees $-1$ 和 $1$ 都有非零 cohomology。点上的 perverse t-structure 就是 standard t-structure，故该 pushforward 不是 perverse。于是 convolution 的 properness 只允许比较 $!$ 与 $\ast$；保持 perversity 还需要 smallness、semismallness 或其他 t-exactness 输入。

## 本章小结

本附录把普通 pull--push、equivariant descent、support-properness、三重 correspondence coherence 和 perverse t-exactness 分成了不同证明责任。后续章节必须先构造 twisted external product，再说明目标映射在哪个有限支撑上 proper；结合性由明确的三重与四重几何给出，不能由“乘法结合”一句话代替。
