# 第四章：Schubert 几何、Hecke categories 与 Kazhdan-Lusztig 基

Hecke algebra 的乘法把两个 double cosets 合成一个线性组合，但系数本身没有解释为何非负，也看不见 Schubert variety 的奇点。把 $B\backslash G/B$ 上的 constructible complexes 沿乘法 correspondence 卷积，会把每个 basis element 提升成一个对象，把乘法系数提升成分解重数。标准对象记录开 cell 的延拓，IC 层则把闭包奇点纳入其中；两者在 Grothendieck group 中分别对应标准基和 Kazhdan--Lusztig 基。有限秩模型 $SL_2/B\simeq\mathbb P^1$ 将检验单位、二次关系和 IC normalization，而一般 KL 识别仍明确作为 mixed/graded 几何的外部输入。

## 4.1 Hecke algebra

**定义 4.1.** 令 $(W,S)$ 为 Coxeter system。Hecke algebra $\mathcal H_W$ 是 $\mathbb Z[v,v^{-1}]$-代数，生成元 $\{T_s\}_{s\in S}$ 满足 braid relations，并满足 quadratic relation
$$
(T_s-v)(T_s+v^{-1})=0.
$$
对 reduced expression $w=s_1\cdots s_r$，定义
$$
T_w=T_{s_1}\cdots T_{s_r}.
$$
由 braid relations，$T_w$ 与 reduced expression 选择无关。

**外部输入定理 4.2.** $\{T_w\}_{w\in W}$ 构成 $\mathcal H_W$ 的 $\mathbb Z[v,v^{-1}]$-基。Kazhdan-Lusztig basis $\{C_w\}$ 由 bar-invariance 和 triangularity 唯一刻画。  
来源：Kazhdan-Lusztig。

## 4.2 $B$-equivariant sheaves on $G/B$

**定义 4.3.** 有限 Hecke category 的 constructible 版本定义为
$$
\mathsf H_G=D^b_B(G/B,E),
$$
其中 $B$ 左作用于 $G/B$。等价地，它是 double quotient stack
$$
D^b(B\backslash G/B,E)
$$
上的 constructible derived category。

**命题 4.4.** $\mathsf H_G$ 的 simple perverse objects 由 $\operatorname{IC}_w$ 标号，其中 $w\in W$。

**证明.** 由 Bruhat decomposition，$B$ 在 $G/B$ 上的轨道为 $X_w$。每个 $X_w\simeq\mathbb A^{\ell(w)}$ 单连通，并且在通常系数语境中只有平凡 irreducible equivariant local system。由 BBD 的 simple perverse sheaf 分类，simple perverse objects 是这些轨道上 irreducible local systems 的 middle extensions。因此得到 $\operatorname{IC}_w$。这里使用了外部输入定理 1.13 和 3.15。$\square$

**定义 4.5.** 标准对象和余标准对象记为
$$
\Delta_w=j_{w!}E_{X_w}[\ell(w)],\qquad
\nabla_w=j_{w\ast}E_{X_w}[\ell(w)],
$$
其中 $j_w:X_w\hookrightarrow G/B$。

## 4.3 卷积

为了定义卷积，把 $G/B$ 上的 $B$-equivariant sheaves 等价看作 $B$-bi-equivariant sheaves on $G$，即 $D^b(B\backslash G/B)$。

**定义 4.6.** 卷积 correspondence 为
$$
B\backslash G/B \times B\backslash G/B
\xleftarrow{\quad p\quad}
B\backslash G\times^B G/B
\xrightarrow{\quad m\quad}
B\backslash G/B,
$$
其中 $G\times^B G=(G\times G)/B$，右侧 $B$ 作用为
$$
(g_1,g_2)\cdot b=(g_1b,b^{-1}g_2),
$$
而 $m[g_1,g_2]=g_1g_2$。

对 $\mathcal F,\mathcal G\in\mathsf H_G$，定义
$$
\mathcal F\star\mathcal G=m_!p^\ast(\mathcal F\boxtimes\mathcal G).
$$
在 proper setting 或有限 flag variety 情形中也常用 $m_\ast$；本章采用 $m_!$ 并在 proper 时识别 $m_!=m_\ast$。

**命题 4.7.** 卷积 $\star$ 在六函子 formalism 的标准假设下是结合的，即存在自然同构
$$
(\mathcal F\star\mathcal G)\star\mathcal K\simeq
\mathcal F\star(\mathcal G\star\mathcal K).
$$

**证明.** 三重卷积由 stack
$$
B\backslash G\times^B G\times^B G/B
$$
和乘法映射 $[g_1,g_2,g_3]\mapsto g_1g_2g_3$ 控制。两种加括号方式分别对应先对前两个或后两个因子取 fiber product correspondence。由附录 A 的命题 A.15，correspondence 复合与 functor 复合自然同构；而群乘法的结合律给出两种三重 correspondence 的目标映射相同。因此得到自然 associator。$\square$

**定义 4.8.** 单位对象为
$$
\mathbf 1=\operatorname{IC}_e,
$$
其中 $e\in W$ 对应闭轨道 $B/B\subset G/B$。

**命题 4.9.** $\mathbf 1$ 是卷积单位。

**证明.** $\operatorname{IC}_e$ 是支撑在单位 double coset $B\subset G$ 上的 skyscraper 型对象。卷积 correspondence 中与单位 double coset 相乘不改变另一个 double coset；对应的 correspondence 等同于恒等 correspondence
$$
B\backslash G/B \xleftarrow{\operatorname{id}} B\backslash G/B \xrightarrow{\operatorname{id}} B\backslash G/B.
$$
由恒等 correspondence 的 functor 为 identity，得到 $\mathbf 1\star\mathcal F\simeq\mathcal F$ 和 $\mathcal F\star\mathbf 1\simeq\mathcal F$。$\square$

结合性和单位说明这里确有 monoidal category，但还没有看到 Hecke 二次关系。最小 Schubert closure $\overline X_s\simeq\mathbb P^1$ 已经包含开 cell 与闭点两个相对位置，它的二重卷积正好产生第一项 lower contribution。

## 4.4 标准对象的简单反射计算

**定义 4.10.** 对 simple reflection $s$，记
$$
\Delta_s=j_{s!}E_{X_s}[1],\qquad \nabla_s=j_{s\ast}E_{X_s}[1],
$$
其中 $X_s\simeq\mathbb A^1$。

**命题 4.11.** 在 mixed 或 graded Hecke category 的 Grothendieck group 中，$\Delta_s\star\Delta_s$ 对应 Hecke algebra 中标准基元素 $T_s^2$。若只在未分级 constructible category 中工作，则这里应理解为把 $v$ 专门化后的影子。

**证明.** 标准对象 $\Delta_w$ 的类对应标准基 $T_w$ 是 graded/mixed Hecke categorification 的 normalization 之一，其中 Tate twist 或 grading shift 记录参数 $v$。卷积 functor 在 Grothendieck group 上给出乘法，因为 distinguished triangles 的类满足 additivity，且卷积是三角函子。因此
$$
[\Delta_s\star\Delta_s]=[\Delta_s]\,[\Delta_s]=T_s^2.
$$
若采用关系
$$
(T_s-v)(T_s+v^{-1})=0,
$$
则
$$
T_s^2=(v-v^{-1})T_s+1.
$$
几何上右侧对应卷积分解中的开轨道贡献和单位轨道贡献。完整对象级分解需要 mixed grading 或 parity formalism。$\square$

**例 4.12.** 对 $G=SL_2$，$G/B\simeq\mathbb P^1$，$\Delta_s$ 是 $\mathbb A^1$ 上常值 sheaf 的 extension by zero，shift 为 $[1]$。卷积 $\Delta_s\star\Delta_s$ 的支撑仍在两个 Schubert strata 上，Grothendieck group 计算给出
$$
[\Delta_s\star\Delta_s]=1+(v-v^{-1})[\Delta_s]
$$
在标准 Hecke normalization 下。

## 4.5 Grothendieck group 和 KL 基

**定义 4.13.** 令 $K_0(\mathsf H_G)$ 为 $\mathsf H_G$ 的 split Grothendieck group，并把 shift 作用规范为
$$
[\mathcal F[1]]= -[\mathcal F]
$$
或在 graded 版本中引入 $v$ 记录 Tate twist/shift。具体 convention 在 mixed sheaf 版本中更自然；本章只记录 decategorification 入口。

未分级 Betti category 会丢掉 $v$ 所记录的权或 grading，因此不能单凭前面的卷积定义识别完整 KL 基。下面的定理恰好补上 mixed/graded 结构与 IC 分解之间的深层联系。

**外部输入定理 4.14.** 在合适的 mixed 或 graded sheaf theory 中，映射
$$
\mathcal H_W\longrightarrow K_0(\mathsf H_G)
$$
把 Kazhdan-Lusztig basis element $C_w$ 送到 $[\operatorname{IC}_w]$，把标准基送到标准对象类，并与卷积乘法相容。

该定理依赖 purity、decomposition theorem 和 Kazhdan-Lusztig 的 Hecke algebra formalism。当前作为外部输入，不在本章重证。

定理的内容不只是给两个基重新命名：IC stalk 的分次维数成为从标准对象到 self-dual 基的系数，于是代数中的多项式开始测量具体奇点。最低秩时闭包光滑，没有高阶奇点贡献，计算应退化为 $P_{e,s}=1$。

**推论 4.15.** Kazhdan-Lusztig 多项式的系数可解释为 Schubert variety 的 intersection cohomology stalk 维数，具体 shift 和 $v$ convention 由定理 4.14 的 normalization 决定。

**证明.** 在定理 4.14 下，$\operatorname{IC}_w$ 在标准对象基中的展开系数对应 $C_w$ 在标准基中的展开系数。标准对象的限制记录 Schubert cell 上的局部贡献，而 IC sheaf 的 stalk cohomology 给出这些系数。完整等式需要 mixed sheaf normalization，故本推论仍依赖外部输入定理 4.14。$\square$

## 4.6 低阶例子：$SL_2$

**例 4.16.** 对 $G=SL_2$，$W=\{e,s\}$。Hecke algebra 由 $T_s$ 生成，满足
$$
(T_s-v)(T_s+v^{-1})=0.
$$
Schubert varieties 为点 $\overline X_e=X_e$ 和 $\overline X_s=\mathbb P^1$。两者都光滑，因此
$$
\operatorname{IC}_e=E_{\{pt\}},\qquad
\operatorname{IC}_s=E_{\mathbb P^1}[1].
$$
由于无奇点贡献，非平凡 Kazhdan-Lusztig polynomial 为 $P_{e,s}=1$。

**证明.** 第一章例 1.18 给出 Schubert 分层。点和 $\mathbb P^1$ 都光滑，第三章命题 3.11 说明常值 sheaf 按维数 shift 后 perverse。$\overline X_s$ 的 open stratum 为 $\mathbb A^1$，其闭包光滑，middle extension 是整空间上的 shifted constant sheaf。KL polynomial 的值由外部输入定理 4.14 和 IC stalk 无高阶奇点贡献得到。$\square$

卷积把 double-coset 乘法提升为函子，$SL_2$ 的点与射影直线则显示单位项和简单反射项怎样同时出现。一般 Schubert closure 的奇点使 IC stalk 不再平凡，Kazhdan--Lusztig 多项式由此进入。下一章把同一套 pull--intersect--push 机制移到 nilpotent cone 上；在那里卷积的 top Borel--Moore homology 将产生 Weyl group 本身。

## 练习

**练习 4.1.** 对 $G=SL_3$ 写出 $B$ 在 $G/B$ 上的六个轨道，并标出维数。

**练习 4.2.** 展开定义 4.6，证明 $m[g_1,g_2]=g_1g_2$ 对 $B$-quotient well-defined。

**练习 4.3.** 对简单反射 $s$，计算 $\Delta_s\star\Delta_s$ 在 Grothendieck group 中对应的 Hecke algebra 元素。

**练习 4.4.** 对 $SL_2$，直接用两条 Schubert strata 写出 $\Delta_s$、$\nabla_s$ 和 $\operatorname{IC}_s$ 的限制。
