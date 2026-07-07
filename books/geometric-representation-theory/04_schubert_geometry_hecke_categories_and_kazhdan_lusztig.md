# 第四章：Schubert 几何、Hecke categories 与 Kazhdan-Lusztig 基

## 本章目标

本章把第一章的 Schubert 分解和第三章的 equivariant sheaves 结合起来，构造有限型 Hecke category 的基本版本，并说明 Kazhdan-Lusztig 基如何由 intersection complexes 给出。大型几何定理仍作为外部输入。

## 依赖前置知识

需要第一章的 Bruhat decomposition、第三章的 perverse sheaves 和附录 A 的卷积 correspondence。

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

## 4.4 标准对象的简单反射计算

**定义 4.10.** 对 simple reflection $s$，记
$$
\Delta_s=j_{s!}E_{X_s}[1],\qquad \nabla_s=j_{s\ast}E_{X_s}[1],
$$
其中 $X_s\simeq\mathbb A^1$。

**命题 4.11.** 在 Grothendieck group 中，$\Delta_s\star\Delta_s$ 对应 Hecke algebra 中标准基元素 $T_s^2$。

**证明.** 标准对象 $\Delta_w$ 的类对应标准基 $T_w$ 是 Hecke categorification 的 normalization 之一。卷积 functor 在 Grothendieck group 上给出乘法，因为 distinguished triangles 的类满足 additivity，且卷积是三角函子。因此
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

**外部输入定理 4.14.** 在合适的 mixed 或 graded sheaf theory 中，映射
$$
\mathcal H_W\longrightarrow K_0(\mathsf H_G)
$$
把 Kazhdan-Lusztig basis element $C_w$ 送到 $[\operatorname{IC}_w]$，把标准基送到标准对象类，并与卷积乘法相容。

该定理依赖 purity、decomposition theorem 和 Kazhdan-Lusztig 的 Hecke algebra formalism。当前作为外部输入，不在本章重证。

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

## 本章小结

本章构造了有限 Hecke category 的 constructible sheaf 版本，写出卷积 correspondence 并证明卷积的结合性和单位性质。IC sheaves 与 Kazhdan-Lusztig basis 的对应、KL 多项式的 stalk 解释和 positivity 仍是外部输入，后续需要 theorem locator。

## 练习

**练习 4.1.** 对 $G=SL_3$ 写出 $B$ 在 $G/B$ 上的六个轨道，并标出维数。

**练习 4.2.** 展开定义 4.6，证明 $m[g_1,g_2]=g_1g_2$ 对 $B$-quotient well-defined。

**练习 4.3.** 对简单反射 $s$，计算 $\Delta_s\star\Delta_s$ 在 Grothendieck group 中对应的 Hecke algebra 元素。

**练习 4.4.** 对 $SL_2$，直接用两条 Schubert strata 写出 $\Delta_s$、$\nabla_s$ 和 $\operatorname{IC}_s$ 的限制。
