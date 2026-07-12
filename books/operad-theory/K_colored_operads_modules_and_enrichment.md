# 附录 K：Colored operad、模结构与 enriched 版本

## 本附录目标

第五章已经给出 colored operad 的基本定义。本附录补充四个严格化点：

1. $C$-轮廓群胚的骨架和稳定子。
2. Colored substitution product 的纤维群胚/coend 公式。
3. 由 colored operad 编码代数同态、模、双模和代数-模对。
4. Enriched colored operad 的定义边界。

本附录仍以集合值为主；线性和 dg 版本通过把 $\mathbf{Set}$ 替换为相应对称幺半范畴得到，但模型结构问题回到附录 G。

## K.1 轮廓群胚与稳定子

**定义 K.1.** 固定颜色集合 $C$。一个 $C$-轮廓是 $(S,\kappa;c)$，其中 $S$ 是有限集，$\kappa:S\to C$，$c\in C$。轮廓同构
$$
(S,\kappa;c)\to(T,\lambda;d)
$$
是双射 $\varphi:S\to T$，满足 $d=c$ 且 $\lambda\varphi=\kappa$。轮廓群胚记为 $\mathbf B_C$。

**命题 K.2.** 选择骨架后，$\mathbf B_C$ 等价于所有有限颜色词和输出颜色的群胚并：
$$
\coprod_{n\ge0}\coprod_{(c_1,\ldots,c_n;c)\in C^{n+1}}
B\operatorname{Aut}(c_1,\ldots,c_n),
$$
其中
$$
\operatorname{Aut}(c_1,\ldots,c_n)=
\{\sigma\in\Sigma_n: c_{\sigma(i)}=c_i\text{ for all }i\}.
$$

**证明.** 任一有限集 $S$ 与 $[n]$ 同构。选择双射 $[n]\cong S$ 后，输入颜色函数变为颜色词 $(c_1,\ldots,c_n)$。改变双射由 $\Sigma_n$ 作用；保持同一颜色词的自同构正是只置换相同颜色位置的子群 $\operatorname{Aut}(c_1,\ldots,c_n)$。输出颜色不能被同构改变。故得到所述骨架。$\square$

**说明 K.3.** Colored arity 公式中的对称群通常不是全 $\Sigma_n$，而是颜色词的稳定子。若颜色词中有重复颜色，稳定子非平凡；若所有输入颜色不同，稳定子可能平凡。

## K.2 Colored substitution 的 coend 口径

**定义 K.4.** 设 $X,Y:\mathbf B_C\to\mathbf{Set}_{\mathcal U}$ 是 colored symmetric sequences。对轮廓 $(S,\kappa;c)$，定义
$$
(X\circ_CY)(S,\kappa;c)
=
\operatorname*{colim}_{(f:S\to T)\in\operatorname{Fib}(S)}
\coprod_{\delta:T\to C}
X(T,\delta;c)
\times
\prod_{t\in T}
Y(f^{-1}(t),\kappa|_{f^{-1}(t)};\delta(t)).
$$
目标双射同时重标号 $T$、$\delta$ 和全部张量因子。函数 $f$ 允许空纤维；若所有 nullary $Y$-项均为空，才可缩成第五章的非空分块特例。

**命题 K.5.** 定义 K.4 关于轮廓同构自然，并给出 $\operatorname{SymSeq}_C$ 上的 bifunctor。

**证明.** 若 $\varphi:(S,\kappa;c)\to(S',\lambda;c)$ 是同构，则把代表映射 $f:S\to T$ 改为 $f\varphi^{-1}:S'\to T$，并在每个纤维上应用 $\varphi$ 的限制。该操作与目标双射关系交换，故通过 colimit。恒等与复合由源重标号和纤维限制的函子性给出。对 $X,Y$ 的态射逐装饰作用，显然与同一商关系相容，因此得到 bifunctor。$\square$

**命题 K.6.** $\circ_C$ 的结合约束由可复合 colored finite maps 的两层共同表示给出，并满足五边形相干性。

**证明.** 两边都由二层映射 $S\xrightarrow{g}U\xrightarrow{p}T$、中间颜色函数 $\epsilon:U\to C$、外层颜色函数 $\delta:T\to C$ 及相容的 $X,Y,Z$ 装饰表示，并对 $U,T$ 的双射取商。一个方向先展开 $(X\circ_CY)(U)$；另一方向把每个 $(pg)^{-1}(t)$ 中的 $Y\circ_CZ$ 目标集合不交并为 $p^{-1}(t)$。这两个操作互逆。四重代入对应三层映射和逐层颜色函数，所有结合路径在该共同数据上相同，故满足五边形。$\square$

## K.3 自由 colored operad

**定义 K.7.** 设 $E$ 是 $C$-colored symmetric sequence。一个 $C$-colored $S$-叶标号树由有根树 $T$、叶标号 $S\cong\operatorname{Leaf}(T)$、边颜色函数
$$
\chi:E(T)\to C
$$
组成，要求每个叶 $s$ 的颜色为 $\kappa(s)$，根边颜色为输出颜色 $c$。一个 $E$-装饰是对每个顶点 $v$ 选取
$$
e_v\in E(\operatorname{In}(v),\chi|_{\operatorname{In}(v)};\chi(\operatorname{out}(v))).
$$

记这类树的群胚为 $\mathbf{Tree}_{C}(S,\kappa;c)$。

**定义 K.8.** 自由 $C$-colored operad 定义为
$$
\mathbb F_C(E)(S,\kappa;c)
=
\int^{T\in\mathbf{Tree}_{C}(S,\kappa;c)}
\prod_{v\in V(T)}
E(\operatorname{In}(v),\chi_v;\chi(\operatorname{out}(v))).
$$

**命题 K.9.** $\mathbb F_C(E)$ 是由 $E$ 生成的自由 $C$-colored operad。

**证明.** 与定理 H.11 的自由单色 operad 证明相同，但每条内部边带颜色。给定 colored symmetric sequence morphism $E\to U\mathcal P$，对每棵 $E$-装饰 colored tree 自底向上复合；内部边颜色保证每次复合的输出颜色与外层输入颜色匹配。Operad 结合律说明结果与收缩顺序无关，等变性说明结果通过树同构商。反过来，任意 operad morphism 限制到 corolla 给出 $E\to U\mathcal P$，且由于所有装饰树由 corolla grafting 生成，该限制唯一决定原 morphism。$\square$

## K.4 编码代数同态

**定义 K.10.** 令 $C=\{A,B\}$。定义 colored operad $\operatorname{MorAss}$，其代数是两个含单位结合代数和一个代数同态 $A\to B$。它由以下生成元和关系给出：

1. $m_A:(A,A)\to A$、$e_A:()\to A$，满足 $\operatorname{Ass}$ 关系；
2. $m_B:(B,B)\to B$、$e_B:()\to B$，满足 $\operatorname{Ass}$ 关系；
3. $f:(A)\to B$；
4. 关系
   $$
   f(m_A(x,y))=m_B(fx,fy),\qquad f(e_A)=e_B.
   $$

**命题 K.11.** $\operatorname{MorAss}$-代数等价于含单位结合代数同态。

**证明.** 代数结构在颜色 $A$ 和 $B$ 上给出两个集合或模对象。生成元 $m_A,e_A$ 与关系使 $A$ 成为含单位结合代数；$m_B,e_B$ 同理。生成元 $f$ 给出函数或线性映射 $A\to B$。最后两条关系正是保持乘法与单位。反向由任一代数同态按生成元泛性质给出 colored operad 代数。$\square$

## K.5 模、双模与代数-模对

**定义 K.12.** 令 $C=\{A,M\}$。定义 colored operad $\operatorname{LMod}$，其代数是一个含单位结合代数 $A$ 与一个左 $A$-模 $M$。生成元为
$$
m_A:(A,A)\to A,\qquad e_A:()\to A,\qquad
\lambda:(A,M)\to M.
$$
关系为 $\operatorname{Ass}$ 的结合单位关系，以及
$$
\lambda(m_A(a,b),x)=\lambda(a,\lambda(b,x)),
\qquad
\lambda(e_A,x)=x.
$$

**命题 K.13.** 在线性语境中，$\operatorname{LMod}$-代数等价于含单位结合 $R$-代数 $A$ 与左 $A$-模 $M$。

**证明.** 生成元和关系直接给出双线性乘法、单位和左作用。第一组关系是代数公理；第二组关系是模结合律和单位律。反向由任一代数和左模通过生成元关系泛性质给出 colored operad morphism。$\square$

**定义 K.14.** 令 $C=\{A,B,M\}$。双模 operad $\operatorname{Bimod}$ 由两个结合代数颜色 $A,B$ 和一个对象 $M$ 组成，带生成元
$$
\lambda:(A,M)\to M,\qquad
\rho:(M,B)\to M,
$$
并施加左模、右模和左右作用交换关系
$$
\rho(\lambda(a,x),b)=\lambda(a,\rho(x,b)).
$$

**命题 K.15.** 在线性语境中，$\operatorname{Bimod}$-代数等价于三元组 $(A,B,M)$，其中 $A,B$ 是含单位结合 $R$-代数，$M$ 是 $(A,B)$-双模。

**证明.** 与命题 K.13 相同；新增交换关系正是双模公理中左右作用相容性。$\square$

## K.6 Enriched colored operad

**定义 K.16.** 设 $(\mathcal V,\otimes,\mathbb 1)$ 是 cocomplete 对称幺半范畴，并且张量积分别保持小 colimits。一个 $\mathcal V$-enriched $C$-colored symmetric sequence 是函子
$$
X:\mathbf B_C\to\mathcal V.
$$
代入乘积定义为
$$
(X\circ_CY)(S,\kappa;c)
=
\operatorname*{colim}_{(f:S\to T)\in\operatorname{Fib}(S)}
\coprod_{\delta:T\to C}
X(T,\delta;c)
\otimes
\bigotimes_{t\in T}
Y(f^{-1}(t),\kappa|_{f^{-1}(t)};\delta(t)).
$$
这里 colimit 在 $\mathcal V$ 中取，目标双射同步重标号 $T$、$\delta$ 与张量因子；空纤维保留 nullary $Y$-项。若所有这些 nullary 项均为初对象，才可改写为非空 colored 分块公式。
$\mathcal V$-enriched colored operad 是该代入乘积下的幺半对象。

**说明 K.17.** 当 $\mathcal V=\mathbf{Mod}_R$ 时得到线性 colored operad；当 $\mathcal V=\mathbf{Ch}_k$ 时得到 dg colored operad；当 $\mathcal V=\mathbf{sSet}$ 或 $\mathbf{Top}$ 时得到 simplicial 或 topological colored operad。

**命题 K.18.** 在定义 K.16 的假设下，$\circ_C$ 在 $\mathcal V$-值 colored symmetric sequences 上满足结合律和单位律。

**证明.** 两个加括号方向都由可复合映射 $S\to U\to T$、逐层颜色函数和同一组 $X,Y,Z$ 装饰表示。唯一新增点是要把
$$
X\otimes\bigotimes_B\left(Y_B\otimes\bigotimes_D Z_D\right)
$$
与
$$
\left(X\otimes\bigotimes_B Y_B\right)\otimes\bigotimes_D Z_D
$$
通过 $\mathcal V$ 的幺半结合约束和 braiding 相识别，并要求张量积保持公式中的 coproduct 与群胚 colimit。右单位强制 $S\to T$ 为双射，左单位强制 $T$ 为单点，这两项也覆盖 $S=\varnothing$。Mac Lane 相干性与多层映射复合的严格结合性保证五边形和三角形交换。$\square$

## K.7 模型结构边界

**警告 K.19.** Enriched colored operad 的定义不自动给出其代数范畴的模型结构。若 $\mathcal V$ 是模型范畴，需要额外检查附录 G 的 admissibility 条件。特别是 colored operads 的自由代数包含多颜色的对称幂和 coinvariants；这些在一般底环或非 cofibrant 情况下可能不保持弱等价。

**外部输入定理 K.20.** 在满足 Pavlov-Scholbach 或 Berger-Moerdijk 型 admissibility 假设的对称幺半模型范畴中，small colored symmetric operads 的代数范畴可获得 transferred 模型结构。精确陈述依赖 symmetric h-monoidality、symmetric flatness、tractability 和 smallness 等条件。

## K.8 本附录小结

Colored operad 是多对象、多类型代数系统的基本语言。它不仅编码多元运算，也编码“哪些输入类型可以组合到哪个输出类型”。代数同态、左模、双模和范畴对象都可由 colored operad 生成元关系描述。Enriched 版本把 Hom 集替换为 $\mathcal V$ 中对象，但一旦进入同伦语境，admissibility 和 rectification 条件必须另行检查。
