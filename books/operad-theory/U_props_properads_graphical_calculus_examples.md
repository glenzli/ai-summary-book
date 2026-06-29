# 附录 U：PROP、properad 与 wheeled 图计算样例

本附录补充第七章，把 PROP/properad 的公理写成可检查的图计算。完整自由 properad、自由 PROP 和 wheeled graph complex 的构造仍作为外部输入；本附录只证明由定义直接推出的低阶公式。

## U.1 PROP 中的 interchange law

设 $\mathsf P$ 是 PROP。取
$$
f_1\in\mathsf P(m_1,n_1),\quad
f_2\in\mathsf P(m_2,n_2),
$$
$$
g_1\in\mathsf P(n_1,r_1),\quad
g_2\in\mathsf P(n_2,r_2).
$$
则 interchange law 为
$$
(f_1\otimes f_2)\circ(g_1\otimes g_2)
=
(f_1\circ g_1)\otimes(f_2\circ g_2).
$$

**命题 U.1.** 在 endomorphism PROP
$$
\operatorname{End}_V(m,n)=\operatorname{Hom}(V^{\otimes n},V^{\otimes m})
$$
中，interchange law 成立。

**证明.** 对纯张量
$$
x=x_1\otimes x_2\in V^{\otimes r_1}\otimes V^{\otimes r_2}
$$
有
$$
(g_1\otimes g_2)(x)=g_1(x_1)\otimes g_2(x_2).
$$
再作用 $f_1\otimes f_2$ 得
$$
f_1(g_1(x_1))\otimes f_2(g_2(x_2)).
$$
右侧
$$
((f_1\circ g_1)\otimes(f_2\circ g_2))(x)
$$
给出同一纯张量。由线性性推广到所有元素。$\square$

## U.2 双代数兼容关系的坐标检查

设 $V$ 是 $R$-模，$\mu:V\otimes V\to V$，$\Delta:V\to V\otimes V$。双代数兼容关系在 PROP 中写作
$$
\Delta\circ\mu
=
(\mu\otimes\mu)\circ
(\operatorname{id}\otimes\tau\otimes\operatorname{id})
\circ
(\Delta\otimes\Delta).
$$

**命题 U.2.** 对 $x,y\in V$，该关系等价于 Sweedler 公式
$$
\Delta(xy)=\sum x_{(1)}y_{(1)}\otimes x_{(2)}y_{(2)}.
$$

**证明.** 左侧为 $\Delta(\mu(x\otimes y))=\Delta(xy)$。右侧先给出
$$
(\Delta\otimes\Delta)(x\otimes y)
=
\sum x_{(1)}\otimes x_{(2)}\otimes y_{(1)}\otimes y_{(2)}.
$$
中间置换 $\operatorname{id}\otimes\tau\otimes\operatorname{id}$ 把第二、三因子交换，得到
$$
\sum x_{(1)}\otimes y_{(1)}\otimes x_{(2)}\otimes y_{(2)}.
$$
再作用 $\mu\otimes\mu$ 得到
$$
\sum x_{(1)}y_{(1)}\otimes x_{(2)}y_{(2)}.
$$
两边相等正是所需公式。$\square$

## U.3 Frobenius 关系的 PROP 形式

一个 Frobenius algebra 同时有乘法 $\mu$ 和余乘法 $\Delta$，满足
$$
(\mu\otimes\operatorname{id})\circ(\operatorname{id}\otimes\Delta)
=
\Delta\circ\mu
=
(\operatorname{id}\otimes\mu)\circ(\Delta\otimes\operatorname{id}).
$$

**说明 U.3.** 这与双代数兼容不同。双代数要求 $\Delta$ 是代数同态；Frobenius 关系要求 $\Delta$ 是 $A$-bimodule map。二者对应不同 PROP。

**命题 U.4.** 若 $A$ 是有限维 Frobenius algebra，其非退化 pairing
$$
\langle-,-\rangle:A\otimes A\to k
$$
与乘法满足
$$
\langle ab,c\rangle=\langle a,bc\rangle,
$$
则由 pairing 的共评价定义的 $\Delta$ 满足 Frobenius 关系。

**证明边界.** 证明需要选择 dual bases 并用非退化 pairing 验证三个 maps $A^{\otimes2}\to A^{\otimes2}$ 在任意测试元素下 pairing 相同。该计算属于 Frobenius algebra 标准线性代数。本书只把它作为 PROP 关系的识别样例，不在后续证明链中使用。$\square$

## U.4 Properad 的连通图复合

考虑两个 operations
$$
p\in\mathcal P(2,1),\qquad q\in\mathcal P(1,2).
$$
把 $q$ 的唯一输出接到 $p$ 的一个输入，得到连通图，整体类型为
$$
(2,\ 2)
$$
或在本书约定中为两输入两输出运算，具体排列取决于接线位置。

**命题 U.5.** properad 复合记录接线图，而不只记录代数表达式。

**证明.** Properad 的复合由连通 directed graph $G$ 指定。即使顶点装饰相同，不同接线位置给出的 graph morphism 可能不同，并通过输入输出重标号相联系。定义 7.14 的图复合 $\mu_G$ 以 $G$ 为索引，因此接线是结构的一部分。$\square$

**说明 U.6.** Operad 树是 special case：每个顶点只有一个输出，且整体图有一个输出。Properad 允许多个输出，因此不能用普通 rooted tree 完全编码。

## U.5 PROP 与 properad 的不连通差异

设 $p\in\mathcal P(m,n)$，$q\in\mathcal P(m',n')$ 是 properad 中两个 operation。如果只有 properad 结构，没有额外 PROP 结构，则
$$
p\otimes q
$$
不是 properad 的基本复合，因为对应图不连通。

**命题 U.7.** 由 PROP 遗忘得到的 properad 不能反推出水平张量的全部数据，除非给出自由 PROP 或额外结构。

**证明.** Properad 只记录连通图复合。PROP 的水平张量把两个不连通组件并排，得到一个从 $n+n'$ 输入到 $m+m'$ 输出的运算。该运算不由任何连通图复合给出。若只知道 properad 的连通复合，缺少不连通并排操作的指定。自由 PROP 构造正是把这些不连通并排形式加入并施加相干关系。$\square$

## U.6 Wheeled contraction 的 trace 公式

设 $V$ 是有限维 $k$-向量空间。对
$$
f:V^{\otimes n}\to V^{\otimes m}
$$
定义把第 $i$ 个输入与第 $j$ 个输出 contraction 的 map
$$
\operatorname{tr}_i^j(f):V^{\otimes(n-1)}\to V^{\otimes(m-1)}.
$$
选取基 $(e_a)$ 和对偶基 $(e^a)$，公式为
$$
\operatorname{tr}_i^j(f)(x_1,\ldots,\widehat{x_i},\ldots,x_n)
=
\sum_a
(\operatorname{id}^{\otimes j-1}\otimes e^a\otimes
\operatorname{id}^{\otimes m-j})
f(x_1,\ldots,e_a,\ldots,x_n),
$$
其中输出第 $j$ 位被 $e^a$ 评价后删除。

**命题 U.8.** 该定义不依赖所选基。

**证明.** 元素
$$
\sum_a e_a\otimes e^a\in V\otimes V^\vee
$$
是 identity endomorphism $\operatorname{id}_V\in V\otimes V^\vee$ 对应的 canonical coevaluation element。换基不改变 identity endomorphism，因此 contraction 公式不依赖基。$\square$

**警告 U.9.** 若 $V$ 不是 dualizable 对象，则 coevaluation/evaluation 不一定存在，trace 公式无意义。Wheeled endomorphism properad 必须附带 dualizability 或 trace 假设。

## U.7 图替换结合律

设 $G$ 是连通 directed graph。对每个顶点 $v\in V(G)$，取连通 graph $H_v$，其外腿类型与 $v$ 的输入输出匹配。把 $H_v$ 替换到 $v$ 得到总图
$$
G(H_v).
$$

**命题 U.10.** Properad 的图替换公理断言：
$$
\mu_G\big(\mu_{H_v}((p_w)_{w\in V(H_v)})\big)_{v\in V(G)}
=
\mu_{G(H_v)}((p_w)_{v,w}).
$$

**证明.** 这是定义 7.14 的相干性公理按符号展开。左侧先在每个小图 $H_v$ 内复合，再按外图 $G$ 复合。右侧在替换后的总图中一次性复合。Properad 公理要求二者相等。$\square$

**说明 U.11.** 对 operad，$G$ 和 $H_v$ 都是 rooted trees，该公式退化为树代入结合律。对 PROP，还需加入不连通水平张量与垂直复合的 interchange law。

## U.8 使用检查表

使用 PROP/properad 语言时必须说明：

1. 输入输出约定是 $\mathsf P(m,n)=\operatorname{Hom}(n,m)$ 还是相反；
2. 是否允许 arity/coarity $0$；
3. 是 PROP、properad、wheeled properad 还是 wheeled PROP；
4. 图是否必须连通、无环、有向；
5. 是否需要水平张量；
6. 是否需要 trace 或 dualizability；
7. 自由 PROP/properad 构造是否作为外部输入。

## U.9 小结

PROP 的核心计算是垂直复合、水平张量和 interchange law。Properad 保留多输入多输出和连通图复合，但不含任意不连通水平张量。Wheeled 结构进一步允许输出接回输入，因而必须有 trace 或 dualizability。双代数、Frobenius 代数和 trace 操作展示了这些差异在具体代数结构中的表现。
