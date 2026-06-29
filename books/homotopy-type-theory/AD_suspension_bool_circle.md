# 附录 AD：二点类型悬挂与圆的等价

本附录补全例 10.5：
$$
\mathsf{susp}(\mathbf 2)\simeq\mathbb S^1.
$$
记二点类型为 $\mathbf 2$，其元素为 $0_{\mathbf 2},1_{\mathbf 2}$。悬挂的点构造子为
$$
\mathsf{north},\mathsf{south}:\mathsf{susp}(\mathbf 2),
$$
路径构造子为
$$
\mathsf{merid}(i):\mathsf{north}=\mathsf{south}
\qquad(i:\mathbf 2).
$$

## AD.1 从悬挂到圆

**定义 AD.1.** 定义
$$
F:\mathsf{susp}(\mathbf 2)\to\mathbb S^1
$$
由悬挂递归给出：
$$
F(\mathsf{north})\coloneqq\mathsf{base},
\qquad
F(\mathsf{south})\coloneqq\mathsf{base},
$$
并在 meridian 上规定
$$
\mathsf{ap}_F(\mathsf{merid}(0_{\mathbf 2}))\coloneqq\mathsf{refl}_{\mathsf{base}},
$$
$$
\mathsf{ap}_F(\mathsf{merid}(1_{\mathbf 2}))\coloneqq\mathsf{loop}.
$$

## AD.2 从圆到悬挂

**定义 AD.2.** 定义
$$
G:\mathbb S^1\to\mathsf{susp}(\mathbf 2)
$$
由圆递归给出：
$$
G(\mathsf{base})\coloneqq\mathsf{north},
$$
且
$$
\mathsf{ap}_G(\mathsf{loop})
\coloneqq
\mathsf{merid}(1_{\mathbf 2})\cdot\mathsf{merid}(0_{\mathbf 2})^{-1}.
$$
右边是 $\mathsf{north}=\mathsf{north}$ 的 loop。

## AD.3 第一个复合

**命题 AD.3.** 有同伦
$$
F\circ G\sim\mathsf{id}_{\mathbb S^1}.
$$

**证明.** 对圆作依赖消去。基点处，
$$
F(G(\mathsf{base}))\equiv F(\mathsf{north})\equiv\mathsf{base},
$$
取反身路径。

需要检查 loop 上的相容性。展开定义，
$$
\mathsf{ap}_{F\circ G}(\mathsf{loop})
=
\mathsf{ap}_F(\mathsf{ap}_G(\mathsf{loop}))
$$
化为
$$
\mathsf{ap}_F(\mathsf{merid}(1)\cdot\mathsf{merid}(0)^{-1})
=
\mathsf{loop}.
$$
由 $\mathsf{ap}$ 保持复合与逆，
$$
\mathsf{ap}_F(\mathsf{merid}(1))\cdot
\mathsf{ap}_F(\mathsf{merid}(0))^{-1}
=
\mathsf{loop}\cdot\mathsf{refl}^{-1}
=
\mathsf{loop}.
$$
这正是圆依赖消去所需的路径构造子相容性。$\square$

## AD.4 第二个复合

**命题 AD.4.** 有同伦
$$
G\circ F\sim\mathsf{id}_{\mathsf{susp}(\mathbf 2)}.
$$

**证明.** 对悬挂作依赖消去，目标族为
$$
P(z)\coloneqq G(F(z))=z.
$$
在两个点上定义：
$$
H_{\mathsf{north}}\coloneqq\mathsf{refl}_{\mathsf{north}},
$$
$$
H_{\mathsf{south}}\coloneqq\mathsf{merid}(0_{\mathbf 2}):\mathsf{north}=\mathsf{south}.
$$
需对每个 $i:\mathbf 2$ 检查沿 $\mathsf{merid}(i)$ 的依赖相容性。

当 $i=0_{\mathbf 2}$ 时，$F$ 把 $\mathsf{merid}(0)$ 送到反身路径；transport 计算化为说明
$$
\mathsf{refl}_{\mathsf{north}}
$$
沿 $\mathsf{merid}(0)$ 后得到 $\mathsf{merid}(0)$，这是路径族 $P$ 的标准 transport 计算。

当 $i=1_{\mathbf 2}$ 时，$F$ 把 $\mathsf{merid}(1)$ 送到 $\mathsf{loop}$，而 $G$ 把 $\mathsf{loop}$ 送到
$$
\mathsf{merid}(1)\cdot\mathsf{merid}(0)^{-1}.
$$
展开 dependent action 后，需要证明的路径代数等式等价于
$$
(\mathsf{merid}(1)\cdot\mathsf{merid}(0)^{-1})\cdot\mathsf{merid}(0)
=
\mathsf{merid}(1),
$$
这由结合律和右逆律得到。$\square$

## AD.5 等价

**定理 AD.5.** 有等价
$$
\mathsf{susp}(\mathbf 2)\simeq\mathbb S^1.
$$

**证明.** 函数 $F$ 与 $G$ 由 AD.1、AD.2 定义。AD.3、AD.4 给出双向同伦，即 $F$ 有准逆。由 G.7 中准逆推出等价，得到所需等价。$\square$

**依赖说明。** 本证明使用圆和悬挂的递归/依赖消去规则，以及路径复合的结合、单位和逆律。若在 propositional HIT computation 口径下机器化，需要显式插入圆与悬挂路径构造子的计算路径。
