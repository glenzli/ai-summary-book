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
所用的 meridian 数据 $m_F:\prod_{i:\mathbf 2}(\mathsf{base}=\mathsf{base})$ 由布尔消去定义为
$$
m_F(0_{\mathbf 2})\coloneqq\mathsf{refl}_{\mathsf{base}},
\qquad
m_F(1_{\mathbf 2})\coloneqq\mathsf{loop}.
$$
令
$$
F\coloneqq
\mathsf{suspRec}(\mathsf{base},\mathsf{base},m_F).
$$
由 L.18 的点 $\beta$ 规则，
$$
F(\mathsf{north})\equiv\mathsf{base},
\qquad
F(\mathsf{south})\equiv\mathsf{base}.
$$
meridian 上不是定义等号，而是以下两个命名计算路径：
$$
\beta^F_0\coloneqq
\beta^{\mathsf{suspRec}}_{\mathsf{merid}}(0_{\mathbf 2}):
\mathsf{ap}_F(\mathsf{merid}(0_{\mathbf 2}))
=\mathsf{refl}_{\mathsf{base}},
$$
$$
\beta^F_1\coloneqq
\beta^{\mathsf{suspRec}}_{\mathsf{merid}}(1_{\mathbf 2}):
\mathsf{ap}_F(\mathsf{merid}(1_{\mathbf 2}))
=\mathsf{loop}.
$$

## AD.2 从圆到悬挂

**定义 AD.2.** 定义
$$
G:\mathbb S^1\to\mathsf{susp}(\mathbf 2)
$$
先令
$$
\ell_G\coloneqq
\mathsf{merid}(1_{\mathbf 2})\cdot\mathsf{merid}(0_{\mathbf 2})^{-1}
:
\mathsf{north}=\mathsf{north}.
$$
由 L.15 的圆递归，以点数据 $\mathsf{north}$ 和 loop 数据 $\ell_G$ 定义 $G$。于是
$$
G(\mathsf{base})\equiv\mathsf{north},
$$
并有命名的 propositional 计算路径
$$
\beta^G_{\mathsf{loop}}\coloneqq\beta_{\mathsf{loop}}:
\mathsf{ap}_G(\mathsf{loop})=\ell_G.
$$

## AD.3 第一个复合

**命题 AD.3.** 有同伦
$$
F\circ G\sim\mathsf{id}_{\mathbb S^1}.
$$

**证明.** 令 $Q(x)\coloneqq F(G(x))=x$。对圆作依赖消去；基点数据取
$$
k_{\mathsf{base}}\coloneqq\mathsf{refl}_{\mathsf{base}}:Q(\mathsf{base}),
$$
其中两次点计算都是 judgmental。

依赖消去在 loop 上所需的 transport 条件，由路径族 $Q$ 的标准 transport 公式等价地化为
$$
\mathsf{ap}_{F\circ G}(\mathsf{loop})
=
\mathsf{loop}.
$$
函数复合的 $\mathsf{ap}$ 定律先把左边改写为
$$
\mathsf{ap}_F(\mathsf{ap}_G(\mathsf{loop})).
$$
沿 $\mathsf{ap}_{\lambda q.\,\mathsf{ap}_F(q)}(\beta^G_{\mathsf{loop}})$ 改写，再用 $\mathsf{ap}$ 保持复合与逆，得到
$$
\mathsf{ap}_F(\mathsf{merid}(1))\cdot
\mathsf{ap}_F(\mathsf{merid}(0))^{-1}
$$
并分别沿 $\beta^F_1$、$\beta^F_0$ 改写为 $\mathsf{loop}\cdot\mathsf{refl}^{-1}=\mathsf{loop}$。记由此得到的 dependent loop 数据为 $m_K$。L.16 给出
$$
K:\prod_{x:\mathbb S^1}Q(x),
$$
其命名计算路径为
$$
\beta^K_{\mathsf{loop}}:
\mathsf{apd}_{K}(\mathsf{loop})=m_K.
$$
于是 $K$ 即所需同伦。$\square$

## AD.4 第二个复合

**命题 AD.4.** 有同伦
$$
G\circ F\sim\mathsf{id}_{\mathsf{susp}(\mathbf 2)}.
$$

**证明.** 对悬挂作依赖消去，目标族为
$$
P(z)\coloneqq G(F(z))=z.
$$
在两个点上定义
$$
h_{\mathsf{north}}\coloneqq\mathsf{refl}_{\mathsf{north}},
$$
$$
h_{\mathsf{south}}\coloneqq\mathsf{merid}(0_{\mathbf 2}):\mathsf{north}=\mathsf{south}.
$$
需构造
$$
m_H(i):
\mathsf{transport}^{P}(\mathsf{merid}(i),h_{\mathsf{north}})
=h_{\mathsf{south}}
$$
对每个 $i:\mathbf 2$ 成立。对路径族 $P(z)=(G(F(z))=z)$ 使用标准 transport 公式后，该条件等价于自然性方程
$$
\mathsf{ap}_{G\circ F}(\mathsf{merid}(i))\cdot h_{\mathsf{south}}
=
h_{\mathsf{north}}\cdot\mathsf{merid}(i).
$$

当 $i=0_{\mathbf 2}$ 时，沿 $\beta^F_0$ 改写 $\mathsf{ap}_F(\mathsf{merid}(0))$，再用 $G$ 的点计算，方程化为 $\mathsf{refl}\cdot\mathsf{merid}(0)=\mathsf{refl}\cdot\mathsf{merid}(0)$。

当 $i=1_{\mathbf 2}$ 时，依次沿 $\beta^F_1$ 与 $\beta^G_{\mathsf{loop}}$ 改写，左边化为
$$
(\mathsf{merid}(1)\cdot\mathsf{merid}(0)^{-1})\cdot\mathsf{merid}(0)
$$
而右边化为 $\mathsf{merid}(1)$；结合律和右逆律给出所需路径。用 $\mathbf 2$ 的消去原则把两种情形组装成 $m_H$。

现在定义
$$
H\coloneqq\mathsf{suspInd}_{P}
(h_{\mathsf{north}},h_{\mathsf{south}},m_H).
$$
L.18 给出 judgmental 点计算和命名的 propositional meridian 计算路径
$$
\beta^H_{\mathsf{merid}}(i)\coloneqq
\beta^{\mathsf{suspInd}}_{\mathsf{merid}}(i):
\mathsf{apd}_{H}(\mathsf{merid}(i))=m_H(i).
$$
因此 $H:G\circ F\sim\mathsf{id}_{\mathsf{susp}(\mathbf 2)}$。$\square$

## AD.5 等价

**定理 AD.5.** 有等价
$$
\mathsf{susp}(\mathbf 2)\simeq\mathbb S^1.
$$

**证明.** 函数 $F$ 与 $G$ 由 AD.1、AD.2 定义。AD.3、AD.4 给出双向同伦，即 $F$ 有准逆。由 G.7 中准逆推出等价，得到所需等价。$\square$

**依赖说明。** 本证明使用圆和悬挂的递归/依赖消去规则，以及路径复合的结合、单位和逆律。所有路径构造子计算均通过 $\beta^F_0$、$\beta^F_1$、$\beta^G_{\mathsf{loop}}$、$\beta^K_{\mathsf{loop}}$ 与 $\beta^H_{\mathsf{merid}}$ 这些命名的 propositional 路径完成；没有把 meridian 或 loop 计算当作 judgmental equality。
