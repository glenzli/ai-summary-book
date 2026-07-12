# 附录 AI：Pushout 的等价不变性证明核

本附录补全命题 10.12 的 pushout 情形。一般同伦余极限的函子性可由相同模式推广；本书在基础层只需要 pushout、cofiber、wedge 等由 pushout 生成的情形。

## AI.1 Span 的等价

设有两个 span：
$$
A\xrightarrow{f}B,\qquad A\xrightarrow{g}C,
$$
$$
A'\xrightarrow{f'}B',\qquad A'\xrightarrow{g'}C'.
$$
记其 pushout 为
$$
P\coloneqq\mathsf{pushout}(f,g),
\qquad
P'\coloneqq\mathsf{pushout}(f',g').
$$

**定义 AI.1（span 等价）.** 一个 span 等价由等价
$$
\alpha:A\simeq A',
\qquad
\beta:B\simeq B',
\qquad
\gamma:C\simeq C'
$$
和相干同伦
$$
H_f:\prod_{a:A}\beta(f(a))=f'(\alpha(a)),
$$
$$
H_g:\prod_{a:A}\gamma(g(a))=g'(\alpha(a))
$$
组成。

## AI.2 正向映射

**定义 AI.2（pushout 正向映射）.** 给定 AI.1 的数据，定义
$$
\Phi:P\to P'
$$
所用的点数据为
$$
u_\Phi(b)\coloneqq\mathsf{inl}'(\beta(b)),
\qquad
v_\Phi(c)\coloneqq\mathsf{inr}'(\gamma(c)).
$$
对 glue 构造子，定义路径数据
$$
h_\Phi(a)\coloneqq
\mathsf{ap}_{\mathsf{inl}'}(H_f(a))
\cdot
\mathsf{glue}'(\alpha(a))
\cdot
\mathsf{ap}_{\mathsf{inr}'}(H_g(a))^{-1}.
$$
它具有所需类型
$$
\mathsf{inl}'(\beta(f(a)))=\mathsf{inr}'(\gamma(g(a))).
$$
令
$$
\Phi\coloneqq\mathsf{pushRec}(u_\Phi,v_\Phi,h_\Phi).
$$
于是点计算为 judgmental，而 glue 计算是命名路径
$$
\beta^\Phi_{\mathsf{glue}}(a)\coloneqq
\beta^{\mathsf{pushRec}}_{\mathsf{glue}}(a):
\mathsf{ap}_{\Phi}(\mathsf{glue}(a))=h_\Phi(a).
$$

## AI.3 反向映射

设 $\alpha^{-1},\beta^{-1},\gamma^{-1}$ 为等价的逆函数，并用等价逆的左右同伦把 AI.1 的两个相干方块反向运输，得到
$$
H_f':\prod_{a':A'}\beta^{-1}(f'(a'))=f(\alpha^{-1}(a')),
$$
$$
H_g':\prod_{a':A'}\gamma^{-1}(g'(a'))=g(\alpha^{-1}(a')).
$$

**定义 AI.3（pushout 反向映射）.** 定义
$$
\Psi:P'\to P
$$
的点数据为
$$
u_\Psi(b')\coloneqq\mathsf{inl}(\beta^{-1}(b')),
\qquad
v_\Psi(c')\coloneqq\mathsf{inr}(\gamma^{-1}(c')).
$$
glue 数据取
$$
h_\Psi(a')\coloneqq
\mathsf{ap}_{\mathsf{inl}}(H_f'(a'))
\cdot
\mathsf{glue}(\alpha^{-1}(a'))
\cdot
\mathsf{ap}_{\mathsf{inr}}(H_g'(a'))^{-1}.
$$
令
$$
\Psi\coloneqq\mathsf{pushRec}(u_\Psi,v_\Psi,h_\Psi).
$$
它在点上 judgmentally 计算，并有命名的 propositional glue 计算
$$
\beta^\Psi_{\mathsf{glue}}(a')\coloneqq
\beta^{\mathsf{pushRec}}_{\mathsf{glue}}(a'):
\mathsf{ap}_{\Psi}(\mathsf{glue}'(a'))=h_\Psi(a').
$$

## AI.4 两个复合

**命题 AI.4.** 有同伦
$$
\Psi\circ\Phi\sim\mathsf{id}_P.
$$

**证明核.** 令
$$
Q(z)\coloneqq\Psi(\Phi(z))=z.
$$
对 $P$ 作 pushout 依赖消去。

在左点构造子上，取 $\beta$ 的逆律经 $\mathsf{ap}_{\mathsf{inl}}$ 的像作为
$$
k_{\mathsf{inl}}(b):
\mathsf{inl}(\beta^{-1}(\beta(b)))=\mathsf{inl}(b).
$$

在右点构造子上，同理由 $\gamma$ 的逆律得到
$$
k_{\mathsf{inr}}(c):
\mathsf{inr}(\gamma^{-1}(\gamma(c)))=\mathsf{inr}(c).
$$

在 glue 构造子上，依赖消去要求一个明确的相容项
$$
m_K(a):
\mathsf{transport}^{Q}
(\mathsf{glue}(a),k_{\mathsf{inl}}(f(a)))
=k_{\mathsf{inr}}(g(a)).
$$
用路径族 $Q$ 的 transport 公式把它改写为路径代数方块。展开 $\mathsf{ap}_{\Psi\circ\Phi}(\mathsf{glue}(a))$ 时，先沿 $\beta^\Phi_{\mathsf{glue}}(a)$ 把内层像改写为 $h_\Phi(a)$；再用 $\mathsf{ap}_{\Psi}$ 保持复合，并沿 $\beta^\Psi_{\mathsf{glue}}(\alpha(a))$ 改写其中的 $\mathsf{glue}'(\alpha(a))$。剩余目标由以下事实组成：

1.  $\alpha,\beta,\gamma$ 的左右逆同伦；
2.  $H_f,H_g$ 与反向相干 $H_f',H_g'$ 的定义相容；
3.  命名计算路径 $\beta^\Phi_{\mathsf{glue}}$ 与 $\beta^\Psi_{\mathsf{glue}}$；
4.  路径复合的结合律、单位律和逆律。

对等价 $\alpha,\beta,\gamma$ 作等价归纳，可把该相容性化到三者均为恒等等价、$H_f,H_g$ 为反身同伦的情形；点数据 judgmentally 化为恒等，glue 数据经上述两个命名 $\beta$ 路径 propositionally 化为原 glue，目标为反身路径。这给出 $m_K$。

令
$$
K\coloneqq\mathsf{pushInd}_{Q}
(k_{\mathsf{inl}},k_{\mathsf{inr}},m_K).
$$
L.20 给出 judgmental 点计算和命名的 glue 计算路径
$$
\beta^K_{\mathsf{glue}}(a)\coloneqq
\beta^{\mathsf{pushInd}}_{\mathsf{glue}}(a):
\mathsf{apd}_{K}(\mathsf{glue}(a))=m_K(a).
$$
故 $K$ 是所需同伦。$\square$

**命题 AI.5.** 有同伦
$$
\Phi\circ\Psi\sim\mathsf{id}_{P'}.
$$

**证明核.** 与 AI.4 对称。对族 $Q'(z')\coloneqq\Phi(\Psi(z'))=z'$ 作 pushout 依赖消去。把 $\beta$、$\gamma$ 的另一侧逆律分别经 $\mathsf{ap}_{\mathsf{inl}'}$、$\mathsf{ap}_{\mathsf{inr}'}$ 后得到的点数据记为
$$
\ell_{\mathsf{inl}}(b'):
\mathsf{inl}'(\beta(\beta^{-1}(b')))=\mathsf{inl}'(b'),
$$
$$
\ell_{\mathsf{inr}}(c'):
\mathsf{inr}'(\gamma(\gamma^{-1}(c')))=\mathsf{inr}'(c').
$$
使用 $\alpha,\beta,\gamma$ 的另一侧逆律，并沿 $\beta^\Psi_{\mathsf{glue}}(a')$、$\beta^\Phi_{\mathsf{glue}}(\alpha^{-1}(a'))$ 改写两个递归子的 glue 像，得到相容项 $m_L(a')$。定义 $L\coloneqq\mathsf{pushInd}_{Q'}(\ell_{\mathsf{inl}},\ell_{\mathsf{inr}},m_L)$，并保留命名计算路径
$$
\beta^L_{\mathsf{glue}}(a'):
\mathsf{apd}_{L}(\mathsf{glue}'(a'))=m_L(a').
$$
于是 $L$ 给出所需同伦。$\square$

## AI.5 等价不变性

**定理 AI.6（pushout 等价不变性）.** 若两个 span 由 AI.1 的数据等价，则
$$
\mathsf{pushout}(f,g)\simeq\mathsf{pushout}(f',g').
$$

**证明.** $\Phi$ 与 $\Psi$ 由 AI.2、AI.3 给出。AI.4、AI.5 证明它们互为准逆。由 G.7 中准逆推出等价，得到所需等价。$\square$

**推论 AI.7（cofiber、wedge 的等价不变性）.** cofiber、wedge 和由有限次 pushout、和类型、单位类型组合得到的基础同伦余极限构造，在输入图等价替换下保持等价。

**证明.** 展开这些构造为 pushout 表达式，逐次应用 AI.6。$\square$
