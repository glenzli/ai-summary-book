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
由 pushout 递归：
$$
\Phi(\mathsf{inl}(b))\coloneqq\mathsf{inl}'(\beta(b)),
$$
$$
\Phi(\mathsf{inr}(c))\coloneqq\mathsf{inr}'(\gamma(c)).
$$
对 glue 构造子，需给出路径
$$
\mathsf{inl}'(\beta(f(a)))=\mathsf{inr}'(\gamma(g(a))).
$$
取
$$
\mathsf{ap}_{\mathsf{inl}'}(H_f(a))
\cdot
\mathsf{glue}'(\alpha(a))
\cdot
\mathsf{ap}_{\mathsf{inr}'}(H_g(a))^{-1}.
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
由 pushout 递归：
$$
\Psi(\mathsf{inl}'(b'))\coloneqq\mathsf{inl}(\beta^{-1}(b')),
$$
$$
\Psi(\mathsf{inr}'(c'))\coloneqq\mathsf{inr}(\gamma^{-1}(c')).
$$
对 glue 构造子取路径
$$
\mathsf{ap}_{\mathsf{inl}}(H_f'(a'))
\cdot
\mathsf{glue}(\alpha^{-1}(a'))
\cdot
\mathsf{ap}_{\mathsf{inr}}(H_g'(a'))^{-1}.
$$

## AI.4 两个复合

**命题 AI.4.** 有同伦
$$
\Psi\circ\Phi\sim\mathsf{id}_P.
$$

**证明核.** 对 $P$ 作 pushout 依赖消去。

在左点构造子上，目标为
$$
\mathsf{inl}(\beta^{-1}(\beta(b)))=\mathsf{inl}(b),
$$
由 $\mathsf{ap}_{\mathsf{inl}}$ 作用于 $\beta$ 的逆律得到。

在右点构造子上，目标为
$$
\mathsf{inr}(\gamma^{-1}(\gamma(c)))=\mathsf{inr}(c),
$$
由 $\gamma$ 的逆律得到。

在 glue 构造子上，需要验证一个路径代数相容方块。展开 $\Phi$ 与 $\Psi$ 对 glue 的定义后，目标由以下事实组成：

1.  $\alpha,\beta,\gamma$ 的左右逆同伦；
2.  $H_f,H_g$ 与反向相干 $H_f',H_g'$ 的定义相容；
3.  pushout 的 $\mathsf{glue}$ 计算规则；
4.  路径复合的结合律、单位律和逆律。

对等价 $\alpha,\beta,\gamma$ 作等价归纳，可把该相容性化到三者均为恒等等价、$H_f,H_g$ 为反身同伦的情形；此时 $\Phi,\Psi$ 对点和 glue 都 judgmentally/propositionally 化为恒等，目标为反身路径。$\square$

**命题 AI.5.** 有同伦
$$
\Phi\circ\Psi\sim\mathsf{id}_{P'}.
$$

**证明核.** 与 AI.4 对称，对 $P'$ 作 pushout 依赖消去，并使用 $\alpha,\beta,\gamma$ 的另一侧逆律。$\square$

## AI.5 等价不变性

**定理 AI.6（pushout 等价不变性）.** 若两个 span 由 AI.1 的数据等价，则
$$
\mathsf{pushout}(f,g)\simeq\mathsf{pushout}(f',g').
$$

**证明.** $\Phi$ 与 $\Psi$ 由 AI.2、AI.3 给出。AI.4、AI.5 证明它们互为准逆。由 G.7 中准逆推出等价，得到所需等价。$\square$

**推论 AI.7（cofiber、wedge 的等价不变性）.** cofiber、wedge 和由有限次 pushout、和类型、单位类型组合得到的基础同伦余极限构造，在输入图等价替换下保持等价。

**证明.** 展开这些构造为 pushout 表达式，逐次应用 AI.6。$\square$
