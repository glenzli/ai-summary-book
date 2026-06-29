# 附录 M：整数对象与 Successor 等价

## 目标

本附录补齐第十一章计算圆基本群时所需的整数对象。圆的 encode-decode 证明不能只使用集合商意义下的整数，因为 decode 需要从整数消去到 loop space，而 loop space 在证明完成前尚未知道是集合。因此本附录把两种整数呈现分开：

1.  **归纳整数** $\mathbb Z_{\mathsf{ind}}$：用于圆的覆盖、loop 幂和 encode-decode。它有普通归纳消去原则，可消去到任意类型族。
2.  **集合商整数** $\mathbb Z_{\mathsf{q}}$：用于集合层代数和第八章商类型练习。它展示传统 Grothendieck 群完成口径，但消去原则只适合集合值或命题值目标。

本书在第十一章默认把 $\mathbb Z$ 解释为 $\mathbb Z_{\mathsf{ind}}$，除非上下文明确说“商整数”。

## M.1 自然数的集合性与算术引理

**引理 M.1（自然数的 no-confusion 与集合性）.** $\mathbb N$ 是集合，即
$$
\mathsf{isSet}(\mathbb N).
$$

**证明.** 见附录 AE.4。证明使用自然数路径的 encode-decode：构造 code 族 $\mathsf{Code}_{\mathbb N}(m,n)$，证明 $(m=n)\simeq\mathsf{Code}_{\mathbb N}(m,n)$，再由 code 只归约到 $\mathbf 0$ 或 $\mathbf 1$ 推出路径类型是命题。$\square$

**引理 M.2（自然数加法规则）.** 本书使用以下自然数加法引理：

1.  $m+0\equiv m$；
2.  $m+\mathsf{succ}(n)\equiv\mathsf{succ}(m+n)$；
3.  $0+n=n$；
4.  $\mathsf{succ}(m)+n=\mathsf{succ}(m+n)$；
5.  $(a+b)+c=a+(b+c)$；
6.  $a+b=b+a$；
7.  若 $x+k=y+k$，则 $x=y$。

**证明.** 第 1、2 条是定义 3.13 的 judgmental 规则，第 3 条是命题 3.15。第 4 条对 $n$ 归纳：基步是 $\mathsf{refl}_{\mathsf{succ}(m)}$；归纳步中两边分别化为 successor，使用 $\mathsf{ap}_{\mathsf{succ}}$ 作用于归纳假设。第 5 条对 $c$ 归纳；基步由第 1 条给出，归纳步由第 2 条和 $\mathsf{ap}_{\mathsf{succ}}$ 作用于归纳假设给出。第 6 条对 $b$ 归纳，基步用第 1、3 条，归纳步用第 2、4 条和 $\mathsf{ap}_{\mathsf{succ}}$。第 7 条对 $k$ 归纳；基步为原路径，归纳步中路径形如 $\mathsf{succ}(x+k)=\mathsf{succ}(y+k)$，由自然数 no-confusion 得 $x+k=y+k$，再用归纳假设。$\square$

## M.2 归纳整数

**定义 M.3（归纳整数）.** 定义
$$
\mathbb Z_{\mathsf{ind}}\coloneqq\mathbb N+\mathbb N.
$$
记
$$
\mathsf{neg}(n)\coloneqq\mathsf{inl}(n),
\qquad
\mathsf{pos}(n)\coloneqq\mathsf{inr}(n).
$$
语义约定为 $\mathsf{neg}(n)$ 表示 $-(n+1)$，$\mathsf{pos}(n)$ 表示 $n$。定义
$$
0_{\mathbb Z}\coloneqq\mathsf{pos}(0).
$$

**命题 M.4（归纳整数的消去原则）.** 若 $P:\mathbb Z_{\mathsf{ind}}\to\mathcal U_i$，并给出
$$
u:\prod_{n:\mathbb N}P(\mathsf{neg}(n)),
\qquad
v:\prod_{n:\mathbb N}P(\mathsf{pos}(n)),
$$
则有
$$
\mathsf{ind}_{\mathbb Z}(u,v):\prod_{z:\mathbb Z_{\mathsf{ind}}}P(z)
$$
并在两个注入上满足和类型的 beta 规则。

**证明.** 这是和类型消去规则 3.7 的直接实例。$\square$

**命题 M.5（归纳整数是集合）.** $\mathbb Z_{\mathsf{ind}}$ 是集合。

**证明.** 由引理 M.1，$\mathbb N$ 是集合。附录 AE.8 证明和类型保持集合性，因此
$$
\mathbb Z_{\mathsf{ind}}\equiv\mathbb N+\mathbb N
$$
是集合。$\square$

## M.3 Successor 与 predecessor

**定义 M.6（整数 successor）.** 定义
$$
\mathsf{succ}_{\mathbb Z}:\mathbb Z_{\mathsf{ind}}\to\mathbb Z_{\mathsf{ind}}
$$
为
$$
\mathsf{succ}_{\mathbb Z}(\mathsf{neg}(0))\coloneqq\mathsf{pos}(0),
$$
$$
\mathsf{succ}_{\mathbb Z}(\mathsf{neg}(\mathsf{succ}(n)))\coloneqq\mathsf{neg}(n),
$$
$$
\mathsf{succ}_{\mathbb Z}(\mathsf{pos}(n))\coloneqq\mathsf{pos}(\mathsf{succ}(n)).
$$

**定义 M.7（整数 predecessor）.** 定义
$$
\mathsf{pred}_{\mathbb Z}:\mathbb Z_{\mathsf{ind}}\to\mathbb Z_{\mathsf{ind}}
$$
为
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{pos}(0))\coloneqq\mathsf{neg}(0),
$$
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{pos}(\mathsf{succ}(n)))\coloneqq\mathsf{pos}(n),
$$
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{neg}(n))\coloneqq\mathsf{neg}(\mathsf{succ}(n)).
$$

**定理 M.8（successor 与 predecessor 互逆）.** 有同伦
$$
\alpha:\prod_{z:\mathbb Z_{\mathsf{ind}}}
\mathsf{pred}_{\mathbb Z}(\mathsf{succ}_{\mathbb Z}(z))=z
$$
和
$$
\beta:\prod_{z:\mathbb Z_{\mathsf{ind}}}
\mathsf{succ}_{\mathbb Z}(\mathsf{pred}_{\mathbb Z}(z))=z.
$$

**证明.** 对 $z$ 作和类型消去。

若 $z\equiv\mathsf{pos}(0)$，则
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{succ}_{\mathbb Z}(\mathsf{pos}(0)))
\equiv
\mathsf{pred}_{\mathbb Z}(\mathsf{pos}(\mathsf{succ}(0)))
\equiv\mathsf{pos}(0),
$$
取反身路径。若 $z\equiv\mathsf{pos}(\mathsf{succ}(n))$，同样按定义两步化为 $\mathsf{pos}(\mathsf{succ}(n))$。

若 $z\equiv\mathsf{neg}(0)$，则
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{succ}_{\mathbb Z}(\mathsf{neg}(0)))
\equiv
\mathsf{pred}_{\mathbb Z}(\mathsf{pos}(0))
\equiv\mathsf{neg}(0).
$$
若 $z\equiv\mathsf{neg}(\mathsf{succ}(n))$，则
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{succ}_{\mathbb Z}(\mathsf{neg}(\mathsf{succ}(n))))
\equiv
\mathsf{pred}_{\mathbb Z}(\mathsf{neg}(n))
\equiv\mathsf{neg}(\mathsf{succ}(n)).
$$
这些给出 $\alpha$。

$\beta$ 的证明相同：分别对 $\mathsf{pos}(0)$、$\mathsf{pos}(\mathsf{succ}(n))$、$\mathsf{neg}(n)$ 展开定义，目标均化为反身路径。$\square$

**定理 M.9（successor 是等价）.** 函数
$$
\mathsf{succ}_{\mathbb Z}:\mathbb Z_{\mathsf{ind}}\to\mathbb Z_{\mathsf{ind}}
$$
是等价。

**证明（书内证明）.** 由定理 M.8，$\mathsf{pred}_{\mathbb Z}$ 是 $\mathsf{succ}_{\mathbb Z}$ 的准逆。用推论 G.7 把准逆提升为 fiber 可收缩意义下的等价；其中准逆到半伴随等价的相干化公式已在定理 G.4 展开。$\square$

**定义 M.10.** 记
$$
\mathsf{succEquiv}_{\mathbb Z}:\mathbb Z_{\mathsf{ind}}\simeq\mathbb Z_{\mathsf{ind}}
$$
为定理 M.9 得到的自等价。

## M.4 整数加法与 loop 幂

**定义 M.11（迭代）.** 对任意函数 $f:A\to A$，定义
$$
f^{[0]}(a)\coloneqq a,
\qquad
f^{[\mathsf{succ}(n)]}(a)\coloneqq f(f^{[n]}(a)).
$$

**定义 M.12（整数加法）.** 对 $z,w:\mathbb Z_{\mathsf{ind}}$，定义
$$
z+w:\mathbb Z_{\mathsf{ind}}
$$
为对 $w$ 消去：
$$
z+\mathsf{pos}(n)\coloneqq
(\mathsf{succ}_{\mathbb Z})^{[n]}(z),
$$
$$
z+\mathsf{neg}(n)\coloneqq
(\mathsf{pred}_{\mathbb Z})^{[\mathsf{succ}(n)]}(z).
$$

**命题 M.13（加法的基础计算规则）.** 对任意 $z$ 有
$$
z+0_{\mathbb Z}\equiv z,
$$
$$
z+\mathsf{pos}(\mathsf{succ}(n))
\equiv
\mathsf{succ}_{\mathbb Z}(z+\mathsf{pos}(n)),
$$
$$
z+\mathsf{neg}(0)\equiv\mathsf{pred}_{\mathbb Z}(z),
$$
$$
z+\mathsf{neg}(\mathsf{succ}(n))
\equiv
\mathsf{pred}_{\mathbb Z}(z+\mathsf{neg}(n)).
$$

**证明.** 全部是定义 M.11-M.12 的递归计算规则。$\square$

**命题 M.14（加法群律）.** $\mathbb Z_{\mathsf{ind}}$ 带有通常的交换群结构，其中单位是 $0_{\mathbb Z}$，逆元由符号翻转给出。

**证明.** 定义逆元
$$
-\mathsf{pos}(0)\coloneqq\mathsf{pos}(0),
$$
$$
-\mathsf{pos}(\mathsf{succ}(n))\coloneqq\mathsf{neg}(n),
$$
$$
-\mathsf{neg}(n)\coloneqq\mathsf{pos}(\mathsf{succ}(n)).
$$
完整证明见附录 W。证明路线如下：先证明加法与右侧 successor/predecessor 相容，再证明左侧 successor/predecessor 平移相容；由这些引理推出左右单位律、结合律和交换律；最后用自然数归纳证明
$$
\mathsf{pos}(\mathsf{succ}(n))+\mathsf{neg}(n)=0_{\mathbb Z}
$$
并由交换律得到所有符号情形的逆元律。该证明只使用 M.8、M.13、自然数归纳、整数归纳和路径代数，不使用单值性或选择原则。$\square$

**定义 M.15（loop 幂）.** 若 $X:\mathcal U$，$x:X$，并给出 loop
$$
\ell:x=x,
$$
则定义
$$
\ell^{(-)}:\mathbb Z_{\mathsf{ind}}\to(x=x)
$$
为
$$
\ell^{\mathsf{pos}(0)}\coloneqq\mathsf{refl}_x,
$$
$$
\ell^{\mathsf{pos}(\mathsf{succ}(n))}
\coloneqq
\ell^{\mathsf{pos}(n)}\cdot \ell,
$$
$$
\ell^{\mathsf{neg}(0)}\coloneqq\ell^{-1},
$$
$$
\ell^{\mathsf{neg}(\mathsf{succ}(n))}
\coloneqq
\ell^{\mathsf{neg}(n)}\cdot \ell^{-1}.
$$

**说明 M.16.** 这是第十一章 decode 构造需要的函数。它依赖 $\mathbb Z_{\mathsf{ind}}$ 的普通归纳消去；若改用商整数 $\mathbb Z_{\mathsf{q}}$，由于目标 $(x=x)$ 未必是集合，集合商递归原则不能直接给出该定义。

## M.5 集合商整数

**定义 M.17（整数关系）.** 在 $\mathbb N\times\mathbb N$ 上定义关系
$$
(a,b)\sim(c,d)\quad:\!\!\equiv\quad a+d=c+b.
$$

**命题 M.18.** $\sim$ 是命题值等价关系。

**证明.** 命题值性由引理 M.1 给出，因为 $a+d=c+b$ 是 $\mathbb N$ 中的路径。反身性取
$$
a+b=a+b.
$$
对称性取路径逆。传递性如下。设
$$
p:a+d=c+b,\qquad q:c+f=e+d.
$$
要证 $a+f=e+b$。由引理 M.2 的结合律、交换律和路径复合得
$$
(a+f)+d
=a+(f+d)
=a+(d+f)
=(a+d)+f
=(c+b)+f
=c+(b+f)
=c+(f+b)
=(c+f)+b
=(e+d)+b
=e+(d+b)
=e+(b+d)
=(e+b)+d.
$$
再对右侧公共加数 $d$ 使用引理 M.2 的消去律，得到 $a+f=e+b$。$\square$

**定义 M.19（集合商整数）.** 定义
$$
\mathbb Z_{\mathsf{q}}\coloneqq(\mathbb N\times\mathbb N)/{\sim}.
$$
商类记为
$$
[a,b].
$$

**定义 M.20（商整数 successor 与 predecessor）.** 由集合商递归定义
$$
\mathsf{succ}_{\mathsf{q}}([a,b])\coloneqq[\mathsf{succ}(a),b],
$$
$$
\mathsf{pred}_{\mathsf{q}}([a,b])\coloneqq[a,\mathsf{succ}(b)].
$$
良定义性分别由以下蕴含给出：
$$
a+d=c+b
\Rightarrow
\mathsf{succ}(a)+d=\mathsf{succ}(c)+b,
$$
$$
a+d=c+b
\Rightarrow
a+\mathsf{succ}(d)=c+\mathsf{succ}(b),
$$
它们由引理 M.2 的 successor 加法规则和 $\mathsf{ap}_{\mathsf{succ}}$ 得到。

**命题 M.21（商整数 successor 是等价）.** $\mathsf{succ}_{\mathsf{q}}$ 是自等价。

**证明.** $\mathsf{pred}_{\mathsf{q}}$ 是其准逆。对代表元有
$$
\mathsf{succ}_{\mathsf{q}}(\mathsf{pred}_{\mathsf{q}}([a,b]))
=[\mathsf{succ}(a),\mathsf{succ}(b)]
=[a,b],
$$
因为
$$
\mathsf{succ}(a)+b=a+\mathsf{succ}(b).
$$
另一方向同样给出
$$
\mathsf{pred}_{\mathsf{q}}(\mathsf{succ}_{\mathsf{q}}([a,b]))
=[\mathsf{succ}(a),\mathsf{succ}(b)]
=[a,b].
$$
从代表元提升到商类型上的同伦使用附录 L.13 的命题值依赖消去；目标路径类型是命题，因为 $\mathbb Z_{\mathsf{q}}$ 由 L.11 是集合。最后用推论 G.7 把准逆提升为等价。$\square$

## M.6 与第十一章的接口

第十一章圆覆盖使用以下数据：
$$
\mathbb Z\coloneqq\mathbb Z_{\mathsf{ind}},
\qquad
0_{\mathbb Z}\coloneqq\mathsf{pos}(0),
\qquad
\mathsf{succEquiv}_{\mathbb Z}:\mathbb Z\simeq\mathbb Z.
$$
通过单值性得到路径
$$
\mathsf{ua}(\mathsf{succEquiv}_{\mathbb Z}):\mathbb Z=\mathbb Z
$$
作为圆递归到宇宙时 loop 构造子的像。decode 使用 M.15 的 loop 幂。商整数 $\mathbb Z_{\mathsf{q}}$ 只在集合层代数比较和商类型章节中使用。
