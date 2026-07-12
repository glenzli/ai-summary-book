# 附录 U：预层范畴与 Yoneda 嵌入

## U.0 目标

附录 Q 已证明 Yoneda 引理的核心等价
$$
\mathsf{Nat}(y(c),P)\simeq P(c).
$$
本附录把它提升为范畴论表述：

1. 定义集合值预层范畴 $\mathsf{PSh}(\mathcal C)$；
2. 证明其 Hom 类型是集合；
3. 定义自然变换的恒等和复合；
4. 定义 Yoneda 嵌入
   $$
   y:\mathcal C\to\mathsf{PSh}(\mathcal C);
   $$
5. 证明 $y$ fully faithful。

本附录仍只使用预范畴、函数外延性和 Hom 集合性，不使用 $\mathcal C$ 的单值性。

## U.1 预层范畴的对象与 Hom

设 $\mathcal C$ 是预范畴。对象定义为附录 Q.1 的集合值预层：
$$
\mathsf{PSh}(\mathcal C)_0\coloneqq
\{P:\mathcal C^{op}\to\mathsf{Set}\}.
$$

对预层 $P,Q$，定义 Hom：
$$
\mathsf{PSh}(\mathcal C)(P,Q)
\coloneqq
\mathsf{Nat}(P,Q),
$$
其中 $\mathsf{Nat}(P,Q)$ 是定义 Q.3 的自然变换类型。

**命题 U.1（自然变换 Hom 是集合）。** 对任意预层 $P,Q$，
$$
\mathsf{isSet}(\mathsf{Nat}(P,Q)).
$$

**证明.** 取 $\alpha,\beta:\mathsf{Nat}(P,Q)$。要证 $\alpha=\beta$ 是命题。

由命题 Q.4，自然变换路径由分量路径决定；更具体地，路径 $\alpha=\beta$ 等价于
$$
\prod_{x:\mathcal C_0}\alpha_x=\beta_x.
$$
对固定 $x$，$\alpha_x,\beta_x:P_0(x)\to Q_0(x)$。由于 $Q_0(x)$ 是集合，函数类型 $P_0(x)\to Q_0(x)$ 也是集合：给定两条函数路径，由函数外延性化为逐点二阶路径，而逐点二阶路径由 $Q_0(x)$ 的集合性唯一。因此 $\alpha_x=\beta_x$ 是命题。

命题值依赖函数类型仍是命题，故
$$
\prod_x(\alpha_x=\beta_x)
$$
是命题。经等价反射，$\alpha=\beta$ 是命题。$\square$

## U.2 恒等自然变换

**定义 U.2（恒等自然变换）。** 对预层 $P$，定义
$$
\mathsf{id}_P:P\Rightarrow P
$$
的分量为
$$
(\mathsf{id}_P)_x(u)\coloneqq u.
$$
自然性要求
$$
P(f)((\mathsf{id}_P)_y(u))
=
(\mathsf{id}_P)_x(P(f)(u)),
$$
两边定义相同，取反身路径。

## U.3 自然变换复合

**定义 U.3（垂直复合）。** 给定
$$
\alpha:P\Rightarrow Q,\qquad
\beta:Q\Rightarrow R,
$$
定义
$$
\beta\circ\alpha:P\Rightarrow R
$$
的分量为
$$
(\beta\circ\alpha)_x(u)\coloneqq\beta_x(\alpha_x(u)).
$$

自然性如下。对 $f:\mathcal C(x,y)$ 和 $u:P_0(y)$，需证
$$
R(f)(\beta_y(\alpha_y(u)))
=
\beta_x(\alpha_x(P(f)(u))).
$$
由 $\beta$ 的自然性，有
$$
R(f)(\beta_y(\alpha_y(u)))
=
\beta_x(Q(f)(\alpha_y(u))).
$$
由 $\alpha$ 的自然性，有
$$
Q(f)(\alpha_y(u))=\alpha_x(P(f)(u)).
$$
对该路径作用 $\mathsf{ap}_{\beta_x}$，并与上一条路径复合，得到所需自然性。

**命题 U.4（复合律和单位律）。** 自然变换的垂直复合满足结合律和左右单位律。

**证明.** 三条律都用命题 Q.4 化为分量函数相等，再由函数外延性化为逐点相等。逐点处均为函数复合的定义相等。因此取反身路径。自然性证明分量由命题性自动一致。$\square$

## U.4 预层范畴

**定理 U.5（预层范畴）。** 数据
$$
\mathsf{PSh}(\mathcal C)
$$
构成预范畴。

**证明.** 对象、Hom 分别由 U.1 定义。Hom 集合性由命题 U.1 给出。恒等态射由定义 U.2 给出，复合由定义 U.3 给出。单位律和结合律由命题 U.4 给出。$\square$

## U.5 Yoneda 嵌入的对象与态射作用

**定义 U.6（Yoneda 嵌入的对象作用）。** 定义
$$
y_0:\mathcal C_0\to\mathsf{PSh}(\mathcal C)_0
$$
为
$$
y_0(c)\coloneqq y(c),
$$
其中 $y(c)=\mathcal C(-,c)$ 是定义 Q.2 的可表预层。

**定义 U.7（Yoneda 嵌入的态射作用）。** 对态射
$$
h:\mathcal C(c,d),
$$
定义自然变换
$$
y(h):y(c)\Rightarrow y(d)
$$
的分量为
$$
y(h)_x:\mathcal C(x,c)\to\mathcal C(x,d),
\qquad
y(h)_x(g)\coloneqq h\circ g.
$$

自然性为：对 $f:\mathcal C(x,y)$ 和 $g:\mathcal C(y,c)$，需证
$$
y(d)(f)(h\circ g)=y(h)_x(y(c)(f)(g)).
$$
左边为
$$
(h\circ g)\circ f,
$$
右边为
$$
h\circ(g\circ f).
$$
由预范畴结合律得到路径。

## U.6 Yoneda 是函子

**命题 U.8（Yoneda 保恒等）。** 对每个 $c$，
$$
y(\mathsf{id}_c)=\mathsf{id}_{y(c)}.
$$

**证明.** 由命题 Q.4，只需比较分量。固定 $x$，再由函数外延性，只需对 $g:\mathcal C(x,c)$ 证明
$$
\mathsf{id}_c\circ g=g,
$$
这是预范畴左单位律。$\square$

**命题 U.9（Yoneda 保复合）。** 对
$$
f:\mathcal C(c,d),\qquad g:\mathcal C(d,e),
$$
有
$$
y(g\circ f)=y(g)\circ y(f).
$$

**证明.** 由命题 Q.4 和函数外延性，逐点化为对 $h:\mathcal C(x,c)$ 证明
$$
(g\circ f)\circ h=g\circ(f\circ h),
$$
这是预范畴结合律的逆方向或正方向，取决于本书复合记号的括号约定。按 P.1 的约定，结合律给出
$$
g\circ(f\circ h)=(g\circ f)\circ h,
$$
必要时取路径逆。$\square$

**定义 U.10（Yoneda 嵌入）。** 定义
$$
y:\mathcal C\to\mathsf{PSh}(\mathcal C)
$$
为对象作用 U.6、态射作用 U.7，函子律由 U.8-U.9 给出。

## U.7 Fully faithful

对 $c,d:\mathcal C_0$，Yoneda 嵌入诱导函数
$$
y_{c,d}:\mathcal C(c,d)\to
\mathsf{PSh}(\mathcal C)(y(c),y(d))
$$
即
$$
h\mapsto y(h).
$$

**定理 U.11（Yoneda 嵌入 fully faithful）。** 对任意 $c,d$，函数 $y_{c,d}$ 是等价。

**证明.** 附录 Q.10 给出等价
$$
\mathsf{Nat}(y(c),y(d))\simeq\mathcal C(c,d)
$$
其正向映射为 evaluation：
$$
\alpha\mapsto\alpha_c(\mathsf{id}_c).
$$
其反向映射把 $h:\mathcal C(c,d)$ 送到自然变换
$$
g:x\to c\mapsto h\circ g,
$$
这正是定义 U.7 的 $y(h)$。因此 $y_{c,d}$ 是该等价的逆方向，故为等价。$\square$

**推论 U.12。** 若以 fully faithful 定义为 Hom 映射全为等价，则 Yoneda 嵌入 $y:\mathcal C\to\mathsf{PSh}(\mathcal C)$ fully faithful。

**证明.** 逐对象对 $(c,d)$ 应用定理 U.11。$\square$
