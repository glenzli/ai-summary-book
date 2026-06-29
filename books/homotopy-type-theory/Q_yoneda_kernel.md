# 附录 Q：Yoneda 引理证明核

## 目标

本附录补齐第十四章的 Yoneda 引理。我们在预范畴 $\mathcal C$ 上处理集合值预层
$$
P:\mathcal C^{op}\to\mathsf{Set}.
$$
证明只使用函数外延性、Hom 集合性和预范畴律，不使用单值范畴条件。

## Q.1 集合值预层

**定义 Q.1（预层）.** 设 $\mathcal C$ 是预范畴。一个集合值预层 $P$ 包含：

1.  对象函数
    $$
    P_0:\mathcal C_0\to\mathcal U;
    $$
2.  集合性
    $$
    P_{\mathsf{set}}(x):\mathsf{isSet}(P_0(x));
    $$
3.  反变作用：对 $f:\mathcal C(x,y)$，有函数
    $$
    P(f):P_0(y)\to P_0(x);
    $$
4.  恒等律
    $$
    P(\mathsf{id}_x)=\mathsf{id}_{P_0(x)};
    $$
5.  复合律
    $$
    P(g\circ f)=P(f)\circ P(g)
    $$
    对 $f:\mathcal C(x,y)$、$g:\mathcal C(y,z)$ 成立。

其中第 4、5 条是函数路径，使用函数外延性表达；由于每个 $P_0(x)$ 是集合，这些律的证明分量是命题。

**定义 Q.2（可表预层）.** 对 $c:\mathcal C_0$，定义
$$
y(c):\mathcal C^{op}\to\mathsf{Set}
$$
如下：
$$
y(c)_0(x)\coloneqq\mathcal C(x,c),
$$
对 $f:\mathcal C(x,y)$，
$$
y(c)(f):\mathcal C(y,c)\to\mathcal C(x,c),
\qquad
y(c)(f)(h)\coloneqq h\circ f.
$$
恒等律和复合律分别由预范畴的单位律和结合律给出。

## Q.2 自然变换

**定义 Q.3（自然变换）.** 设 $P,Q$ 是预层。自然变换 $\alpha:P\Rightarrow Q$ 由以下数据组成：
$$
\alpha_x:P_0(x)\to Q_0(x)
$$
以及对每个 $f:\mathcal C(x,y)$ 的自然性路径
$$
Q(f)(\alpha_y(u))=\alpha_x(P(f)(u))
$$
其中 $u:P_0(y)$。记自然变换类型为
$$
\mathsf{Nat}(P,Q).
$$

**命题 Q.4（自然变换相等由分量决定）.** 若 $\alpha,\beta:\mathsf{Nat}(P,Q)$，则给出
$$
\prod_{x:\mathcal C_0}\alpha_x=\beta_x
$$
即可得到 $\alpha=\beta$。

**证明.** 自然变换是分量函数族加自然性证明的 $\Sigma$ 类型。自然性证明的目标位于 $Q_0(x)$ 的路径类型中；由于 $Q_0(x)$ 是集合，该路径类型是命题。因此自然性证明分量由命题性唯一。分量函数族的路径由函数外延性从逐对象路径得到。$\square$

## Q.3 Yoneda 映射

固定预层 $P$ 和对象 $c:\mathcal C_0$。

**定义 Q.5（Yoneda evaluation）.** 定义
$$
\Phi:\mathsf{Nat}(y(c),P)\to P_0(c)
$$
为
$$
\Phi(\alpha)\coloneqq\alpha_c(\mathsf{id}_c).
$$

**定义 Q.6（Yoneda extension）.** 对 $u:P_0(c)$，定义自然变换
$$
\Psi(u):y(c)\Rightarrow P
$$
的分量为
$$
\Psi(u)_x:\mathcal C(x,c)\to P_0(x),
\qquad
\Psi(u)_x(g)\coloneqq P(g)(u).
$$
自然性如下。对 $f:\mathcal C(x,y)$ 和 $h:\mathcal C(y,c)$，需证
$$
P(f)(\Psi(u)_y(h))=\Psi(u)_x(h\circ f).
$$
左边展开为
$$
P(f)(P(h)(u)),
$$
由预层复合律
$$
P(h\circ f)=P(f)\circ P(h)
$$
得到
$$
P(f)(P(h)(u))=P(h\circ f)(u),
$$
正是右边。

## Q.4 互逆性

**命题 Q.7（$\Phi\Psi=\mathsf{id}$）.** 对任意 $u:P_0(c)$，
$$
\Phi(\Psi(u))=u.
$$

**证明.** 展开定义：
$$
\Phi(\Psi(u))
=
\Psi(u)_c(\mathsf{id}_c)
=
P(\mathsf{id}_c)(u).
$$
由预层恒等律 $P(\mathsf{id}_c)=\mathsf{id}$，得 $P(\mathsf{id}_c)(u)=u$。$\square$

**命题 Q.8（$\Psi\Phi=\mathsf{id}$）.** 对任意 $\alpha:\mathsf{Nat}(y(c),P)$，
$$
\Psi(\Phi(\alpha))=\alpha.
$$

**证明.** 由命题 Q.4，只需逐对象比较分量函数。固定 $x:\mathcal C_0$，再由函数外延性，只需对 $g:\mathcal C(x,c)$ 比较：
$$
\Psi(\Phi(\alpha))_x(g)=\alpha_x(g).
$$
左边展开为
$$
P(g)(\alpha_c(\mathsf{id}_c)).
$$
把 $\alpha$ 的自然性应用于态射 $g:x\to c$ 和元素 $\mathsf{id}_c:y(c)_0(c)=\mathcal C(c,c)$，得到
$$
P(g)(\alpha_c(\mathsf{id}_c))
=
\alpha_x(y(c)(g)(\mathsf{id}_c)).
$$
而
$$
y(c)(g)(\mathsf{id}_c)=\mathsf{id}_c\circ g=g
$$
由左单位律给出。代回即得目标。$\square$

## Q.5 Yoneda 引理

**定理 Q.9（Yoneda 引理）.** 对任意预范畴 $\mathcal C$、集合值预层 $P$ 和对象 $c:\mathcal C_0$，有等价
$$
\mathsf{Nat}(y(c),P)\simeq P_0(c).
$$

**证明.** 取 $\Phi$ 为正向函数，$\Psi$ 为反向函数。命题 Q.7 和 Q.8 给出双向同伦，因此 $\Phi$ 有准逆。由推论 G.7，$\Phi$ 是等价。$\square$

**推论 Q.10（Yoneda 嵌入的 Hom 等价）.** 对任意 $c,d:\mathcal C_0$，自然变换
$$
y(c)\Rightarrow y(d)
$$
等价于态射
$$
\mathcal C(c,d).
$$

**证明.** 在定理 Q.9 中取 $P=y(d)$，得到
$$
\mathsf{Nat}(y(c),y(d))\simeq y(d)_0(c)\equiv\mathcal C(c,d).
$$
$\square$

**说明 Q.11（fully faithful 版本）.** 推论 Q.10 是 Yoneda 嵌入 fully faithful 的核心。附录 U 定义预层范畴、自然变换复合和 Yoneda 函子，并把推论 Q.10 提升为定理 U.11：$y:\mathcal C\to\mathsf{PSh}(\mathcal C)$ fully faithful。
