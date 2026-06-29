# 附录 W：整数加法群律证明核

本附录补全命题 M.14 的证明。我们在归纳整数
$$
\mathbb Z_{\mathsf{ind}}
$$
上使用附录 M.1-M.4 的构造：
$$
0_{\mathbb Z}\coloneqq\mathsf{pos}(0),
$$
successor $\mathsf{succ}_{\mathbb Z}$、predecessor $\mathsf{pred}_{\mathbb Z}$、二者的互逆同伦
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{succ}_{\mathbb Z}(z))=z,
\qquad
\mathsf{succ}_{\mathbb Z}(\mathsf{pred}_{\mathbb Z}(z))=z
$$
以及定义 M.12 的加法。

证明目标不是重新构造整数，而是说明由 M.12 定义的加法确实给出通常整数群。为避免隐藏 case split，本附录把跨越零的两处计算单独列为引理。

## W.1 右侧 successor 与 predecessor 的计算

**引理 W.1（加法与右侧 successor 相容）.** 对任意
$z,u:\mathbb Z_{\mathsf{ind}}$，
$$
z+\mathsf{succ}_{\mathbb Z}(u)
=
\mathsf{succ}_{\mathbb Z}(z+u).
$$

**证明.** 对 $u$ 作整数情形分析。

若 $u\equiv\mathsf{pos}(n)$，则
$$
\mathsf{succ}_{\mathbb Z}(u)\equiv\mathsf{pos}(\mathsf{succ}(n)),
$$
故结论正是 M.13 的计算式
$$
z+\mathsf{pos}(\mathsf{succ}(n))
\equiv
\mathsf{succ}_{\mathbb Z}(z+\mathsf{pos}(n)).
$$

若 $u\equiv\mathsf{neg}(0)$，则
$$
\mathsf{succ}_{\mathbb Z}(u)\equiv 0_{\mathbb Z}.
$$
左边 judgmentally 化为 $z$，右边为
$$
\mathsf{succ}_{\mathbb Z}(z+\mathsf{neg}(0))
\equiv
\mathsf{succ}_{\mathbb Z}(\mathsf{pred}_{\mathbb Z}(z)),
$$
由 M.8 的 $\mathsf{succ}\circ\mathsf{pred}$ 互逆同伦等于 $z$。

若 $u\equiv\mathsf{neg}(\mathsf{succ}(n))$，则
$$
\mathsf{succ}_{\mathbb Z}(u)\equiv \mathsf{neg}(n).
$$
左边为 $z+\mathsf{neg}(n)$。右边由 M.13 化为
$$
\mathsf{succ}_{\mathbb Z}(z+\mathsf{neg}(\mathsf{succ}(n)))
\equiv
\mathsf{succ}_{\mathbb Z}
  (\mathsf{pred}_{\mathbb Z}(z+\mathsf{neg}(n))),
$$
再由 M.8 化为 $z+\mathsf{neg}(n)$。$\square$

**引理 W.2（加法与右侧 predecessor 相容）.** 对任意
$z,u:\mathbb Z_{\mathsf{ind}}$，
$$
z+\mathsf{pred}_{\mathbb Z}(u)
=
\mathsf{pred}_{\mathbb Z}(z+u).
$$

**证明.** 对 $u$ 作整数情形分析。

若 $u\equiv\mathsf{pos}(0)$，则
$$
\mathsf{pred}_{\mathbb Z}(u)\equiv \mathsf{neg}(0).
$$
左边与右边都由 M.13 化为 $\mathsf{pred}_{\mathbb Z}(z)$。

若 $u\equiv\mathsf{pos}(\mathsf{succ}(n))$，则
$$
\mathsf{pred}_{\mathbb Z}(u)\equiv\mathsf{pos}(n).
$$
左边为 $z+\mathsf{pos}(n)$。右边为
$$
\mathsf{pred}_{\mathbb Z}(z+\mathsf{pos}(\mathsf{succ}(n)))
\equiv
\mathsf{pred}_{\mathbb Z}
  (\mathsf{succ}_{\mathbb Z}(z+\mathsf{pos}(n))),
$$
由 M.8 化为 $z+\mathsf{pos}(n)$。

若 $u\equiv\mathsf{neg}(n)$，则
$$
\mathsf{pred}_{\mathbb Z}(u)\equiv\mathsf{neg}(\mathsf{succ}(n)),
$$
故结论正是 M.13 的计算式
$$
z+\mathsf{neg}(\mathsf{succ}(n))
\equiv
\mathsf{pred}_{\mathbb Z}(z+\mathsf{neg}(n)).
$$
$\square$

## W.2 左侧 successor 与 predecessor 的计算

**引理 W.3（左侧 successor 平移）.** 对任意
$z,w:\mathbb Z_{\mathsf{ind}}$，
$$
\mathsf{succ}_{\mathbb Z}(z)+w
=
\mathsf{succ}_{\mathbb Z}(z+w).
$$

**证明.** 对 $w$ 依定义 M.12 作归纳。

当 $w\equiv 0_{\mathbb Z}$ 时，两边都化为
$\mathsf{succ}_{\mathbb Z}(z)$。

当 $w\equiv\mathsf{pos}(\mathsf{succ}(n))$ 时，归纳假设为
$$
\mathsf{succ}_{\mathbb Z}(z)+\mathsf{pos}(n)
=
\mathsf{succ}_{\mathbb Z}(z+\mathsf{pos}(n)).
$$
两边按 M.13 化为上述等式两端再各作用一次
$\mathsf{succ}_{\mathbb Z}$。

当 $w\equiv\mathsf{neg}(0)$ 时，左边化为
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{succ}_{\mathbb Z}(z)),
$$
右边化为
$$
\mathsf{succ}_{\mathbb Z}(\mathsf{pred}_{\mathbb Z}(z)).
$$
二者分别由 M.8 化为 $z$。

当 $w\equiv\mathsf{neg}(\mathsf{succ}(n))$ 时，归纳假设给出
$$
\mathsf{succ}_{\mathbb Z}(z)+\mathsf{neg}(n)
=
\mathsf{succ}_{\mathbb Z}(z+\mathsf{neg}(n)).
$$
两边按 M.13 化为上式两端再各作用
$\mathsf{pred}_{\mathbb Z}$；右边再用 M.8 识别
$$
\mathsf{pred}_{\mathbb Z}
  (\mathsf{succ}_{\mathbb Z}(z+\mathsf{neg}(n)))
=
z+\mathsf{neg}(n),
$$
与
$$
\mathsf{succ}_{\mathbb Z}
  (\mathsf{pred}_{\mathbb Z}(z+\mathsf{neg}(n)))
=
z+\mathsf{neg}(n)
$$
同由 M.8 给出。$\square$

**引理 W.4（左侧 predecessor 平移）.** 对任意
$z,w:\mathbb Z_{\mathsf{ind}}$，
$$
\mathsf{pred}_{\mathbb Z}(z)+w
=
\mathsf{pred}_{\mathbb Z}(z+w).
$$

**证明.** 对 $w$ 作整数归纳。

当 $w\equiv 0_{\mathbb Z}$ 时，两边都化为
$\mathsf{pred}_{\mathbb Z}(z)$。

当 $w\equiv\mathsf{pos}(\mathsf{succ}(n))$ 时，归纳假设为
$$
\mathsf{pred}_{\mathbb Z}(z)+\mathsf{pos}(n)
=
\mathsf{pred}_{\mathbb Z}(z+\mathsf{pos}(n)).
$$
左边按 M.13 化为
$$
\mathsf{succ}_{\mathbb Z}
  (\mathsf{pred}_{\mathbb Z}(z)+\mathsf{pos}(n)),
$$
再用归纳假设得到
$$
\mathsf{succ}_{\mathbb Z}
  (\mathsf{pred}_{\mathbb Z}(z+\mathsf{pos}(n))).
$$
右边按 M.13 化为
$$
\mathsf{pred}_{\mathbb Z}
  (\mathsf{succ}_{\mathbb Z}(z+\mathsf{pos}(n))).
$$
两者分别由 M.8 化为同一项 $z+\mathsf{pos}(n)$。

当 $w\equiv\mathsf{neg}(0)$ 时，左边为
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{pred}_{\mathbb Z}(z)),
$$
右边为
$$
\mathsf{pred}_{\mathbb Z}(z+\mathsf{neg}(0))
\equiv
\mathsf{pred}_{\mathbb Z}(\mathsf{pred}_{\mathbb Z}(z)).
$$

当 $w\equiv\mathsf{neg}(\mathsf{succ}(n))$ 时，归纳假设给出
$$
\mathsf{pred}_{\mathbb Z}(z)+\mathsf{neg}(n)
=
\mathsf{pred}_{\mathbb Z}(z+\mathsf{neg}(n)).
$$
两边按 M.13 化为上式两端再各作用一次
$\mathsf{pred}_{\mathbb Z}$。$\square$

## W.3 单位律

**引理 W.5（右单位律）.** 对任意 $z$，
$$
z+0_{\mathbb Z}=z.
$$

**证明.** 这是 M.13 第一条计算规则。$\square$

**引理 W.6（左单位律）.** 对任意 $z$，
$$
0_{\mathbb Z}+z=z.
$$

**证明.** 对 $z$ 作整数归纳。

当 $z\equiv\mathsf{pos}(0)$ 时为 reflexivity。若
$z\equiv\mathsf{pos}(\mathsf{succ}(n))$，则
$$
0+\mathsf{pos}(\mathsf{succ}(n))
\equiv
\mathsf{succ}_{\mathbb Z}(0+\mathsf{pos}(n)),
$$
由归纳假设化为
$\mathsf{succ}_{\mathbb Z}(\mathsf{pos}(n))
\equiv\mathsf{pos}(\mathsf{succ}(n))$。

若 $z\equiv\mathsf{neg}(0)$，则
$$
0+\mathsf{neg}(0)
\equiv
\mathsf{pred}_{\mathbb Z}(0)
\equiv
\mathsf{neg}(0).
$$
若 $z\equiv\mathsf{neg}(\mathsf{succ}(n))$，则
$$
0+\mathsf{neg}(\mathsf{succ}(n))
\equiv
\mathsf{pred}_{\mathbb Z}(0+\mathsf{neg}(n)),
$$
由归纳假设化为
$\mathsf{pred}_{\mathbb Z}(\mathsf{neg}(n))
\equiv\mathsf{neg}(\mathsf{succ}(n))$。$\square$

## W.4 结合律

**定理 W.7（结合律）.** 对任意 $x,y,z:\mathbb Z_{\mathsf{ind}}$，
$$
(x+y)+z=x+(y+z).
$$

**证明.** 对第三个变量 $z$ 作整数归纳。

零情形由右单位律 W.5 得到。

正 successor 步设
$$
(x+y)+u=x+(y+u).
$$
则
$$
\begin{aligned}
(x+y)+\mathsf{succ}_{\mathbb Z}(u)
&=
\mathsf{succ}_{\mathbb Z}((x+y)+u) && \text{由 W.1}\\
&=
\mathsf{succ}_{\mathbb Z}(x+(y+u)) && \text{由归纳假设}\\
&=
x+\mathsf{succ}_{\mathbb Z}(y+u) && \text{由 W.1 的逆路径}\\
&=
x+(y+\mathsf{succ}_{\mathbb Z}(u)) && \text{由 W.1 的逆路径。}
\end{aligned}
$$

负 predecessor 步完全相同，把 W.1 换成 W.2：
$$
\begin{aligned}
(x+y)+\mathsf{pred}_{\mathbb Z}(u)
&=
\mathsf{pred}_{\mathbb Z}((x+y)+u)\\
&=
\mathsf{pred}_{\mathbb Z}(x+(y+u))\\
&=
x+\mathsf{pred}_{\mathbb Z}(y+u)\\
&=
x+(y+\mathsf{pred}_{\mathbb Z}(u)).
\end{aligned}
$$
这覆盖 $\mathsf{pos}$ 与 $\mathsf{neg}$ 两支，因为归纳整数的非零构造正是 successor 或 predecessor 迭代。$\square$

## W.5 交换律

**定理 W.8（交换律）.** 对任意 $x,y:\mathbb Z_{\mathsf{ind}}$，
$$
x+y=y+x.
$$

**证明.** 对 $y$ 作整数归纳。

零情形由 W.5 与 W.6 给出：
$$
x+0=x=0+x.
$$

successor 步中，归纳假设为 $x+u=u+x$。于是
$$
\begin{aligned}
x+\mathsf{succ}_{\mathbb Z}(u)
&=
\mathsf{succ}_{\mathbb Z}(x+u) && \text{W.1}\\
&=
\mathsf{succ}_{\mathbb Z}(u+x) && \text{归纳假设}\\
&=
\mathsf{succ}_{\mathbb Z}(u)+x && \text{W.3 的逆路径。}
\end{aligned}
$$

predecessor 步中同理：
$$
\begin{aligned}
x+\mathsf{pred}_{\mathbb Z}(u)
&=
\mathsf{pred}_{\mathbb Z}(x+u) && \text{W.2}\\
&=
\mathsf{pred}_{\mathbb Z}(u+x) && \text{归纳假设}\\
&=
\mathsf{pred}_{\mathbb Z}(u)+x && \text{W.4 的逆路径。}
\end{aligned}
$$
因此交换律成立。$\square$

## W.6 逆元律

**定义 W.9（符号翻转）.** 定义
$$
-\mathsf{pos}(0)\coloneqq\mathsf{pos}(0),
$$
$$
-\mathsf{pos}(\mathsf{succ}(n))\coloneqq\mathsf{neg}(n),
$$
$$
-\mathsf{neg}(n)\coloneqq\mathsf{pos}(\mathsf{succ}(n)).
$$

**引理 W.10（正整数抵消）.** 对任意 $n:\mathbb N$，
$$
\mathsf{pos}(\mathsf{succ}(n))+\mathsf{neg}(n)=0_{\mathbb Z}.
$$

**证明.** 对 $n$ 作自然数归纳。

当 $n\equiv 0$ 时，
$$
\mathsf{pos}(1)+\mathsf{neg}(0)
\equiv
\mathsf{pred}_{\mathbb Z}(\mathsf{pos}(1))
\equiv
0_{\mathbb Z}.
$$

若结论对 $n$ 成立，则
$$
\begin{aligned}
\mathsf{pos}(\mathsf{succ}(\mathsf{succ}(n)))
  +\mathsf{neg}(\mathsf{succ}(n))
&=
\mathsf{pred}_{\mathbb Z}
  (\mathsf{pos}(\mathsf{succ}(\mathsf{succ}(n)))+\mathsf{neg}(n))
  && \text{M.13}\\
&=
\mathsf{pred}_{\mathbb Z}
  (\mathsf{succ}_{\mathbb Z}(\mathsf{pos}(\mathsf{succ}(n))+\mathsf{neg}(n)))
  && \text{W.3}\\
&=
\mathsf{pred}_{\mathbb Z}(\mathsf{succ}_{\mathbb Z}(0_{\mathbb Z}))
  && \text{归纳假设}\\
&=
0_{\mathbb Z}.
\end{aligned}
$$
$\square$

**定理 W.11（右逆元律）.** 对任意 $z:\mathbb Z_{\mathsf{ind}}$，
$$
z+(-z)=0_{\mathbb Z}.
$$

**证明.** 对 $z$ 分情形。

若 $z\equiv\mathsf{pos}(0)$，则结论为右单位律。

若 $z\equiv\mathsf{pos}(\mathsf{succ}(n))$，则 $-z\equiv\mathsf{neg}(n)$，结论正是 W.10。

若 $z\equiv\mathsf{neg}(n)$，则 $-z\equiv\mathsf{pos}(\mathsf{succ}(n))$。由交换律 W.8，
$$
\mathsf{neg}(n)+\mathsf{pos}(\mathsf{succ}(n))
=
\mathsf{pos}(\mathsf{succ}(n))+\mathsf{neg}(n),
$$
再由 W.10 得到 $0_{\mathbb Z}$。$\square$

**定理 W.12（左逆元律）.** 对任意 $z:\mathbb Z_{\mathsf{ind}}$，
$$
(-z)+z=0_{\mathbb Z}.
$$

**证明.** 由交换律 W.8 把左边改写为 $z+(-z)$，再用 W.11。$\square$

## W.7 整数加法群

**定理 W.13（整数加法群律）.** 类型 $\mathbb Z_{\mathsf{ind}}$ 与运算 $+$、单位
$0_{\mathbb Z}$、逆元 $(-)$ 构成群，并且该群为交换群。

**证明.** 群律分别为右单位 W.5、左单位 W.6、结合律 W.7、右逆元 W.11、左逆元 W.12。交换性为 W.8。所有证明只使用 M.8 的 successor/predecessor 互逆同伦、M.13 的加法计算规则、自然数归纳、整数归纳和路径代数。$\square$

**依赖说明。** 本附录不使用单值性、函数外延性、截断、商类型或选择原则。逐项展开时，唯一需要注意的是 W.3-W.4 中对“整数归纳”的表达可选择直接按
$\mathsf{pos}/\mathsf{neg}$ 构造子分支写出，也可先定义双向整数归纳原则；两种写法的证明项等价。
