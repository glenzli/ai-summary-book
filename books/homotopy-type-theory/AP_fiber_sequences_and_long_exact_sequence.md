# 附录 AP：Fiber Sequence 与同伦群长正合列

本附录补齐第十二章和附录 AL 留下的关键接口：如何从 pointed map 得到 homotopy fiber，如何得到 connecting map，以及长正合列的精确定理形态。完整证明需要大量 pointed transport 和截断相干；本附录把核心构造和 exactness 证明核写出。

## AP.1 Pointed fiber

**定义 AP.1（pointed map）.** Pointed map
$$
f:(E,e_0)\to_\ast(B,b_0)
$$
由函数 $f:E\to B$ 和路径
$$
f_\ast:f(e_0)=b_0
$$
组成。

**定义 AP.2（homotopy fiber）.** $f$ 在基点 $b_0$ 处的 homotopy fiber 为
$$
F_f\coloneqq\mathsf{fib}_f(b_0)
=\sum_{e:E}f(e)=b_0.
$$
其基点为
$$
f_0\coloneqq(e_0,f_\ast).
$$

**定义 AP.3（fiber inclusion）.** 定义 pointed map
$$
i_f:F_f\to_\ast E
$$
为第一投影
$$
i_f(e,p)\coloneqq e,
$$
基点路径为 $\mathsf{refl}_{e_0}$。

## AP.2 Loop of a fiber

**命题 AP.4（fiber 中路径的刻画）.** 对 $(e,p),(e',p'):F_f$，有等价
$$
((e,p)=(e',p'))
\simeq
\sum_{q:e=e'}\mathsf{transport}^{\lambda x.\,f(x)=b_0}(q,p)=p'.
$$

**证明.** 这是 $\Sigma$-路径等价的直接实例，见附录 D.2。$\square$

**命题 AP.5（基点 fiber loop）.** 有等价
$$
\Omega(F_f,f_0)
\simeq
\sum_{\ell:e_0=e_0}
f_\ast^{-1}\cdot\mathsf{ap}_f(\ell)\cdot f_\ast
=\mathsf{refl}_{b_0}.
$$

**证明（证明核）.** 对 AP.4 取 $(e,p)=(e',p')=(e_0,f_\ast)$。transport 的标准计算给出
$$
\mathsf{transport}^{\lambda x.\,f(x)=b_0}(\ell,f_\ast)
=
\mathsf{ap}_f(\ell)^{-1}\cdot f_\ast
$$
或按本书复合方向的等价形式。把该等式整理为
$$
f_\ast^{-1}\cdot\mathsf{ap}_f(\ell)\cdot f_\ast
=\mathsf{refl}_{b_0}.
$$
不同文献中复合方向可能相反；逐项展开时需用附录 A 的 transport-paths 公式改写。$\square$

## AP.3 Connecting map

**定义 AP.6（connecting map）.** 对 $n\ge0$，定义
$$
\partial_n:\pi_{n+1}(B,b_0)\to\pi_n(F_f,f_0)
$$
为如下构造在 $0$-截断上的诱导：一个 $(n+1)$-loop
$$
\alpha:\Omega^{n+1}(B,b_0)
$$
可视为 $\Omega^n(\Omega B,\mathsf{refl})$ 的点；由 AP.5 中 fiber loop 对应的约束，把 $\alpha$ 提升为 $\Omega^n(F_f,f_0)$。

**说明.** 对 $n=0$，$\partial_0:\pi_1(B)\to\pi_0(F_f)$ 把基点处 loop $\alpha:b_0=b_0$ 送到 fiber 中点
$$
(e_0,f_\ast\cdot\alpha).
$$
高阶情形是该构造的迭代 loop 版本。

**证明义务 AP.7（pointed 相干）.** 定义 AP.6 必须检查：

1.  代表 loop 的同伦相等给出 fiber loop 的相等；
2.  对 $n\ge1$，$\partial_n$ 保持群运算；
3.  对 $n\ge2$，它是阿贝尔群同态；
4.  截断下降合法，即目标同伦群为集合。

这些义务由函数外延性、路径代数、Eckmann-Hilton 和截断递归完成。

## AP.4 长正合列

**定理 AP.8（homotopy fiber sequence 的长正合列，证明核 / 外部输入）.** 对 pointed map
$$
F_f\xrightarrow{i_f}E\xrightarrow{f}B
$$
有长序列
$$
\cdots\to
\pi_{n+1}(B)
\xrightarrow{\partial_n}
\pi_n(F_f)
\xrightarrow{(i_f)_\ast}
\pi_n(E)
\xrightarrow{f_\ast}
\pi_n(B)
\xrightarrow{\partial_{n-1}}
\pi_{n-1}(F_f)
\to\cdots
$$
并且它在每一项 exact。

**证明核.** exactness 分三段：

1.  $f_\ast\circ(i_f)_\ast=0$：fiber 元 $(e,p)$ 的投影经 $f$ 后由路径 $p:f(e)=b_0$ 收缩到基点，因此诱导的 loop 在 $\pi_n(B)$ 中为单位。
2.  若 $\ell:\Omega^nE$ 在 $\pi_n(B)$ 中为单位，则存在 null-homotopy $H$ of $f_\ast(\ell)$；把 $\ell$ 与 $H$ 组成 AP.5 型的 fiber loop，得到 $\pi_n(F_f)$ 中的前像。
3.  $\partial_n$ 的像等于 $(i_f)_\ast$ 的 kernel：边界元素投影到 $E$ 后由基点常值 loop 给出单位；反向使用上一项中 null-homotopy 的边界数据。

所有等式最后都在 $0$-截断的同伦群中，因此 exactness 以命题形式表达。严格 proof term 需要把 loop space 的 iterated equivalence、transport 与 group operation compatibility 固定下来。

## AP.5 Fiber sequence

**定义 AP.9（fiber sequence）.** 一个序列
$$
F\xrightarrow{i}E\xrightarrow{p}B
$$
称为 fiber sequence，若给定 pointed 等价
$$
F\simeq_\ast F_p
$$
并且 $i$ 与 $F_p\to E$ 在 pointed 同伦下相容。

**推论 AP.10（fiber sequence 的长正合列）.** 若 $F\to E\to B$ 是 fiber sequence，则有 AP.8 的长正合列，其中 $F_f$ 通过给定 pointed 等价替换为 $F$。

**证明.** pointed 等价诱导同伦群同构；用该同构把 AP.8 的长正合列 transport 到 $F$。exactness 是命题，沿同构保持。$\square$

## AP.6 Hopf fibration 的接口

**事实 AP.11.** 对 Hopf fibration
$$
\mathbb S^1\to\mathbb S^3\to\mathbb S^2
$$
应用 AP.10 得到低阶同伦群之间的长正合列。结合
$$
\pi_2(\mathbb S^3)=0,\qquad
\pi_1(\mathbb S^1)\cong\mathbb Z
$$
等输入，可推出
$$
\pi_3(\mathbb S^2)\cong\mathbb Z.
$$

**边界.** 本书已证明 $\pi_1(\mathbb S^1)\cong\mathbb Z$；但 $\pi_2(\mathbb S^3)=0$、Hopf fibration 的完整 fiber sequence 相干和相关低阶球面连通性仍为高级合成同伦论输入。
