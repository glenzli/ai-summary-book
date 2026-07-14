# 第五章：Fiber、等价与等价的等价定义

“$f:A\to B$ 可逆”若只写成存在 $g:B\to A$ 以及两条逆律，会把逆函数及相干证明一起当成额外结构；同一个 $f$ 可能因此携带多组准逆数据。更稳定的做法是逐个检查目标点 $y:B$：所有原像及其等式证明组成 fiber，而 fiber 可收缩恰好表达“原像存在，并且连同证明一起唯一”。

本章采用这一 fiber 定义，再从收缩中心读出准逆，并比较半伴随等价等相干版本。所有构造依赖前四章的 $\Sigma$ 路径、transport 和可收缩性，不使用单值性。这样建立的等价概念既能复合、取逆，又能保持同伦层级，正好成为下一章比较类型路径与类型等价的右端对象。

## 5.1 Fiber

**定义 5.1.** 设 $f:A\to B$，$y:B$。$f$ 在 $y$ 处的 fiber 定义为
$$
\mathsf{fib}_f(y)\coloneqq \sum_{x:A}(f(x)=y).
$$

fiber 的项是二元组 $(x,p)$，其中 $x:A$ 且 $p:f(x)=y$。

**定义 5.2.** 函数 $f:A\to B$ 是等价，若
$$
\mathsf{isEquiv}(f)\coloneqq \prod_{y:B}\mathsf{isContr}(\mathsf{fib}_f(y)).
$$
等价类型定义为
$$
A\simeq B\coloneqq \sum_{f:A\to B}\mathsf{isEquiv}(f).
$$

**例 5.3.** 恒等函数 $\mathsf{id}_A:A\to A$ 是等价。

**证明（书内证明）.** 给定 $y:A$，需证明
$$
\sum_{x:A}(x=y)
$$
可收缩。这是附录 D.9 的右端点版本。直接证明如下：取中心 $(y,\mathsf{refl}_y)$。给定 $(x,p)$，由 $\Sigma$ 路径刻画，只需给出 $r:y=x$ 和
$$
\mathsf{transport}^{\lambda z.\,z=y}(r,\mathsf{refl}_y)=p.
$$
取 $r\coloneqq p^{-1}$；第二个目标对 $p$ 作路径归纳，反身情形化为 $\mathsf{refl}_y=\mathsf{refl}_y$。$\square$

## 5.2 准逆

**定义 5.4.** 函数 $f:A\to B$ 有准逆（quasi-inverse），若
$$
\mathsf{qinv}(f)\coloneqq
\sum_{g:B\to A}\left(\prod_{x:A}g(f(x))=x\right)\times
\left(\prod_{y:B}f(g(y))=y\right).
$$

**命题 5.5（等价推出准逆）.** 若 $\mathsf{isEquiv}(f)$，则 $\mathsf{qinv}(f)$。

**证明（书内证明）.** 见附录 D.14。核心构造是：对每个 $y:B$ 取 fiber 的收缩中心 $(g(y),\epsilon_y)$，由此得到候选逆 $g:B\to A$ 和右逆同伦 $\epsilon$。对每个 $x:A$，fiber $\mathsf{fib}_f(f(x))$ 的收缩性连接
$$
(g(f(x)),\epsilon_{f(x)})
\quad\text{与}\quad
(x,\mathsf{refl}_{f(x)}),
$$
对该路径取第一投影即得到左逆同伦 $g(f(x))=x$。$\square$

**警告 5.6.** 准逆的类型 $\mathsf{qinv}(f)$ 本身通常不是命题，因此不适合作为 $\mathsf{isEquiv}(f)$ 的最终定义。fiber 可收缩定义的优点是它是性质，而不是额外结构。

## 5.3 伴随等价与半伴随等价

**定义 5.7.** 伴随等价数据由 $f:A\to B$、$g:B\to A$、同伦
$$
\eta:\prod_{x:A}g(f(x))=x,\qquad
\epsilon:\prod_{y:B}f(g(y))=y
$$
以及一个三角相干条件组成。半伴随等价（half-adjoint equivalence）选择其中一个三角相干条件作为额外路径。

**定理 5.8（等价定义的比较）.** 对任意 $f:A\to B$，以下数据互相蕴含：

1.  $\mathsf{isEquiv}(f)$；
2.  $f$ 有半伴随等价结构；
3.  $f$ 有左右可收缩的 fiber；
4.  $f$ 有带适当相干条件的准逆。

**证明（书内证明）.** 见附录 G.7。$\mathsf{isEquiv}(f)$ 推出准逆由附录 D.14 给出；准逆相干化为半伴随等价见附录 G.4；半伴随等价推出 fiber 可收缩意义下的等价见附录 G.6。完整相干化公式依赖路径复合方向，附录 G 按本书约定给出证明和相干条件。

## 5.4 等价的基本稳定性

**命题 5.9（等价复合）.** 若 $f:A\to B$ 与 $g:B\to C$ 都是等价，则 $g\circ f:A\to C$ 是等价。

**证明（书内证明）.** 见附录 E.7。证明按 fiber 定义进行：对每个 $c:C$，把
$$
\mathsf{fib}_{g\circ f}(c)
$$
分解为
$$
\sum_{w:\mathsf{fib}_g(c)}\mathsf{fib}_f(\mathsf{pr}_1(w)).
$$
外层 fiber 因 $g$ 是等价而可收缩，内层 fiber 因 $f$ 是等价而可收缩；可收缩基底上的可收缩纤维总空间可收缩。$\square$

**命题 5.10（等价逆）.** 若 $e:A\simeq B$，则存在 $e^{-1}:B\simeq A$。

**证明（书内证明）.** 见附录 G.8。从 $e$ 的 fiber 可收缩数据得到准逆 $g:B\to A$；把同一组左右逆同伦交换方向，得到 $g$ 的准逆数据；再由准逆推出等价。$\square$

**定义 5.11.** 等价 $e:A\simeq B$ 的底层函数记为 $\mathsf{pr}_1(e)$，也常简记为 $e$。在需要区分时，本书写 $e.1$ 表示函数，$e.2$ 表示等价性证明。

## 5.5 等价与同伦层级

**命题 5.12（等价保持可收缩性）.** 若 $A\simeq B$ 且 $A$ 可收缩，则 $B$ 可收缩。

**证明（书内证明，使用等价 fiber 定义）.** 设等价底层函数为 $f:A\to B$。给定 $b:B$，fiber $\mathsf{fib}_f(b)$ 可收缩，因此有中心 $(a,p)$，其中 $p:f(a)=b$。若 $A$ 可收缩，取中心 $a_0$，则 $f(a_0)$ 可作为 $B$ 的中心。对任意 $b:B$，由上面的 $(a,p)$ 和 $A$ 中 $a_0=a$ 的路径，经 $\mathsf{ap}_f$ 后与 $p$ 复合，得到 $f(a_0)=b$。$\square$

**命题 5.13（等价保持同伦层级）.** 若 $A\simeq B$ 且 $A$ 具有 $h$-level $n$，则 $B$ 具有 $h$-level $n$。

**证明（书内证明）.** 见附录 G.10。对 $n$ 归纳。基步是命题 5.12。归纳步使用等价诱导路径空间等价，并对路径空间应用归纳假设。路径空间等价的构造见附录 G.9。$\square$

## 5.6 Fiber 观点留下的结论

恒等函数的 fiber 是基于端点的路径总空间，因而可收缩；一般等价则要求每个 fiber 都具有同样的唯一性。这个定义把“是等价”与具体选择的逆数据分开，并使复合、逆和同伦层级保持性都能归约到可收缩总空间。第六章的单值性不会重新定义等价，而是断言宇宙中的路径恰好由这里已经建立的等价检测。

## 练习

**练习 5.1.** 证明若 $f:A\to B$ 是等价，则对任意 $b:B$，存在 $a:A$ 使 $f(a)=b$。

**练习 5.2.** 证明若 $A$ 可收缩，则任意 $B\to A$ 在适当条件下具有唯一性性质。

**练习 5.3.** 设 $f:A\to B$ 与 $g:B\to C$ 有准逆，写出 $g\circ f$ 的准逆数据。

**练习 5.4.** 证明恒等等价是等价复合的左右单位，结论应写成等价类型中的路径或结构同伦。
