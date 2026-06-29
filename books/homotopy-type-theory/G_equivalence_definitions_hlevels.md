# 附录 G：等价定义与同伦层级证明核

## 目标

本附录补全第五章剩余的基础证明：准逆推出 fiber 可收缩意义下的等价、等价的逆、等价诱导路径空间等价，以及等价保持同伦层级。除明确说明外，不使用单值性。

## G.1 同伦函数的 fiber 比较

**定义 G.1.** 设 $f,g:A\to B$，并有同伦
$$
H:\prod_{x:A}f(x)=g(x).
$$
对任意 $y:B$，定义
$$
\Phi_H:\mathsf{fib}_f(y)\to\mathsf{fib}_g(y)
$$
为
$$
\Phi_H(x,p)\coloneqq (x,H(x)^{-1}\cdot p),
$$
其中 $p:f(x)=y$。定义反向
$$
\Psi_H:\mathsf{fib}_g(y)\to\mathsf{fib}_f(y)
$$
为
$$
\Psi_H(x,q)\coloneqq (x,H(x)\cdot q).
$$

**定理 G.2（同伦函数有等价 fiber）.** $\Phi_H$ 与 $\Psi_H$ 给出 $\mathsf{fib}_f(y)$ 与 $\mathsf{fib}_g(y)$ 的双向同伦。因此若其中一个 fiber 可收缩，另一个也可收缩。

**证明.** 对 $(x,p):\mathsf{fib}_f(y)$，复合 $\Psi_H(\Phi_H(x,p))$ 的第二分量为
$$
H(x)\cdot(H(x)^{-1}\cdot p).
$$
由结合律和逆律化为 $p$。用 $\Sigma$ 路径刻画得到 $\Psi_H(\Phi_H(x,p))=(x,p)$。另一方向同理，使用 $H(x)^{-1}\cdot(H(x)\cdot q)=q$。可收缩性由 retract 保持可收缩或双向同伦直接转移。$\square$

## G.2 准逆推出等价

**定理 G.3（准逆推出 fiber 可收缩等价）.** 若 $f:A\to B$ 有准逆
$$
g:B\to A,\qquad
\eta:\prod_{x:A}g(f(x))=x,\qquad
\epsilon:\prod_{y:B}f(g(y))=y,
$$
则 $f$ 是等价。

**证明.** 固定 $y:B$。我们证明 $\mathsf{fib}_f(y)$ 可收缩。取中心
$$
(g(y),\epsilon_y).
$$
给定任意 $(x,p):\mathsf{fib}_f(y)$，需构造
$$
(g(y),\epsilon_y)=(x,p).
$$
由 $\Sigma$ 路径刻画，只需给出路径
$$
r:g(y)=x
$$
以及 transport 后第二分量等于 $p$ 的路径。令
$$
r\coloneqq \mathsf{ap}_g(p^{-1})\cdot\eta_x:
g(y)=x.
$$
第二分量相容性对 $p:f(x)=y$ 作路径归纳。反身情形中 $y\equiv f(x)$，中心第一分量为 $g(f(x))$，路径 $r$ 化为 $\eta_x$。目标变为说明沿 $\eta_x:g(f(x))=x$ transport $\epsilon_{f(x)}:f(g(f(x)))=f(x)$ 得到 $\mathsf{refl}_{f(x)}$。这是准逆三角相干的一个命题形式；若原始准逆数据没有给出该相干，可通过把 $\epsilon$ 替换为相干化后的 $\epsilon'$ 得到半伴随等价数据。

因此严格地说，普通准逆推出 fiber 可收缩需要先进行相干化。相干化的构造见定理 G.4。$\square$

**定理 G.4（准逆相干化为半伴随等价）.** 若 $f$ 有准逆，则存在 $g,\eta,\epsilon$ 以及三角相干
$$
\tau:\prod_{x:A}\mathsf{ap}_f(\eta_x)=\epsilon_{f(x)}
$$
或等价方向的相干条件，使 $f$ 成为半伴随等价。

**证明.** 设准逆数据为
$$
g:B\to A,\qquad
\eta:\prod_{x:A}g(f(x))=x,\qquad
\epsilon:\prod_{y:B}f(g(y))=y.
$$
保留 $g$ 和 $\eta$，把右逆同伦替换为
$$
\epsilon'_y
\coloneqq
\epsilon_{f(g(y))}^{-1}
\cdot
\bigl(\mathsf{ap}_f(\eta_{g(y)})\cdot\epsilon_y\bigr).
$$
类型检查如下：
$$
\epsilon_{f(g(y))}^{-1}:f(g(y))=f(g(f(g(y)))),
$$
$$
\mathsf{ap}_f(\eta_{g(y)}):
f(g(f(g(y))))=f(g(y)),
$$
$$
\epsilon_y:f(g(y))=y.
$$
故 $\epsilon'_y:f(g(y))=y$。

需构造
$$
\tau_x:\mathsf{ap}_f(\eta_x)=\epsilon'_{f(x)}.
$$
展开右侧得
$$
\epsilon'_{f(x)}
=
\epsilon_{f(g(f(x)))}^{-1}
\cdot
\bigl(\mathsf{ap}_f(\eta_{g(f(x))})\cdot\epsilon_{f(x)}\bigr).
$$
准逆左同伦 $\eta:g\circ f\sim\mathsf{id}_A$ 对映射 $g\circ f$ 的自然性给出
$$
\eta_{g(f(x))}
=
\mathsf{ap}_g(\mathsf{ap}_f(\eta_x))
$$
作为从 $g(f(g(f(x))))$ 到 $g(f(x))$ 的路径。把 $f$ 作用到该等式，并用右同伦 $\epsilon:f\circ g\sim\mathsf{id}_B$ 的自然性，得到
$$
\mathsf{ap}_f(\eta_{g(f(x))})\cdot\epsilon_{f(x)}
=
\epsilon_{f(g(f(x)))}\cdot\mathsf{ap}_f(\eta_x).
$$
代回 $\epsilon'_{f(x)}$ 后，右侧化为
$$
\epsilon_{f(g(f(x)))}^{-1}
\cdot
\bigl(\epsilon_{f(g(f(x)))}\cdot\mathsf{ap}_f(\eta_x)\bigr),
$$
由路径结合律和左逆律等于 $\mathsf{ap}_f(\eta_x)$。取该路径的逆即得
$$
\mathsf{ap}_f(\eta_x)=\epsilon'_{f(x)}.
$$
因此 $(g,\eta,\epsilon',\tau)$ 是半伴随等价数据。$\square$

## G.3 半伴随等价推出 fiber 可收缩

**定义 G.5.** 函数 $f:A\to B$ 的半伴随等价结构由
$$
g:B\to A,\quad
\eta:\prod_{x:A}g(f(x))=x,\quad
\epsilon:\prod_{y:B}f(g(y))=y
$$
和三角相干
$$
\tau:\prod_{x:A}\mathsf{ap}_f(\eta_x)=\epsilon_{f(x)}
$$
组成。若采用相反复合约定，$\tau$ 的方向相应取逆。

**定理 G.6（半伴随等价推出等价）.** 若 $f$ 有半伴随等价结构，则 $\mathsf{isEquiv}(f)$。

**证明.** 固定 $y:B$。取 fiber 中心 $(g(y),\epsilon_y)$。对任意 $(x,p):\mathsf{fib}_f(y)$，按 G.3 构造第一分量路径
$$
r\coloneqq\mathsf{ap}_g(p^{-1})\cdot\eta_x:g(y)=x.
$$
用 $\Sigma$ 路径刻画，剩余目标为第二分量相容性。对 $p$ 作路径归纳，化为 $y\equiv f(x)$ 的情形；此时第二分量相容性正由三角相干 $\tau_x$ 和 transport 计算给出。故 fiber 中任意点等于中心。$\square$

**推论 G.7（等价定义比较）.** 对任意 $f:A\to B$：

1.  $\mathsf{isEquiv}(f)$ 推出 $\mathsf{qinv}(f)$；
2.  $\mathsf{qinv}(f)$ 推出半伴随等价结构；
3.  半伴随等价结构推出 $\mathsf{isEquiv}(f)$。

**证明.** 第 1 条为附录 D.14。第 2 条为定理 G.4。第 3 条为定理 G.6。$\square$

## G.4 等价的逆

**定理 G.8（等价的逆仍是等价）.** 若 $f:A\to B$ 是等价，则由附录 D.14 得到的逆函数 $g:B\to A$ 也是等价。

**证明.** 由 $\mathsf{isEquiv}(f)$ 得 $\mathsf{qinv}(f)$，即有
$$
g:B\to A,\quad
\eta:\prod_{x:A}g(f(x))=x,\quad
\epsilon:\prod_{y:B}f(g(y))=y.
$$
把同一组数据交换左右，可看作 $g$ 的准逆数据：其逆函数为 $f$，左/右逆同伦分别为 $\epsilon$ 与 $\eta$。由推论 G.7，准逆推出等价，故 $g$ 是等价。$\square$

## G.5 等价诱导路径空间等价

**定理 G.9（等价作用于路径空间）.** 若 $e:A\simeq B$，底层函数为 $f:A\to B$，则对任意 $x,y:A$，函数
$$
\mathsf{ap}_f:(x=y)\to(f(x)=f(y))
$$
是等价。

**证明.** 由定理 G.8 取 $f$ 的逆等价，底层函数为 $g:B\to A$，并有同伦
$$
\eta:\prod_{z:A}g(f(z))=z.
$$
定义反向函数
$$
\Theta:(f(x)=f(y))\to(x=y)
$$
为
$$
\Theta(q)\coloneqq \eta_x^{-1}\cdot\mathsf{ap}_g(q)\cdot\eta_y.
$$
对 $p:x=y$，证明 $\Theta(\mathsf{ap}_f(p))=p$，对 $p$ 作路径归纳即可。对 $q:f(x)=f(y)$，证明 $\mathsf{ap}_f(\Theta(q))=q$，对 $q$ 作路径归纳后化为三角相干和路径代数。使用半伴随相干化后的逆数据可直接完成。由准逆到等价比较，$\mathsf{ap}_f$ 是等价。$\square$

## G.6 等价保持同伦层级

**定理 G.10（等价保持同伦层级）.** 若 $A\simeq B$ 且 $A$ 具有 $h$-level $n$，则 $B$ 具有 $h$-level $n$。

**证明.** 对 $n$ 归纳。

基步 $n=0$：这是第五章命题 5.12，等价保持可收缩性。

归纳步：假设结论对 $n$ 成立。设 $A$ 具有 $h$-level $n+1$，取 $b,b':B$。由等价逆得到 $g:B\to A$。路径空间
$$
b=b'
$$
与
$$
g(b)=g(b')
$$
等价，理由是定理 G.9 应用于逆等价 $g$。由于 $A$ 具有 $h$-level $n+1$，路径空间 $g(b)=g(b')$ 具有 $h$-level $n$。由归纳假设沿路径空间等价传回，$b=b'$ 具有 $h$-level $n$。因此 $B$ 具有 $h$-level $n+1$。$\square$
