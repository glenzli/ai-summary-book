# 附录 E：等价证明核

## 目标

本附录把第五章中关于等价稳定性的证明展开。基础定义固定为
$$
\mathsf{isEquiv}(f)\coloneqq\prod_{y}\mathsf{isContr}(\mathsf{fib}_f(y)).
$$
因此证明一个函数是等价，就是逐点证明它的 fiber 可收缩。

## E.1 可收缩基底上的可收缩纤维

**定理 E.1（可收缩 total space）.** 设 $A:\mathcal U$，$B:A\to\mathcal U$。若 $A$ 可收缩，并且每个 $B(a)$ 可收缩，则
$$
\sum_{a:A}B(a)
$$
可收缩。

**证明.** 设 $A$ 的收缩数据为 $(a_0,H)$，其中
$$
H:\prod_{a:A}(a_0=a).
$$
对每个 $a:A$，设 $B(a)$ 的收缩中心为 $b_a:B(a)$，收缩证明为
$$
K_a:\prod_{u:B(a)}(b_a=u).
$$
取总空间中心为
$$
(a_0,b_{a_0}).
$$
对任意 $(a,b):\sum_{a:A}B(a)$，由 $\Sigma$ 路径刻画，只需给出
$$
p:a_0=a
$$
和
$$
\mathsf{transport}^{B}(p,b_{a_0})=b.
$$
取 $p\coloneqq H(a)$。第二个目标的两边都是 $B(a)$ 的项；由 $K_a$ 得到路径
$$
K_a(\mathsf{transport}^{B}(H(a),b_{a_0}))^{-1}\cdot K_a(b)
$$
或直接由 $B(a)$ 的命题性得到。于是总空间中任意点都等于中心。$\square$

**推论 E.2（可收缩基底上的性质纤维）.** 若 $A$ 可收缩，且每个 $B(a)$ 是命题，并且某个 $B(a_0)$ 有点，则 $\sum_{a:A}B(a)$ 可收缩。

**证明.** 因为 $A$ 可收缩，任意 $B(a)$ 由 transport 得到点；命题且有点推出可收缩。再用定理 E.1。$\square$

## E.2 复合函数的 fiber 分解

设
$$
f:A\to B,\qquad g:B\to C.
$$
对 $c:C$，定义辅助类型
$$
T_c\coloneqq \sum_{w:\mathsf{fib}_g(c)}\mathsf{fib}_f(\mathsf{pr}_1(w)).
$$
写出元素即为
$$
((b,q),(a,r))
$$
其中
$$
b:B,\quad q:g(b)=c,\quad a:A,\quad r:f(a)=b.
$$

**定义 E.3（复合 fiber 到分解 fiber）.** 定义
$$
\Phi_c:\mathsf{fib}_{g\circ f}(c)\to T_c
$$
如下。给定 $(a,p)$，其中 $p:g(f(a))=c$，令
$$
\Phi_c(a,p)\coloneqq ((f(a),p),(a,\mathsf{refl}_{f(a)})).
$$

**定义 E.4（分解 fiber 到复合 fiber）.** 定义
$$
\Psi_c:T_c\to\mathsf{fib}_{g\circ f}(c)
$$
如下。给定 $((b,q),(a,r))$，其中 $r:f(a)=b$ 且 $q:g(b)=c$，令
$$
\Psi_c((b,q),(a,r))\coloneqq
(a,\mathsf{ap}_g(r)\cdot q).
$$

**定理 E.5（复合 fiber 分解）.** 对每个 $c:C$，$\Phi_c$ 与 $\Psi_c$ 给出双向同伦：
$$
\Psi_c\circ\Phi_c\sim\mathsf{id}_{\mathsf{fib}_{g\circ f}(c)},
$$
$$
\Phi_c\circ\Psi_c\sim\mathsf{id}_{T_c}.
$$

**证明.** 证明两个复合与恒等同伦。

先看 $\Psi_c\circ\Phi_c$。对 $(a,p):\mathsf{fib}_{g\circ f}(c)$，
$$
\Psi_c(\Phi_c(a,p))
=
(a,\mathsf{ap}_g(\mathsf{refl}_{f(a)})\cdot p).
$$
由 $\mathsf{ap}$ 的反身路径计算，第二分量为 $\mathsf{refl}_{g(f(a))}\cdot p$。由左单位律得它等于 $p$。再由 $\Sigma$ 路径刻画，得到
$$
\Psi_c(\Phi_c(a,p))=(a,p).
$$

再看 $\Phi_c\circ\Psi_c$。给定 $((b,q),(a,r)):T_c$，目标是
$$
((f(a),\mathsf{ap}_g(r)\cdot q),(a,\mathsf{refl}_{f(a)}))
=
((b,q),(a,r)).
$$
对路径 $r:f(a)=b$ 作路径归纳。反身情形中 $b\equiv f(a)$ 且 $r\equiv\mathsf{refl}_{f(a)}$。目标化为比较
$$
((f(a),\mathsf{refl}_{g(f(a))}\cdot q),(a,\mathsf{refl}_{f(a)}))
$$
与
$$
((f(a),q),(a,\mathsf{refl}_{f(a)})).
$$
第一层差异由左单位律 $\mathsf{refl}\cdot q=q$ 给出；第二层 fiber 分量在 transport 后仍为反身路径。应用 $\Sigma$ 路径刻画两次得到所需路径。

因此 $\Phi_c$ 和 $\Psi_c$ 互为准逆意义下的双向同伦。这里不把它升级为 $\mathsf{isEquiv}$，以避免依赖第五章等价定义比较定理。$\square$

**定理 E.6（可收缩类型的 retract 仍可收缩）.** 设 $A,B:\mathcal U$，有函数
$$
i:A\to B,\qquad r:B\to A
$$
和同伦
$$
H:\prod_{a:A}r(i(a))=a.
$$
若 $B$ 可收缩，则 $A$ 可收缩。

**证明.** 设 $B$ 的收缩中心为 $b_0$，收缩同伦为 $K:\prod_{b:B}(b_0=b)$。取 $A$ 的中心为 $r(b_0)$。对任意 $a:A$，有
$$
K(i(a)):b_0=i(a).
$$
对它应用 $\mathsf{ap}_r$ 得
$$
\mathsf{ap}_r(K(i(a))):r(b_0)=r(i(a)).
$$
再与 $H(a):r(i(a))=a$ 复合，得到 $r(b_0)=a$。$\square$

## E.3 等价的复合

**定理 E.7（等价复合）.** 若 $f:A\to B$ 与 $g:B\to C$ 都是等价，则
$$
g\circ f:A\to C
$$
是等价。

**证明.** 取任意 $c:C$。需证明
$$
\mathsf{fib}_{g\circ f}(c)
$$
可收缩。由定理 E.5，足以证明
$$
T_c=\sum_{w:\mathsf{fib}_g(c)}\mathsf{fib}_f(\mathsf{pr}_1(w))
$$
可收缩。

因为 $g$ 是等价，$\mathsf{fib}_g(c)$ 可收缩。对任意
$$
w:\mathsf{fib}_g(c),
$$
其第一投影 $\mathsf{pr}_1(w):B$。因为 $f$ 是等价，$\mathsf{fib}_f(\mathsf{pr}_1(w))$ 可收缩。于是由定理 E.1，$T_c$ 可收缩。

由定理 E.5，$\mathsf{fib}_{g\circ f}(c)$ 是 $T_c$ 的 retract：取
$$
i\coloneqq\Phi_c,\qquad r\coloneqq\Psi_c,
$$
而 $\Psi_c\circ\Phi_c\sim\mathsf{id}$ 给出 retract 同伦。由定理 E.6，$\mathsf{fib}_{g\circ f}(c)$ 可收缩。因为 $c$ 任意，$g\circ f$ 是等价。$\square$

## E.4 等价的逆方向

**定理 E.8（等价给出的逆函数也是等价）.** 若 $f:A\to B$ 是等价，并用附录 D.14 取其准逆 $g:B\to A$，则 $g$ 也是等价。

**证明.** 见附录 G.8。由 $\mathsf{isEquiv}(f)$ 和 D.14 得到准逆数据
$$
g:B\to A,\qquad
\eta:\prod_{x:A}g(f(x))=x,\qquad
\epsilon:\prod_{y:B}f(g(y))=y.
$$
把同一组数据交换左右，可看作 $g$ 的准逆数据，其逆函数为 $f$，左右逆同伦分别为 $\epsilon$ 与 $\eta$。由 G.7 中“准逆推出等价”的比较定理，$g$ 是等价。$\square$
