# 附录 D：基础证明核

## 目标

本附录把正文反复使用的基础证明展开到“可审读证明蓝图”的粒度。除特别说明外，本附录只使用第一至二章的 $\Pi$、$\Sigma$、恒等类型、路径归纳、transport、路径复合和 $\mathsf{ap}$，不使用函数外延性、单值性、截断或高阶归纳类型。

## D.1 固定端点路径归纳

**定理 D.1（固定左端点路径归纳）.** 设 $A:\mathcal U_i$，$a:A$，并有族
$$
y:A,\ p:a=y\vdash C(y,p):\mathcal U_j.
$$
若有
$$
c:C(a,\mathsf{refl}_a),
$$
则存在
$$
\mathsf{basedJ}_{a,C}(c):\prod_{y:A}\prod_{p:a=y}C(y,p)
$$
并严格满足 J 的 $\beta$-计算
$$
\mathsf{basedJ}_{a,C}(c,a,\mathsf{refl}_a)\equiv c
$$
作为 judgmental equality。

**证明.** 按 HoTT Book §1.12.2，从完整 J 先构造一个量化所有固定端点族的通用 motive。令
$$
D(x,y,p)\coloneqq
\prod_{E:\prod_{z:A}\bigl((x=z)\to\mathcal U_j\bigr)}
\bigl(E(x,\mathsf{refl}_x)\to E(y,p)\bigr).
$$
这里必须显式提高 motive 的宇宙层级。因为 $\mathcal U_j:\mathcal U_{j+1}$，族空间
$$
\prod_{z:A}\bigl((x=z)\to\mathcal U_j\bigr)
$$
位于 $\mathcal U_{\max(i,j+1)}$，从而 $D(x,y,p):\mathcal U_{\max(i,j+1)}$。这是 J 向更高 universe 的一次消去，不是累积性，也没有把 $E(y,p):\mathcal U_j$ 隐式提升到该 universe。

反身分支定义为
$$
d(x)\coloneqq\lambda E.\,\lambda e.\,e:
D(x,x,\mathsf{refl}_x).
$$
完整 J 给出
$$
\mathsf J_D(d,x,y,p):D(x,y,p)
$$
并有严格计算
$$
\mathsf J_D(d,x,x,\mathsf{refl}_x)\equiv d(x).
$$
现在才固定题设的 $a$、$C$ 与 $c$，定义
$$
\mathsf{basedJ}_{a,C}(c,y,p)
\coloneqq
\mathsf J_D(d,a,y,p)(C)(c).
$$
在反身构造子上，逐次按 J 与函数应用的 judgmental $\beta$-规则计算：
$$
\begin{aligned}
\mathsf{basedJ}_{a,C}(c,a,\mathsf{refl}_a)
&\equiv \mathsf J_D(d,a,a,\mathsf{refl}_a)(C)(c)\\
&\equiv d(a)(C)(c)\\
&\equiv c.
\end{aligned}
$$
因此所需固定端点原则连同严格 $\beta$ 一并由完整 J 导出；证明不经过路径复合、单位律或 transport。$\square$

**说明 D.2.** 许多证明助手把固定端点路径归纳作为库引理提供。正文使用它时，应把它视为本定理的派生规则，而不是新的公理。

## D.2 $\Sigma$ 类型中的路径

**定义 D.3（总空间路径的编码）.** 设 $B:A\to\mathcal U$，$u=(a,b)$、$v=(a',b'):\sum_{x:A}B(x)$。定义
$$
\mathsf{code}_{\Sigma}(u,v)\coloneqq
\sum_{p:a=a'}\mathsf{transport}^{B}(p,b)=b'.
$$

**定义 D.4（由编码得到路径）.** 给定
$$
(p,q):\mathsf{code}_{\Sigma}((a,b),(a',b')),
$$
定义
$$
\mathsf{pairpath}(p,q):(a,b)=(a',b')
$$
如下：对 $p:a=a'$ 作路径归纳。反身情形中 $a'\equiv a$，且
$$
q:\mathsf{transport}^{B}(\mathsf{refl}_a,b)=b'
$$
按 transport 计算化为 $q:b=b'$。再对 $q$ 作路径归纳；反身情形中目标是 $(a,b)=(a,b)$，取 $\mathsf{refl}_{(a,b)}$。

**定义 D.5（由路径得到编码）.** 给定
$$
r:(a,b)=(a',b'),
$$
对 $r$ 作路径归纳。反身情形取
$$
(\mathsf{refl}_a,\mathsf{refl}_b):
\sum_{p:a=a}\mathsf{transport}^{B}(p,b)=b.
$$

**定理 D.6（$\Sigma$ 路径刻画）.** 有等价
$$
((a,b)=(a',b'))\simeq
\sum_{p:a=a'}\mathsf{transport}^{B}(p,b)=b'.
$$

**证明.** 两个方向由定义 D.4 和 D.5 给出。证明两个复合为恒等：对路径或编码作归纳。对路径方向，归纳到反身路径，两个方向均按计算规则化为反身路径。对编码方向，先对第一分量 $p$ 归纳，再对第二分量 $q$ 归纳，归纳终点也化为反身路径。$\square$

**推论 D.7（非依赖积路径）.** 对 $A\times B$ 中的元素 $(a,b)$ 与 $(a',b')$，有等价
$$
((a,b)=(a',b'))\simeq (a=a')\times(b=b').
$$

**证明.** 在定理 D.6 中令族 $B$ 为常值族。transport 沿常值族的路径 propositionally 等于恒等；用路径归纳把该命题化简即可。$\square$

## D.3 可收缩 total space

**定理 D.8（基点路径纤维可收缩）.** 对任意 $A:\mathcal U$ 和 $a:A$，类型
$$
\sum_{x:A}(a=x)
$$
可收缩。

**证明.** 取中心
$$
(a,\mathsf{refl}_a).
$$
给定任意 $(x,p):\sum_{x:A}(a=x)$，需构造
$$
(a,\mathsf{refl}_a)=(x,p).
$$
由 $\Sigma$ 路径刻画，只需给出
$$
r:a=x
$$
以及
$$
\mathsf{transport}^{\lambda z.\,a=z}(r,\mathsf{refl}_a)=p.
$$
取 $r\coloneqq p$。第二个目标对 $p$ 作路径归纳；反身情形中它化为
$$
\mathsf{refl}_a=\mathsf{refl}_a,
$$
取反身路径。$\square$

**推论 D.9（恒等函数是等价）.** 对任意 $A$，$\mathsf{id}_A:A\to A$ 是等价。

**证明.** 对 $y:A$，其 fiber 为
$$
\sum_{x:A}(x=y).
$$
由定理 D.8 对基点 $y$ 得到 $\sum_{x:A}(y=x)$ 可收缩。用路径逆给出与 $\sum_{x:A}(x=y)$ 的等价，或直接把定理 D.8 的证明按右端点版本重写。故每个 fiber 可收缩。$\square$

## D.4 命题性与 $\Sigma$

**定理 D.10（命题纤维的依赖和）.** 若 $A$ 是命题，且对每个 $x:A$，$B(x)$ 是命题，则
$$
\sum_{x:A}B(x)
$$
是命题。

**证明.** 取两点 $(a,b)$ 与 $(a',b')$。因为 $A$ 是命题，有
$$
p:a=a'.
$$
由 $\Sigma$ 路径刻画，只需给出
$$
\mathsf{transport}^{B}(p,b)=b'.
$$
而 $B(a')$ 是命题，且 $\mathsf{transport}^{B}(p,b):B(a')$、$b':B(a')$，故有该路径。$\square$

**定理 D.11（可收缩性的命题性，需函数外延性）.** 假设函数外延性。则 $\mathsf{isContr}(A)$ 是命题。

**证明.** 取两个收缩数据 $(c,H)$ 与 $(c',H')$。由 $H(c')$ 得路径 $p:c=c'$。用 $\Sigma$ 路径刻画，需证明沿 $p$ transport 后的收缩证明等于 $H'$。这是函数类型中的路径；由函数外延性，化为对每个 $x:A$ 的路径。目标位于路径类型 $c'=x$ 中。该路径类型是命题，因为可收缩类型的路径空间可由路径归纳和收缩数据证明为可收缩。展开方式为：任意两条 $r,s:c'=x$，先对 $r$ 作路径归纳，再用收缩中心的唯一性把 $s$ 化到反身情形。$\square$

**说明 D.12.** 定理 D.11 的证明是 HoTT 中典型的“性质是命题”证明：先用结构路径刻画，再用函数外延性和路径空间收缩。若当前章节尚未引入函数外延性，只能把它作为后续定理引用。

## D.5 Fiber 与等价的基本证明

**定理 D.13（fiber 中中心给出满射见证）.** 若 $f:A\to B$ 且 $\prod_{y:B}\mathsf{isContr}(\mathsf{fib}_f(y))$，则对每个 $y:B$ 有
$$
\sum_{x:A}f(x)=y.
$$

**证明.** 直接取 $\mathsf{fib}_f(y)$ 的收缩中心。$\square$

**定理 D.14（等价诱导准逆）.** 若 $f:A\to B$ 且 $\mathsf{isEquiv}(f)$，则 $\mathsf{qinv}(f)$。

**证明.** 对每个 $y:B$，令
$$
(g(y),\epsilon_y)
$$
为 $\mathsf{fib}_f(y)$ 的收缩中心。于是得到 $g:B\to A$ 和
$$
\epsilon:\prod_{y:B}f(g(y))=y.
$$
对 $x:A$，考虑 fiber $\mathsf{fib}_f(f(x))$ 中的两点
$$
(g(f(x)),\epsilon_{f(x)})
\quad\text{和}\quad
(x,\mathsf{refl}_{f(x)}).
$$
该 fiber 可收缩，所以二者有路径。对该路径作用第一投影，得到
$$
\eta_x:g(f(x))=x.
$$
于是 $(g,\eta,\epsilon)$ 是准逆数据。$\square$

**说明 D.15.** 从准逆反推 fiber 可收缩也成立；完整路线见附录 G.4-G.7。关键点是普通准逆需要先相干化为半伴随等价，再由半伴随相干证明每个 fiber 可收缩。
