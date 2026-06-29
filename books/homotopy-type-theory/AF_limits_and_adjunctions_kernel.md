# 附录 AF：终对象唯一性与伴随形式证明核

本附录补全第十四章 14.6 与 14.8 的证明核。全篇在附录 P、X 的预范畴和函子范畴口径下工作；Hom 类型为集合，自然变换相等由逐分量相等给出。

## AF.1 终对象唯一性

设 $\mathcal C$ 为预范畴。

**定义 AF.1（终对象）.** 对象 $t:\mathcal C$ 是终对象，若
$$
\mathsf{isTerminal}(t)\coloneqq
\prod_{x:\mathcal C}\mathsf{isContr}(\mathcal C(x,t)).
$$

**引理 AF.2（终对象之间唯一同构）.** 若 $t,u$ 都是终对象，则
$$
t\cong u.
$$

**证明.** 由 $u$ 终，Hom $\mathcal C(t,u)$ 可收缩，取其中心
$$
f:t\to u.
$$
由 $t$ 终，Hom $\mathcal C(u,t)$ 可收缩，取其中心
$$
g:u\to t.
$$
要证明 $g\circ f=\mathsf{id}_t$，注意二者都是 $\mathcal C(t,t)$ 的元素，而 $t$ 终说明 $\mathcal C(t,t)$ 可收缩。类似地，$f\circ g=\mathsf{id}_u$ 由 $u$ 终和 $\mathcal C(u,u)$ 可收缩得到。因此 $(f,g)$ 给出同构。$\square$

**定理 AF.3（单值范畴中终对象唯一到路径）.** 若 $\mathcal C$ 是单值范畴，且 $t,u$ 是终对象，则
$$
t=u.
$$

**证明.** 由 AF.2 得同构 $t\cong u$。单值范畴给出
$$
(t=u)\simeq(t\cong u)
$$
的等价，取其逆方向作用于该同构，得到对象路径。$\square$

**推论 AF.4（极限对象唯一性）.** 若某图的锥范畴是单值范畴，则该图的极限对象唯一到路径。

**证明.** 极限定义为锥范畴中的终对象，直接应用 AF.3。$\square$

## AF.2 伴随：Hom 等价形式到单位/余单位

设
$$
F:\mathcal C\to\mathcal D,\qquad
G:\mathcal D\to\mathcal C
$$
为函子。

**定义 AF.5（Hom 等价形式）.** 一个伴随 Hom 等价形式由自然于 $c:\mathcal C$、$d:\mathcal D$ 的等价族组成：
$$
\Phi_{c,d}:\mathcal D(Fc,d)\simeq\mathcal C(c,Gd).
$$

**定义 AF.6（单位与余单位）.** 给定 AF.5，定义单位
$$
\eta_c:c\to GFc
$$
为
$$
\eta_c\coloneqq\Phi_{c,Fc}(\mathsf{id}_{Fc}),
$$
定义余单位
$$
\epsilon_d:FGd\to d
$$
为
$$
\epsilon_d\coloneqq\Phi^{-1}_{Gd,d}(\mathsf{id}_{Gd}).
$$

**命题 AF.7（三角恒等式）.** AF.6 的 $\eta,\epsilon$ 满足
$$
\epsilon_{Fc}\circ F(\eta_c)=\mathsf{id}_{Fc},
$$
$$
G(\epsilon_d)\circ\eta_{Gd}=\mathsf{id}_{Gd}.
$$

**证明核.** 第一条是 $\Phi_{c,Fc}$ 下 $\eta_c$ 的定义与 $\Phi^{-1}$ 的自然性在 $f=\mathsf{id}_{Fc}$ 处的计算。更显式地，$\Phi$ 的自然性说明对任意
$$
h:Fc\to d
$$
有
$$
\Phi(h)=G(h)\circ\eta_c.
$$
取 $d=Fc$、$h=\mathsf{id}_{Fc}$ 得 $\eta_c=\Phi(\mathsf{id})$。再把 $F(\eta_c)$ 代入 $\Phi^{-1}$ 的对应公式，可得
$$
\epsilon_{Fc}\circ F(\eta_c)=\mathsf{id}_{Fc}.
$$
第二条对偶，使用
$$
\Phi^{-1}(k)=\epsilon_d\circ F(k)
$$
并取 $k=\mathsf{id}_{Gd}$。所有等式位于 Hom 集合中，自然性证明的高阶相容由 Hom 集合性消去。$\square$

## AF.3 伴随：单位/余单位到 Hom 等价形式

**定义 AF.8（单位/余单位形式）.** 一个单位/余单位形式由自然变换
$$
\eta:\mathsf{Id}_{\mathcal C}\Rightarrow GF,
\qquad
\epsilon:FG\Rightarrow\mathsf{Id}_{\mathcal D}
$$
和三角恒等式
$$
\epsilon_{Fc}\circ F(\eta_c)=\mathsf{id}_{Fc},
$$
$$
G(\epsilon_d)\circ\eta_{Gd}=\mathsf{id}_{Gd}
$$
组成。

**定义 AF.9（由单位/余单位定义 Hom 映射）.** 给定 AF.8，定义
$$
\Phi_{c,d}:\mathcal D(Fc,d)\to\mathcal C(c,Gd)
$$
为
$$
\Phi_{c,d}(f)\coloneqq G(f)\circ\eta_c.
$$
定义反向
$$
\Psi_{c,d}:\mathcal C(c,Gd)\to\mathcal D(Fc,d)
$$
为
$$
\Psi_{c,d}(g)\coloneqq \epsilon_d\circ F(g).
$$

**命题 AF.10（Hom 映射互逆）.** 对任意 $c,d$，$\Phi_{c,d}$ 与 $\Psi_{c,d}$ 互为逆。

**证明.** 对 $f:Fc\to d$，
$$
\begin{aligned}
\Psi(\Phi(f))
&=\epsilon_d\circ F(G(f)\circ\eta_c)\\
&=\epsilon_d\circ F(G(f))\circ F(\eta_c)\\
&=f\circ\epsilon_{Fc}\circ F(\eta_c)\\
&=f\circ\mathsf{id}_{Fc}\\
&=f.
\end{aligned}
$$
第三步用余单位 $\epsilon$ 的自然性，第四步用第一三角恒等式。

对 $g:c\to Gd$，
$$
\begin{aligned}
\Phi(\Psi(g))
&=G(\epsilon_d\circ F(g))\circ\eta_c\\
&=G(\epsilon_d)\circ G(F(g))\circ\eta_c\\
&=G(\epsilon_d)\circ\eta_{Gd}\circ g\\
&=\mathsf{id}_{Gd}\circ g\\
&=g.
\end{aligned}
$$
第三步用单位 $\eta$ 的自然性，第四步用第二三角恒等式。$\square$

**定理 AF.11（伴随两种形式等价）.** Hom 等价形式 AF.5 与单位/余单位形式 AF.8 互相构造。

**证明.** AF.5 到 AF.8 由 AF.6 和 AF.7 给出。AF.8 到 AF.5 由 AF.9 和 AF.10 给出等价族；自然性由函子律、$\eta$ 与 $\epsilon$ 的自然性以及 Hom 集合性证明。两种构造互逆在 Hom 集合层面逐点验证；自然变换和等价证明分量的高阶相容由附录 X.4 的自然变换路径原则与 $\mathsf{isEquiv}$ 的命题性消去。$\square$
