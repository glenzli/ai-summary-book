# 附录 P：预范畴与单值范畴证明核

## 目标

本附录补齐第十三章的核心证明义务：预范畴的精确定义、对象路径到同构的映射 $\mathsf{idtoiso}$、同构证明的命题性、集合范畴的单值性，以及由结构等同性原则得到的代数结构范畴单值性。

本附录只处理 Hom 类型为集合的 1-范畴口径，不处理 $(\infty,1)$-范畴。

## P.1 预范畴记录

**定义 P.1（预范畴）.** 一个预范畴 $\mathcal C$ 由以下数据组成：

1.  对象类型 $\mathcal C_0:\mathcal U$；
2.  Hom 族
    $$
    \mathcal C(-,-):\mathcal C_0\to\mathcal C_0\to\mathcal V;
    $$
3.  Hom 集合性
    $$
    \mathsf{homset}_{x,y}:\mathsf{isSet}(\mathcal C(x,y));
    $$
4.  恒等态射
    $$
    \mathsf{id}_x:\mathcal C(x,x);
    $$
5.  复合
    $$
    (g:\mathcal C(y,z))\mapsto(f:\mathcal C(x,y))\mapsto g\circ f:\mathcal C(x,z);
    $$
6.  单位律
    $$
    \mathsf{id}_y\circ f=f,\qquad f\circ\mathsf{id}_x=f;
    $$
7.  结合律
    $$
    h\circ(g\circ f)=(h\circ g)\circ f.
    $$

由于每个 Hom 是集合，单位律和结合律的目标是命题。因此预范畴结构中这些律的证明分量不会在路径比较中产生额外高阶结构。

## P.2 同构

**定义 P.2（isIso 与 iso）.** 对态射 $f:\mathcal C(x,y)$，定义
$$
\mathsf{isIso}(f)\coloneqq
\sum_{g:\mathcal C(y,x)}
\bigl((g\circ f=\mathsf{id}_x)\times(f\circ g=\mathsf{id}_y)\bigr).
$$
对象 $x,y$ 的同构类型定义为
$$
x\cong y\coloneqq
\sum_{f:\mathcal C(x,y)}\mathsf{isIso}(f).
$$

**命题 P.3（isIso 是命题）.** 对任意 $f:\mathcal C(x,y)$，$\mathsf{isIso}(f)$ 是命题。

**证明.** 取两个逆数据
$$
(g,\ell,r),(g',\ell',r'):\mathsf{isIso}(f),
$$
其中
$$
\ell:g\circ f=\mathsf{id}_x,\quad r:f\circ g=\mathsf{id}_y
$$
以及对应的 $\ell',r'$。先证 $g=g'$：
$$
g
=
g\circ\mathsf{id}_y
=
g\circ(f\circ g')
=
(g\circ f)\circ g'
=
\mathsf{id}_x\circ g'
=
g'.
$$
每一步分别使用右单位律、$r'^{-1}$、结合律、$\ell$ 和左单位律。得到 $p:g=g'$ 后，由 Hom 集合性，逆律证明所在路径类型是命题，因此 $\ell,\ell'$ 与 $r,r'$ 的相容性自动给出。用 $\Sigma$ 路径刻画得到两个逆数据相等。$\square$

**推论 P.4（iso 的路径由底层态射路径决定）.** 若 $u,v:x\cong y$，则路径 $u=v$ 等价于其底层态射相等。

**证明.** $x\cong y$ 是 $\sum_{f:\mathcal C(x,y)}\mathsf{isIso}(f)$。由命题 P.3，第二分量是命题，故 $\Sigma$ 路径只需比较第一分量。$\square$

## P.3 idtoiso

**定义 P.5（idtoiso）.** 对 $x,y:\mathcal C_0$，定义
$$
\mathsf{idtoiso}_{x,y}:(x=y)\to(x\cong y)
$$
为对路径 $p:x=y$ 作路径归纳。反身情形定义为恒等同构：
$$
\mathsf{idtoiso}_{x,x}(\mathsf{refl}_x)
\coloneqq
(\mathsf{id}_x,\mathsf{id}_x,\lambda_x,\rho_x),
$$
其中 $\lambda_x:\mathsf{id}_x\circ\mathsf{id}_x=\mathsf{id}_x$ 和 $\rho_x:\mathsf{id}_x\circ\mathsf{id}_x=\mathsf{id}_x$ 分别由预范畴单位律给出。

**定义 P.6（单值范畴）.** 预范畴 $\mathcal C$ 是单值范畴，若
$$
\mathsf{isUnivalentCat}(\mathcal C)
\coloneqq
\prod_{x,y:\mathcal C_0}\mathsf{isEquiv}(\mathsf{idtoiso}_{x,y}).
$$

**命题 P.7（单值范畴中的路径-同构等价）.** 若 $\mathcal C$ 是单值范畴，则
$$
(x=y)\simeq(x\cong y)
$$
自然于 $x,y:\mathcal C_0$。

**证明.** 直接由定义 P.6，取底层函数为 $\mathsf{idtoiso}_{x,y}$。$\square$

## P.4 集合范畴

**定义 P.8（集合范畴）.** 固定宇宙 $\mathcal U$。定义 $\mathsf{Set}_{\mathcal U}$：

1.  对象为
    $$
    \sum_{A:\mathcal U}\mathsf{isSet}(A);
    $$
2.  Hom 为普通函数
    $$
    \mathsf{Set}_{\mathcal U}((A,s),(B,t))\coloneqq A\to B;
    $$
3.  恒等和复合为函数恒等与函数复合。

Hom 集合性由函数外延性和 $B$ 是集合推出：若 $f,g:A\to B$，则 $f=g$ 的路径空间等价于逐点路径族；逐点路径位于集合 $B$ 的路径空间，因此是命题。

**命题 P.9（集合范畴的同构等价于类型等价）.** 对集合对象 $(A,s),(B,t)$，有等价
$$
((A,s)\cong(B,t))\simeq(A\simeq B).
$$

**证明.** 一个同构给出函数 $f:A\to B$、函数 $g:B\to A$ 和双向复合等于恒等函数的路径。这正是 $f$ 的准逆数据。由推论 G.7 得到 $f$ 是等价，故得到 $A\simeq B$。

反向地，等价 $e:A\simeq B$ 给出底层函数 $f:A\to B$ 和由附录 D.14 得到的逆函数及双向同伦；这些正是集合范畴中的同构数据。

两侧互逆：同构类型中逆数据是命题（P.3），等价结构 $\mathsf{isEquiv}(f)$ 也是命题（由 fiber 可收缩性和函数外延性）。因此只需检查底层函数，二者均为反身。$\square$

**定理 P.10（集合范畴是单值范畴）.** 假设单值性，则 $\mathsf{Set}_{\mathcal U}$ 是单值范畴。

**证明.** 对对象 $(A,s),(B,t)$，由 $\Sigma$ 路径刻画和推论 O.4，路径
$$
(A,s)=(B,t)
$$
等价于 $A=B$。由单值性，$A=B$ 等价于 $A\simeq B$。由命题 P.9，$A\simeq B$ 等价于 $(A,s)\cong(B,t)$。复合这些等价得到
$$
((A,s)=(B,t))\simeq((A,s)\cong(B,t)).
$$
该复合在反身路径上把 $\mathsf{refl}_{(A,s)}$ 送到恒等同构；由路径归纳，这与 $\mathsf{idtoiso}$ 相同。因此 $\mathsf{idtoiso}$ 是等价。$\square$

## P.5 代数结构范畴

**定理 P.11（命题性公理结构的范畴单值性）.** 设结构由有限运算、常元和命题性公理给出，且结构同态是保持全部运算和常元的函数。若结构对象路径与结构同构由附录 I-J 的 SIP 等价，则相应结构范畴是单值范畴。

**证明.** 见附录 AG.8。对结构对象 $X,Y$，$\mathsf{idtoiso}_{X,Y}$ 从对象路径诱导结构同构。附录 I.3 和 AG.7 给出对象路径与传统结构同构的等价，并且在反身路径上计算为恒等结构同构。由与定理 P.10 相同的路径归纳比较，该等价就是 $\mathsf{idtoiso}$，故 $\mathsf{idtoiso}$ 是等价。群范畴是 AG.9 的实例。$\square$
