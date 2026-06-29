# 附录 J：同伦转移的树公式与最小模型约定

## 本附录目标

第十三章陈述了同伦转移定理。本附录把实际计算中使用的树公式、side conditions 和符号边界集中写出。它的作用是让读者能从一个 contraction 出发，明确写出低阶 transferred operations，并知道哪些高阶符号仍依赖外部输入定理。

## J.1 Contraction 与 side conditions

**定义 J.1.** 一个 contraction 数据写作
$$
\left(
H\xrightarrow{i}A\xrightarrow{p}H,\ h:A\to A[1]
\right),
$$
满足
$$
pi=\operatorname{id}_H,\qquad
ip-\operatorname{id}_A=d_Ah+hd_A.
$$
若还满足
$$
hi=0,\qquad ph=0,\qquad h^2=0,
$$
则称为 normalized contraction 或 strong deformation retract data。

**命题 J.2.** 在 normalized contraction 中，
$$
pdh=0,\qquad phd=0,\qquad hdi=0,\qquad dhi=0.
$$

**证明.** 因为 $p$ 是链映射，$pd=d_Hp$。由 $ph=0$ 得 $pdh=d_Hph=0$。同理 $hd i=h i d_H=0$，因为 $hi=0$。其余两个等式分别由 $ph=0$ 与 $hi=0$ 直接在链映射关系两侧复合得到。$\square$

**外部输入命题 J.3（side condition normalization）.** 在通常的链复形语境中，任意 contraction 可替换为同伦等价的 normalized contraction。具体公式依赖 perturbation lemma 或标准 homological algebra 调整。本书使用 normalized contraction 做树公式；若输入数据未 normalized，应先引用该命题进行替换。

## J.2 平面二叉树与 $A_\infty$ 转移

**定义 J.4.** 令 $\operatorname{PBT}_n$ 为有 $n$ 个叶的平面二叉树集合。每个 $T\in\operatorname{PBT}_n$ 的叶按平面顺序标为 $1,\ldots,n$。若根顶点的左、右子树分别为 $T_L,T_R$，写作
$$
T=T_L\vee T_R.
$$

**定义 J.5.** 设 $(A,d,\mu)$ 是 dg associative algebra，并给定 normalized contraction
$$
H\xrightarrow{i}A\xrightarrow{p}H,\qquad h:A\to A[1].
$$
对每棵平面二叉树 $T$ 定义两个映射：
$$
\Phi_T:H^{\otimes n}\to A,\qquad
M_T:H^{\otimes n}\to H.
$$
递归如下。

若 $T$ 是单叶，则
$$
\Phi_T=i.
$$
若 $T=T_L\vee T_R$ 且 $T$ 有至少两个叶，则
$$
\Phi_T=h\,\mu(\Phi_{T_L}\otimes\Phi_{T_R}),
$$
而根部输出定义为
$$
M_T=p\,\mu(\Phi_{T_L}\otimes\Phi_{T_R}).
$$
所有张量符号按约定 E.1 和约定 E.3 的 Koszul rule 解释。

**定义 J.6.** 转移的未悬挂 $A_\infty$ 运算在本书中写作
$$
m_n^H=\sum_{T\in\operatorname{PBT}_n}\epsilon(T)M_T,
$$
其中 $\epsilon(T)$ 是由定义 E.18--定义 E.23 的 suspended convention 决定的符号。低阶时取
$$
m_1^H=d_H,\qquad
m_2^H=p\mu(i\otimes i).
$$

**命题 J.7.** 对 $n=3$，平面二叉树公式给出
$$
m_3^H(x,y,z)
=
\epsilon_L\,p\mu(h\mu(ix,iy),iz)
+\epsilon_R\,p\mu(ix,h\mu(iy,iz)),
$$
其中 $\epsilon_L,\epsilon_R\in\{\pm1\}$ 由所选 suspended convention 决定。

**证明.** $\operatorname{PBT}_3$ 有两棵树：$(12)3$ 与 $1(23)$。对第一棵树，左子树有两个叶，故
$$
\Phi_{T_L}=h\mu(i\otimes i),
$$
右子树为单叶，故 $\Phi_{T_R}=i$，根部输出为
$$
p\mu(h\mu(ix,iy),iz).
$$
第二棵树同理给出
$$
p\mu(ix,h\mu(iy,iz)).
$$
符号由把未悬挂公式从 suspended 公式转回时产生。$\square$

**命题 J.8.** 若 $h=0$ 且 $ip=\operatorname{id}_A$，则 $m_n^H=0$ 对所有 $n\ge3$。

**证明.** 任意 $T\in\operatorname{PBT}_n$ 且 $n\ge3$，至少有一条内部边不连接根部。定义 J.5 在每条内部边上放入 $h$。若 $h=0$，则对应 $M_T=0$。故求和为零。$\square$

## J.3 $A_\infty$-morphism 分量

**定义 J.9.** 转移 quasi-isomorphism
$$
I_\infty:H\rightsquigarrow A
$$
的树分量写作
$$
I_n=\sum_{T\in\operatorname{PBT}_n}\epsilon_I(T)\Phi_T.
$$
特别地
$$
I_1=i,
$$
而对 $n=2$，
$$
I_2(x,y)=\epsilon\,h\mu(ix,iy).
$$

**命题 J.10.** $I_1=i$ 是 quasi-isomorphism。

**证明.** 由 contraction 等式，$i$ 与 $p$ 在同调上互为逆；这已经在命题 13.3 中证明。$\square$

**外部输入定理 J.11（$A_\infty$ homotopy transfer with signs）.** 在 normalized contraction 下，定义 J.6 与 J.9 的树公式，配合标准 suspended signs，给出 $H$ 上的 $A_\infty$-代数结构和 $A_\infty$ quasi-isomorphism
$$
I_\infty:H\rightsquigarrow A.
$$

**定位说明.** Markl MHT-6 支撑 strongly homotopy associative structures 在 chain homotopy equivalence 下的存在性转移。定义 J.6 和定义 J.9 的具体 sign convention 由定义 E.18--定义 E.23 与附录 W 的悬挂约定固定；Kadeishvili/Loday--Vallette/Fresse 的未悬挂公式作为 convention translation 使用。
该定理的完整证明依赖 homological perturbation lemma 或 bar construction 上 coderivation 的转移。

## J.4 低阶 $A_\infty$ 恒等式的检查

**命题 J.12.** 若 $A$ 是 dg associative algebra，则转移公式满足 $A_\infty$ 恒等式的 $n=2$ 部分：
$$
d_Hm_2^H=m_2^H(d_H\otimes1)+(-1)^{|x|}m_2^H(1\otimes d_H)
$$
在元素 $x\otimes y$ 上成立。

**证明.** 展开左侧：
$$
d_Hp\mu(ix,iy)=p\,d_A\mu(ix,iy),
$$
因为 $p$ 是链映射。$\mu$ 是链映射，所以
$$
d_A\mu(ix,iy)=\mu(d_Aix,iy)+(-1)^{|x|}\mu(ix,d_Aiy).
$$
又 $i$ 是链映射，$d_Aix=i d_Hx$，$d_Aiy=i d_Hy$。代回即得所需公式。$\square$

**命题 J.13.** $m_3^H$ 的两棵树项正是 $m_2^H$ 结合律失败的链同伦修正项。

**证明.** 比较
$$
m_2^H(m_2^H(x,y),z)
=p\mu(i p\mu(ix,iy),iz)
$$
与
$$
m_2^H(x,m_2^H(y,z))
=p\mu(ix,i p\mu(iy,iz)).
$$
使用 contraction 恒等式
$$
ip=\operatorname{id}_A+d_Ah+hd_A.
$$
把 $ip$ 替换到两式中。含 $\operatorname{id}_A$ 的部分由 $\mu$ 的严格结合律相消。剩余含 $d_Ah$ 与 $hd_A$ 的部分可重写为 $d_Hm_3^H$、$m_3^H(d_H\otimes1\otimes1)$ 等边界项。完整符号由定义 E.18--定义 E.23 的 suspended convention 给出。$\square$

## J.5 $L_\infty$ 转移与反对称化

**定义 J.14.** 设 $(\mathfrak g,d,[-,-])$ 是 dg Lie algebra，并给定 normalized contraction
$$
H\xrightarrow{i}\mathfrak g\xrightarrow{p}H,\qquad h:\mathfrak g\to\mathfrak g[1].
$$
对有根二叉树 $T$，定义 $\Psi_T$ 与 $L_T$：叶放入 $i$，内部顶点放入 bracket，内部边放入 $h$，根部放入 $p$。

**定义 J.15.** $L_\infty$ 转移运算写作
$$
\ell_n^H(x_1,\ldots,x_n)
=
\sum_{T}\sum_{\sigma\in\operatorname{Sh}(T)}
\epsilon(T,\sigma;x_\bullet)
L_T(x_{\sigma(1)},\ldots,x_{\sigma(n)}),
$$
其中外层对非平面有根二叉树求和，内层对相应 shuffle 求和。符号由 Koszul 反对称化和附录 E 的悬挂约定共同决定。

**命题 J.16.** 低阶运算为
$$
\ell_1^H=d_H,\qquad
\ell_2^H(x,y)=p[ix,iy],
$$
且 $\ell_2^H$ 满足 graded antisymmetry。

**证明.** 第一式来自一元结构。第二式来自唯一二叶树。由于 dg Lie bracket 满足
$$
[ix,iy]=-(-1)^{|x||y|}[iy,ix],
$$
并且 $p$ 是次数 $0$ 线性映射，$\ell_2^H$ 满足同一反对称关系。$\square$

**外部输入定理 J.17（$L_\infty$ homotopy transfer with signs）.** 定义 J.15 的树反对称化公式给出 $H$ 上的 $L_\infty$-结构，并且 $i$ 延拓为 $L_\infty$ quasi-isomorphism
$$
H\rightsquigarrow\mathfrak g.
$$
完整证明依赖 operadic homotopy transfer theorem，或由 $A_\infty$ 型 coderivation 转移在 cocommutative coalgebra 上的反对称版本推出。

**定位说明.** Markl MHT-6 支撑 strongly homotopy Lie structures 在 chain homotopy equivalence 下的存在性转移。定义 J.15 的 shuffle signs 与反对称化公式由定义 E.18--定义 E.23 与附录 W 的悬挂约定固定；Merkulov/Loday--Vallette/Fresse 的未悬挂公式作为 convention translation 使用。

## J.6 Minimal models 与选择依赖

**定义 J.18.** 若 $H=H_\*(A)$ 且 $d_H=0$，转移得到的 $A_\infty$ 或 $L_\infty$ 结构称为一个 minimal model。

**外部输入定理 J.19（minimal model uniqueness）.** 在域上，两个由不同 normalized contractions 得到的 minimal $A_\infty$-models 或 $L_\infty$-models 由 $\infty$-isomorphism 相连。该 $\infty$-isomorphism 的线性项为同调上的恒等映射。

**警告 J.20.** 单个高阶运算 $m_n$ 的具体公式依赖 contraction 选择；其在 $\infty$-isomorphism 类中的信息才是同伦不变量。Massey product 与 $m_3$ 的关系也受代表元和 null-homotopy 选择影响，不能把某一组树公式的数值直接称为不变量。

## J.7 本附录小结

同伦转移的可计算核心是：

1. 选择 normalized contraction；
2. 用平面二叉树写 $A_\infty$ 运算；
3. 用有根树加 shuffle 反对称化写 $L_\infty$ 运算；
4. 用定义 E.18--定义 E.23 的 suspended convention 管理符号；
5. 把全高阶恒等式和唯一性作为外部输入定理。

本附录使第十三章的树公式具备明确的数据来源和低阶检查，但不替代 homological perturbation lemma 的完整证明。
