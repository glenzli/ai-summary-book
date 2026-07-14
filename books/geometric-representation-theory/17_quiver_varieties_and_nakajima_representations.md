# 第十七章：Quiver varieties 与 Nakajima 表示构造

Springer resolution 从一对“幂零算子与稳定旗标”产生 Weyl group 表示；Nakajima quiver variety 把同一思想推广为“线性映射组、moment-map 方程与稳定性”的 Hamiltonian reduction。顶点维数 $v$ 决定被约化的 gauge group，framing $w$ 决定最高权，而改变某个 $v_i$ 的 Hecke correspondence 扮演 Chevalley generator。抽象定义中最容易丢失的是 moment map、GIT stability 与 cotangent 几何之间的关系。为此将单顶点无边 quiver 算到底：一般商是 $T^*\operatorname{Gr}(v,W)$，在 $v=1,w=2$ 时可用矩阵坐标得到 $T^*\mathbb P^1\to\mathcal N(\mathfrak{sl}_2)$，正好重现第五章的最低秩 Springer resolution。

## 17.1 Quiver 数据

**定义 17.1.** 一个 quiver $Q=(I,H)$ 由顶点集 $I$ 和箭头集 $H$ 构成。给定 dimension vectors $v,w\in\mathbb Z_{\ge0}^I$，取向量空间
$$
V_i\simeq\mathbb C^{v_i},\qquad W_i\simeq\mathbb C^{w_i}.
$$

**定义 17.2.** framed doubled quiver representation space 为
$$
\mathbf M(v,w)=
\bigoplus_{h\in H}\operatorname{Hom}(V_{s(h)},V_{t(h)})
\oplus
\bigoplus_{h\in H}\operatorname{Hom}(V_{t(h)},V_{s(h)})
\oplus
\bigoplus_i\operatorname{Hom}(W_i,V_i)
\oplus
\bigoplus_i\operatorname{Hom}(V_i,W_i).
$$
群
$$
G_v=\prod_i GL(V_i)
$$
按基变换作用。

**定义 17.3.** $\mathbf M(v,w)$ 有自然 symplectic form，moment map 记为
$$
\mu:\mathbf M(v,w)\to\mathfrak g_v^\ast.
$$
Nakajima quiver variety 定义为 GIT quotient
$$
\mathfrak M_\theta(v,w)=\mu^{-1}(0)^{\theta\text{-ss}}/\!/G_v.
$$

这里 $\theta$ 是 $G_v$ 的 character；不同 chambers 可给出不同稳定性与 birational models。以下单顶点例子选择使 framing map $i:W\to V$ 满射的 chamber。只有在 semistable points 没有非平凡 stabilizer 时，GIT quotient 才可直接按几何轨道商理解。

**定义 17.4.** 写一个点为
$$
(B_h,B_{\bar h},i,j),
$$
其中 $B_h:V_{s(h)}\to V_{t(h)}$，$B_{\bar h}:V_{t(h)}\to V_{s(h)}$，$i:W_i\to V_i$，$j:V_i\to W_i$。在顶点 $i$ 处，moment map 分量形式为
$$
\mu_i=\sum_{h:t(h)=i}B_hB_{\bar h}
-\sum_{h:s(h)=i}B_{\bar h}B_h+i_i j_i.
$$
符号依 quiver orientation convention 可能整体改变。

**命题 17.5.** moment map 方程在每个顶点 $i$ 处具有“入箭头复合减出箭头复合加 framing 项”的形式。

**证明.** $G_v$ 的 Lie algebra 为 $\bigoplus_i\operatorname{End}(V_i)$。对 $\xi_i\in\operatorname{End}(V_i)$，infinitesimal action 在所有以 $i$ 为端点的箭头映射上给出左乘或右乘项，在 framing maps 上给出相应基变换项。moment map 由 symplectic form 与 infinitesimal action 配对定义，因此每个 $\xi_i$ 的系数即为该顶点处的矩阵方程。$\square$

## 17.2 基本例子

**例 17.6.** 对单顶点无边 quiver，取 $\dim V=v$、$\dim W=w$。则
$$
\mathbf M(v,w)=\operatorname{Hom}(W,V)\oplus\operatorname{Hom}(V,W)
$$
点记作 $(i,j)$，moment map 为
$$
\mu(i,j)=ij\in\operatorname{End}(V).
$$
在合适稳定性条件下，$\mathfrak M(v,w)$ 与 $T^\ast\operatorname{Gr}(v,W)$ 同构。

**命题 17.7.** 在上述例子中，若稳定性要求 $i:W\to V$ 满射，则 quotient by $GL(V)$ 的底空间为 Grassmannian $\operatorname{Gr}(v,W)$。

**证明.** 满射 $i:W\to V$ 的 kernel 是 $W$ 中 codimension $v$ 的子空间，等价于选择 $W$ 的 $v$ 维 quotient。两个满射在 $GL(V)$ 作用下等价当且仅当它们有同一 kernel。故商为 quotient Grassmannian，也可与 $v$ 维子空间 Grassmannian 对偶识别。加入 $j$ 和方程 $ij=0$ 后得到 cotangent direction。$\square$

上述 cotangent 识别可以直接从 tangent space 推出。令 $K=\ker i$，则 quotient Grassmannian 在 $i:W\twoheadrightarrow V$ 处的 tangent space 是
$$
T_{[i]}\operatorname{Gr}(v,W)\simeq\operatorname{Hom}(K,V).
$$
方程 $ij=0$ 等价于 $\operatorname{im}j\subset K$，故 $j$ 给出 $V\to K$。取 trace pairing
$$
\operatorname{Hom}(K,V)\times\operatorname{Hom}(V,K)\longrightarrow\mathbb C,
\qquad (a,j)\longmapsto\operatorname{tr}_V(a\circ j),
$$
便把 $j$ 识别为该 tangent space 的 covector。这个识别与 $GL(V)$-换基相容，因此下降到 quotient，得到
$$
\mathfrak M_\theta(v,w)\simeq T^*\operatorname{Gr}(v,W).
$$

**例 17.7.1（两个坐标图上的 $T^*\mathbb P^1$）.** 取 $V=\mathbb C$、$W=\mathbb C^2$。写
$$
i=(a\ b):\mathbb C^2\to\mathbb C,
\qquad
j=\begin{pmatrix}c\\d\end{pmatrix}:\mathbb C\to\mathbb C^2.
$$
Moment-map 方程为
$$
ij=ac+bd=0,
$$
稳定性是 $(a,b)\ne(0,0)$，而 $t\in GL(V)=\mathbb C^\times$ 的作用为
$$
t\cdot(i,j)=(ti,jt^{-1}).
$$
在 $a\ne0$ 的图上取 gauge $a=1$，置 $x=b/a$。变换后的 $j$ 唯一写成
$$
j=\begin{pmatrix}-xp\\p\end{pmatrix},
$$
所以该图的商坐标为 $(x,p)\in\mathbb A^2$。在 $b\ne0$ 的图上取 $b=1$，置 $y=a/b$，并写
$$
j=\begin{pmatrix}q\\-yq\end{pmatrix}.
$$
重叠处 $y=x^{-1}$、$q=-x^2p$，这正是 covector 在坐标变换 $y=x^{-1}$ 下的过渡公式。因此两个图粘成 $T^*\mathbb P^1$，零截面由 $j=0$ 给出。

**命题 17.7.2（到最低秩 nilpotent cone 的映射）.** 在例 17.7.1 中，公式
$$
\pi(i,j)=ji\in\operatorname{End}(W)
$$
下降为 proper birational morphism
$$
\pi:T^*\mathbb P^1\longrightarrow
\mathcal N(\mathfrak{sl}_2),
$$
其零元 fiber 是 $\mathbb P^1$，非零 nilpotent 元的 fiber 是一点。

**证明.** 换基后 $(jt^{-1})(ti)=ji$，故公式对 $GL(V)$-轨道不变。又由 $ij=0$，
$$
(ji)^2=j(ij)i=0,
\qquad
\operatorname{tr}(ji)=\operatorname{tr}(ij)=0,
$$
所以 $ji\in\mathcal N(\mathfrak{sl}_2)$。令 $\ell=\ker i$；它对每个稳定点都是 $W$ 中的直线。由 $ij=0$ 有 $\operatorname{im}j\subset\ell$，而 $i$ 满射给出 $\operatorname{im}(ji)=\operatorname{im}j$，故
$$
\operatorname{im}(ji)\subset\ell\subset\ker(ji).
$$
对二维空间上的非零 nilpotent $A$，有 $\operatorname{im}A=\ker A$，故 $\ell$ 被 $A$ 唯一确定。因而 $\pi$ 在非零 orbit 上为同构。若 $ji=0$，由于 $i$ 满射可推出 $j=0$，剩余数据只是 quotient line $i$，所以 fiber 为 $\mathbb P^1$。把 $T^*\mathbb P^1$ 写成
$$
\{(A,\ell)\in\mathcal N\times\mathbb P^1
\mid \operatorname{im}A\subset\ell\subset\ker A\}
$$
后，$\pi$ 是到第一因子的投影；第二因子 projective，故 $\pi$ proper。它在稠密非零 orbit 上为同构，因此 birational。$\square$

## 17.3 表示构造

**外部输入定理 17.8.** Nakajima quiver varieties 的 Borel-Moore homology 或 middle homology 携带 Kac-Moody algebra $\mathfrak g_Q$ 的表示；在固定 framing $w$ 后，直和
$$
\bigoplus_v H_\ast(\mathfrak L(v,w))
$$
给出最高权为 $w$ 的 integrable representation。

**例 17.8.1（$A_1$ 的三个权空间）.** 对单顶点无边 quiver 取 $w=2$。只有 $v=0,1,2$ 给出非空 quotient Grassmannians，分别为点、$\mathbb P^1$、点。到 affine quotient 的零 fiber $\mathfrak L(v,2)$ 是 $T^*\operatorname{Gr}(v,2)$ 的零截面，因为 $ji=0$ 与 $i$ 满射合起来强迫 $j=0$。因此三个 $\mathfrak L(v,2)$ 都不可约，其 top Borel--Moore homology 各有一个 fundamental class，权依次为
$$
2,\qquad 0,\qquad -2.
$$
Nakajima 定理把它们组成最高权 $2$ 的 $\mathfrak{sl}_2$-表示。这里权的排列由几何直接给出，Chevalley relations 与作用系数仍由外部输入定理 17.10 保证。

**定义 17.9.** Hecke correspondence between quiver varieties 通过改变 $v$ by simple root $\alpha_i$ 定义，产生 Chevalley generators $e_i,f_i$ 的几何算子。其点通常包括两个 quiver variety 点和一个 $I$-graded subspace
$$
V'\subset V
$$
使得 $\dim V/V'$ 为 simple root 方向。

**外部输入定理 17.10.** 这些 Hecke correspondences 满足 Kac-Moody relations，并实现相应 highest weight representation。

## 17.4 与 Springer 和 Satake 的接口

**边界说明 17.11.** 某些 quiver varieties 同构于 affine Grassmannian slices、Slodowy slices 或 instanton moduli spaces。这些同构支撑了 quiver variety、geometric Satake、symplectic duality 和 Coulomb branch 之间的联系，但每个同构都需要独立定理。

单顶点模型把三层结构逐一显出：moment equation $ij=0$ 把 framing 的反向映射变成 cotangent covector，稳定商把满射 $W\to V$ 变成 Grassmannian，而 invariant $ji$ 给出到 affine nilpotent cone 的 projective resolution。$v=0,1,2$ 的零 fibers 已经呈现最高权 $2$ 表示的三个权。一般 quiver 中，Hecke correspondences 改变一个顶点的 $v_i$，其关系由 Nakajima 定理控制；下一章将用 KLR algebra 把这些生成元与关系完全代数化。

## 练习

**练习 17.1.** 对单顶点无边 quiver，写出 $\mathbf M(v,w)$ 和 moment map。

**练习 17.2.** 对 $A_1$ quiver，解释 quiver variety 与 cotangent bundle of Grassmannian 的关系。

**练习 17.3.** 写出 Hecke correspondence 改变 dimension vector 的基本图。

**练习 17.4.** 对单顶点无边 quiver，证明 $ij=0$ 正好使 $j$ 分解为 $V\to\ker i$，并用 trace pairing 将它识别为 quotient Grassmannian 上的 cotangent covector。

**练习 17.5.** 在例 17.7.1 的两个坐标图中验证 $q=-x^2p$，并直接算出 $ji$ 的矩阵，检查其平方为零。
