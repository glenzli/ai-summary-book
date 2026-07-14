# 第七章：stops、partially wrapped categories 与 localization

Wrapped 范畴让 Lagrangian 在正方向不断移动，但实际几何常要求它不能越过无穷远的某些 Legendrian 区域。这样的禁区不是附加装饰：增大禁区会删除 Hamiltonian chords，移除禁区则会让原先围住禁区的 linking disks 变成可缩信息。Ganatra--Pardon--Shende 的 stop-removal 定理把这一变化精确识别为范畴局部化。本章在第六章的 Liouville sector 模型上加入 stop，并用第一章的 perfect/Morita 商说明为何“允许更多 wrapping”对应杀掉一个由 linking disks 生成的厚子范畴。

## 7.1 Stops

**定义 7.1.** 设 $M$ 是 Liouville manifold，其 contact boundary at infinity 记为 $\partial_\infty M$。一个 stop 是闭子集
$$
\mathfrak f\subset\partial_\infty M
$$
通常要求为 mostly Legendrian 或带有足够好的 stratified Legendrian 结构。它指定 wrapping 过程中不允许穿过的无穷远障碍。

**定义 7.2.** stopped Liouville manifold 是二元组 $(M,\mathfrak f)$。其 partially wrapped Fukaya category 记为
$$
\mathcal W(M,\mathfrak f).
$$
对象为避免 stop 的 admissible Lagrangians，morphisms 由不穿过 $\mathfrak f$ 的 positive wrapping 定义。

**解释 7.3.** 若 $\mathfrak f=\varnothing$，则 $\mathcal W(M,\mathfrak f)=\mathcal W(M)$。若 stop 增大，允许的 wrapping 减少，category 通常变小。

## 7.2 Linking disks

**定义 7.4.** 设 $\Lambda\subset\mathfrak f$ 是 stop 的光滑 Legendrian stratum。围绕 $\Lambda$ 的 linking disk 是一个小的 exact Lagrangian disk $D_\Lambda$，其边界在无穷远处 link 住 $\Lambda$。

**外部输入定理 7.5（linking disk generation near stops）.** 在 mostly Legendrian stop 假设下，stop 附近新增或移除的信息由相应 Legendrian strata 的 linking disks 控制。
来源：Ganatra-Pardon-Shende 的 partially wrapped generation 和 stop removal 理论。

**解释 7.6.** Linking disks 是 stop 的范畴影子。移除 stop 时，它们会成为被局部化掉的对象。

## 7.3 Stop removal equals localization

**外部输入定理 7.7（stop removal）.** 设 $\mathfrak f\subset\mathfrak g\subset\partial_\infty M$ 是合适 stops。则存在 functor
$$
\mathcal W(M,\mathfrak g)\to \mathcal W(M,\mathfrak f),
$$
并且在适当假设下它把 $\mathcal W(M,\mathfrak f)$ 表为 $\mathcal W(M,\mathfrak g)$ 对由 $\mathfrak g\setminus\mathfrak f$ 的 linking disks 生成的子范畴的 localization。

**范畴解释 7.8.** 若 $\mathcal D\subset\mathcal C$ 是由 linking disks 生成的 full subcategory，则 stop removal 的形式为
$$
\mathcal C/\mathcal D\simeq \mathcal W(M,\mathfrak f)
$$
在 Morita 或 pretriangulated quotient 口径下成立。

**命题 7.9.** 假设定理 7.7 成立。若对象 $K$ 属于被 linking disks split-generate 的子范畴，则其像在 stop-removed category 中为零对象。

**证明.** localization functor 按定义把局部化子范畴 $\mathcal D$ 的对象送到零。若 $K$ 属于 $\mathcal D$ 的 split-closure，则 $K$ 由 $\mathcal D$ 经过 shifts、cones 和 direct summands 得到；三角函子保持这些操作，并把 $\mathcal D$ 的对象送零，因此把 $K$ 送零。证毕。

局部化的作用可以先在一个完全代数的模型中看清。令 $\mathcal C$ 是只有两个
对象 $P,Q$ 的 dg category，所有 morphisms 集中在次数 $0$，且
$$
\operatorname{hom}(P,P)=k e_P,\qquad
\operatorname{hom}(Q,Q)=k e_Q,\qquad
\operatorname{hom}(P,Q)=k u,\qquad
\operatorname{hom}(Q,P)=0.
\tag{7.1}
$$
它是 $A_2$ quiver $P\to Q$ 的有向模型。

**例 7.9A（杀掉末端对象）.** 在 idempotent-complete pretriangulated 口径中，
$$
\operatorname{Perf}(\mathcal C)/\operatorname{thick}(Q)
\simeq \operatorname{Perf}(k).
\tag{7.2}
$$
右边的生成元对应 $P$ 的像。

**证明.** $P,Q$ 的 representables 厚生成 $\operatorname{Perf}(\mathcal C)$；
商中 $Q$ 为零，所以 $P$ 的像厚生成商。又因
$\operatorname{hom}^\ast(Q,P)=0$，通过 $Q$ 的 roofs 不会给 $P$ 产生新的
endomorphisms，故商中 $\operatorname{hom}^\ast(P,P)=k$。由单生成元的
derived Morita 理论，商与 $\operatorname{Perf}(k)$ 等价。证毕。

几何中的 linking disk 扮演 (7.2) 中 $Q$ 的角色。真实 stop-removal 定理的
困难在于证明几何函子的核恰由这些 linking disks 生成；一旦这一外部输入
成立，商后的形式推理与例 7.9A 相同。

## 7.4 Viterbo functor 与 Liouville 子域

**定义 7.10.** 若 $U\subset M$ 是 Liouville subdomain，Viterbo transfer
指从大空间到小空间的限制型结构。在采用 Sylvan 的 partially wrapped
模型并满足其定义域假设时，范畴方向写作
$$
\mathcal W(M)\to\mathcal W(U)
$$
；它与 Liouville sector inclusion 的协变函子
$\mathcal W(U)\to\mathcal W(M)$ 方向相反。

**警告 7.11.** 文献中 Viterbo functor 的方向依赖采用的是 covariant sector inclusion、restriction 还是 module-category 口径。本书每次使用都会明确方向，不使用“自然限制”这种未定向说法。

**外部输入定理 7.12（Viterbo 与 localization）.** 设 Liouville domains
$U\subset M$ 在 Sylvan 的构造中都 independently Weinstein。则相应
Viterbo functor 是 homological epimorphism，亦即传到 module categories 后为
localization；若余补 cobordism 还满足 Ganatra--Pardon--Shende 的 Weinstein
假设，则有更强的 genuine localization 版本。
来源：Sylvan, *Orlov and Viterbo functors in partially wrapped Fukaya
categories*，以及 GPS stop-removal/localization 定理。

## 7.5 局部化方块与镜像函子

Functorial HMS 试图把几何操作与范畴函子相匹配。例如：

- A-side stop removal 对应 B-side quotient 或 open restriction；
- A-side sector inclusion 对应 B-side pushforward 或 extension by zero；
- Orlov functor 对应 Landau-Ginzburg/Calabi-Yau correspondence 中的 functor；
- Viterbo transfer 对应 restriction to open subvarieties 或 localization of modules。

**定义 7.13.** 一个 functorial HMS square 是交换到指定自然同构或同伦的图：
$$
\begin{array}{ccc}
\mathcal W(A_1) & \xrightarrow{\Phi_A} & \mathcal W(A_2)\\
\downarrow\simeq & & \downarrow\simeq\\
\mathcal B_1 & \xrightarrow{\Phi_B} & \mathcal B_2 .
\end{array}
$$
其中竖直箭头为 HMS 等价，水平箭头由几何操作诱导。

**命题 7.14.** 若 functorial HMS square 中三条边是 Morita equivalences，且方块在 perfect module categories 中交换，则第四条边的 Morita 类型由其他三条边确定。

**证明.** 在 Morita homotopy category 中，等价态射可逆。若
$$
E_2\circ \Phi_A\simeq \Phi_B\circ E_1
$$
且 $E_1,E_2$ 为等价，则
$$
\Phi_A\simeq E_2^{-1}\circ \Phi_B\circ E_1.
$$
因此 $\Phi_A$ 的 Morita 类型由右侧确定。证毕。

Stop 由此成为一种可操作的无穷远边界条件：它的局部范畴信息由 linking disks 检测，移除 stop 则在 perfect/Morita 层面把这些对象强制为零。这个商结构不仅比较两个 wrapped 范畴，也给出了之后函子化 HMS 的基本方块；镜像一侧必须找到具有同一核与同一局部化性质的限制、商或 Orlov 型函子。

## 练习

**练习 7.1.** 说明为什么 stop 变大时允许的 wrapping 变少。

**练习 7.2.** 在一个二维 Liouville sector 的图像中画出 stop 和 linking disk，并用文字描述其作用。

**练习 7.3.** 证明命题 7.9 中 split-closure 对零对象的稳定性。

**练习 7.4.** 构造一个 functorial HMS square 的形式例子，并标出每条边的几何来源。
