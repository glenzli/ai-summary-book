# 练习解答与提示

本文件给出主体章节练习的解答要点。完整出版版应把每题扩成独立证明；当前版本保证每题至少有可核验路线。

## 第零章

**0.1.** 缺少增强类别、系数、brane data、分次、等价类型和候选函子。只写三角范畴等价不能检测 Hochschild/Morita 数据。

**0.2.** quasi-equivalence 给出 morphism complexes 的 quasi-isomorphism；取 $H^0$ 得 morphism 集合双射。本质满是定义的一部分。

**0.3.** Holomorphic polygons 给出 $\mu^d$，$d\ge3$。这些高阶项控制链级结合失败，不能由普通二元复合记录。

## 第一章

**1.1.** 单位 $1_X$ 是闭零次元素。其 cohomology class 对任意 $[f]$ 满足 $[1_Y\circ f]=[f]=[f\circ1_X]$。

**1.2.** 单对象 dg category 的 right modules 即 right dg $A$-modules；perfect objects 是由 $A_A$ 通过 shifts、cones、direct summands 生成的 dg modules。

**1.3.** $d=1$ 给 $\mu^1{}^2=0$；$d=2$ 给 Leibniz；$d=3$ 给 associator 为 $\mu^3$ 的边界。

**1.4.** quasi-equivalence 保持 representables，并延拓到 shifts、cones 和 direct summands，因此诱导 perfect categories 的 quasi-equivalence。

## 第二章

**2.1.** 导出范畴按定义把 quasi-isomorphisms 形式反演，因此 quasi-isomorphism 的像为同构。

**2.2.** $X=\operatorname{Spec}A$ 时，perfect complexes 是局部有限投射分解；等价于 perfect dg $A$-modules。

**2.3.** 用投影公式把 $p_1^\ast E\otimes\Delta_\ast\mathcal O_X$ 化为 $\Delta_\ast E$，再用 $p_2\Delta=\operatorname{id}$。

**2.4.** 例如 nodal curve 或 dual numbers $\operatorname{Spec}k[\epsilon]/(\epsilon^2)$。残差模可能无有限投射分解，故属于 $\mathrm D^b\operatorname{Coh}$ 但不 perfect。

## 第三章

**3.1.** 对 isotropic $W$，有 $W\subset W^\omega$，而 $\dim W+\dim W^\omega=2n$，故 $\dim W\le n$。

**3.2.** 零截面与小 Hamiltonian 图像交点对应 Morse function 的临界点。Floer cochains 由这些临界点生成。

**3.3.** 能量等于 $\int u^\ast\omega=\int_{\partial u}\lambda$，边界上 $\lambda=df_L$，得到 action 差。

**3.4.** Disk bubbling 会成为一维模空间紧化中的额外边界项，使 $\mu^1\mu^1$ 不再唯一对应 broken strips。

## 第四章

**4.1.** 稳定多边形的维数公式适用于 $d\ge2$：$d=2$ 是三角形，$d=3$ 是四边形的一维模空间。$d=1$ 的 strip 是 Floer differential 的单独模型，不应直接代入稳定多边形公式。

**4.2.** $\mu^2$ 由三角形计数。其链级 associator 由四边形边界控制，所以只在 cohomology 上严格结合。

**4.3.** 一个边界分量对应输入中连续一段先合成为 $\mu^s$，再插入外层 $\mu^{r+1+t}$。

**4.4.** 非紧 Lagrangian 经 Hamiltonian wrapping 后，交点由时间一 Hamiltonian chords 替代。

## 第五章

**5.1.** 固定 valuation 截断只含有限多项；乘积中低于任意能量界的指数来自有限组合，因此仍趋向无穷。

**5.2.** 采用附录 B 的 suspended convention。零输入时，对每个对象 $X$
只有
$$
b_1(b_0(X))=0.
$$
对齐次 $x\in s\operatorname{hom}(X_0,X_1)$，一个输入时恰有
$$
b_1b_1(x)+b_2(b_0(X_1),x)
+(-1)^{|x|}b_2(x,b_0(X_0))=0.
$$
这是全部三项，不存在省略的 higher-curvature insertion；特别地，$b_0=0$
时得到 $b_1^2=0$。Desuspension 的符号由 (B.7a) 固定。

**5.3.** 见附录 I：$\mu_b^0=0$ 后，变形低阶方程只剩 $\mu_b^1\mu_b^1=0$。

**5.4.** $W'=1-x^{-2}$，critical points 为 $x=\pm1$；Jacobian ring 为 $k[x^{\pm1}]/(x-x^{-1})\cong k[x]/(x^2-1)$。

## 第六章

**6.1.** 在边界上 $\alpha=\lambda|_{\partial M}$，Liouville 向外性给 $\alpha\wedge(d\alpha)^{n-1}$ 非零。

**6.2.** 对径向 Hamiltonian，chords 对应 Reeb chords；在 $T^\ast S^1$ 中可按绕行次数标记。

**6.3.** wrapped morphism 是 cofinal Hamiltonians 的极限；不同 Hamiltonian 间必须用 continuation maps 比较。

**6.4.** compact category 通常对象紧、morphisms 由交点生成；wrapped 允许非紧对象，morphisms 由 Hamiltonian chords 生成。

## 第七章

**7.1.** Stop 是 wrapping 不可穿越的障碍。Stop 越大，允许路径越少。

**7.2.** 在二维图中 stop 是无穷远端点；linking disk 是靠近该端点并绕住它的小弧。

**7.3.** 零对象在三角范畴中对 shifts、cones、direct summands 稳定。

**7.4.** 例：stop removal square。A-side 为 $\mathcal W(M,\mathfrak g)\to\mathcal W(M,\mathfrak f)$，B-side 为 quotient 或 open restriction。

## 第八章

**8.1.** 指定 A-side 几何、B-side 几何、两边增强范畴、系数、分次、等价类型和候选函子。

**8.2.** quasi-equivalence 保持 morphism cohomology；properness 保证 Euler 交错和有限。

**8.3.** $K_0$ 同构只记录对象的加性关系，不记录 morphism complexes 和高阶复合。

**8.4.** 按模板 8.15 列九项；关键是生成对象、endomorphism algebra 和外部输入。

## 第九章

**9.1.** $|2\cdot5-3\cdot1|=7$。

**9.2.** 椭圆曲线上 $\chi(E,F)=\operatorname{rank}(E)\deg F-\operatorname{rank}(F)\deg E$。

**9.3.** lift 到 $\mathbb R^2$ 后，三条直线围成仿射三角形；面积给 Novikov 权重。

**9.4.** $T^2\to S^1$ 的 fiber 是 $S^1$，对偶 fiber 参数为 rank-one local systems。

## 第十章

**10.1.** 对 $\mathbb P^2$，$W=x+y+q/(xy)$；critical equations 为 $x=q/(xy)$、$y=q/(xy)$，故 $x=y$ 且 $x^3=q$。

**10.2.** 见附录 J：critical values 为 $\pm2q^{1/2}$。

**10.3.** Laurent variables 可逆，$z_i\partial_i$ 是 torus-invariant vector field 对应的导数。

**10.4.** A-side LG 为 $(\mathbb C^\ast,z+qz^{-1})$，B-side 为 $\operatorname{Perf}(\mathbb P^1)$，生成元为两个 thimbles 与 $(\mathcal O,\mathcal O(1))$。

## 第十一章

**11.1.** 同 10.2。

**11.2.** 方向条件排除逆向 morphisms，使高阶复合只能沿对象顺序递增。

**11.3.** 绕临界值的单值化在 fiber 上为沿 vanishing cycle 的 Dehn twist。

**11.4.** 使用 Beilinson 分解或 $\mathbb P^1$ 上任意 coherent sheaf 有由 $\mathcal O,\mathcal O(1)$ 生成的有限分解；作为外部输入。

## 第十二章

**12.1.** $K_{X_d}=\mathcal O(d-n-1)|_{X_d}$，故 $d=n+1$ 时 Calabi-Yau。

**12.2.** Serre functor 由 category 内部确定，增强等价必须保持；Calabi-Yau 维数由其 shift 检测。

**12.3.** 列 degeneration、A-side category、B-side MF/sheaf category、生成对象、endomorphism algebra、外部输入。

**12.4.** 核查 polytope 条件、smoothness、coefficient field、category model、generation theorem 和 theorem statement。

## 第十三章

**13.1.** $x+y=1$ 给 $y=1-x$，且 $x,y\ne0$，故 $x\in\mathbb C\setminus\{0,1\}$。

**13.2.** 二项式 tropical hypersurface 是两仿射函数相等的超平面；三项式给 tropical line。

**13.3.** 它断言 wrapped categories 可由 sectorial pieces 的 homotopy colimit 重建，非形式。

**13.4.** 写 A/B 两个 Cech diagrams，逐点等价并取 homotopy colimit。

## 第十四章

**14.1.** 记 $L=-\otimes^{\mathbf L}_{\mathcal A_{\mathcal G}}\mathcal A$
为 extension，$R=i^*$ 为 restriction。令 $\mathcal T\subset
\operatorname{Perf}(\mathcal A)$ 由 counit $LR(M)\to M$ 为
quasi-isomorphism 的对象组成。因 $L,R$ exact 且保持有限直和与 retracts，
$\mathcal T$ 是厚子范畴；Yoneda fully faithfulness 表明它包含所有 $Y_G$。
由 (14.1)，$\mathcal T=\operatorname{Perf}(\mathcal A)$。同理，unit
$N\to RL(N)$ 为 quasi-isomorphism 的对象在
$\operatorname{Perf}(\mathcal A_{\mathcal G})$ 中构成厚子范畴，并包含该
category 的全部 representables；故它也是全范畴。于是 unit 与 counit 处处
为 quasi-isomorphisms，$L,R$ 互为 quasi-inverse。

**14.2.** (14.3) 把 Hochschild degree $q$ 送到
$SH^{q+n}(M)$。要命中 $SH^0(M)$，必须有 $q+n=0$，故所需 class 位于
$HH_{-n}(\mathcal B)$。

**14.3.** 取 full subcategory
$\mathcal W_{\mathcal G}\subset\mathcal W(M)$、小严格含单位 B-side category
$\mathcal C$ 及 full subcategory $\mathcal C_{\mathcal H}$。依次验证：存在
$\alpha\in HH_{-n}(\mathcal W_{\mathcal G})$ 使
$\mathcal{OC}(\alpha)=1_{SH}$；$\mathcal H$ split-generates $\mathcal C$；
存在保持 units 与对象标号的 strictly unital quasi-equivalence
$$
\mathcal W_{\mathcal G}\simeq_{\mathrm{qe}}\mathcal C_{\mathcal H}.
$$
定理 14.7 给出 $\mathcal G$ 对 $\mathcal W(M)$ 的 split-generation，第二项
给出 B-side split-generation，第三项比较两个 generating full
subcategories。命题 14.9 因而给出
$\mathcal W(M)\simeq_{\mathrm{Morita}}\mathcal C$；结论不是 raw categories
的 quasi-equivalence。

**14.4.** 写 $e_1=(1,0)$、$e_2=(0,1)$，则
$k\times k=ke_1\oplus ke_2$，global unit 是 $1=e_1+e_2$。子空间（也是理想）
$ke_1$ 非零并含 $e_1$，但不含 $e_1+e_2$。因此 open-closed image 若为
$ke_1$，最多在另行引用 summand generation theorem 后推出第一个
idempotent summand 的生成；它不能满足定理 14.7 的 global-unit hypothesis。
若 image 含 $(1,1)$，则 exact wrapped generation criterion 才推出两个
summands 组成的全范畴被 split-generate。

## 第十五章

**15.1.** 两开集覆盖有 $\mathcal W(X_1)$、$\mathcal W(X_2)$、$\mathcal W(X_{12})$，箭头从交到各开集再到全局。

**15.2.** sector inclusion 诱导协变 functor，故是 cosheaf 型 gluing。

**15.3.** 局部生成对象在 diagram 的 homotopy colimit 中生成 colimit category。

**15.4.** Pair-of-pants pieces 上证明局部 HMS，沿交叠比较，再用 descent。

## 第十六章

**16.1.** 常值 sheaf 的 microsupport 是零截面；support 只记录底空间位置。

**16.2.** 对开区间 stratum，标准为 extension by zero，余标准为 direct image。

**16.3.** conormal covectors 湮灭 tangent directions；其维数为 $\dim Q$，且 canonical symplectic form 限制为零。

**16.4.** 先用 microlocal equivalence 把 $\mathcal W$ 转成 sheaves，再与 B-side coherent category 比较。

## 第十七章

**17.1.** 取 stop removal square；四个顶点为移除前后 A-side categories 与相应 B-side quotient/open categories。

**17.2.** Orlov functor 在不同语境中源靶不同；不声明方向会导致相反 functor。

**17.3.** 增强自然变换诱导 $K$、HH、Euler 上相等映射。

**17.4.** 两者都把一个子范畴中的对象强制为零。

## 第十八章

**18.1.** Hochschild chains 可由 perfect bimodules 表述，Morita 等价保持双模范畴。

**18.2.** 椭圆曲线有 $h^{0,0}=h^{1,1}=h^{1,0}=h^{0,1}=1$。

**18.3.** HKR 裸同构不保持 Mukai pairing；Todd 修正恢复 pairing 相容。

**18.4.** 例如椭圆曲线：交点数/Euler、theta 乘法、Serre functor、HH 维数。

## 第十九章

**19.1.** $F_f=\{x^a+y^b=\epsilon\}\cap B_\delta(0)$。

**19.2.** MF 是 $\mathbb Z/2$-分次并满足 $d^2=f$；普通复形满足 $d^2=0$。

**19.3.** Rabinowitz HMS 为 Morita 等价，HH Morita invariance 给同构。

**19.4.** Rabinowitz category 记录 contact/Reeb/Rabinowitz action 数据，不等同 wrapped category。

## 第二十章

**20.1.** 需 theorem statement、假设、证明依赖、符号翻译和 locator。

**20.2.** 例如稳定条件跨墙时 spherical twist 作用在 derived category 上。

**20.3.** BPS category 没有唯一模型，需指定 DT/CoHA/MF/Fukaya 等版本。

**20.4.** 例：统一 stopped wrapped categories 与 BPS wall-crossing 的 functorial HMS 框架；障碍是模型和 functor 方向不统一。
