# 附录 F：凝聚基础的形式化证明义务

## F.0 目标

第四卷前几章给出形式化蓝图，但还偏路线图。本附录把第一卷基础定理拆成更接近 proof assistant 的证明义务：对象、字段、引理、依赖和输出。它不写 Lean 代码，而是给可逐项翻译的数学规格。

## F.1 站点与覆盖

形式化一个小站点需要以下数据：

1. 小范畴 \(\mathcal C\)；
2. 对每个 \(U\in\mathcal C\)，覆盖筛或覆盖族集合；
3. 恒等覆盖；
4. 覆盖对拉回稳定；
5. 覆盖的传递性；
6. 小性证明。

**证明义务 F.1（有限覆盖族版本）。** 若使用有限覆盖族而非筛，则需额外证明：

$$
\{U_i\to U\}_{i\in I}
$$

与其生成筛定义同一 sheaf 条件。

**证明.** 覆盖族生成的筛由所有通过某个 \(U_i\to U\) 分解的态射组成。若预层满足覆盖族的匹配族粘合，则对生成筛上的匹配族，先限制到有限族得到粘合，再用自然性检查它对整个筛匹配。反向由筛 sheaf 条件限制到生成族。证毕。

## F.2 Sheaf 条件的等化子规格

对集合值预层 \(F\)，有限覆盖 \(\{U_i\to U\}\) 的 sheaf 条件可形式化为等化子：

$$
F(U)\to\prod_iF(U_i)\rightrightarrows\prod_{i,j}F(U_i\times_UU_j).
$$

**证明义务 F.2（唯一性与存在性拆分）。**

1. separated 性：该箭头为单射；
2. existence：每个匹配族位于该箭头像中；
3. uniqueness：separated 性给粘合唯一。

**证明.** 单射即两个截面在覆盖上相等则相等。存在性即匹配族可粘合。等化子条件同时编码二者。证毕。

## F.3 可表预层是 sheaf

紧 Hausdorff 站点中的可表预层

$$
h_T(S)=\operatorname{Hom}_{\mathbf{CHaus}}(S,T)
$$

应形式化为 sheaf。

**证明义务 F.3（商映射粘合）。** 若

$$
q:\coprod_iS_i\to S
$$

是有限联合满射覆盖，则 \(q\) 是 quotient map。

**证明.** \(\coprod_iS_i\) 紧，\(S\) Hausdorff。连续满射从紧空间到 Hausdorff 空间是闭映射，因此 quotient。证毕。

**推论 F.4.** \(h_T\) 满足 sheaf 条件。

**证明.** 覆盖上给出的相容连续映射等价于一个集合映射 \(\coprod_iS_i\to T\)，它在 \(q\) 的纤维上常值。由 quotient 性，唯一下降为连续 \(S\to T\)。证毕。

## F.4 凝聚阿贝尔群的阿贝尔范畴结构

形式化 \(\mathbf{CondAb}\) 的关键不是逐对象定义 kernel 和 cokernel，而是证明 sheafification 正合。

**证明义务 F.5（预 sheaf 阿贝尔群范畴）。** 预 sheaf 阿贝尔群范畴是 Grothendieck 阿贝尔范畴，极限和余极限逐对象计算。

**证明.** 它是函子范畴 \(\operatorname{Fun}(\mathcal C^{op},\mathbf{Ab})\)。函子范畴中的阿贝尔结构逐对象给出，生成元由可表自由阿贝尔群给出，filtered colimit 逐对象 exact。证毕。

**证明义务 F.6（sheafification exact）。** sheafification

$$
a:\operatorname{PSh}(\mathcal C,\mathbf{Ab})\to\operatorname{Sh}(\mathcal C,\mathbf{Ab})
$$

保持有限极限和有限余极限。

**证明.** 左伴随性给有限余极限。有限极限由 plus 构造和阿贝尔群值匹配族逐覆盖计算给出；附录 H 已给出数学证明。形式化时需把“局部为零”和“覆盖细化”写成两个 lemma。证毕。

## F.5 ED 测试对象与投射性

**证明义务 F.7（自由对象泛性质）。** 对极不连通 \(E\)，证明

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline E],A)\cong A(E).
$$

**证明.** 先由可表 sheaf 得 \(\operatorname{Hom}(\underline E,U(A))=U(A)(E)\)，再由自由阿贝尔群左伴随得到阿贝尔群对象版本。证毕。

**证明义务 F.8（投射性）。** 若 \(A\to B\) 是 sheaf 满射，则

$$
A(E)\to B(E)
$$

满射。

**证明.** sheaf 满射给覆盖 \(E'\to E\) 上的局部提升。Gleason 投射性给截面 \(E\to E'\)。拉回局部提升得到全局提升。证毕。

## F.6 Ext/Tor 形式化接口

形式化 Ext/Tor 需要：

1. 足够投射对象；
2. 投射分解存在；
3. 链同伦比较定理；
4. Hom 复形和 tensor 复形；
5. quasi-isomorphism 不依赖选择。

**证明义务 F.9（比较定理）。** 任意两个投射分解 \(P_\bullet\to A\)、\(Q_\bullet\to A\) 之间存在链映射提升 identity，且任意两个提升链同伦。

**证明.** 按次数归纳。第 \(n\) 步使用 \(P_n\) 投射性提升到 \(Q\) 中对应 cycle 的前像。链同伦同理，用投射性逐阶解边界方程。证毕。

## 练习

1. 将 F.2 的等化子条件改写成两个命题：separated 与 gluing。
2. 证明紧到 Hausdorff 的连续满射是 quotient map。
3. 写出 sheafification exact 证明中“局部为零”的形式化 lemma。
4. 说明 Ext 定义为什么依赖投射分解比较定理。
