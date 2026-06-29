# 第二卷资料源

第二卷继续使用第一卷资料，并把重点放在 solid、analytic、liquid 和几何应用上。

## 核心来源

1. Peter Scholze, *Lectures on Condensed Mathematics*, arXiv:2605.03658.  
   链接：<https://arxiv.org/abs/2605.03658>  
   用途：solid abelian groups、solid tensor product、analytic rings、condensed/solid 向量空间和相干对偶入口。

2. Dustin Clausen and Peter Scholze, *Condensed Mathematics and Complex Geometry*, arXiv:2605.11731.  
   链接：<https://arxiv.org/abs/2605.11731>  
   用途：复几何和解析几何方向。

3. Dagur Asgeirsson, *Towards solid abelian groups: A formal proof of Nöbeling's theorem*, arXiv:2309.07252.  
   链接：<https://arxiv.org/abs/2309.07252>  
   用途：Nöbeling 定理、solid 计算背景和形式化证明线索。

4. Dagur Asgeirsson, Riccardo Brasca, Nikolas Kuhn, Filippo Alberto Edoardo Nuccio Mortarino Majno di Capriglio, Adam Topaz, *Categorical Foundations of Formalized Condensed Mathematics*, arXiv:2407.12840.  
   链接：<https://arxiv.org/abs/2407.12840>  
   用途：站点基础和形式化背景；第二卷只在回溯基础时使用。

## 引用纪律

- 对第一卷已有结论，引用第一卷章节。
- 对 Scholze 讲义中的长定理，标明“输入定理”。
- 对第二卷自行证明的命题，必须写出所用第一卷定理。
- 任何涉及 liquid 或复几何的陈述，若只是路线图，必须明确写为路线图。

## 章节依赖映射

- 第 1-2 章主要依赖 Scholze 讲义中的 solid abelian groups 与 solid tensor product。
- 第 3-4 章主要依赖 Scholze 讲义中的 analytic rings 和 Bousfield localization 观点。
- 第 5 章主要依赖 Scholze 的 liquid vector spaces 构造。
- 第 6-7 章主要依赖 Scholze 讲义中的离散 Huber pair、解析环全局化和相干对偶。
- 第 8 章主要依赖 Clausen-Scholze 复几何讲义，当前只给范畴语言和目标定理路线图。
- 附录 A-B 主要用于把输入定理的证明路线和公式类型检查显式化，不引入新的外部来源。
- 附录 C 使用 presentable stable category 与 Bousfield localization 的标准形式定理；作为范畴论输入使用。
- 附录 D 不引入新定理，只把第二卷已经使用的输入定理拆成更精确的引用颗粒。
- 附录 E 不引入新的 condensed 数学输入；它证明局部化、张量理想、幺半下降和相对张量积的范畴论引理。所需外部前提仍是附录 C-D 中标出的 presentable localization 与 Scholze 核心输入。
- 附录 F 使用 presentable adjoint functor theorem、Brown representability、闭对称幺半范畴和投影公式的标准形式理论；它不构造 Scholze 的 $f_!$，只证明接受 $f_!$ 后的形式推论。
- 附录 G 使用 Cech nerve、totalization、ordinary sheaf descent 和稳定范畴值 descent 的形式理论；rational Cech descent 本身仍是附录 D.7 登记的 Scholze 输入定理。
- 附录 H 使用紧生成三角范畴、localizing subcategory、紧对象和生成元检验的标准形式理论；具体 solid/analytic 生成元来自 Scholze 输入定理，不在本附录构造。
- 附录 I 使用 analytic ring 的 cone 判别、Bousfield localization、张量理想和 rational descent 的形式边界；它是检查表和失败模式，不证明具体 Scholze analytic ring。
- 附录 J 使用拓扑向量空间凝聚化、Banach 空间非闭像、Fréchet 空间和 Fredholm 条件的标准分析背景；liquid realization 仍作为 Scholze/Clausen-Scholze 输入。
- 附录 K 使用 presentable stable category、Bousfield localization、张量理想、对称幺半局部化和 bar construction 的标准形式理论；solid/analytic 核的张量理想性仍作为 Scholze 输入。
- 附录 L 使用闭对称幺半范畴、内部 Hom、dualizable 对象和伴随函子的标准形式理论；几何中的 $f^!$ 比较仍需 properness/perfectness 等输入。
- 附录 M 使用 solid Dirac-to-measure cone、Bousfield localization 和张量理想输入，补 solidification 的生成核和完备化口径。
- 附录 N 使用 analytic ring cone、rational localization、Čech nerve 和 totalization，补 analytic descent 与 rational cover gluing 的证明义务。
- 附录 O 使用可展示稳定范畴、反射局部化、正合局部化、kernel 和 local objects 的一般理论，补 Bousfield localization 的书内形式推论。
- 附录 P 使用 Fréchet 空间、闭值域、Fredholm 复形、Hodge 分解和 liquid realization，补复几何分析对象进入 analytic/liquid 范畴前的类型检查。
- 附录 Q 使用第二卷附录 C、K、M 的 Bousfield localization、张量理想和 solidification 语言，把 solid theory 收束为主定理包；solid 反射存在性、张量理想性和 profinite 测度张量公式仍是 Scholze 输入。
- 附录 R 使用第二卷附录 I、N 的 analytic cone、analyticization、Huber pair 和 rational descent 语言，把 analytic theory 收束为主定理包；analytic ring localization 与 rational Čech descent 仍是 Scholze 输入。
- 附录 S 使用第二卷附录 J、P 的 liquid realization、Fréchet 闭值域和 Fredholm 复形语言，把 liquid theory 收束为主定理包；\(p\)-liquid 测度理论和 realization 仍是 Scholze/Clausen-Scholze 输入。
- 附录 T 不引入新输入；它把附录 Q、R、S 合成为 solid/analytic/liquid 主线闭包定理，并规定第三卷应用这些结构时的类型检查规则。
- 附录 U 不引入新输入；它按出版级标准审查第二卷主线的定义、输入、证明、边界、接口和练习答案状态。
- 附录 V 使用 presentable localization theorem 和 \(D(\mathbf{CondAb})\) 的可展示稳定性，把 solidification 反射存在性拆成集合生成局部化、Dirac-to-measure cone 正交和 Scholze 识别定理三部分；前两类形式后果书内证明，识别定理仍为 Scholze 输入。
- 附录 W 使用 localizing subcategory 与张量理想的生成元判别，把 solid kernel 张量理想性归约为 profinite 测度张量计算；生成元判别书内证明，profinite 测度张量公式仍为 Scholze 输入。
- 附录 X 使用附录 V、K 的局部化和幺半下降形式，把 analytic localization 分解为 analytic cone、反射局部化、analyticization 泛性质和 analytic tensor；analytic ring 公理推出张量相容仍为 Scholze 输入。
- 附录 Y 使用 Čech nerve、mapping-space descent、compact generation descent 和 rational acyclicity 语言，把 rational Čech descent 分解为形式范畴论部分和 Huber rational acyclicity 输入。
- 附录 Z 使用 compact Hausdorff quotient descent、Fréchet 闭值域和 Fredholm 复形语言，把 liquid realization 拆成拓扑向量空间凝聚化、realization exactness 输入和 Dolbeault cohomology 形式比较。
- 附录 AA 使用本卷和第三卷的输入登记表、主定理包和证明模块，把 Scholze 与 Clausen-Scholze 核心定理整理成主线图谱；它不新增输入，只规定这些定理在本书中的核心地位和依赖边界。
