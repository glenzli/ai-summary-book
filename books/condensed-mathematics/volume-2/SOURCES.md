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
- 附录 G 使用 Cech nerve、totalization、ordinary sheaf descent 和稳定范畴值 descent 的形式理论；rational Cech descent 本身仍是附录 D.6 登记的 Scholze 输入定理。
- 附录 H 使用紧生成三角范畴、localizing subcategory、紧对象和生成元检验的标准形式理论；具体 solid/analytic 生成元来自 Scholze 输入定理，不在本附录构造。
- 附录 I 使用 analytic ring 的 cone 判别、Bousfield localization、张量理想和 rational descent 的形式边界；它是检查表和失败模式，不证明具体 Scholze analytic ring。
- 附录 J 使用拓扑向量空间凝聚化、Banach 空间非闭像、Fréchet 空间和 Fredholm 条件的标准分析背景；liquid realization 仍作为 Scholze/Clausen-Scholze 输入。
- 附录 K 使用 presentable stable category、Bousfield localization、张量理想、对称幺半局部化和 bar construction 的标准形式理论；solid/analytic 核的张量理想性仍作为 Scholze 输入。
- 附录 L 使用闭对称幺半范畴、内部 Hom、dualizable 对象和伴随函子的标准形式理论；几何中的 $f^!$ 比较仍需 properness/perfectness 等输入。
