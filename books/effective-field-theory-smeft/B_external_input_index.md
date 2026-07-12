# 附录 B：外部输入与计算索引

本附录是正文外部输入的依赖台账。`输入键` 对应正文粗体标签，`source key` 对应 `SOURCES.md`；“不覆盖”列是定理可用范围的一部分，不得在后文省略。

## B.1 EFT 基础外部输入

| 输入键 | 本书采用的精确版本与假设 | 正文用途 | source key 与 locator | 不覆盖 |
| --- | --- | --- | --- | --- |
| EFT-DEC | 可重整化 UV 理论；固定低能重整化条件；外动量低于重质量；耦合不随重质量增长 | 外部输入定理 1.6 | EFT-AC，Phys. Rev. D 11 (1975), pp. 2856-2861 | 自发破缺造成的 non-decoupling、强耦合、未知非微扰质量隙 |
| EFT-EQ | 局域、扰动可逆、保持渐近 pole、无 anomalous Jacobian；比较适当归一化的 on-shell S-matrix | 外部输入定理 2.7A、命题 4.4 的量子层面 | EFT-KOS，pp. 529-549；EFT-ARZT；EFT-EQ-HO，Sec. 2 | 非局域/不可逆变换、off-shell Green 函数不变性、异常 Jacobian |
| EFT-EQ-HO | 作用量与场重定义按同一小参数逐阶展开，二阶及以上保留诱导项 | 警告 4.5B；命题 4.5A 的量子解释 | EFT-EQ-HO，Secs. 3、5，Apps. B、D | 只用“修正后 EOM”替代显式高阶 field redefinition；任意非微扰重参数化 |
| EFT-REGIONS | 指定微扰 Feynman integral、质量/动量 hierarchy 和 dimensional/analytic regulator；逐区域齐次展开 | 外部输入方法 2.7C 和命题 2.7B 的 hard/soft 分解来源 | EFT-REGIONS，Sec. 2 的步骤、Sec. 3 的 threshold 形式化 | 任意非微扰 QFT 的收敛定理；未枚举区域的阈值问题 |
| EFT-REN | 无 anomaly 的精确对称性；局域微扰反项；Ward/Slavnov--Taylor identities 在选定方案中可恢复 | 命题 3.7、外部输入 3.7A、外部输入边界 4.13 | EFT-BRST，pp. 21-79；SMEFT 具体后果另见 SMEFT-JMT-I Sec. 3 | anomalous symmetry、Gribov/非微扰 BRST 问题、未指定 evanescent finite scheme |
| EFT-TOPOLOGY | 总导数删除要求 current 全局定义且边界积分为零；非平凡 bundle 可有非零拓扑数 | 反例 4.2A | EFT-BRST，pp. 93-115；SMEFT-WARSAW，Sec. 2 对规范拓扑项的边界说明 | 本书不分类四维规范丛或重建 instanton measure |

Weinberg 的 phenomenological-Lagrangian 思想和 EFT-BURGESS 仍作为定义与教材背景，但不承担上述任何外部定理的证明责任。

## B.2 SMEFT 外部输入

| 输入键 | 本书采用的精确版本与假设 | 正文用途 | source key 与 locator | 不覆盖 |
| --- | --- | --- | --- | --- |
| SMEFT-D5 | 无额外轻场的 SM field content；局域 Lorentz/$G_{\rm SM}$ invariant；按 IBP/EOM quotient；保留 flavor 对称性；非自伴 Weinberg pair 只计一个结构代表，但 Hermitian 拉氏量显式含 dagger | 外部输入定理 6.2 | SMEFT-WARSAW，Sec. 3、Eq. (3.1)；历史源 SMEFT-W79 | 含轻 $\nu_R$ 的 EFT、非线性 HEFT、维数七以上分类 |
| SMEFT-WARSAW6 | baryon number 守恒；维数六；未破缺相；59 个结构不展开 flavor，且每个非自伴 dagger pair 只计一个代表；h.c. 不另计但在拉氏量中恢复 | 第 6、7、13、20 章及算符台账 | SMEFT-WARSAW，摘要、Sec. 3、Tables 2-3；独立性论证见 Secs. 5-7 | baryon-number violating 四个额外类型、完整三代实参数秩证明、dimension eight |
| SMEFT-RGE6 | baryon-number conserving Warsaw basis；一次维数六插入；one loop；dimensional regularization 与 $\overline{\rm MS}$ | 外部输入 3.9、第 15 章 | SMEFT-JMT-I（formalism/$\lambda$）、SMEFT-JMT-II（Yukawa）、SMEFT-JMT-III（gauge）；三篇合并 | 双插入、维数八 anomalous dimensions、不同 evanescent finite projections |
| SMEFT-EOM-RG | 在上述线性维数六方案中扩大到 EOM operators；EOM 子空间在 operator RGE 下不变 | 外部输入 4.9B、命题 4.9A 的 SMEFT 实例 | SMEFT-JMT-I，Sec. 3、Eqs. (3.8)-(3.11) | $p=4$ 非线性 mixing、dimension eight、任意 gauge/renormalization scheme |
| SMEFT-BROKEN | Warsaw conventions、$R_\xi$ gauges、给定输入与场重归一化 | 破缺相 Feynman-rule 外部边界 | SMEFT-BROKEN | 本书不把规则表当作 basis-independent 定义 |
| SMEFT-D8 | SM field content 下的 dimension-eight 分类 | 第 12 章研究边界与 $p=4$ 遗漏项 | SMEFT-D8 | 本书不逐项重建全基、双插入 RGE 或全局拟合 |
| SMEFT-MODERN | 各来源明确声明的 EFT/basis/flavor/scale/truncation 口径 | 第 19、22 章应用地图 | SMEFT-WORKFLOW、SMEFT-ATLAS、SMEFT-SNOWMASS | 任何随版本变化的数值拟合都须另给数据与访问日期 |

## B.3 工具边界

工具链可用于核算和拟合，但不得替代定义。引用任何工具结果时必须记录：

1.  EFT 类型；
2.  算符基；
3.  flavor 假设；
4.  保留的 $(p,L)$、evanescent/EOM projection 和 input scheme；
5.  $M_{\rm gap}$ 假设、$\Lambda_{\rm ref}$、$\mu_{\rm match}$ 与 $\mu_{\rm obs}$；
6.  工具版本、模型文件 hash 与阈值顺序；
7.  输入数据集、协方差与统计口径。

## B.4 现代方向的纳入标准

一个新方向若要进入本书正文，至少需要满足：

1.  有明确 EFT 类型、自由度、对称性和算符商/基；
2.  区分 $M_{\rm gap}$、$\Lambda_{\rm ref}$、$\mu_{\rm match}$ 和观测尺度；
3.  说明 flavor、CP、baryon number 与 lepton number 假设；
4.  能指出主导算符族、保留的 $(p,L)$ 和多次插入规则；
5.  能说明截断误差、$Q/M_{\rm gap}$ 有效性域和外部数据版本；
6.  每个外部输入都有正文用途、source key、精确 locator 和不覆盖范围。

不满足这些条件的内容只进入资料源或研究边界，不进入主线定理链。
