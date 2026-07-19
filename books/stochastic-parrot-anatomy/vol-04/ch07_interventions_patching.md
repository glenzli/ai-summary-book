# 第七章 Ablation、Patching 与因果追踪

观察到某 activation 与行为相关后，下一步是修改它并看输出怎样变化。因为模型前向是已知计算图，这类实验可以建立模型内部的干预效应；但替换值、分布外程度和替代路径决定结论强度。

## 7.1 Ablation

对组件 activation $a$，常见 ablation 包括：

- zero ablation：$a\leftarrow0$；
- mean ablation：$a\leftarrow\mathbb E[a]$；
- resample ablation：从匹配对照样本取 $a'$；
- shuffle：在 batch 内打乱 activation；
- edge ablation：只删除从某来源到某目标的贡献。

zero 可能是训练分布外状态；mean 会抹去方差且未必对应真实样本；resample 更自然，却也带入对照样本的其他信息。结论应注明具体干预。

## 7.2 Necessity 与 Sufficiency

若删除组件使行为显著下降，它对该输入分布具有某种必要性；若只保留组件或把它加入基线能恢复行为，它支持某种充分性。

神经网络有冗余和备份路径，因此：

- ablation 无效不证明组件未参与；
- 单组件 steering 有效不证明正常运行依赖它；
- 必要且充分通常只相对于所用干预、模型和数据集成立。

## 7.3 Clean/Corrupt Activation Patching

构造 clean 输入 $x_c$，模型在目标任务上成功；构造 minimal corrupt 输入 $x_r$，目标答案改变或行为失败。记录 clean activation $a_c^{\ell,p}$，在 corrupt run 的某位置替换：

$$
a_r^{\ell,p}\leftarrow a_c^{\ell,p}.
$$

若目标 logit difference 恢复，说明该 site 携带能因果推动 clean 行为的信息。

可定义归一化恢复率

$$
R=\frac{S_{patched}-S_{corrupt}}
{S_{clean}-S_{corrupt}},
$$

但分母接近零时不稳定，$R>1$ 也可能出现。应同时报告原始分数。

## 7.4 Patching 回答的不是“信息在哪里”全部含义

patch effect 大可能表示：

- 该位置生成了关键信息；
- 信息只是在这里传递；
- 替换同时修复多个混杂属性；
- patch 把下游状态推向 clean manifold。

要区分生成与传递，需要在层和位置上扫描、比较 upstream/downstream sites，并进一步做 path patching。

## 7.5 Causal Tracing

事实回忆研究常先扰动输入 embedding 使回答损坏，再把某层某位置的 clean activation 恢复，观察事实 logit 恢复。热图定位信息通过哪些位置和层传播。

这种方法建立特定扰动下的中介路径，不证明“事实永久存储在某一层”。模型参数、prompt 关系和分布式表示共同参与事实生成。

## 7.6 Activation Steering

沿方向 $v$ 修改 residual state：

$$
h'=h+\alpha v.
$$

若行为随 $\alpha$ 系统变化，说明该方向具有干预控制力。steering 常用于情感、拒答、风格或事实方向。

需要检查：

- 小 $\alpha$ 到大 $\alpha$ 的剂量反应；
- activation norm 与 LayerNorm 影响；
- 非目标能力和流畅性损伤；
- 跨模板、语言和主题泛化；
- 反方向 $-v$ 是否产生相反效果；
- matched random direction 基线。

## 7.7 Mediation 视角

把输入处理 $T$、内部变量 $M$、输出 $Y$ 视为计算图节点，patching 试图测量经 $M$ 的路径效应。但神经网络中 mediator 维度高、组件相互作用强，传统线性中介分解的无交互假设通常不成立。

更可靠的表述是“在指定替换操作下，$M$ 的值变化使输出指标改变多少”，而不是声称得到唯一自然间接效应。

## 7.8 Off-manifold 问题

把 activation 任意置零或叠加大向量，可能产生训练时从未出现的组合。下游输出变化可能来自异常状态，而非目标概念。

缓解方法包括：

- 从真实对照 run resample；
- 用生成模型或 SAE 重构到 activation manifold；
- 匹配 activation norm 与协方差；
- 使用小幅干预并画剂量曲线；
- 检查非目标 logits 和困惑度；
- 用不同干预方式复验同一假说。

## 7.9 Compensation 与 Backup

单次前向 ablation 通常不让模型有时间重新训练，但同层其他并行组件可以在原模型中已经提供冗余。逐组件 ablation 效应之和也不等于联合 ablation 效应：

$$
\Delta(A\cup B)
\ne\Delta(A)+\Delta(B).
$$

分析回路时应测试组合删除、恢复和 interaction，而不是把每个组件 credit 强行相加到 100%。

## 7.10 干预实验的最小报告

- clean/corrupt 数据如何配对；
- hook site 与张量形状；
- 替换、缩放或投影操作；
- 目标行为的连续指标；
- 随机和结构基线；
- 原始模型性能与干预后副作用；
- 跨样本效应分布，不只展示最好案例；
- 是否在假说提出后的 held-out 集验证。

## 7.11 结论

干预把“相关 activation”推进到“该变量在指定操作下影响行为”。要进一步说明多个变量怎样依次实现计算，需要将节点干预扩展为路径、边和回路假说。
