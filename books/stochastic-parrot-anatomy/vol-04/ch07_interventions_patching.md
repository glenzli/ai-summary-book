# 第七章 Ablation、Patching 与因果追踪

模型前向传播是已知计算图，因此研究者可以把内部节点设置为指定值并重新计算下游输出。这比相关读出更接近因果实验；但结论仍相对于节点定义、替换值、输入分布与允许干预族，不能自动解释为人类概念的唯一自然中介。

## 7.1 前向图作为确定性结构模型

把内部节点按拓扑顺序写成

$$
U_k=f_k(\operatorname{pa}(U_k);\theta),
$$

最终目标为 $S=g(U_1,\ldots,U_K)$。对节点 $U_j$ 施加

$$
\operatorname{do}(U_j\leftarrow u)
$$

表示忽略其原结构方程，固定为 $u$，其余下游节点按原模型重算。

模型内部不存在传统观察研究中的未知边；但“把什么视为节点”和“赋什么值”仍由研究者决定。把 residual vector、单 head result、一个投影或 SAE latent 作为 $U_j$，得到的是不同干预。

## 7.2 单样本与分布效应

对输入 $x$，节点干预的个体效应为

$$
\Delta_I(x)=S(M_{I}(x))-S(M(x)).
$$

分布平均效应为

$$
\tau_I(\mathcal D_{\mathrm{eval}})
=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}[\Delta_I(x)].
$$

应报告均值、分位数、符号一致率和 cluster bootstrap 区间。平均接近零可能来自正负效应抵消；只展示最大恢复样本则会严重选择偏差。

模型是确定性的并不消除抽样不确定性：$\mathcal D_{\mathrm{eval}}$ 上的有限样本仍决定我们能否推广。

## 7.3 Ablation 及其基线语义

常见 ablation 为：

- zero：$u\leftarrow0$；
- mean：$u\leftarrow\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}[U(x)]$；
- resample：$u\leftarrow U(x')$，$x'$ 来自匹配对照；
- shuffle：在 batch 或 group 内重排；
- projection：删除子空间 $P_Vu$；
- edge ablation：只阻断 sender 到 receiver 的贡献。

这些操作估计不同量。zero 问“没有该向量加数时怎样”，mean 问“替换为总体中心怎样”，resample 问“替换为对照实例状态怎样”。它们不应被平均成一个无条件必要性分数。

mean 可能从不在自然数据出现；resample 较接近真实 activation，却可能携带对照样本的多种属性。选择基线必须对应要消除的因素。

## 7.4 Clean/corrupt activation patching

构造 clean 输入 $x_c$ 与 corrupt 输入 $x_r$，目标 score 满足 $S_c>S_r$。缓存 site $u$ 的 clean 值 $u_c$，在 corrupt run 中替换：

$$
u_r\leftarrow u_c.
$$

原始恢复量为

$$
\Delta^{patch}_u
=S(M(x_r;\operatorname{do}(u\leftarrow u_c)))-S_r.
$$

常见归一化恢复率

$$
R_u=\frac{S_{patched}-S_r}{S_c-S_r}
$$

仅在分母远离零时稳定，且可小于 $0$ 或大于 $1$。必须同时报告 $S_c,S_r,S_{patched}$ 与未归一化 effect。

## 7.5 Corruption 是实验的一半

patch effect 依赖 clean/corrupt 差异。若 corrupt 同时改变主体、句长与答案位置，patch 恢复可能修复任何一个因素。

必要设计包括：

- 多种 corruption 机制；
- token 数、位置与频率匹配；
- semantic 与 surface corrupt 分离；
- clean/corrupt 双向 patch；
- 在原模型均成功但答案不同的 pairs 上测试；
- 检查 corruption 是否造成广泛 loss 崩坏。

“某 site 携带答案”应进一步区分它生成答案信息、传递上游信息，还是仅把整体 state 拉回 clean 区域。

## 7.6 Necessity 与 Sufficiency 的操作定义

对候选组件集 $C$，设 $M_{\setminus C}$ 为按规定删除 $C$ 的模型，$M_C$ 为只保留 $C$ 的受限模型。可定义

$$
N_C=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[S(M(x))-S(M_{\setminus C}(x))],
$$

$$
Q_C=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[S(M_C(x))-S(M_{base}(x))].
$$

$N_C$ 是指定删除下的必要性证据，$Q_C$ 是指定基线与保留规则下的充分性证据。神经网络有冗余：删除单组件无效不证明未参与；只保留一个 direction 后 steering 有效也不证明自然前向依赖它。

必要与充分都不是组件的内在二元属性，而是 $(C,\mathcal D_{\mathrm{eval}},S,I)$ 的关系。

## 7.7 Total、Direct 与路径效应

普通节点 patch 允许替换值沿全部下游路径传播，测量 total intervention effect。若只允许 sender $A$ 的变化经 receiver $B$ 到达输出，则研究 path-specific effect。

线性无交互图中可以把效应相加；一般网络中

$$
\Delta(A\cup B)
\ne\Delta(A)+\Delta(B).
$$

LayerNorm、softmax 和门控使路径相互依赖。所谓 direct/indirect effect 必须由具体边阻断与缓存协议定义，不能直接套用线性中介公式。

## 7.8 Path patching 与状态一致性

path patching 通常经历：

1. clean 与 corrupt run 缓存 sender；
2. 建立含 clean sender 的中间 run；
3. 只把该变化对指定 receiver 的输入传入最终 run；
4. 其他 sender-to-receiver edges 保持 corrupt；
5. 比较目标 score。

不同实现对 receiver 的哪些输入固定、是否重新计算 attention pattern、是否包含 residual identity 并不相同。论文应给出计算图示意或伪代码。

混合多个 run 的节点可能形成没有任何自然输入产生的状态。path-specific 结论因此是“该人工组合中的传递效应”，而非自动可识别的自然路径效应。

## 7.9 Causal tracing

causal tracing 在位置和层上扫描 clean restoration，形成恢复热图。它适合定位事实回忆或实体信息的候选路径。

热图峰值支持：在给定 corruption 下恢复该 site 能有效恢复目标。它不证明：

- 事实永久或唯一存储在该层；
- 该层独立完成事实检索；
- 不同 prompt、关系和语言使用同一路径；
- 参数编辑该层不会有非局部副作用。

位置—层扫描还涉及多重发现，应在独立 pairs 上复验峰值，而不是在同一热图上发现和验证。

## 7.10 Activation steering 与剂量反应

沿方向 $v$ 修改

$$
h'=h+\alpha v
$$

并估计剂量曲线

$$
g(\alpha)=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[S(M_{\alpha v}(x))].
$$

一个可信控制方向应在小幅区间表现出可重复、方向一致的 effect，并报告：

- $v$ 与 $h$ 的 norm 比；
- 正负 $\alpha$；
- 多层与多位置；
- matched random directions；
- perplexity、格式与非目标能力；
- 大 $\alpha$ 下的分布外崩坏。

steerability 证明方向具有控制作用；只有自然 activation、下游读取和删除证据共同成立时，才支持正常机制参与。

## 7.11 Off-manifold 的定义与诊断

自然 activation 集记为

$$
\mathcal M_{\mathrm{eval}}
=\{u(x):x\sim\mathcal D_{\mathrm{eval}}\}.
$$

任意干预值 $u'$ 可能远离 $\mathcal M_{\mathrm{eval}}$。高维数据没有可靠统一距离，常用诊断包括：

- 与邻近自然 activation 的 Mahalanobis/knn 距离；
- norm、均值和协方差偏移；
- 下游 next-token loss 或 hidden-state density proxy；
- 非目标 logits 的广泛变化；
- 用不同分布内 resampling 得到的结果一致性。

SAE 重构或生成式 projection 也只是另一个模型定义的近似 manifold，不会自动解决问题。

缓解策略是使用真实对照 resample、小幅剂量、norm/covariance matching、多个干预实现与副作用检查。最终结论仍应注明干预族。

## 7.12 Causal abstraction 与 interchange intervention

设低层模型输入空间为 $\mathcal X_M$，内部状态 $U(x)\in\mathcal U$，输出位于 $\mathcal O_M$；高层假说 $H$ 的输入空间为 $\mathcal X_H$，内部变量 $Z\in\mathcal Z$，输出位于 $\mathcal O_H$。给出输入映射 $\alpha_X:\mathcal X_M\to\mathcal X_H$、内部对齐 $\alpha_U:\mathcal U\to\mathcal Z$ 与输出比较映射 $\alpha_O:\mathcal O_M\to\mathcal O_H$。若两模型共用输入空间，可取 $\alpha_X$ 为恒等映射。interchange intervention 从 source input $x_s$ 取内部变量值，替换 base input $x_b$ 的对应表示，并检查低层模型输出经 $\alpha_O$ 后是否匹配高层模型的同类反事实。

给 $\mathcal O_H$ 指定度量或任务损失 $d_O$，并固定容差 $\epsilon\ge0$。单个 pair 的一致性判据为

$$
d_O\!\left(
\alpha_O\bigl(M_{U\leftarrow U(x_s)}(x_b)\bigr),
H_{Z\leftarrow\alpha_U(U(x_s))}
\bigl(\alpha_X(x_b)\bigr)
\right)
\le \epsilon.
$$

再令 $(X_b,X_s)\sim\Pi_{\mathrm{pair}}$ 为预先声明的 base/source 配对分布，定义

$$
\operatorname{IIA}_\epsilon
=P_{(X_b,X_s)\sim\Pi_{\mathrm{pair}}}
\left[
d_O\!\left(
\alpha_O(M_{U\leftarrow U(X_s)}(X_b)),
H_{Z\leftarrow\alpha_U(U(X_s))}(\alpha_X(X_b))
\right)\le\epsilon
\right].
$$

高 $\operatorname{IIA}_\epsilon$ 只支持这些输入、内部与输出映射在 $\Pi_{\mathrm{pair}}$ 和给定容差下的近似因果抽象。若允许任意高容量映射，对齐可能变得空泛；必须限制映射类别、在 held-out interchange pairs 上估计置信区间，并比较随机或错误高层图。离散输出常取 $d_O(a,b)=\mathbf 1\{a\ne b\}$、$\epsilon=0$，此时该定义退化为通常的 interchange-intervention accuracy。

## 7.13 干预下的可识别性

若两个机制假说对所有已测试干预给同样输出，它们仍不可识别。增加更多节点扫描未必解决问题；关键是设计使假说预测相反的干预。

常见不可识别情形：

- 冗余路径使单点 ablation 都无效；
- patch 同时携带多个属性；
- 只看最终 token，忽略中间响应差异；
- 对齐映射容量足以拟合所有 interchange pairs；
- 干预集合没有覆盖组合或反方向操作。

研究报告应列出仍与数据相容的主要替代假说，而不是把“找到一个有效干预”写成唯一解释。

## 7.14 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| ablation | 组件在删除下是否必要 | zero/mean/resample；$N_C$ | 多基线、副作用、组合删除 | 指定删除效应 | 无效即未参与；基线依赖 |
| activation patching | clean state 能否恢复 corrupt 行为 | run 间替换；$\Delta^{patch}$ | corruption、双向 patch、raw score | 指定 site 的恢复效应 | 信息生成位置或唯一存储 |
| path patching | effect 是否经指定 edge/path | 多 run 边隔离；路径恢复 | 实现图、状态一致性、交互 | 人工路径协议下的 effect | 唯一自然中介 |
| steering | 方向是否具有控制能力 | $h+\alpha v$；剂量曲线 | random direction、负向、副作用 | 指定方向控制效应 | 正常前向依赖 |
| interchange intervention | 表示是否实现高层变量 | source/base 交换；IIA | 映射容量、错误图、held-out pairs | 受限映射类的因果抽象 | 唯一或完整算法 |

干预路线能建立模型内部的操作性因果效应。它的严谨性来自明确的赋值与对照，而不是把 `do` 符号写在任意 activation 上。下一章把节点和边组织为可组合的回路假说。
