# 第十章 训练动力学与机制形成

最终 checkpoint 只展示训练终点。保存训练轨迹并在每个 checkpoint 重复行为、读出与干预实验，可以研究 feature 和 circuit 何时形成、何时分化，以及后训练怎样改变已有机制。最大的困难不是画时间曲线，而是确认不同时间和不同 seed 上比较的是同一功能对象。

## 10.1 纵向研究单位

设训练 run $r$ 在 step $t$ 的参数为 $\theta_{r,t}$。对固定评估分布 $\mathcal D_{\mathrm{eval}}$，记录：

$$
B_{r,t}=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[S(M_{\theta_{r,t}}(x))],
$$

$$
P_{r,t}=\text{probe/readout metric},
\qquad
I_{r,t}=\text{intervention effect},
$$

$$
C_{r,t}=\text{circuit fidelity or structure}.
$$

行为、可读性与因果作用可以在不同时间出现。只追踪 probe accuracy 会把“信息已可读”误写成“机制已被使用”。

step、seen tokens、optimizer updates 和有效计算量不是同一时间轴。跨规模比较应至少用 seen tokens 与训练损失对齐，并说明 batch 和数据 curriculum。

## 10.2 Checkpoint 密度与变化点

若保存点间隔过大，平滑变化会看似突然。对候选变化点 $t^*$，应在附近增加 checkpoint 密度，并同时检查连续指标：loss、margin、activation selectivity、ablation effect 与 circuit fidelity。

可把变化量写为

$$
\Delta_t X=X_{t+1}-X_t
$$

并用多 seed 的 change-point model 估计转变区间。事后选取最陡区间会高估突变，应在独立 runs 上复验。

## 10.3 坐标对齐问题

同一连续 run 的参数坐标固定，但 neuron 功能仍可漂移；独立 seeds 还存在置换、旋转与不同冗余分解。直接比较 `neuron 42` 通常只对同一 run 有坐标意义。

常用对齐对象：

- weight matching；
- activation correlation；
- CKA/CCA 或 Procrustes 子空间；
- SAE decoder 与 firing-set matching；
- head 的 attention/OV 功能签名；
- circuit role 与 intervention signature。

对齐算法可能强行制造连续性。应在未用于匹配的 anchor inputs 和 intervention effects 上验证。

## 10.4 Feature trajectory

给 checkpoint $t$ 的 feature set $F_t$，在相邻时间构造相似度

$$
K_{ij}^{t,t+1}
=w_d\cos(d_i^t,d_j^{t+1})
+w_a\operatorname{corr}(a_i^t,a_j^{t+1})
+w_I\operatorname{sim}(I_i^t,I_j^{t+1}).
$$

activation correlation 只在两个 activation 向量的经验方差都为正时定义；dead/constant feature 应标作不可匹配或由单独规则处理，不能把未定义相关系数置为零后静默参加 assignment。三项在组合前须规范到可比较尺度。通过 assignment 得到候选轨迹后，权重 $w$ 与阈值仍会影响“出生、死亡、分裂、合并”的判断，必须做敏感性分析。

feature identity 更适合定义为多维功能签名的连续性，而不是 decoder cosine 单项。split 后的多个后代可能共同保留旧 feature 功能，此时一对一匹配不再合适。

## 10.5 Grokking 与受控算法任务

grokking 指某些任务中模型先拟合训练集，较晚才显著改善验证性能。机制分析可追踪记忆表示被更一般算法 circuit 替代或压过的过程。

要声称 grokking，应报告：

- 训练与验证曲线；
- 数据规模、正则和优化设置；
- 多 seed 的发生率与时间；
- 连续 margin 而不只是 exact match；
- 回路级变化与行为变化的先后关系。

受控模运算中的机制结果提供存在性案例，不可直接外推为大模型能力形成的一般规律。

## 10.6 “涌现”与测量尺度

离散指标会把连续 logit 改善变成阈值跳跃。例如答案 margin 从负变正时 exact match 突然翻转。应联合观察

$$
m_t=z_{correct,t}-\max_{j\ne correct}z_{j,t}
$$

与准确率。

真正的机制重组可能表现为：旧 circuit fidelity 降低、新 circuit fidelity 上升、组合干预 interaction 改变。单一能力曲线不足以区分坐标连续增强和算法切换。

## 10.7 Circuit 形成与角色稳定

对 circuit 中功能角色 $R_1,\ldots,R_k$，每个角色由行为签名、输入条件、输出方向和干预 effect 定义。跨 checkpoint/seed 比较时，先在每个模型独立定位满足角色标准的组件，再比较连接图。

可定义经角色对齐后的 edge Jaccard：

$$
J(C_1,C_2)
=\frac{|E_1\cap E_2|}{|E_1\cup E_2|},
$$

该式只在 $E_1\cup E_2\ne\varnothing$ 时定义；若两个候选 circuit 都无边，应报告“空图退化”，而不是用任意约定制造高相似度。Jaccard 还忽略 edge weight 与等价冗余。还应比较两个 circuit 对共同 intervention battery 的响应向量：

$$
\rho_I(C_1,C_2)
=\operatorname{corr}
(\Delta S_{C_1}^{1:m},\Delta S_{C_2}^{1:m}).
$$

若任一响应向量的经验方差为零，$\rho_I$ 未定义；应同时报告两个响应向量及其退化状态。结构相似、功能不同和结构不同、功能等价都可能发生。

## 10.8 跨 Seed、训练阶段与规模的四层稳定性

机制稳定性至少分为：

1. **within-checkpoint**：同一模型跨输入；
2. **across-time**：同一 run 跨 checkpoints；
3. **across-seed**：同配方独立训练；
4. **across-scale/architecture**：不同宽度、深度或架构。

每升一级都需要新的对齐假设。[Tigges 等（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47c7edadfee365b394b2a3bd416048da-Abstract-Conference.html)在特定 decoder-only 模型、任务与规模范围内发现功能组件和高层算法具有一定训练/规模一致性；这支持继续研究小模型迁移，但不证明任意机制普遍可迁移。

最少报告每个层级的 run 数、失败 run、对齐规则和稳定性分布。一个 seed 的精细故事不能承担 population-level 结论。

## 10.9 Base、SFT 与偏好训练的机制差异

继续预训练、SFT 和偏好优化可能：

- 复用已有知识 features，只改 late-layer policy；
- 改变 feature threshold 或 token-position gating；
- 新增拒答和格式 circuits；
- 改写同一方向的 downstream reader；
- 造成旧能力 interference。

比较时要使用模型各自正确的 chat template，并同时在 base-style 与 assistant-style prompts 上测量。否则 prompt 适配差异会被误归因权重变化。

同初始化的连续 fine-tuning 允许直接研究 $\Delta\theta$；独立基础模型之间的参数差没有自然坐标对应。

## 10.10 Model diffing

对连续训练的两个模型 $A,B$，可从三层比较：

1. 参数：$\Delta\theta=\theta_B-\theta_A$；
2. activation：$\Delta h(x)=h_B(x)-h_A(x)$；
3. causal response：把 $B$ 的 state patch 入 $A$ 或反向 patch。

双向 cross-model patch 需要表示对齐。若同层 activation 尺度或 basis 改变，直接替换会 off-manifold。可先用受限线性 map 对齐，再用 held-out reconstruction 与 intervention 验证；map 太强又会引入可识别性问题。

## 10.11 训练数据归因的层级

“哪条数据造成 feature $u$”比“哪条数据与 $u$ 相似”强得多。可分为：

- retrieval similarity：找到表面或 embedding 近邻；
- gradient similarity：训练样本梯度与目标梯度对齐；
- influence approximation：近似删除样本对参数/目标的影响；
- data ablation/reweighting：改变训练集并重复训练；
- exact replay：在可控小模型中追踪更新。

前两项是相关证据。大模型非凸优化、optimizer state、数据顺序和样本交互使经典 influence 假设脆弱。强因果主张需要受控增删数据和多个训练 repeats。

## 10.12 发展性因果主张

时间先后可以排除“结果晚于原因”的某些故事，但不能单独建立原因。若 feature readout 在行为前出现，可能是前置表示，也可能是无用副产物；若 ablation effect 与行为同步增强，证据更强。

一个纵向机制假说应同时预测：

- feature selectivity 的出现时间；
- downstream reader 的形成时间；
- intervention effect 的剂量变化；
- 行为 margin 的变化；
- 数据或正则干预改变训练时序的方向。

最后一项通过训练干预把时间相关推进到学习机制证据。

## 10.13 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| checkpoint tracking | 指标何时出现 | 重复 probe/patch；时间曲线 | checkpoint 密度、连续指标、多 seed | 同 run 的发展轨迹 | 训练原因 |
| feature matching | 两时点单位是否对应 | assignment；decoder/firing/effect | held-out anchors、split/merge | 匹配规则下的连续性 | 唯一身份 |
| circuit stability | 高层算法是否复现 | 角色对齐、edge/response similarity | 独立发现、失败 runs | 指定范围的机制稳定 | 跨架构普遍性 |
| model diffing | 训练阶段改变了什么 | 参数/activation/cross-patch | template、basis、双向 patch | 连续模型差异路径 | 独立模型直接参数语义 |
| data attribution | 数据怎样影响机制 | gradient/influence/retraining | optimizer path、重复训练 | 依方法强度的影响证据 | 检索相似即因果来源 |

训练动力学把静态解释扩展为机制形成史。可靠结论通常在功能角色层稳定、在具体坐标层不稳定；教材应保留这两层，而不是为获得整齐故事强行把不同 seed 的 neuron 一一对应。
