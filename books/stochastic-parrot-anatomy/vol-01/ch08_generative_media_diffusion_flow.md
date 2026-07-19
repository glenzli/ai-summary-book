# 第八章 生成媒体：从自回归到 Diffusion 与 Flow

文本模型通常生成离散 token；图像、音频和视频原本是高维连续信号。生成媒体的核心问题是：怎样定义一个可训练的分布，使模型既能产生多样样本，又能接受文本、图像、结构或动作条件。

本章比较几条主要生成路线，重点放在它们怎样表示数据、训练什么目标、推理时执行什么过程。具体采样轨迹将在卷二再次从一次运行的角度展开。

## 8.1 生成模型在学习什么

给定数据分布 $p_{\mathrm{data}}(x)$ 和条件 $c$，生成模型希望构造 $p_\theta(x\mid c)$，或至少构造一个其终点分布逼近数据条件分布的采样过程。这里的“构造分布”有三种不同含义：能够求归一化概率或密度，能够计算一个变分下界，或者只能通过隐式过程得到样本。不同模型族主要区别在于：

- 是否显式计算似然；
- 是否引入潜变量；
- 生成是单次、逐位置还是多步迭代；
- 训练目标怎样连接到最终样本分布；
- 条件控制在何处进入模型。

“能生成逼真图片”是经验结果，不等于模型恢复了唯一真实数据分布。连续数据的密度还依赖量化与测度约定；对同一图像改变位深或加入极小噪声，数值似然就可能显著变化。有限数据、目标函数、采样器和评价方法共同决定可观察结果。

## 8.2 自回归媒体模型

图像、音频或视频可以先离散化为 token 序列，再使用链式分解

$$
p_\theta(x_1,\ldots,x_T\mid c)
=
\prod_{t=1}^T
p_\theta(x_t\mid x_{<t},c).
$$

在 teacher forcing 下，负对数似然分解为各位置交叉熵之和，训练可以并行；推理时第 $t$ 项依赖已经生成的 $x_{<t}$，因此存在串行关键路径。二维图像常先由向量量化器或其他 codec 变成离散索引，再按 raster、分块或多尺度顺序排列。序列化方式会给二维或时空结构施加人为顺序，也决定哪些 token 可以直接使用同一前缀缓存。分层 token、并行预测和 coarse-to-fine 解码可以缓解成本，但不会自动消除顺序建模的假设。

## 8.3 VAE 与潜空间

变分自编码器用近似后验 $q_\phi(z\mid x)$ 把数据映射到潜变量，用解码器 $p_\theta(x\mid z)$ 重建，并给定先验 $p(z)$。从边际似然出发，插入 $q_\phi$ 后有恒等式

$$
\log p_\theta(x)
=\mathcal L_{\mathrm{ELBO}}(x)
+D_{\mathrm{KL}}
\bigl(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\bigr).
$$

由于 KL 散度非负，得到证据下界

$$
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}
\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr).
$$

第一项奖励在后验样本上的条件对数似然，第二项约束近似后验接近先验。等号成立当且仅当近似后验几乎处处等于真实后验。对高斯后验常用重参数化

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I),
$$

把随机性移到与参数无关的 $\varepsilon$ 上，从而对 Monte Carlo 估计反向传播。

把 KL 项乘以 $\beta\ne1$ 会改变率—失真折中；这不再是上述精确分解中的标准 ELBO。KL 过强可使解码器忽略 $z$，形成 posterior collapse；重建权重过强则可能得到难以从先验采样的潜空间。VAE 的重要工程作用不只在于独立生成：高质量自动编码器可以把像素压缩到更小潜空间，让后续 diffusion 或 flow 模型在潜变量上工作。

## 8.4 GAN 的对抗训练

生成器 $G$ 把噪声映射为样本，判别器 $D$ 区分真实与生成数据。经典极小极大目标为

$$
\min_G\max_D
\mathbb E_{x\sim p_{\mathrm{data}}}\log D(x)
+
\mathbb E_{z\sim p(z)}\log(1-D(G(z))).
$$

若 $p_{\mathrm{data}}$ 与生成分布 $p_g$ 对同一测度有密度，并在固定 $G$ 时把 $D$ 优化到函数空间最优，则逐点最优判别器为

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}.
$$

代回原目标得到

$$
-\log 4
+2D_{\mathrm{JS}}(p_{\mathrm{data}}\,\|\,p_g).
$$

这项结论解释了理想极限，却不能直接证明有限网络、交替梯度训练会收敛。数据与生成分布支撑集近乎分离时，判别器容易饱和，生成器梯度可能信息不足；实际训练因而使用非饱和损失、正则化或不同分布距离。GAN 可以一次前向产生样本，但样本逼真度与模式覆盖仍是不同目标。

## 8.5 Diffusion 的前向破坏与反向生成

去噪 diffusion 先定义一个固定前向 Markov 过程，把数据逐渐加噪。给定 $\beta_t\in(0,1)$，令 $\alpha_t=1-\beta_t$、$\bar\alpha_0=1$ 且 $\bar\alpha_t=\prod_{s=1}^t\alpha_s$，则

$$
q(x_t\mid x_{t-1})
=\mathcal N
\left(x_t;\sqrt{\alpha_t}x_{t-1},\beta_tI\right).
$$

高斯闭包给出从 $x_0$ 到任意时刻的边际分布：

$$
q(x_t\mid x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
$$

于是可以直接采样

$$
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
$$

在给定 $x_0$ 时，前向后验仍为高斯：

$$
q(x_{t-1}\mid x_t,x_0)
=\mathcal N(x_{t-1};\widetilde\mu_t,\widetilde\beta_tI),
$$

其中

$$
\widetilde\mu_t
=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
\qquad
\widetilde\beta_t
=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

对 $t\ge2$，这通常是非退化高斯。$t=1$ 时 $\widetilde\beta_1=0$ 且 $\widetilde\mu_1=x_0$，上式应解释为位于 $x_0$ 的 Dirac 测度，而不是具有普通 Lebesgue 密度的高斯。

反向模型以

$$
p_\theta(x_{t-1}\mid x_t,c)
=\mathcal N\bigl(x_{t-1};\mu_\theta(x_t,t,c),\Sigma_\theta(x_t,t,c)\bigr)
$$

近似无法直接计算的反向条件分布。对变分下界中各步 KL 项做参数化和加权，可以得到不同训练目标；常用的简化噪声预测目标是

$$
\mathbb E_{\substack{(x_0,c)\sim p_{\mathrm{data}},\,t\sim\rho\\
\varepsilon\sim\mathcal N(0,I)}}
\left[
\left\|
\varepsilon-\varepsilon_\theta(x_t,t,c)
\right\|_2^2
\right],
\qquad
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\,\varepsilon.
$$

这里 $\rho$ 是声明的时间步采样分布，常见选择是离散均匀分布。简化 MSE 与完整变分目标的权重并不完全相同。时间步采样分布、噪声调度和损失权重共同决定模型在哪些信噪比区域投入梯度预算。

### 8.5.1 噪声、干净样本、score 与 velocity

写 $a_t=\sqrt{\bar\alpha_t}$、$s_t=\sqrt{1-\bar\alpha_t}$，则 $x_t=a_tx_0+s_t\varepsilon$。同一个网络输出可以有多种参数化：

- $\varepsilon$-prediction 直接估计噪声；
- $x_0$-prediction 直接估计干净样本；
- score prediction 估计 $\nabla_{x_t}\log q_t(x_t)$；
- $v$-prediction 以 $v=a_t\varepsilon-s_tx_0$ 为目标。

若条件密度 $q_t(\cdot\mid c)$ 可微且相关随机变量具有有限二阶矩，则在理想条件均方误差下，最优噪声预测器满足

$$
\varepsilon^*(x_t,t,c)
=\mathbb E[\varepsilon\mid x_t,t,c]
=-s_t\nabla_{x_t}\log q_t(x_t\mid c).
$$

无条件模型可删去 $c$，得到相应的 $q_t(x_t)$ 版本；条件训练中不能把 $c$ 从条件期望或 score 密度中省略。

对 $v$ 参数化，由 $a_t^2+s_t^2=1$ 可反解

$$
x_0=a_tx_t-s_tv,
\qquad
\varepsilon=s_tx_t+a_tv.
$$

这些参数化在精确预测时可以互换，但有限网络、有限精度和不同 SNR 权重下的优化性质并不相同。把“预测噪声”当成模型只学会了噪声，会混淆训练坐标与所估计的数据 score。

## 8.6 Latent Diffusion

直接在高分辨率像素上迭代很昂贵。Latent Diffusion 先用编码器得到 $z_0=E(x_0)$，在潜空间执行加噪和去噪，再用解码器还原 $\widehat x=D(z_0)$。

这形成三个可分离误差源：编码器或量化缩放可能丢失细节，去噪模型可能生成错误潜结构，解码器可能引入纹理伪影。潜空间中的欧氏距离也不自动等于感知距离；自动编码器的损失定义了哪些差异会被优先保留。只检查最终图片很难判断问题来自哪一层。

## 8.7 条件与 Guidance

文本条件通常由文本编码器产生，并通过 cross-attention 或联合 Transformer 进入去噪网络。Classifier-free guidance 把有条件和无条件预测组合：

$$
\widehat\varepsilon
=
\varepsilon_\theta(x_t,t,\varnothing)
+w\bigl(
\varepsilon_\theta(x_t,t,c)
-\varepsilon_\theta(x_t,t,\varnothing)
\bigr).
$$

若网络输出与 score 成线性对应，则相同组合可写成

$$
s_{\mathrm{guided}}(x_t,c)
=s_{\mathrm{uncond}}(x_t)
+w\bigl(s_{\mathrm{cond}}(x_t,c)-s_{\mathrm{uncond}}(x_t)\bigr).
$$

在 score 精确且条件密度正则时，差值对应 $\nabla_{x_t}\log p_t(c\mid x_t)$；因此 guidance 可理解为强化条件似然，而不是产生新的观测证据。较大 $w$ 往往增强条件服从，却可能降低多样性、造成过饱和或放大伪影。不同实现对 $w$ 的零点约定并不统一，比较数值前必须读清公式。guidance scale 是推理算法的一部分，不是训练后无代价增加的“理解强度”。

## 8.8 DiT 与 Flow Matching

去噪网络不必是 U-Net。Diffusion Transformer 把潜变量切成 patch token，并用 Transformer 处理时间与条件信息。这使生成媒体更容易共享大规模 Transformer 的训练和并行基础设施。

Flow matching 从基分布 $p_0$ 到数据分布 $p_1$ 选择一族概率路径 $p_t$，并学习随时间变化的速度场

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c),
$$

使每个条件 $c$ 下的密度满足连续性方程

$$
\partial_t p_t(x\mid c)
+\nabla_x\!\cdot\!\bigl(p_t(x\mid c)v_t(x,c)\bigr)=0.
$$

以独立耦合的线性插值为例，令 $t\sim\operatorname{Uniform}[0,1]$，从目标数据联合分布采样 $(X,c)$，再独立采样 $Z\sim p_0$，并假设 $\mathbb E\|X\|_2^2+\mathbb E\|Z\|_2^2<\infty$，从而 $U:=X-Z\in L^2$。取 $X_t=(1-t)Z+tX$，条件路径速度为 $U$。回归目标

$$
\mathcal L_{\mathrm{FM}}
=\mathbb E_{\substack{t\sim\operatorname{Uniform}[0,1],\,(X,c)\sim p_{\mathrm{data}}\\
Z\sim p_0,\,Z\perp(X,c)}}
\left[\left\|v_\theta(X_t,t,c)-U\right\|_2^2\right]
$$

的 $L^2$ 总体最优解是 $v^*(x,t,c)=\mathbb E[U\mid X_t=x,t,c]$，等式按联合分布几乎处处理解。在条件密度与速度足以使连续性方程成立的正则条件下，这个边际速度逐条件输运 $p_t(\cdot\mid c)$；它不要求网络从一个混合点唯一恢复配对端点。耦合和路径的选择会改变速度场的弯曲程度与数值积分难度。

Flow matching 与 diffusion 在连续时间、score 和概率流视角下有紧密关系，但“速度回归”“噪声回归”和“score matching”是不同目标。只有明确路径、参数化与转换关系后才能比较。

## 8.9 训练网络不等于采样器

生成时从基噪声开始，调度器或数值求解器反复调用同一网络。需要区分：

- **祖先式 DDPM 采样**按学习到的反向 Markov 核逐步加入规定方差的随机性；
- **DDIM 型采样**可在同一边际加噪参数化下采用非 Markov、甚至确定性的轨迹，并跳过时间步；
- **反向时间 SDE**保留随机扩散项，**概率流 ODE**使用确定性动力系统。在 score 精确且方程精确求解时，两者具有相同的时间边际分布，并不表示有限步样本逐点相同；
- 高阶 ODE/SDE 求解器减少网络调用次数，但离散误差、阈值处理和 guidance 会改变实际分布。

因此报告生成系统时至少要给出模型参数化、噪声或概率路径、时间步网格、求解器、网络调用次数、随机种子与 guidance。只写“使用同一 diffusion 模型”不足以复现实验。

## 8.10 视频为什么更难

视频模型还要处理时间一致性、身份保持、运动、镜头变化和长程事件。潜变量可写成带时间维的张量，模型既要建立单帧空间结构，也要建立跨帧依赖。常见方案包括时空注意力、分层生成、图像模型初始化和在时间维上追加模块。

漂亮的短片不自动证明物理理解。评价至少分开：单帧质量、跨帧一致性、提示服从、运动合理性、对象持续性和长时间事件结构。下一章会进一步说明，视频预测与可用于行动规划的世界模型仍是不同目标。

## 8.11 离散状态也可以迭代去噪

扩散并不限于连续像素。若状态空间大小为 $K$，可用转移矩阵 $Q_t\in[0,1]^{K\times K}$ 定义

$$
q(x_t\mid x_{t-1})
=\operatorname{Cat}\bigl(x_t;x_{t-1}^{\top}Q_t\bigr),
\qquad
q(x_t\mid x_0)
=\operatorname{Cat}\bigl(x_t;x_0^{\top}\overline Q_t\bigr),
$$

其中 $\overline Q_t=Q_1Q_2\cdots Q_t$。均匀替换、离散高斯邻域和吸收态 `[MASK]` 对应不同 $Q_t$。反向模型学习 $p_\theta(x_{t-1}\mid x_t,c)$，或通过预测 $x_0$ 构造反向核。

masked diffusion 可以逐步遮蔽 token，再学习反向恢复；生成时模型可以在多个位置之间反复修订，而不是永远从左到右追加。

这改变了执行结构：自回归模型每步固定一个新前缀，masked diffusion 维护一整个尚未完成的序列状态。许多实现会在一次反向步中近似地独立预测多个位置，位置间依赖通过后续迭代修正。两者都可以使用 Transformer，却具有不同训练目标、条件独立近似、缓存方式和停止规则。它们在卷二中会被并排展开，在卷三中再比较概率分解。

## 8.12 生成质量怎样测量

没有单个指标同时测量逼真度、覆盖、条件服从与记忆。以图像为例，FID 把真实样本与生成样本映射到固定特征空间，拟合均值 $\mu_r,\mu_g$ 与协方差 $\Sigma_r,\Sigma_g$，再计算

$$
\operatorname{FID}
=\|\mu_r-\mu_g\|_2^2
+\operatorname{tr}\!\left(
\Sigma_r+\Sigma_g
-2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}
\right).
$$

它只比较选定特征的前两阶矩，且有限样本估计有偏；不同特征网络、分辨率与样本数不可直接混比。KID 使用特征上的核 MMD 并可构造无偏估计。precision/recall 型指标试图区分样本质量与模式覆盖，但仍依赖特征空间和邻域估计。原始定义与有限样本讨论见 [FID](SOURCE_NOTES.md#ref-heusel-fid-2017)和 [KID](SOURCE_NOTES.md#ref-binkowski-kid-2018)。

条件生成还要单独评估文本—媒体一致性、组合属性、空间关系、文字可读性和人类偏好。视频再增加运动、身份持续、时间顺序和音画同步。最后还应做最近邻与片段匹配审计，检查高分是否来自训练样本近似复现。指标是测量协议，不是模型质量的坐标系真值。

## 8.13 训练数据与来源边界

生成媒体数据包含图像、视频、音频、文字说明、编辑记录和授权关系。裁剪、重采样、自动描述、去重和过滤会改变训练分布。模型可以复现风格或片段，却不能仅由输出反推出完整训练来源；没有复现也不能证明某项数据未被使用。

因此模型发布应区分模型工件、自动编码器、文本编码器、训练快照和安全过滤器。第十二章将统一讨论这些生命周期对象。

本章的主要技术谱系见 [VAE](SOURCE_NOTES.md#ref-kingma-vae-2013)、[GAN](SOURCE_NOTES.md#ref-goodfellow-gan-2014)、[DDPM](SOURCE_NOTES.md#ref-ho-ddpm-2020)、[Score SDE](SOURCE_NOTES.md#ref-song-score-sde-2021)、[Latent Diffusion](SOURCE_NOTES.md#ref-rombach-2022)、[DiT](SOURCE_NOTES.md#ref-peebles-xie-2023)、[Flow Matching](SOURCE_NOTES.md#ref-lipman-2022)与[离散 diffusion](SOURCE_NOTES.md#ref-austin-d3pm-2021)。这些文献给出模型与实验，不把任一评测指标提升为生成质量的充分定义。
