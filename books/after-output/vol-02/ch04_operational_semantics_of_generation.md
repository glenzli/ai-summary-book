# 第四章：自回归生成的小步语义

第三章把运行写成轨迹，却把“生成一句话”留成了一个粗粒度事件。现在放大这一步。航班查询和文件写入的结果已经进入模型上下文，生成器接下来要为“SP404 已取消；已写入 trip.md。”产生第一章的 token 序列

$$
v_\star=(101,102,103,104,105,106,107),
$$

并以 EOS 结束。我们需要知道：一次 token 选择怎样改变状态，模型或选择器失败时轨迹在哪里结束，以及固定选择输入后“确定生成”究竟是一条什么定理。第一章的 admissible domain、第二章的 Result 类型和第三章的 LTS 将在这里组合起来。

## 4.1 静态数据与类型

固定有限 token 集 $\mathbb V$、特殊 token
$\mathtt{EOS}\in\mathbb V$，并令
$\mathbb V_c=\mathbb V\setminus\{\mathtt{EOS}\}$。
固定输入前缀 $x\in\mathbb V^*$、选择状态集 $R$、模型错误集 $E_M$ 与选择错误集 $E_S$。
令

$$
E=(\{\mathsf{model}\}\times E_M)
\sqcup(\{\mathsf{selector}\}\times E_S),
$$

并以 $\iota_M:E_M\to E$、$\iota_S:E_S\to E$ 记两个不交注入。这样即使两个错误集合含有同名值，事件仍记录错误来自模型求值还是选择器。

模型求值接口是总函数

$$
M:\mathbb V^*\to
\operatorname{Result}(\mathbb R^{\mathbb V},E_M),
$$

其中 $\mathbb R^{\mathbb V}$ 是以 token 为索引的实数分数向量。上下文过长、非法设备状态等可作为 $E_M$ 中错误返回。该接口只建模“返回或显式失败”；若实现可能挂起，需在更细状态机中加入计算中状态与发散轨迹。

选择器是总函数

$$
S:\mathbb R^{\mathbb V}\times R
\to\operatorname{Result}(\mathbb V\times R,E_S).
$$

贪心解码可取 $R$ 为单点，并固定 $\arg\max$ 并列规则。采样实现把尚未消费的随机输入或 PRNG 状态放入 $R$。此处的 $S$ 是实现映射，不是第五章的抽象随机核。

## 4.2 运行配置

状态集合由配置

$$
c=(y,r,n,q)
$$

组成，其中：

- $y\in\mathbb V_c^*$ 是已生成但不含 EOS 的后缀；
- $r\in R$ 是选择状态；
- $n\in\mathbb N$ 是剩余 token 预算；
- 状态标记
  $$
  q\in
  \{\operatorname{run}\}
  \sqcup\{\operatorname{done}(d):d\in D\}
  \sqcup\{\operatorname{error}(e):e\in E\},
  $$
  $D=\{\operatorname{eos},\operatorname{length}\}$。

事件标签包括
$\operatorname{token}(v)$、$\operatorname{finish}(d)$ 与
$\operatorname{fail}(e)$。正常终止集明确取为

$$
F_{\mathrm{gen}}=
\{(y,r,n,\operatorname{done}(d)):
y\in\mathbb V_c^*,\ r\in R,\ n\in\mathbb N,\ d\in D\}.
$$

done 和 error 配置都无后继；前者属于 $F_{\mathrm{gen}}$，后者不属于，因而分别是第三章意义下的正常终止与带错误信息的卡死最大状态。

## 4.3 转移规则

当 $q=\operatorname{run}$ 时，规则按下列互斥顺序定义：

1. 若 $n=0$，
   $$
   (y,r,0,\operatorname{run})
   \overset{\operatorname{finish}(\operatorname{length})}{\longrightarrow}
   (y,r,0,\operatorname{done}(\operatorname{length})).
   $$
2. 若 $n>0$ 且 $M(xy)=\operatorname{Err}(e_M)$，则
   $$
   (y,r,n,\operatorname{run})
   \overset{\operatorname{fail}(\iota_M(e_M))}{\longrightarrow}
   (y,r,n,\operatorname{error}(\iota_M(e_M))).
   $$
3. 若 $n>0$、$M(xy)=\operatorname{Ok}(z)$ 且
   $S(z,r)=\operatorname{Err}(e_S)$，则
   $$
   (y,r,n,\operatorname{run})
   \overset{\operatorname{fail}(\iota_S(e_S))}{\longrightarrow}
   (y,r,n,\operatorname{error}(\iota_S(e_S))).
   $$
4. 若 $n>0$、$M(xy)=\operatorname{Ok}(z)$ 且
   $S(z,r)=\operatorname{Ok}(\mathtt{EOS},r')$，则
   $$
   (y,r,n,\operatorname{run})
   \overset{\operatorname{finish}(\operatorname{eos})}{\longrightarrow}
   (y,r',n-1,\operatorname{done}(\operatorname{eos})).
   $$
5. 若 $n>0$、$M(xy)=\operatorname{Ok}(z)$、
   $S(z,r)=\operatorname{Ok}(v,r')$ 且 $v\in\mathbb V_c$，则
   $$
   (y,r,n,\operatorname{run})
   \overset{\operatorname{token}(v)}{\longrightarrow}
   (yv,r',n-1,\operatorname{run}).
   $$

Result 构造子的互斥性、$\mathtt{EOS}\notin\mathbb V_c$ 和 $n=0$ 与 $n>0$ 的互斥性保证规则不重叠。五条规则还写出了完整前提与目标配置，所以规则 4、5 中的 $z,r'$ 不依赖隐含的前一条分支。

贯穿案例给出一个可以逐步核对的有限实例。令 $x_\star$ 包含用户请求、查询结果
`Cancelled` 和写入重试的成功确认；令 $N=8$，并固定模型与选择状态，使七次内容选择依次返回 $v_\star$ 的七个 ID，第八次返回 EOS。忽略每步变化但此处已经固定的选择状态，唯一轨迹的关键分量为：

| 转移后步数 | 事件 | 内容后缀 $y$ | 剩余预算 |
|---:|---|---|---:|
| 0 | 初态 | $\epsilon$ | 8 |
| 1 | $\operatorname{token}(101)$ | $(101)$ | 7 |
| 2 | $\operatorname{token}(102)$ | $(101,102)$ | 6 |
| 3--7 | 依次选择 $103,\ldots,107$ | $v_\star$ | 1 |
| 8 | $\operatorname{finish}(\operatorname{eos})$ | $v_\star$ | 0 |

由第一章的玩具 tokenizer，$\operatorname{Dec}_{\Theta_\star}(v_\star)=u_\star$，即“SP404 已取消；已写入 trip.md。”。表格不是另一套语义，而是五条规则在一组完全指定输入上的展开。

## 4.4 固定选择输入后的确定性

**定理 4.1（生成 LTS 强确定）.** 对固定
$(\mathbb V,\mathtt{EOS},x,M,S)$，上述带标签转移系统强确定；每个 run 配置恰有一个后继，每个 done 或 error 配置无后继。

**证明.** 任取配置。done 与 error 没有规则。对 run 配置：

- 若 $n=0$，仅规则 1 适用；
- 若 $n>0$，总函数 $M$ 有唯一 Result。Err 时仅规则 2 适用；Ok 时得到唯一 $z$；
- 对该 $z$，总函数 $S$ 有唯一 Result。Err 时仅规则 3 适用；Ok 时得到唯一 $(v,r')$；
- 由 $\mathbb V=\mathbb V_c\sqcup\{\mathtt{EOS}\}$，规则 4 与 5 恰有一个适用。

因此 run 状态恰有一个带标签后继，其他状态无后继，满足定义 3.2。证毕。

模型分数允许多个 token 同分，或抽象分布允许多个 token 有正概率，都不反驳本定理；选择器的并列规则及本次选择输入已经被固定。

所以贯穿案例中的“唯一轨迹”不是从模型分数本身推出的。它依赖 $M$ 返回唯一 Result、$S$ 把并列和随机输入消解为唯一 Result，以及初始选择状态已经固定。第五章将忘掉这部分选择状态；同一个生成器随即不再由单条轨迹表示，而由路径上的概率测度表示。

## 4.5 前缀单调性

定义 $u\preceq v$ 当且仅当存在 $w\in\mathbb V_c^*$ 使 $v=uw$。

**命题 4.2（生成后缀单调）.** 若
$(y,r,n,q)\to(y',r',n',q')$，则 $y\preceq y'$。若事件为
$\operatorname{token}(v)$，则 $y'=yv$ 且 $|y'|=|y|+1$；其余事件满足 $y'=y$。

**证明.** 逐条检查五条规则。只有规则 5 改变 $y$，且只在右侧追加一个
$v\in\mathbb V_c$；其他规则保持 $y$。证毕。

该命题只关于 token 缓冲。支持 delete、replace 或服务端 speculative token 撤回的协议必须把“候选 token”和“已提交 token”分成不同状态；第七章处理这一点。

## 4.6 长度预算与终止

**定理 4.3（有界步数终止）.** 从初始配置
$(\epsilon,r_0,N,\operatorname{run})$ 出发，唯一最大轨迹在至多
$N+1$ 个转移后进入 done 或 error。

**证明.** 由定理 4.1，run 状态恰有一个后继。每个预算正的非错误步骤或者立即进入 done/error，或者由规则 5 把预算减一。因而若前 $N$ 步都由规则 5 产生内容 token，预算变为零，下一步必由规则 1 进入 done。故最大轨迹有限，长度至多 $N+1$。证毕。

这是抽象 LTS 的步数结论，不是墙钟时间界。若实现 $M$ 或 $S$ 的计算本身可能无限运行，则它不满足本章所声明的“总函数返回”接口，应在实现语义中显式加入该发散。

## 4.7 从 token 到文本

设 tokenizer 配置为 $\Theta$，并假设生成后缀 $y$ 落在
$\operatorname{AdmTok}_\Theta$ 中。文本观察是部分函数

$$
\operatorname{text}_\Theta(y)
=\operatorname{Dec}_\Theta(y).
$$

若某个生成前缀不在 admissible domain，系统必须选择：

- 把它标为解码错误；
- 定义特殊 token 的删除或转义规则；
- 在字节层积累，直到形成合法 UTF-8 序列。

不能一面允许任意 $y\in\mathbb V_c^*$，一面无条件声称最终 Unicode 文本存在。

## 4.8 停止串是状态机

文本停止串检测依赖已经解码的后缀、规范化策略、跨 token 边界匹配以及是否保留命中串。严格实现应把识别器状态 $m\in Q_{\mathrm{stop}}$ 加入配置，并定义确定转移

$$
\delta_{\mathrm{stop}}:
Q_{\mathrm{stop}}\times\mathbb V_c
\to Q_{\mathrm{stop}}\times
\{\operatorname{continue},\operatorname{stop}\}.
$$

若在 Unicode 文本层匹配，$\delta_{\mathrm{stop}}$ 还需携带未完成的字节或规范化缓冲。不同识别层即使给出同一可见文本，也可能消费不同 token、记录不同 logprob，并具有不同 finish reason。

在本章的具体运行中，$v_\star$ 属于
$\operatorname{AdmTok}_{\Theta_\star}$，EOS 由 token 层识别，因而文本观察有值且轨迹在八步内结束。换一个停止层或允许 tokenizer 产生域外序列，这两个结论便需要重新检查。更重要的是，表格只记录已经固定选择状态后的那一次运行；它没有给“生成这句话的概率”下定义。下一章保留同一个状态空间，却把下一步从函数改为随机核，并把最终文本作为轨迹分布的可测投影。

## 练习

**练习 4.1.** 为贪心解码定义单点选择状态和确定并列规则，并验证选择器是总函数。

**练习 4.2.** 扩展状态以允许结构化工具调用请求，但把工具响应作为显式外部输入，从而保持固定输入下的强确定性。

**练习 4.3.** 取消长度预算并允许内容 token 永续生成，构造永不产生 EOS 的唯一无限轨迹。

**练习 4.4.** 给出 token 层停止模式与 Unicode 文本层停止串产生不同消费轨迹的例子。

**练习 4.5.** 扩展状态为候选缓冲和已提交缓冲以支持流式撤回；指出命题 4.2 对哪一个缓冲仍成立。


## 生成语义审计接口

### S6.4 自回归生成的小步语义


令状态包含前缀 $v_{<t}$、上下文 $c$、解码器状态 $d_t$、随机源状态 $\xi_t$ 和停止状态 $q_t$。一步生成可分解为：

1. 模型给出 logits $z_t$；
2. 解码器用 $D$ 产生候选 token $v_t$；
3. 停止规则判断是否提交、截断或终止；
4. 状态更新为 $v_{\le t}$。

若 $D$ 是采样解码，则还需随机流；若 $D$ 是 greedy，仍需处理并列最大、浮点差异和停止规则。这里的“生成”只是 token 层事件，不能推出事实已被核验。
