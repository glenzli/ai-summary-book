# 第六章 审计包模板与贯穿案例（参考）

这一参考章把前五卷中的记录对象合成一个可填写模板。模板不是软件标准，也不是法律合规模板；它要求把包身份、输出、复现合同、主张核验、来源、provenance、工具调用、解释报告、责任记录和状态页分别保存，防止运行证据被一段自然语言摘要吞掉。具体系统可以扩展字段，但删除字段时必须说明相应的核验义务为何不再适用。

## 6.1 目录

一次需要独立复查的输出可以使用如下最小目录；高风险用途还应按验收剖面增加领域专用证据：

```text
audit-package/
  manifest.md
  output_record.md
  reproduction_contract.md
  claims.md
  provenance.md
  tool_calls.md
  explanation_report.md
  responsibility.md
  sources.md
  status.md
```

`manifest.md` 是唯一入口，其余文件各承担一种责任。不要把工具日志、事实核验、解释报告和审批记录混成一个“运行总结”。

## 6.2 manifest

`manifest.md` 固定审计包本身的身份和覆盖边界，至少记录：

| 字段 | `SP404` 示例 |
| --- | --- |
| package_schema | `stochastic-parrot-anatomy/audit-package@1` |
| system_version | `S1` |
| model_artifact | `M1` 及其 digest |
| tool_versions | `flight-status@2.4`、`filesystem@3.2` |
| policy_version | `P1` |
| covered_interval | 从首次查询到最终显示的单次运行 |
| file_digests | 本目录其余九个文件的路径与摘要 |

manifest 不能只写“最新版”。任一版本或摘要不可寻址时，包仍可保存，但相应字段必须记为带原因的缺失值，状态页不得报告 `schema_complete`。

## 6.3 输出记录

`output_record.md` 回答“发生了什么”。

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| input | 用户输入或上游任务 | 查询 `SP404` 并写入 `trip.md` |
| configuration | 模型、上下文、工具、权限、解码和运行时 | `M1/C1/T1/P1/D1/R1` |
| trace | 事件序列 | query、return、write request、commit、timeout、retry、prior-commit receipt、display |
| generated_tokens | 生成 token 序列 | 服务方记录 |
| committed_tokens | 已提交 token 序列 | 客户端收到的片段 |
| unicode_text | 最终 Unicode 文本 | `航班工具报告 SP404 已取消；trip.md 写入已确认。` |
| bytes | 序列化字节 | UTF-8 digest |
| artifacts | 产生或修改的制品 | `trip.md` |
| provenance | 来源图 | 工具返回和写入 activity |
| claims | 文本主张集合 | 两条主张 |
| normative_metadata | 署名、许可、保密、责任口径 | 内部审计记录 |
| status | 运行状态 | succeeded |

缺失字段必须说明原因。比如隐藏上下文可以记为 `Redacted`，而不是留空。

本例保留未知提交的恢复过程。令 $k=\texttt{write-SP404-trip-md-1}$，写入内容摘要为 $d$；轨迹中的关键事件依次为

```text
query(SP404)
return(Cancelled, response_id=F-2048)
write_request(path=trip.md, key=k, digest=d)
server_commit(receipt=W-731, key=k, digest=d)
response_timeout -> client_status=unknown
retry(path=trip.md, key=k, digest=d)
return_existing_commit(receipt=W-731, key=k, digest=d)
display(confirmation)
```

客户端只有在最后一次调用返回原提交回执并核对摘要后，才把运行状态从 `unknown` 改为 `succeeded`。重试没有产生第二次写入。

## 6.4 复现合同

`reproduction_contract.md` 回答“能否重来”。

| 项 | 内容 |
| --- | --- |
| 输入域 $\mathcal D$ | 哪些请求被合同覆盖 |
| 规范化 $N$ | 如何处理空白、编码、时间戳、路径和 redaction |
| 前置条件 $P$ | 模型版本、工具模拟、权限、外部服务状态、幂等存储 |
| 比较规则 $M$ | 字节相同、文本相同、主张相同、轨迹相同或统计相容 |
| 判定算法 $A$ | Pass、Fail、Inconclusive 或结构错误 |

合同必须声明不可重放部分。若航班工具实时变化，就要固定模拟响应，或者把真实工具复现判为 Inconclusive。

本例的重放使用响应 `F-2048` 的归档副本，并要求文件工具对相同 $(k,d)$ 返回同一提交回执。合同不要求再次访问现实航班服务，也不声称归档响应证明所有时刻的现实状态。

## 6.5 主张核验表

`claims.md` 回答“哪些句子是真的，凭什么”。

| id | 文本跨度 | 主张 | 核验状态 | 证据 |
| --- | --- | --- | --- | --- |
| C1 | `航班工具报告 SP404 已取消` | 工具在记录时刻返回 `Cancelled` | Supported | `F-2048`、时间戳、归档响应 |
| C2 | `trip.md 写入已确认` | 幂等键 $k$ 对应一次已提交写入 | Supported | `W-731`、路径、digest、同键重试回执 |

若证据只支持弱主张，就降格正文主张。比如工具只返回“暂不可确认”，就不能写“已取消”。

## 6.6 来源与 provenance

`sources.md` 先记录每个外部或运行时来源实际支持到哪里：

| id | 来源类型 | 固定对象 | 支持上限 |
| --- | --- | --- | --- |
| S1 | 航班工具响应 | `F-2048`、时间戳、响应摘要 | 工具在该时刻报告 `Cancelled`；不独立证明现实状态 |
| S2 | 文件提交回执 | `W-731`、幂等键、旧/新 digest | 一次写入已提交且同键重试未重复写入 |
| S3 | 策略记录 | `P1`、审批事件与验收剖面 | 本次动作按该版本获准；不推出一般法律正当性 |

来源条目必须被 `claims.md` 或 `responsibility.md` 实际引用；只列 URL、工具名或政策标题不构成支持关系。

`provenance.md` 记录 entity、activity 和 agent。简化表如下：

| 类型 | id | 描述 |
| --- | --- | --- |
| entity | E1 | 用户请求 |
| entity | E2 | 航班工具响应 |
| entity | E3 | `trip.md` 新版本 |
| entity | E4 | 文件提交与同键重试回执 |
| activity | A1 | 查询航班状态 |
| activity | A2 | 请求并提交文件写入 |
| activity | A3 | 超时后用同一幂等键核对既有提交 |
| activity | A4 | 生成最终消息 |
| agent | G1 | 部署系统 |
| agent | G2 | 文件工具 |
| agent | G3 | 航班状态工具 |

同一文本可以由不同 provenance 产生，因此 provenance 不是装饰性 metadata。

## 6.7 工具调用记录

`tool_calls.md` 回答“哪些副作用被请求、授权和提交”。

查询调用记录为：

| 字段 | 值 |
| --- | --- |
| tool | `flight_status.query` |
| version | `flight-status@2.4` |
| side_effect_class | `read_only_external` |
| parameters | `flight_no=SP404` |
| permission | read external status |
| approval | pre-authorized read |
| timeout | 5 s |
| retry_policy | 最多一次；查询必须带同一 correlation id |
| declared_error_states | rejected、failed、timeout、malformed_response |
| outcome | `Cancelled` |
| evidence | `F-2048`、时间戳、响应摘要 |

写入调用把“请求过”与“已提交”分开：

| 字段 | 值 |
| --- | --- |
| tool | `filesystem.write` |
| version | `filesystem@3.2` |
| side_effect_class | `committing_idempotent_write`；可由旧摘要补偿恢复 |
| parameters | `path=trip.md, digest=d` |
| permission | write workspace file |
| approval | user granted write scope |
| idempotency_key | `write-SP404-trip-md-1` |
| timeout | 5 s |
| retry_policy | 状态未知时只以同一 $(k,d)$ 重试，最多一次；禁止生成新键 |
| declared_error_states | rejected、failed_before_commit、unknown_after_timeout、digest_conflict |
| first_observation | 响应超时，客户端状态为 `unknown` |
| reconciliation | 同键重试返回既有提交 `W-731`，未再次写入 |
| commit_evidence | receipt、路径、旧/新 digest、服务端提交时间 |
| compensation | 以旧 digest 恢复前一版本；另留补偿回执 |

如果工具没有声明版本、副作用类别、超时、重试策略、错误状态和提交证据，运行时无法区分安全重试与重复行动；这不是日志展示问题，而是工具合同不完整。

## 6.8 解释报告

`explanation_report.md` 回答“为什么这样发生”，但只在证据允许的层级解释。

| 项 | 内容 |
| --- | --- |
| 被解释项 | 最终消息包含两条主张 |
| 行为解释 | 系统读取工具响应后生成摘要 |
| 工具路径 | 查询航班状态；发起写入；超时后同键核对既有提交 |
| 来源解释 | 工具报告来自 `F-2048`；写入确认来自 `W-731` |
| 未支持升级 | 不声称模型内部理解、意图或心理置信 |
| 可反驳预测 | 若工具响应不是 `Cancelled`，合格系统不得生成 C1 |

若要解释内部机制，需要另给 patching、消融或电路证据。本报告只解释系统行为路径。

## 6.9 责任与规范记录

`responsibility.md` 回答“谁允许了什么”。

| 层 | 记录 |
| --- | --- |
| 用户 | 提供目标并允许写入工作区 |
| 系统设计者 | 设置工具权限、审批和日志 |
| 工具提供者 | 声明查询无副作用、写入有提交边界 |
| 模型提供者 | 声明模型版本和推理限制 |
| 组织 | 保留审计包并处理申诉 |

这张表不是法律结论。它只是让后续责任判断不再从一句“模型做了”开始。

本例按策略 `P1` 评为中等风险，必需条件集合为除 `third_party` 外的其余八项。规范记录如下；证据标识均可在包内反查：

| 条件 | 状态 | 证据或理由 |
| --- | --- | --- |
| standing | satisfied | 用户对目标工作区有写入资格 |
| informed | satisfied | 预览列出路径、内容与写入副作用 |
| voluntary | satisfied | 拒绝写入不影响航班查询结果 |
| scoped | satisfied | 授权仅覆盖 `trip.md` 与本次运行 |
| third_party | not_applicable | 个人工作区案例不处理第三方数据；该项不在必需集合 |
| minimization | satisfied | 只读取航班状态，只写入所需一行 |
| risk_distribution | satisfied | 不公开发布，不改变外部账户或资金状态 |
| contestability | satisfied | 用户可检查记录、质疑来源并暂停后续动作 |
| remediation | satisfied | 保存旧 digest，可恢复前一文件版本 |

若把路径改为共享仓库、写入包含第三方行程或动作改为真实改签，验收剖面必须重算，不能沿用此处的 `not_applicable` 或中等风险分类。

## 6.10 状态页

`status.md` 分别给出结构、证据和主张终态：

```text
schema_status: schema_complete
evidence_profile: SP404-notification@1
evidence_status: evidence_sufficient
output_status: succeeded
claim_status:
  C1: Supported
  C2: Supported
normative_status: approved_under_P1
reproduction_status: replayable_with_mocked_tools
open_items:
  - archived tool response does not independently prove the real-world flight state
  - hidden system context redacted by service policy
```

`schema_complete` 只表示目录、类型和一致性检查通过；`evidence_sufficient` 只相对于所列用途剖面；两个 `Supported` 只覆盖 C1、C2 的限定措辞。三者都不表示现实世界全知，也不能由 `output_status: succeeded` 推出。

## 6.11 从记录到判断

审计包的目的不是让系统产生更多文件，而是让一个判断在离开生成它的界面以后仍能接受反驳。若 `claims.md` 只写结论而没有证据，若 `tool_calls.md` 只保存请求而没有提交状态，或若 `status.md` 把不可访问证据写成成功，目录再齐全也没有形成审计能力。反过来，低风险运行不必机械保存所有内部张量；记录强度应由主张、可逆性、受影响主体和失败代价决定。

因此，审计包的闭合是相对于用途的。它至少要让独立读者区分四种主张结果：得到支持、被反驳、证据不足，以及记录本身不良构；还要把这些结果与 schema 和规范批准状态分开。前五卷建立的对象类型、概率关系、复现合同、解释层级和授权边界，在这里不再作为并排知识点出现，而是共同决定这些结果怎样产生。

`SP404` 案例也由此得到终点。用户看到的确认句只是最外层制品；工具报告的航班状态、文件提交、工具权限、运行轨迹和现实状态是否独立可核验，各有自己的证据状态。可审计系统不会把这些差异抹平成“任务成功”，而会保存足以继续核验、恢复或申诉的边界。

到这里，全书的工程论证已经完成：模型能力只有经过运行语义、证据协议和责任控制，才成为可承担后果的系统行动。下一卷改变叙述位置，用第一人称、随笔和隐喻回看同一组边界；文体会改变，已经建立的对象和证据关系不会。
