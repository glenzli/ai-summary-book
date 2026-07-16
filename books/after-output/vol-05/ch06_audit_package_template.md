# 第六章 审计包模板与贯穿案例

这一参考章把六卷中的记录对象合成一个可填写模板。模板不是软件标准，也不是法律合规模板；它要求把输出、复现合同、主张核验、provenance、工具调用、解释报告、责任记录和状态页分别保存，防止运行证据被一段自然语言摘要吞掉。具体系统可以扩展字段，但删除字段时必须说明相应的核验义务为何不再适用。

## B.1 目录

一次高风险输出的最小审计包可以使用如下目录：

```text
audit-package/
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

每个文件只承担一种责任。不要把工具日志、事实核验、解释报告和审批记录混成一个“运行总结”。

## B.2 输出记录

`output_record.md` 回答“发生了什么”。

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| input | 用户输入或上游任务 | 查询 `SP404` 并写入 `trip.md` |
| configuration | 模型、上下文、工具、权限、解码和运行时 | `M1/C1/T1/P1/D1/R1` |
| trace | 事件序列 | query、response、write、commit、message |
| generated_tokens | 生成 token 序列 | 服务方记录 |
| committed_tokens | 已提交 token 序列 | 客户端收到的片段 |
| unicode_text | 最终 Unicode 文本 | `SP404 已取消；已写入 trip.md。` |
| bytes | 序列化字节 | UTF-8 digest |
| artifacts | 产生或修改的制品 | `trip.md` |
| provenance | 来源图 | 工具返回和写入 activity |
| claims | 文本主张集合 | 两条主张 |
| normative_metadata | 署名、许可、保密、责任口径 | 内部审计记录 |
| status | 运行状态 | succeeded |

缺失字段必须说明原因。比如隐藏上下文可以记为 `Redacted`，而不是留空。

## B.3 复现合同

`reproduction_contract.md` 回答“能否重来”。

| 项 | 内容 |
| --- | --- |
| 输入域 $\mathcal D$ | 哪些请求被合同覆盖 |
| 规范化 $N$ | 如何处理空白、编码、时间戳、路径和 redaction |
| 前置条件 $P$ | 模型版本、工具模拟、权限、外部服务状态 |
| 比较规则 $M$ | 字节相同、文本相同、主张相同、轨迹相同或统计相容 |
| 判定算法 $A$ | Pass、Fail、Inconclusive 或结构错误 |

合同必须声明不可重放部分。若航班工具实时变化，就要固定模拟响应，或者把真实工具复现判为 Inconclusive。

## B.4 主张核验表

`claims.md` 回答“哪些句子是真的，凭什么”。

| id | 文本跨度 | 主张 | 核验状态 | 证据 |
| --- | --- | --- | --- | --- |
| C1 | `SP404 已取消` | 航班状态为取消 | Supported | 航班工具响应、时间戳 |
| C2 | `已写入 trip.md` | 文件写入已提交 | Supported | 写入日志、路径、digest |

若证据只支持弱主张，就降格正文主张。比如工具只返回“暂不可确认”，就不能写“已取消”。

## B.5 provenance

`provenance.md` 记录 entity、activity 和 agent。简化表如下：

| 类型 | id | 描述 |
| --- | --- | --- |
| entity | E1 | 用户请求 |
| entity | E2 | 航班工具响应 |
| entity | E3 | `trip.md` 新版本 |
| activity | A1 | 查询航班状态 |
| activity | A2 | 写入文件 |
| activity | A3 | 生成最终消息 |
| agent | G1 | 部署系统 |
| agent | G2 | 文件工具 |

同一文本可以由不同 provenance 产生，因此 provenance 不是装饰性 metadata。

## B.6 工具调用记录

`tool_calls.md` 回答“哪些副作用被请求、授权和提交”。

| 字段 | 示例 |
| --- | --- |
| tool | `flight_status.query` |
| parameters | `flight_no=SP404` |
| permission | read external status |
| approval | pre-authorized read |
| response | `Cancelled` |
| commit | no side effect |
| evidence | response id and timestamp |

写入调用另列：

| 字段 | 示例 |
| --- | --- |
| tool | `filesystem.write` |
| parameters | `path=trip.md` |
| permission | write workspace file |
| approval | user granted write scope |
| idempotency_key | `write-SP404-trip-md-1` |
| commit | succeeded |
| evidence | path, digest, write log |

## B.7 解释报告

`explanation_report.md` 回答“为什么这样发生”，但只在证据允许的层级解释。

| 项 | 内容 |
| --- | --- |
| 被解释项 | 最终消息包含两条主张 |
| 行为解释 | 系统读取工具响应后生成摘要 |
| 工具路径 | 查询航班状态，再写入文件 |
| 来源解释 | 取消状态来自航班工具；写入状态来自文件工具 |
| 未支持升级 | 不声称模型内部理解、意图或心理置信 |
| 可反驳预测 | 若工具响应不是 `Cancelled`，合格系统不得生成 C1 |

若要解释内部机制，需要另给 patching、消融或电路证据。本报告只解释系统行为路径。

## B.8 责任记录

`responsibility.md` 回答“谁允许了什么”。

| 层 | 记录 |
| --- | --- |
| 用户 | 提供目标并允许写入工作区 |
| 系统设计者 | 设置工具权限、审批和日志 |
| 工具提供者 | 声明查询无副作用、写入有提交边界 |
| 模型提供者 | 声明模型版本和推理限制 |
| 组织 | 保留审计包并处理申诉 |

这张表不是法律结论。它只是让后续责任判断不再从一句“模型做了”开始。

## B.9 状态页

`status.md` 给出审计包自己的闭合状态：

```text
package_status: complete
output_status: succeeded
claim_status: all supported
reproduction_status: replayable_with_mocked_tools
open_items:
  - real flight API response not independently archived
  - hidden system context redacted by service policy
```

这类状态页应允许 `complete` 与 `open_items` 同时存在。完整表示相对于 schema 已交代边界，不表示现实世界全知。
