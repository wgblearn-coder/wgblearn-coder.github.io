---
title: "消息队列原理与应用：从已支付订单到可靠异步系统"
date: "2026-09-07"
description: "沿餐饮订单已支付后的真实请求链，讲清消息队列的模型、存储、可靠性、幂等、重试、选型与容量治理，并给出可落地的 Java 与 SQL 设计。"
tags: [消息队列, Kafka, RabbitMQ, RocketMQ, Java]
---

订单已经支付，收银台要尽快返回成功；经营统计要汇总，积分要发放，短信或站内信要通知，失败还要能补偿。若支付接口在一个同步请求里依次调用这三个系统，任何一个下游变慢都会拖住用户；若直接开线程“异步一下”，进程崩溃又可能让积分和通知无声丢失。

消息队列解决的是这条业务链中的时间解耦和故障隔离：订单服务提交一条可追踪的业务事件，队列持久化并按规则交给下游消费者。它不能自动让业务获得恰好一次，也不能代替数据库事务、幂等约束和监控。本文用“订单已支付 → 经营统计、积分、通知”贯穿原理与实现，所有容量数字、延迟数字和故障现象都是教学假设，不能当作某个线上系统的实测结果。

读完后应该能记住一句话：先把业务事实可靠地写下来，再让不同消费者各自推进自己的进度；发送、存储、消费、业务提交、外部调用，每一段都要单独定义确认边界。

## 1. 先建立消息队列的概念边界

### 定义、同步与异步

消息是“已经发生的事实”或“请执行的命令”的可序列化记录，通常包含 eventId、eventType、aggregateId、发生时间、schemaVersion、traceId 和业务载荷。队列是保存消息、按投递规则让消费者取得消息的中间系统；生产者发布，消费者处理，代理负责路由、持久化、重试或保留。

同步调用是调用者等待被调用方返回结果。例如订单服务在支付完成后调用积分服务 HTTP 接口，只有收到成功才返回。异步调用是调用者把可继续推进的事实交给中间层后先返回；积分服务稍后消费事件。异步降低了主链路对下游延迟的敏感度，却把“现在失败”变成“之后重试、积压、告警和对账”的运营问题。

### 队列、发布订阅与消费组

传统工作队列通常是一条消息被一个竞争消费者处理：三个积分实例共同消费积分队列，一条消息只由其中一个实例取得。发布订阅则是一条事件被多个逻辑订阅者分别消费：统计、积分、通知各自有独立订阅进度，因此订单事件能被三个系统都看到。

Kafka 的 topic 由 partition 组成。在经典 consumer group 语义下，一个 partition 在同一时刻只分配给一个 group member；同一 group 内扩容到超过 partition 数的消费者不会继续增加并行度。

不同 group 可以分别从自己的 offset 读取同一 topic，因而适合事件流的多订阅。这里引用的是 [Kafka 4.0 Design 中 classic consumer 的基线说明](https://kafka.apache.org/40/design/design/)，不把它泛化成其他新消费协议的结论。

RabbitMQ 常把 exchange、binding、queue 组合成路由拓扑；RocketMQ 则通过 topic、消费组和有序消息等能力组织投递，产品术语不能机械互换。

记录的 offset 标识它在分区中的位置；提交的消费 offset 表示下一条要读取的位置，也不是某条业务记录的“已成功”标记。若批量拉取 offset 10、11、12，10 失败而 11、12 完成，不能提交 13 越过失败的 10；要等 10 也完成后，才能提交 13。并发处理要维护完成集合和最小缺口，或按 key 串行化，不能只看最大已完成 offset。

### 从订单事件画出请求链

支付成功后，订单服务在自己的事务中把订单状态改成 PAID，并准备 `OrderPaid` 事件。可靠方案有两种主流落点：事务内写 outbox 表，再由 publisher/CDC 投递；或使用具备事务消息机制的消息平台，把本地事务结果与消息可见性关联。事件进入 topic 后，统计 group 写入日报聚合，积分 group 写入积分流水，通知 group 调用通知供应商。

每个消费者都要拥有自己的进度、重试和告警。统计短暂故障不应阻塞积分；通知供应商限流时，应能只拖慢通知消费。事件的 `partitionKey=tenantId+orderId` 可以帮助按租户和订单分区，从而让同一订单的支付、退款、取消等事件在一个有序流中出现，但“分区有序”不等于所有消费者之间全局有序。

## 2. 一条消息如何被保存、复制和确认

### 追加日志、页缓存与批量

高吞吐队列的核心常是追加写：生产者把连续消息追加到日志段，offset 或索引把逻辑位置映射到物理文件。追加写避免每条消息都在磁盘上随机找位置；操作系统页缓存会缓存最近读写的文件页。Kafka 4.0 设计文档明确描述了文件系统、页缓存、顺序 I/O 和批量消息集的组合；这是设计推导，不代表任意磁盘、内核和负载都有同样性能。

发送端会把多条记录组成 batch，代理把 batch 追加，消费者一次 fetch 一段数据。批量降低网络往返、系统调用和磁盘小 I/O；压缩整个 batch 通常比逐条压缩有更高的重复利用率。日志段、索引段和页缓存共同作用：索引可能按 offset 建稀疏位置，定位后顺序扫描消息，而不是为每个消费者维护一棵巨大的随机访问树。

### 四层持久性边界

第一层是进程收到数据：生产者把字节交给客户端缓冲区，进程随后崩溃，消息可能尚未离开客户端。第二层是 broker 接收并追加到 leader 日志：leader 已看见，但副本可能尚未跟上。第三层是按复制策略确认：足够副本写入或满足平台规定的 quorum/ISR 条件，消息在限定故障模型下更耐 broker 故障。第四层是保留期与灾备：消息是否仍在 retention 内、磁盘是否可恢复、跨机房副本和恢复点是否满足 RPO。副本确认不是永久备份。

“写入页缓存”也不等于“已经 fsync 到稳定介质”。操作系统可能先把日志页放在内存，机器掉电时仍会丢；是否等待 flush、由 broker 和磁盘策略共同决定。应用要把可接受故障模型写入配置与演练，而不能把一次 API 返回直接称为永久保存。

Kafka 生产者的 `acks`、重试、幂等和事务配置要一起看，具体参数以 [Kafka 3.9 producer configuration](https://kafka.apache.org/39/configuration/producer-configs/) 为准。`acks=all` 还要配合 topic 的 `min.insync.replicas`：教学例子设 RF=3、min.insync.replicas=2 时，至少两个同步副本可写。

失去第二个同步副本会牺牲可写性来保护持久性，不能继续把“all”理解成所有曾经配置的三个副本都在线。若副本数、ISR、unclean leader election、保留策略或灾备没设计好，也不能口头升级成“永不丢失”。

RabbitMQ publisher confirms 是 broker 对发布处理阶段的异步确认，消费者 acknowledgement 是另一方向的确认；官方 [Confirms 文档](https://www.rabbitmq.com/docs/confirms) 把两者分开说明。

它们都不是“下游数据库事务已提交”的确认。

发布 confirm 成功也不必然表示消息已进入目标队列：exchange 可能没有匹配 binding。需要启用 `mandatory` 并处理 broker 返回的 unroutable message；return 与 confirm 都要纳入发布结果，不能只等 confirm 就把 outbox 标为完成。

### 持久性和保留期的边界

`durable`、持久消息、磁盘落盘、复制和保留期分别解决不同问题。消息即使复制成功，也可能按 TTL 或日志保留策略被删除；消费者停机太久，恢复时 offset 可能已经超出保留窗口。跨机房副本、对象存储归档、定期备份和可验证恢复，才是灾备条件的一部分。设计文档必须写清 RPO、RTO、允许丢失的事件类型、保留天数、最大消息大小和恢复时从哪里重放。

教学例子：假设支付事件每秒 200 条、平均 2 KB，日均约 34.6 GB 原始载荷；这是算术示例，未计 header、压缩、复制和索引。若保留 7 天、复制因子为 3，粗略原始容量约为 `34.6 × 7 × 3 = 726.6 GB`，还要留出日志段、峰值和运维余量。不能拿这个数字直接申请生产磁盘。

## 3. 可靠发送：Outbox、事务消息与多 publisher

### 为什么“先写库再发消息”仍会丢

若代码先提交订单，再调用 broker，提交成功而进程在发送前崩溃，订单已是 PAID 却没有事件。若先发消息再提交订单，broker 成功而数据库回滚，消费者看见了不存在的支付。两个独立系统的本地事务无法靠一个普通 try/catch 自动原子化。

### Outbox 表与事务伪 Java

Outbox 的思路是把业务状态和待发事件放进同一个数据库事务。示例表只对 `event_id` 做唯一约束，重复写入时必须识别这一指定约束；不能用 `INSERT IGNORE` 把其他字段错误、截断或非法状态静默吞掉。

```sql
CREATE TABLE outbox_event (
  event_id       VARCHAR(64)  NOT NULL,
  tenant_id      VARCHAR(64)  NOT NULL,
  aggregate_type VARCHAR(64)  NOT NULL,
  aggregate_id   VARCHAR(64)  NOT NULL,
  event_type     VARCHAR(128) NOT NULL,
  schema_version INT          NOT NULL,
  payload_json   JSON         NOT NULL,
  status         VARCHAR(16)  NOT NULL,
  available_at   TIMESTAMP    NOT NULL,
  lease_owner    VARCHAR(128) NULL,
  lease_until    TIMESTAMP    NULL,
  publish_token  BIGINT       NOT NULL DEFAULT 0,
  created_at     TIMESTAMP    NOT NULL,
  published_at   TIMESTAMP    NULL,
  PRIMARY KEY (event_id),
  KEY ix_outbox_ready (status, available_at, created_at)
);
```

```java
@Transactional
public void markPaid(String orderId, String paymentId) {
    String tenantId = auth.tenantId();
    Order order = orderRepository.lockById(tenantId, orderId);
    if (order.status() == PAID) {
        if (!order.paymentId().equals(paymentId)) throw new PaymentConflict();
        return;
    }
    if (order.status() != PENDING_PAYMENT) throw new IllegalStateException("invalid transition");
    order.markPaid(paymentId);
    orderRepository.save(order);

    String eventId = Ids.orderPaid(tenantId, orderId, paymentId);
    OutboxEvent event = OutboxEvent.pending(
        eventId, tenantId, "ORDER", orderId, "OrderPaid", 1,
        json(new OrderPaid(tenantId, orderId, order.memberId(), order.amount(),
                           paymentId, order.version())), clock.instant());
    outboxRepository.insert(event); // event_id 冲突也让整个事务失败并由上层查验
}
```

示例按 MySQL 8 InnoDB 书写，`tenant_id` 是所有查询的租户边界。`OrderPaid` 载荷包含 tenant、member、金额、paymentId 和业务版本，消费者不必猜测缺失字段。只允许 `PENDING_PAYMENT → PAID`；已支付但 paymentId 不同要报警，迟到支付若订单已取消则进入支付对账，而不是强行改回已支付。Outbox 的主键冲突不在事务内吞掉：事务应回滚，由上层按 eventId 查验已提交事实。

### Publisher 租约、token fencing 与 CDC

多个 publisher 可以并行扫描 outbox，但领取任务不能靠“查到后再更新”两步裸奔。常见做法是条件更新领取租约：`status='PENDING' AND (lease_until IS NULL OR lease_until < now())`，写入 `lease_owner`、新的 `lease_until`，并把 `publish_token` 加一。发布者把 token 带进后续更新条件，只有仍持有最新 token 的实例才能把事件标记为已发布。

为什么需要 fencing？实例 A 租约过期后暂停，实例 B 重新领取并发布；A 恢复后若还能把状态覆盖成成功，就会破坏 B 的最新事实。token 是单调栅栏，更新必须满足 `WHERE tenant_id=? AND event_id=? AND publish_token=?`。

它只防止 outbox 旧实例写回旧状态，不能阻止旧 worker 已经向 broker 发出的消息，也不保证发送顺序。租约只能防止正常竞争，不能防止 broker 已接收但客户端超时；因此 publisher 必须接受重复发布，eventId 要稳定，broker 或消费者要有去重策略。

另一条路是 CDC：数据库提交后由日志捕获 outbox 行变更并发布。Debezium 的 [Outbox Event Router](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html) 展示了从 outbox 捕获、用 event id 作为消息标识、用 aggregate id 参与路由的做法。

它降低了自写扫描器的复杂度，但仍要运营连接器 offset、schema 演进、故障重启和下游重复；CDC 不是跨系统恰好一次的证明。

流量很小或基础设施受限时，也可以不用 MQ：在 MySQL 8 InnoDB 中建立 `task` 表，业务事务写入任务，worker 用 `SELECT ... FOR UPDATE SKIP LOCKED` 领取，租约、attempt、next_run_at、状态和死信字段都落库。这仍是可靠异步，只是调度、扫描和扩容由应用负责；任务表也会承受锁竞争、扫描压力和清理成本，不能把它说成免费替代。

### 事务消息何时合适

RocketMQ 事务消息通常先发送半消息，执行本地事务，再让 broker 根据本地结果决定可见性，并在不确定时回查；细节请看 [RocketMQ 事务消息官方文档](https://rocketmq.apache.org/docs/featureBehavior/04transactionmessage/)。

它把消息平台纳入事务协议，适合已有 RocketMQ 体系且团队能运营回查接口的场景。Outbox 则把事实留在业务数据库，便于审计、补发和 CDC。二者都需要幂等、回查或对账，选型要看数据库与平台的责任边界。

## 4. 消费可靠性：ACK、幂等与外部未知结果

### 至多一次、至少一次、恰好一次

至多一次通常是先确认进度再处理：消费者崩溃会丢，但不会因为该消费者重试而重复。至少一次是先处理成功再确认进度：处理后崩溃、确认没发出去，消息会再次投递，所以必须接受重复。恰好一次需要限定边界：消息平台内部可能提供事务读写或幂等生产，但一旦同时更新外部数据库、发送短信或调用支付渠道，就不能轻易宣称端到端恰好一次。

### “业务提交成功后 ACK”与唯一约束

积分消费者应把消费记录和积分流水放进同一数据库事务。独立业务 key 要来自业务语义，例如 `orderId + promotionId`，而不是每次重试都新生成 UUID。消费表可以记录 eventId 追踪事件，积分表的业务唯一约束则真正防止重复发放。

```sql
CREATE TABLE consumer_inbox (
  consumer_name VARCHAR(64) NOT NULL,
  tenant_id     VARCHAR(64) NOT NULL,
  event_id      VARCHAR(64) NOT NULL,
  received_at   TIMESTAMP NOT NULL,
  PRIMARY KEY (consumer_name, tenant_id, event_id)
);

CREATE TABLE points_ledger (
  ledger_id     BIGINT       NOT NULL,
  tenant_id     VARCHAR(64)  NOT NULL,
  member_id     VARCHAR(64)  NOT NULL,
  order_id      VARCHAR(64)  NOT NULL,
  promotion_id  VARCHAR(64)  NOT NULL,
  points        INT          NOT NULL,
  created_at    TIMESTAMP    NOT NULL,
  PRIMARY KEY (ledger_id),
  UNIQUE KEY uk_points_tenant_order_promo (tenant_id, order_id, promotion_id)
);
```

```java
public void onMessage(Event<OrderPaid> event) {
    try {
        transactionTemplate.executeWithoutResult(tx -> handleInTransaction(event));
        listener.ack(event); // 事务提交成功后才 ACK
    } catch (DuplicateKeyException e) {
        if (isInboxPrimaryKey(e) || isPointsBusinessKey(e)) {
            if (sameCommittedBusinessResult(event)) listener.ack(event);
            else throw e; // 同业务不同 eventId 但摘要不一致，必须失败告警
        } else throw e;
    }
}

void handleInTransaction(Event<OrderPaid> event) {
    OrderPaid paid = event.payload();
    inbox.insert(paid.tenantId(), "points", event.eventId(), clock.instant());
    // 按订单保存的积分策略版本计算，不能在重试时套用最新规则
    int points = pointsPolicy.calculateFromOrderSnapshot(paid.tenantId(), paid.orderId());
    pointsLedger.insert(NewLedger.forOrder(paid.tenantId(), paid.orderId(),
        paid.memberId(), "WELCOME_OR_ORDER", points));
    memberRepository.addPoints(paid.tenantId(), paid.memberId(), points);
}
```

不要把所有 DuplicateKeyException 都当成功。事务回滚后，只有精确识别 inbox 主键或 `uk_points_tenant_order_promo` 冲突，才查询已提交行并核对 tenant、member、points、orderId 等摘要；同一业务才 ACK，其他冲突继续失败。这样同业务不同 eventId 的重复不会无限重试，也不会把数据错误吞掉。

批量并发消费时，offset 只能提交已连续完成的前缀；offset 10 失败而 11、12 成功，不能提交 13 越过失败的 10。若积分流水插入成功、余额更新失败，整个 InnoDB 事务回滚；数据库提交成功后 ACK 丢失，重试会被唯一约束挡住。

### 外部 HTTP 的未知结果

通知消费者调用短信供应商时，连接超时并不等于供应商未发送：请求可能已经到达，响应只是在返回途中丢失。此时直接重试可能发两条短信；直接 ACK 又可能漏通知。应使用供应商支持的幂等请求号（如稳定的 `notificationId`），查询原请求状态；若供应商没有查询能力，就把状态标成 `UNKNOWN`，进入人工或定时对账，而不是凭猜测重发。

### ACK 与 publisher confirm 的方向

消费者 ACK 表示“我处理到这里，可以推进投递进度”；publisher confirm 表示“代理处理了我的发布请求”。两者方向相反，必须分别记录。RabbitMQ quorum queue 通过复制日志和仲裁机制提供更强的队列可用性与持久性模型，具体边界见 [Quorum Queues 官方文档](https://www.rabbitmq.com/docs/quorum-queues)；仍需结合磁盘、节点故障、网络分区和消息保留策略评估。

## 5. 顺序、重试、延迟与重放

### 顺序是业务范围内的约束

同一订单可能先支付再退款，也可能支付回调重复到达。以 `orderId` 为 key 让同一订单进入同一 partition/有序队列，只能保证该分区追加顺序；多分区之间、多个 topic 之间和不同消费者 group 之间没有天然全局顺序。消费者还要用版本号或状态机拒绝过期事件。

```sql
UPDATE order_projection
SET status = 'REFUNDED', version = :eventVersion
WHERE tenant_id = :tenantId AND order_id = :orderId
  AND version = :eventVersion - 1
  AND status = 'PAID';
```

返回 0 行时要区分重复、乱序和非法状态：已是 REFUNDED 可能是重复；版本落后说明旧事件；版本跳跃说明前置事件缺失。可以暂停该 key、放入等待队列或请求补数，不能把所有 0 行都当成功。

### 重试退避、抖动和死信

瞬时数据库连接失败适合有限重试；参数校验失败、schema 不兼容和权限错误重试没有意义。退避可以是 `delay = min(cap, base × 2^attempt) + random(0, jitter)`，让不同消费者不要在同一秒形成重试风暴。

每次尝试都要记录 eventId、attempt、错误类型和下次时间，重试预算耗尽后进入死信队列。把失败消息转到独立重试队列会改变它与同 key 后续消息的相对位置；需要顺序的消费者应暂停该 key、使用有序重试调度，或让状态版本检测缺口，不能默认重试队列仍保持原顺序。

死信不是垃圾桶：保留原始 payload、headers、异常摘要、首次失败时间和最后尝试时间，并建立按错误类型的告警。修复代码后，先在隔离 group 或回放 topic 验证，再按时间、tenant、eventType 或 eventId 小批量重放。重放会再次触发副作用，必须依靠业务幂等与人工审批边界；统计类消费者可从历史 offset 重建，通知类消费者通常不能无条件重发。

### 延迟消息与取消的状态 CAS

订单超时取消常见做法是发布一个未来可见的 `CancelIfUnpaid(orderId, deadline, version)`。消息到期时不能只看消息内容，还要原子检查订单当前状态、版本和截止时间：只有 `PENDING_PAYMENT` 且版本仍匹配才改成 `CANCELLED`。支付与取消并发时使用条件更新或行锁，让胜出的状态转换唯一。

```sql
UPDATE orders
SET status = 'CANCELLED', version = version + 1
WHERE tenant_id = :tenantId AND order_id = :orderId
  AND status = 'PENDING_PAYMENT'
  AND version = :expectedVersion
  AND payment_deadline <= CURRENT_TIMESTAMP;
```

更新为 0 行不是异常本身，可能是已支付、已取消或版本变化。RabbitMQ、RocketMQ、Kafka 的延迟实现方式不同；延迟消息只负责把检查推迟，不能替代最终状态 CAS。

## 6. Kafka、RabbitMQ、RocketMQ 如何有条件地选

Kafka 更适合长保留、可回放、高吞吐事件流：统计系统可以按 offset 重建，多个消费组各自追赶。代价是需要认真设计 partition key、rebalance、消费延迟、日志保留和重放隔离；它的经典 group 模型不是“每条消息都广播给同一组内的每个实例”。

RabbitMQ 更适合低延迟工作队列、复杂 exchange 路由、灵活 ack/nack 和短任务分发。Quorum queue 在需要复制和故障恢复的队列场景有价值，但吞吐、磁盘和运维成本要用目标负载验证；不要只因“有确认”就承诺端到端一次。

RocketMQ 适合已经采用其生态、需要顺序消息、事务消息或延迟能力的团队。[RocketMQ FIFO 文档](https://rocketmq.apache.org/docs/featureBehavior/03fifomessage/)强调有序消息的生产和消费约束，顺序范围与队列分配有关；事务消息也需要本地事务状态和回查。它的特性组合很有吸引力，但实际可用性取决于客户端版本、集群部署、监控和团队经验。

选择时先写业务条件：是否要按订单回放七天？是否要复杂路由？能否接受消费组重平衡？是否已有跨机房灾备？是否需要延迟和事务消息？再用同一批代表性订单、峰值、失败注入和恢复演练压测。产品名字不能代替可靠性目标。

## 7. 容量、背压、隔离与可观测性

### 用公式而非感觉估容量

先统一量纲。令入口速率 `λ`、消费速率 `μ` 都是条/秒，单条平均大小为 `s` 字节，则网络入口是 `λ×s` 字节/秒，积压增长速度是 `λ-μ` 条/秒；当 `λ>μ` 时必然增长。设峰值系数为 `p`、复制因子为 `r`、压缩后比例为 `c`、保留秒数为 `T`，保留期间的磁盘粗估为 `λ×s×p×r×c×T`，再加索引、段文件、协议开销和安全余量。

教学算例：峰值 `λ=1000` 条/秒、`μ=600` 条/秒持续 10 分钟，积压是 `(1000-600)×600=240000` 条。恢复时入口降到 200、出口升到 800 条/秒，净清理 600 条/秒，需要 `240000/600=400` 秒。若平均 3 KB、复制因子 3、压缩后 0.45、保留 3 天，磁盘估算约 `1000×3KB×86400×3×0.45×3 ≈ 1.05 TB`。这些是帮助理解变量的假设，不是生产容量建议。

实际还要按最大消息、网络出口、broker 磁盘写入、消费者数据库 TPS 和恢复时长分别核算。

“最老积压时间”比单纯消息数更接近用户影响：`oldest_lag_age = now - oldest_unprocessed_event_time`。如果通知队列有 80 万条但每条很小、最老仅 30 秒，风险可能低于只有 5 万条却最老 2 小时的积分积压。

监控至少包括入口/出口速率、各 partition lag、最老消息年龄、重试率、死信数、消费处理耗时、ACK/confirm 延迟、broker 磁盘水位、复制落后、rebalance 次数和 outbox 未发布年龄。

### 背压与隔离

消费者应设置最大拉取批量、并发上限、数据库连接池上限和单租户配额。下游变慢时暂停或降低拉取速度，让积压可见；无限扩大线程只会把数据库和外部供应商一起压垮。通知、积分、统计用不同 group、队列或资源池隔离，给通知设置供应商 QPS 限制，给统计设置可牺牲的延迟目标。

生产端也要有界：outbox 的未发布行超过阈值时，订单接口可以告警、限流或按业务等级降级。对于支付成功这种事实，不能为了“队列满了”静默丢掉；可以拒绝新请求、延长处理、切换灾备路径或让人工运营介入，具体取决于支付一致性承诺。

### 故障演练要验证什么

演练不应只断 broker。至少注入：数据库提交后 publisher 崩溃、confirm 超时后重复发布、消费者业务提交后 ACK 丢失、外部 HTTP 已成功但响应超时、一个 partition 消费者停止、复制节点故障、保留期边界恢复、schema 不兼容和死信重放。每次记录预期事件数、实际业务流水数、重复数、丢失数、恢复时间和告警到达时间。

## 8. 面向 Agent/RAG 的异步任务与面试表达

### Agent 与 RAG 为什么也需要队列

用户发起“分析本店近三个月复购率”的 Agent 任务时，文档解析、切块、向量化、检索、模型调用都可能耗时，且要受并发和 token 配额限制。可以发布 `RagTaskSubmitted(taskId, tenantId, revision, query)`，worker 按任务版本消费；每次检索要在授权后过滤 tenant、门店和文档 ACL，不能因为消息可见就绕过权限。

任务状态要带版本和取消标记。用户取消任务时设置独立的 `cancel_requested=true` 标志，worker 在 OCR、embedding 批次、检索和模型调用前后检查状态；最终用 CAS 让 `RUNNING` 只能转为 `SUCCEEDED`、`FAILED` 或 `CANCELLED` 中一个。

限流按 tenant、模型、token 和并发槽位分别计算，重试只针对可重试错误；模型超时可能已经产生计费或外部副作用，要记录 `UNKNOWN`；供应商支持查询时查询原请求，不支持时进入对账或明确的重试预算策略，不能假定调用没有发生。

RAG 结果适合保存 query、知识库 revision、检索文档 id、权限快照版本和引用来源，便于重放和审计。增量索引若发现 revision 从 10 跳到 12，不能简单丢掉 11；应暂停消费、拉取缺口或从快照重建。

快照只提供某个一致时点，增量事件负责从快照版本继续推进，二者的边界要记录清楚。缓存 key 至少要包含 tenant、ACL/revision、query 规范化版本和模型配置；消息队列只负责任务推进，不能自动保证答案引用正确。评估时把召回率、答案正确性、引用完整性和任务成功率分开。

### 30 秒表达

“消息队列是一条可恢复的异步事实链：订单事务写 PAID 和 outbox，publisher 允许重复发布，三个消费组各自推进。消费者用 eventId 与业务唯一键幂等，事务提交后 ACK；重试、死信、对账和外部 HTTP 要分别定义边界。”

### 2 分钟表达

“队列的价值是把支付主链路和统计、积分、通知解耦，并提供可观测的积压和恢复点。同步调用会把下游延迟传回收银台；发布订阅让三个消费组各自保留进度。存储层的追加日志、批量、页缓存和副本确认，不能越过业务数据库提交的边界。”

“发送可靠性我会优先选 outbox：业务表与 outbox 同事务提交，publisher 用租约和递增 token 防旧实例覆盖，或用 CDC 捕获。消费端按至少一次设计，inbox 和 orderId+promotionId 唯一约束保证幂等，事务成功后才 ACK。外部短信超时要查询幂等状态。”

“顺序按 orderId 分区，状态转换用版本 CAS；重试用指数退避加抖动，永久错误进死信并隔离重放。Kafka、RabbitMQ、RocketMQ 的选择由回放、路由、延迟、事务消息、运维和压测决定；监控最老积压、复制落后、重试和死信，并演练提交后崩溃与 ACK 丢失。”

### 练习与答案

练习一：订单显示已支付，但积分没有增加，outbox 有一行 PENDING。先查什么？答案：查 publisher 领取租约、发布 confirm/返回、事件在 broker 的位置、积分 group 的 lag 和死信；不要直接改积分表。若确认事件已发布，再查消费者事务日志和唯一键冲突。

练习二：消费者日志显示“处理成功”，但同一订单出现两条积分流水。答案：检查积分流水是否有 `(order_id, promotion_id)` 唯一约束，是否把每次重试生成了不同业务 key，是否在业务提交后才 ACK；eventId inbox 只能辅助追踪，不能代替正确的业务唯一约束。

练习三：通知供应商接口超时，重试前做什么？答案：用稳定 notificationId 查询供应商原请求状态；查不到就进入 UNKNOWN 和对账，不凭网络超时推断未发送。

练习四：通知积压一小时，新增十倍消费者是否一定解决？答案：先看供应商 QPS、数据库连接池、分区数、单条处理耗时和限流；若瓶颈在外部系统，扩容只会放大失败，应先背压、隔离和按租户限流。

### 学习顺序

第一步画出一条订单事件链，标注每个系统的事实、命令、确认和失败恢复。第二步手写 outbox、inbox、业务唯一约束和状态 CAS。第三步用一个本地 broker 验证 batch、partition、consumer group、ACK、重试和重放。

第四步做容量算例，解释最老积压时间和背压阈值。第五步分别演练 publisher 崩溃、消费者 ACK 丢失、外部 HTTP UNKNOWN、复制节点故障和死信回放。最后再比较 Kafka、RabbitMQ、RocketMQ，并把 Agent/RAG 的任务版本、权限过滤、取消和 token 限流映射到同一套边界上。

真正可靠的异步系统不是某个配置项的结果，而是每一段都能回答三个问题：事实写在哪里，成功由谁确认，失败后怎样重试、去重、对账或重放。
  tenant_id     VARCHAR(64)  NOT NULL,
