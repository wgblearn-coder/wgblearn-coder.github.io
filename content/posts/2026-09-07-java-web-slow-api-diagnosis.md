---
title: "Java Web 慢接口怎么查：从线程池到数据库"
date: "2026-09-07"
description: "以 Spring Boot、Tomcat、HikariCP 与 MySQL 8 为例，沿 P99、Trace、线程栈、连接池和锁等待建立一条可复核的慢接口排查链。"
tags: [Java, Spring Boot, 性能排查, MySQL, 分布式系统]
---

线上接口从 200 毫秒变成 5 秒，最容易先怪数据库。请求可能卡在 SQL，也可能还没拿到 Tomcat 工作线程，或拿到线程后一直等 HikariCP 连接。把一次请求经过的队列和资源逐段对齐，结论才站得住。

本文用一个假设的订单查询接口贯穿排查过程：某个订单更新事务长时间持有行锁，其他更新请求等待该行锁并占满共享连接池，后续查询拿不到连接，入口开始积压；超时后客户端重试，又把等待放大。这个场景用于说明方法，数字和输出都是示例，不代表实测结果。

排查时我会按三个问题走：慢在什么位置，谁在等谁，时间线能不能把根因串起来。指标提供线索，Trace 展开路径，线程栈记录现场，数据库视图和执行计划把“等待”落到具体事务或 SQL。

## 先确认慢接口的范围和时间窗口

先把问题写具体：接口路径、HTTP 方法、租户或订单类型、异常起始时间、部署版本和影响比例都留下来。“变慢”还要拆成平均值、P50、P95、P99 与错误率。平均值 400 毫秒、P99 8 秒，往往说明少数请求碰到了锁、连接池或外部依赖，并非所有 SQL 一起变差。

在网关、应用指标和业务日志里使用同一个时间窗口。可以从 Spring Boot Actuator 的 HTTP 请求指标开始，再和数据库、JVM 指标交叉核对；指标名称及配置见 [Spring Boot Actuator metrics 文档](https://docs.spring.io/spring-boot/reference/actuator/metrics.html)。

先看耗时分位数、请求数、5xx 和超时数，再按实例拆开。只有一个实例异常时，先查该实例的线程、JVM 和连接；所有实例同时异常，才把范围扩大到数据库或公共依赖。

不要把客户端总耗时直接当成服务方法耗时。反向代理排队、网络、Tomcat 排队、Controller、连接获取、SQL 和序列化都可能在里面。先为一条慢请求保存 request id 或 trace id，并记录实例 PID、发布版本和时区；后面所有命令的时间都要能映射回这个窗口。

## 用 Trace 找到关键路径和等待点

在 Trace 中先看根 Span 的总时长，再逐层看 HTTP、业务方法、数据库客户端和重试 Span。若 SQL Span 从 3 秒开始，说明应用已经进入数据库调用；若根 Span 很慢但业务 Span 很短，要检查 Tomcat 入队或线程池调度。Trace 没有覆盖线程池排队和连接池获取时，不能据此断言“业务代码只执行了 100 毫秒”。

采集器的语义和观测边界可以对照 [OpenTelemetry observability primer](https://opentelemetry.io/docs/concepts/observability-primer/)。

同一请求中的并行 Span 不可以简单相加。两个查询各执行 300 毫秒，如果并行，墙钟时间可能接近 300 毫秒而不是 600 毫秒；应看父子关系和关键路径。重试也要单独标记：一次用户请求里出现多次相同 SQL，可能是重试，也可能是循环查询或 N+1 访问；需要结合调用位置、参数和重试日志确认。

下面是补充排队与连接获取埋点后的示意时间分解，不是监控工具默认保证提供的 Span：

```text
GET /orders/123          5.2s
  tomcat.queue           1.1s
  orderService           4.1s
    hikari.acquire       3.0s
    SELECT order...      1.0s
    业务与响应处理       0.1s
```

这只能说明主要等待发生在连接获取和 SQL 路径，还不能说明连接为什么没有归还。接下来要把应用线程和连接池状态放到同一时间线上。

## 看 Tomcat 和业务线程池：三次线程栈比一次更可靠

先分别观察 Tomcat 和实际参与请求的业务线程池：活跃线程数与上限、队列长度、排队时间、完成任务速率和拒绝次数。活跃数高且完成速率下降，才需要继续追查线程被什么任务占住；CPU 很低也可能是大量等待。自建线程池不一定自动采集，需确认 Micrometer 注册情况。

先在应用服务器上用 `jcmd -l` 确认 Java 进程。以下命令中的 `<PID>` 是运行 Spring Boot 的 Java PID，`<APP_HOST>` 表示应用服务器 Shell；不要在本地开发机或 MySQL 主机上误运行。

```bash
jcmd <PID> VM.command_line
jcmd <PID> Thread.print -l > /tmp/thread-$(date +%s).txt
```

`Thread.print -l` 会给出线程状态、栈和锁信息。等待 10 秒左右，再执行两次相同命令，形成 t0、t1、t2 三份快照。连续都停在 `com.zaxxer.hikari.pool.HikariPool.getConnection`，且等待线程数增加，支持“连接获取等待”；连续停在同一个 MySQL socket read，才支持“SQL 或数据库响应慢”。只有一次快照不能区分短暂尖峰和持续阻塞。

JDK 诊断工具的适用方式见 [Oracle Diagnostic Tools](https://docs.oracle.com/en/java/javase/13/troubleshoot/diagnostic-tools.html)。

线程状态的字面含义不能直接当成资源结论。`RUNNABLE` 可能正在执行，也可能卡在本地调用或 socket read，不一定吃满 CPU；普通 `WAITING` 常见于空闲线程等任务，也可能确实在等条件变量。线程栈要和 CPU 采样、请求 Trace、等待对象一起看。

若项目使用了 Spring `@Async` 或自定义执行器，只有在实际调用路径出现时才检查那个线程池。可以用 Arthas 连接应用，在 `<APP_HOST>` 执行：

```bash
java -jar arthas-boot.jar <PID>
```

启动命令需要服务器上已有 Arthas 启动包。进入 Arthas 交互终端后执行（业务类与方法名需替换为项目实际值）：

```text
thread -n 10
thread --state WAITING
trace com.example.order.OrderService queryOrder '#cost > 1000' -n 5
```

`thread -n 10` 用于找 CPU 高的线程，`thread --state WAITING` 用于观察等待状态；`trace` 展开的是方法调用耗时。它们都应限制类名和条件，避免在高流量时扩大开销。

Arthas 对 `thread` 和 `trace` 的参数、输出含义分别见[官方 thread 文档](https://arthas.aliyun.com/en/doc/thread.html)与[官方 trace 文档](https://arthas.aliyun.com/doc/trace.html)。

## 连接池要分清 active、idle、pending 和 acquire

HikariCP 监控至少要同时看 `active`、`idle`、`pending`、最大连接数，以及连接获取耗时。Spring Boot 的指标通常以 `hikaricp.connections.active`、`idle`、`pending` 等形式暴露；HikariCP 的配置和行为以[项目文档](https://github.com/brettwooldridge/HikariCP)为准。

`active` 接近 `maximumPoolSize` 且 `pending` 持续上升，说明应用线程在等连接；`idle` 仍很多而 SQL 慢，则更像连接已借出后的数据库执行问题或指标采集口径问题。

借出连接不等于正在执行 SQL。代码可能拿到连接后先做业务计算、等待锁、调用远程服务，或者事务开启后迟迟没有提交。反过来，一次 SQL 也可能在驱动内部等待网络返回。要在连接池指标、事务日志、线程栈和 MySQL 连接列表之间核对。

连接获取超时 `connectionTimeout` 表示等待从池中拿到连接的上限，不是 SQL 执行超时。SQL 超时要看 JDBC、MyBatis、事务或数据库侧配置。把两者混为一谈，会出现“调大 connectionTimeout 修好了数据库”的假象：只是让更多 Tomcat 线程继续等待，峰值时反而拖垮实例。

还要观察连接持有时间（usage）与获取超时次数。长时间不归还可能来自长事务、事务内远程调用或连接泄漏；HikariCP 泄漏检测提示持有过久，只是调查线索，不自动证明泄漏。

可以先用近似关系做容量合理性检查：

```text
平均连接占用数 ≈ 每秒借用次数 × 平均持有时间
```

例如每秒 80 次借用、平均持有 50 毫秒，平均占用约 4 条连接。这个估算隐含稳定流量、持有时间分布平稳等条件，不是直接计算池大小；P99 持有时间、突发流量、事务峰值和数据库上限仍要单独留余量。

## 到 MySQL 8 查慢 SQL、行锁和长事务

以下命令在 `<DB_HOST>` 的 MySQL 8 客户端执行。账号需要具备相应查看权限，结果中的时间是采样时刻，不能替代慢日志和历史指标。

```sql
SHOW FULL PROCESSLIST;

SELECT trx_id, trx_started, trx_state, trx_wait_started,
       trx_mysql_thread_id, trx_query
FROM information_schema.innodb_trx
ORDER BY trx_started;

SELECT * FROM sys.innodb_lock_waits\G
```

`SHOW FULL PROCESSLIST` 中大量 `Waiting for ... lock` 或同一表的长时间 `Query`，说明要进一步看锁；`innodb_trx` 中很早开始、仍处于 `RUNNING` 的事务，可能持有锁或长时间未提交。`sys.innodb_lock_waits` 把等待事务和阻塞事务关联起来：重点找 blocking pid、等待时长、对象和 SQL，而不是只看等待方。

字段和视图说明见 [MySQL sys.innodb_lock_waits](https://dev.mysql.com/doc/refman/8.0/en/sys-innodb-lock-waits.html)。先确认业务归属和回滚风险，再决定是否终止阻塞会话。

拿到 SQL 后先看计划：

```sql
EXPLAIN FORMAT=JSON
SELECT o.id, o.status, o.updated_at
FROM orders o
WHERE o.tenant_id = 42 AND o.status = 'PAID'
ORDER BY o.updated_at DESC LIMIT 50;

EXPLAIN ANALYZE
SELECT o.id, o.status, o.updated_at
FROM orders o
WHERE o.tenant_id = 42 AND o.status = 'PAID'
ORDER BY o.updated_at DESC LIMIT 50;
```

`EXPLAIN` 是优化器估算，关注访问类型、候选索引、估算行数和排序；`EXPLAIN ANALYZE` 会实际执行并返回运行时统计，必须确认语句是只读、数据量可接受且不会造成副作用。若估算行数和实际行数差异很大，检查统计信息、数据分布和组合索引；若计划扫描大量行再排序，评估 `(tenant_id, status, updated_at)` 等索引是否匹配真实过滤条件。索引建议不能脱离写入成本、选择性和锁行为直接上线。

普通 InnoDB 一致性读通常通过 MVCC 读取，不会仅因另一事务的行锁直接阻塞。这里先等行锁的是竞争更新或锁定读，它们占满共享池后，普通查询才会间接等连接。元数据锁则需要另查 `sys.schema_table_lock_waits` 等视图。

在假设案例中，t0 的订单更新事务已经开始 40 秒仍未提交，t1 的查询线程全部在 `getConnection`，MySQL 中又能看到等待事务指向该更新事务。此时“连接池满”是症状，“长事务持锁”才是更靠前的根因；如果只把池从 20 调到 50，可能只是把等待转移到数据库。

## JVM 和 OS 检查用于排除与定位

应用服务器 `<APP_HOST>` 上，先做低成本采样：

```bash
top -H -p <PID>
pidstat -p <PID> -t -u -r 1 10
vmstat 1 10
iostat -xz 1 10
```

`top -H` 或 `pidstat` 看到单线程 CPU 长时间接近一个核，再把线程 ID 转成十六进制与 Java 栈匹配；没有高 CPU 不代表没有阻塞。`vmstat` 的 run queue、上下文切换、内存回收和 swap，`iostat` 的设备利用率与 await，可帮助排除宿主机 CPU、内存和磁盘压力。容器环境还要看 cgroup CPU、内存限制和 throttling，不能只看宿主机总量。

需要确认 GC 时可采集短时 JFR。启动录制后等待 120 秒完成，再执行摘要命令；详细事件用 JDK Mission Control 查看：

```bash
jcmd <PID> JFR.start name=slow-api settings=profile duration=120s filename=/tmp/slow-api.jfr
jfr summary /tmp/slow-api.jfr
```

重点观察 GC pause、分配速率、线程阻塞、Socket 和文件 I/O 事件。一次 Full GC 与慢请求同时出现，只能说明时间相关；必须证明 GC 暂停覆盖关键请求，并且暂停前后堆、分配或回收指标相符，才能把它写成因果。JFR 也可能只是在数据库锁等待期间记录到一次恰好发生的 GC。

## 用时间线确认根因并先止血

把证据整理成一张按时间排序的表：

```text
12:00:00  更新事务 T1 开始，未提交
12:00:05  锁等待增加，连接 active=20、pending=15
12:00:06  Tomcat 工作线程在 HikariPool.getConnection
12:00:08  首批请求 connectionTimeout，客户端开始重试
12:00:10  P99 从 300ms 升到 5s，数据库出现更多等待事务
```

这条链条需要同时得到 Trace、三次线程栈、连接池指标和 MySQL 锁视图支持。若 Trace 显示 SQL 只占 100 毫秒，线程栈却先在连接获取处等待 3 秒，不能把 3 秒写到 SQL 上；若 SQL Span 覆盖 3 秒但没有锁等待记录，还要检查网络、执行计划和数据库资源。

止血按影响面选择：暂停或限流触发重试的入口，设置有边界的重试次数和退避；对订单查询启用降级或读模型；确认事务归属后处理阻塞会话；临时降低并发，保护数据库。调大线程池、连接池或超时时间只能在确认下游仍有余量时使用，否则会把排队长度推到更深一层。任何杀事务或回滚动作都要先评估订单一致性和补偿流程。

修复应回到根因：缩短事务范围，将可移出的远程调用和无关计算移出数据库事务，同时重新确认一致性边界；确保异常路径释放连接并及时提交；为真实过滤条件设计索引并复核计划；把重试改为幂等、限次、带抖动退避；为连接获取、SQL 执行和业务总耗时分别建立指标。修复后用相同流量模型回放，观察 P95/P99、pending、锁等待、事务时长和重试率是否一起下降。

## 把证据串起来，再验收修复

现场可以按下面的顺序收集证据：

1. 固定接口、实例、版本和时间窗口，确认 P95/P99 与错误率。
2. 从 Trace 找根 Span 的关键路径，标记排队、连接获取、SQL、远程调用和重试边界。
3. 在应用主机取三次 `jcmd Thread.print -l`，配合 Arthas 和 CPU 采样识别持续等待点。
4. 对照 HikariCP 的 active、idle、pending、连接获取耗时，判断是池前排队还是借出后的执行。
5. 在 MySQL 查 processlist、`innodb_trx`、`sys.innodb_lock_waits`，再用 `EXPLAIN`/`EXPLAIN ANALYZE` 验证 SQL。
6. 用 JFR、`pidstat`、`vmstat`、`iostat` 排除 JVM 和 OS 资源压力。
7. 按时间线确认根因，先限流和停止重试放大，再做事务、索引和超时修复。

验收不能只看某一次请求变快。在相同并发和数据分布下，P95/P99 要回到目标范围，连接池 pending 在峰值后回落，锁等待和长事务消失，重试率与 5xx 不反弹；还要走一遍超时、重复提交、连接泄漏和回滚路径。否则只是把等待从 Tomcat 挪到了数据库。

### 来源

- [Spring Boot Actuator metrics](https://docs.spring.io/spring-boot/reference/actuator/metrics.html)
- [HikariCP](https://github.com/brettwooldridge/HikariCP)
- [Arthas thread](https://arthas.aliyun.com/en/doc/thread.html) / [trace](https://arthas.aliyun.com/doc/trace.html)
- [Oracle Java Diagnostic Tools](https://docs.oracle.com/en/java/javase/13/troubleshoot/diagnostic-tools.html)
- [MySQL 8.0 sys.innodb_lock_waits](https://dev.mysql.com/doc/refman/8.0/en/sys-innodb-lock-waits.html)
- [OpenTelemetry observability primer](https://opentelemetry.io/docs/concepts/observability-primer/)
