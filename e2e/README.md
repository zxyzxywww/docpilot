# e2e —— 会话生命周期真机验收(手动运行,非 CI)

> 用途:对 **Docker 全家桶真实环境**(真实 qdrant 语料 + 真实 LLM key + 本机 Edge)跑
> "A~H 会话生命周期"38 条断言(新建可见/发送秒建会话/回答中切走不丢/F5 刷新恢复/
> 历史完整性/mode 会话级持久化/draft 无垃圾/失败恢复)。与 `tests/` 的离线 mock 互补,
> 属于**手动验收脚本**,不在 CI 执行(会调用真实付费 API)。

## 前置

1. Docker 全家桶已起并健康(qdrant + api + web,见仓库根 `docker-compose.yml`);
2. `.env` 已配置真实 key(仓库根,已 gitignore);
3. Node 18+(本机,含 `npm install` 权限);本机安装 Edge;
4. docpilot 环境 Python(运行 seed 用,仓库根 README 有环境说明)。

## 运行

```bash
cd e2e
npm install          # 首次;恢复 playwright(node_modules 不入库)
python seed.py all   # 清空会话并造种子:8 个会话(S2/S3/S_fail + E1..E5)
node e2e.js          # playwright 驱动本机 Edge → 期望输出 A~H 结果 38/38 通过
```

> 注:`seed.py` 直连仓库根 `data/db/docpilot.db`(与 api 容器挂载同一份 SQLite);
> `e2e.js` 内 `API/WEB/EDGE` 为常量(默认 localhost:8000 / :3000 / 本机 Edge 路径),
> 换机器只需改这三个常量。失败可重看 `node e2e.js` 的断言输出与截图(screenshot 保存于当前目录)。
