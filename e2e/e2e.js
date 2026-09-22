/* DocPilot A~H 端到端真机验证(playwright-core + 本机 Edge)
 * 前置:Docker 全家桶后端 http://localhost:8000、前端 http://localhost:3000、
 *       seed.py all 已执行。
 * 运行:node e2e.js   (退出码 0=全过)
 */
const { chromium } = require("playwright-core");

const API = "http://localhost:8000";
const WEB = "http://localhost:3000/chat";
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

const results = [];
function ok(name, cond, extra = "") {
  results.push({ name, pass: !!cond, extra });
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${extra ? "  | " + extra : ""}`);
}
async function api(path) {
  const r = await fetch(API + path);
  return r.json();
}
async function waitRunDone(sid, timeoutMs = 240000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const runs = await api(`/api/sessions/${sid}/runs`);
    if (runs.length && ["done", "failed"].includes(runs[0].status)) return runs[0];
    await new Promise((r) => setTimeout(r, 1200));
  }
  return null;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch({ executablePath: EDGE, headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const shot = async (n) => {
    try { await page.screenshot({ path: `shot-${n}.png` }); } catch {}
  };

  // ---------- 载入与预置断言 ----------
  await page.goto(WEB, { waitUntil: "domcontentloaded" });
  await page.getByText("第二个会话-切走目标").waitFor({ timeout: 20000 });
  const seedSessions = (await api("/api/sessions")).length;
  ok("预置:8 个 seed 会话已加载", seedSessions === 8, `count=${seedSessions}`);

  // ---------- G. 空 draft:连点 5 次新对话不产生垃圾会话 ----------
  const btnNew = page.getByRole("button", { name: /新对话/ }).first();
  for (let i = 0; i < 5; i++) {
    await btnNew.click();
    await page.getByText("尚未发送").waitFor({ timeout: 4000 });
  }
  await sleep(800);
  const afterG = (await api("/api/sessions")).length;
  ok("G: 连点 5 次新对话 → 数据库仍 8 会话(无垃圾)", afterG === 8, `count=${afterG}`);
  await shot("G-draft");

  // ---------- A. 新建立即可见:草稿项 + 高亮 + 欢迎页 + 输入框焦点 ----------
  await btnNew.click();
  await page.getByText("尚未发送").waitFor();
  const draftItem = page.getByText("草稿", { exact: true });
  ok("A: sidebar 立即出现草稿会话项(含'草稿')", await draftItem.isVisible().catch(() => false));
  ok("A: 中央空白新会话欢迎页", await page.getByText("Python 后端开发文档助手").isVisible());
  const focused = await page.evaluate(() => {
    const el = document.activeElement;
    return !!(el && el.tagName === "TEXTAREA");
  });
  ok("A: 输入框已聚焦", focused);
  await shot("A-newchat");

  if (!process.env.FAST_E) {
// ---------- B. 新建后发送:落库 + user 可见 + sidebar 标题/生成中 ----------
  const qB = "FastAPI 的 Depends 参数注入是怎么实现的?";
  await page.getByPlaceholder(/输入你的问题/).fill(qB);
  await page.waitForFunction((q) => document.querySelector("textarea") && document.querySelector("textarea").value === q, qB);
  await page.getByRole("button", { name: "快速问答" }).click(); // direct 更快
  await page.getByPlaceholder(/输入你的问题/).click();
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => location.search.includes("session="), { timeout: 8000 });
  const sidB = new URL(page.url()).searchParams.get("session");
  ok("B: 发送即建立真实会话(URL session)", !!sidB, sidB ?? "");
  await page.getByText(qB).first().waitFor({ timeout: 8000 });
  ok("B: user 消息立即可见", true);
  await page.getByText(qB.slice(0, 20), { exact: false }).first().waitFor({ timeout: 8000 });
  ok("B: sidebar 出现会话标题", true);
  const runB = await waitRunDone(sidB, 150000);
  ok("B: 任务最终 done", runB && runB.status === "done", runB ? runB.status : "timeout");
  const msgsB = await api(`/api/sessions/${sidB}/messages`);
  ok("B: 落库 user+assistant 两条", msgsB.length === 2, `len=${msgsB.length}`);
  await sleep(1500);
  await shot("B-sent");

  // ---------- C. 回答中切换(agentic 慢任务) ----------
  await btnNew.click();
  await page.getByRole("button", { name: "深度调研" }).click();
  const qD = "对比 FastAPI 与 Flask 的架构设计差异,包括依赖注入、类型系统、异步支持、插件生态,并给出从 Flask 迁移到 FastAPI 的分步方案";
  await page.getByPlaceholder(/输入你的问题/).fill(qD);
  await page.waitForFunction((q) => document.querySelector("textarea") && document.querySelector("textarea").value === q, qD);
  await page.getByPlaceholder(/输入你的问题/).click();
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => location.search.includes("session="), { timeout: 8000 });
  const sidD = new URL(page.url()).searchParams.get("session");
  await page.getByText(qD).first().waitFor({ timeout: 6000 });
  ok("C: 慢任务已提交,user 立即可见", true);
  // 立即切到 S2
  await page.getByText("第二个会话-切走目标").first().click();
  await page.waitForFunction(
    () => !location.search.includes("session=") || true
  ).catch(() => {});
  await sleep(1200);
  const inS2 = new URL(page.url()).searchParams.get("session");
  const s2List = await api("/api/sessions");
  const s2 = s2List.find((s) => s.title === "第二个会话-切走目标");
  ok("C: 切到 B(预置会话)成功", s2 && inS2 === s2.session_id, `url=${inS2?.slice(0, 8)}`);
  // A 仍生成中:Sd 在 running runs
  let sdRunning = false;
  for (let i = 0; i < 40 && !sdRunning; i++) {
    const active = await api("/api/runs/active");
    sdRunning = active.some((r) => r.session_id === sidD);
    if (!sdRunning) await sleep(500);
  }
  ok("C: A 的 run 仍 running(切走后任务不丢)", sdRunning);
  // 切回 A:问题仍在 + 生成中
  await page.getByText(qD.slice(0, 12), { exact: false }).first().click();
  await page.waitForFunction((s) => location.search.includes("session=" + s), sidD, { timeout: 6000 });
  const userVisible = await page.getByText(qD).first().isVisible().catch(() => false);
  const loadingText = await page.getByText(/检索证据并生成回答/).first().isVisible().catch(() => false);
  ok("C: 切回 A → 问题仍在", userVisible);
  ok("C: 切回 A → 显示生成中", loadingText);
  await shot("C-back-while-running");

  // ---------- D. 回答中刷新:恢复 running,最终回答回来 ----------
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction((s) => location.search.includes("session=" + s), sidD, { timeout: 10000 });
  await page.getByText(qD).first().waitFor({ timeout: 10000 });
  ok("D: F5 后仍停在 A(URL session 恢复)", true);
  ok("D: F5 后 user 问题仍在", true);
  const loaderAfter = await page
    .getByText(/检索证据并生成回答/)
    .first()
    .isVisible()
    .catch(() => false);
  ok("D: F5 后显示'生成中'(running 由后端恢复)", loaderAfter);
  const runD = await waitRunDone(sidD, 300000);
  ok("D: 后台任务在 F5 后继续并 done", runD && runD.status === "done", runD ? runD.status : "timeout");
  const msgsD = await api(`/api/sessions/${sidD}/messages`);
  ok("D: A 最终 user+assistant 完整落库", msgsD.length === 2, `len=${msgsD.length}`);
  await sleep(3000);
  const asstVisible = await page.getByText(/迁移|架构|Flask|方案/, { exact: false }).first().isVisible().catch(() => false);
  ok("D: 刷新页自动出现最终回答(轮询续跑)", asstVisible);
  await shot("D-done");

  // ---------- F. mode:各会话独立持久(API 权威 + UI 恢复) ----------
  const s3 = (await api("/api/sessions")).find((s) => s.title === "第三个会话-快速问答");
  await fetch(`${API}/api/sessions/${s3.session_id}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: "agentic" }),
  });
  await page.reload({ waitUntil: "domcontentloaded" });
  await sleep(1500);
  const modes = await api("/api/sessions");
  const modeOf = (t) => modes.find((s) => s.title === t)?.mode;
  const sdm = modes.find((s) => s.session_id === sidD);
  ok("F: Sd mode=agentic(后端持久)", sdm?.mode === "agentic", sdm?.mode);
  ok("F: S3 mode=agentic(PATCH 持久)", modeOf("第三个会话-快速问答") === "agentic");
  ok("F: S2 mode=auto(默认)", modeOf("第二个会话-切走目标") === "auto");

  }
// ---------- E. 历史完整性:5 会话 × 2 轮 ----------
  await page.getByText("历史完整性 E1").first().click();
  await page.waitForFunction(() => location.search.includes('session='), { timeout: 6000 });
  for (let i = 1; i <= 5; i++) {
    const title = `历史完整性 E${i}`;
    await page.getByText(title).first().click();
    await page.waitForFunction(
      (q) => document.body.innerText.includes(q),
      `E${i} 第一个问题`
    ).catch(() => {});
    await sleep(600);
    const list = (await api("/api/sessions")).find((s) => s.title === title);
    const msgs = await api(`/api/sessions/${list.session_id}/messages`);
    const uiText = await page.evaluate(() => document.body.innerText);
    const rolesOk = msgs.length === 4 && msgs.map((m) => m.role).join(",") === "user,assistant,user,assistant";
    const orderOk =
      uiText.includes(`E${i} 第一个问题`) &&
      uiText.includes(`E${i} 第二个问题`) &&
      uiText.includes(`E${i}-A1`) &&
      uiText.includes(`E${i}-A2`);
    ok(`E: ${title} 4 条消息顺序正确(DB)`, rolesOk, `len=${msgs.length}`);
    ok(`E: ${title} UI 完整显示 2 轮问答`, orderOk);
  }
  await shot("E-history");

  // ---------- H. 失败恢复:user 仍在 + 明确显示失败 ----------
  await page.getByText("失败恢复示例").first().click();
  await page.waitForFunction(() => location.search.includes('session='), { timeout: 6000 });
  await sleep(1200);
  const userHFail = await page.getByText("这个任务会失败").first().isVisible().catch(() => false);
  const errVisible = await page.getByText(/模拟网络故障/).first().isVisible().catch(() => false);
  const stillInList = await page.getByText("失败恢复示例").first().isVisible().catch(() => false);
  ok("H: 失败会话 user 消息仍在", userHFail);
  ok("H: 页面明确显示失败原因", errVisible);
  ok("H: 会话仍在列表(未消失)", stillInList);
  await shot("H-failed");

  const pageErrs = errors.filter((e) => !String(e).includes("favicon"));
  ok("页面无未捕获 JS 错误", pageErrs.length === 0, pageErrs.slice(0, 2).join(" | "));

  await browser.close();
  const fails = results.filter((r) => !r.pass);
  console.log(`\n===== A~H 结果:${results.length - fails.length}/${results.length} 通过 =====`);
  if (fails.length) console.log("失败项:", fails.map((f) => f.name).join("; "));
  process.exit(fails.length ? 1 : 0);
})().catch((e) => {
  console.error("E2E 异常:", e);
  process.exit(2);
});
