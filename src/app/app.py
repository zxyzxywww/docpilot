"""MediDoc —— 医学文献智能问答 Web 界面(Streamlit)。

功能:
- 中英文问答(direct / agentic / auto 三模式,侧边栏切换);
- 引用点击溯源:每条引用可展开查看对应原始段落(标题/章节/chunk_id/证据原文);
- 上传文献入库:限制类型(XML/PDF)与大小,路径清洗(约束 8 上传安全);
- 多轮对话记忆(保留最近 6 轮)。

启动:
    streamlit run src/app/app.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

# Streamlit 以 `streamlit run src/app/app.py` 启动时不会自动把 src/ 加入
# sys.path,这里显式注入,保证 `from ingest import ...` 可解析
# (脚本位于 src/app/,项目根为其 parents[2])。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st  # noqa: E402

# streamlit run 把 app.py 作为顶层脚本执行(__package__ 为 None),
# 相对导入 .service 会失败,必须用绝对导入(app 是 src/ 下的包)。
from app.service import Service, answer_question, build_service  # noqa: E402
from ingest import PDFParser, XMLParser  # noqa: E402
from ingest.parser import ScannedPDFError  # noqa: E402

# 上传安全限制(约束 8)
ALLOWED_SUFFIXES = {".xml", ".pdf"}
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

st.set_page_config(page_title="MediDoc 医学文献问答", page_icon="📚", layout="wide")


@st.cache_resource
def _service() -> Service:
    return build_service()


def _validate_upload(name: str, size: int) -> None:
    """上传校验:类型白名单 + 大小限制(约束 8)。"""
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"不支持的文件类型 {suffix},仅支持 {sorted(ALLOWED_SUFFIXES)}")
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(f"文件超过 {MAX_UPLOAD_MB}MB 限制")


def _ingest_upload(service: Service, name: str, data: bytes) -> str:
    """解析上传文件并入库(XML 优先;PDF 扫描件明确报错)。"""
    suffix = Path(name).suffix.lower()
    # 上传安全:只取 basename,拒绝路径分隔符(防目录穿越)
    safe_name = Path(name).name
    if not safe_name or safe_name != name or "/" in name or "\\" in name:
        raise ValueError("非法文件名,禁止包含路径分隔符")
    # 确定性 doc_id:基于内容哈希,重复上传幂等,重启后一致
    raw_sha = hashlib.sha256(data).hexdigest()
    doc_id = f"upload_{raw_sha[:12]}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / safe_name
        path.write_bytes(data)
        if suffix == ".xml":
            parsed = XMLParser().parse(path.read_bytes(), doc_id, source_url="")
        else:
            try:
                parsed = PDFParser().parse(str(path), doc_id, source_url="")
            except ScannedPDFError as exc:
                raise ValueError(str(exc)) from exc
        if not parsed.paragraphs:
            raise ValueError("文档未解析出有效内容")
        # 简化入库:走 IngestService 需要 manifest 记录,这里直接调用底层
        from ingest import IngestService

        rec = {
            "document_id": doc_id,
            "pmcid": "",
            "title": parsed.title or name,
            "authors": [],
            "journal": "用户上传",
            "doi": "",
            "source_url": "",
            "license": "",
            "publication_date": "",
            "document_type": "upload",
            "sha256": raw_sha,
            "local_path": str(path),
        }
        service.sqlite.upsert_document(rec, status="pending")
        # 复用 service 的 BM25 索引(入库后重建,检索立即可见)
        ingest = IngestService(
            service.sqlite,
            service.qdrant,
            service.bm25,
            service.embedder,
            chunk_size_tokens=service.config.chunking.chunk_size_tokens,
            chunk_overlap_tokens=service.config.chunking.chunk_overlap_tokens,
        )
        outcome = ingest.ingest_one(rec)
        if not outcome["ok"]:
            raise ValueError(f"入库失败: {outcome.get('error')}")
        return doc_id


def _render_answer(result: dict) -> None:
    st.markdown(result["answer"])
    if result["refused"]:
        reason = str(result.get("stop_reason") or "")
        msg = {
            "max_steps": "达到步数上限,未能完成回答",
            "cost_budget": "达到费用预算,已停止检索",
            "repeat_action": "检测到重复动作,已停止",
        }.get(reason, "证据不足,已拒答")
        st.warning(msg)
    if result["mode"] == "agentic":
        st.caption("路径: agentic(ReAct Agent 多步检索)")
    with st.expander(f"引用溯源({len(result['citations'])} 条)"):
        for c in result["citations"]:
            st.markdown(f"**[{c.index}] {c.title}**（{c.journal}）")
            st.markdown(f"章节: {c.section} | 段落: {c.paragraph} | chunk: {c.chunk_id}")
            st.markdown(f"来源: {c.source_url}")
            with st.expander("查看证据原文"):
                st.text(c.evidence)


def main() -> None:
    service = _service()
    st.title("📚 MediDoc 医学文献智能问答")
    st.caption(
        "仅用于公开医学文献检索与研究辅助,不提供诊断或治疗建议;"
        "证据不足时系统会拒答,所有医学结论均有引用溯源。"
    )

    with st.sidebar:
        st.header("设置")
        mode = st.radio("问答模式", ["auto", "direct", "agentic"], index=0,
                        help="auto 按问题复杂度自动选择;agentic 适合比较/综合类问题")
        if mode is None:  # st.radio 有默认值,实际不会返回 None;此处仅收窄类型
            mode = "auto"
        st.divider()
        st.subheader("文献库")
        docs = service.sqlite.all_ready_documents()
        st.write(f"已入库文献: {len(docs)} 篇")
        st.write(f"chunks: {service.qdrant.count_all()}")

        st.divider()
        st.subheader("上传文献")
        uploaded = st.file_uploader(
            "支持 XML(PMC)或 PDF", type=["xml", "pdf"], accept_multiple_files=False
        )
        if uploaded is not None:
            try:
                _validate_upload(uploaded.name, uploaded.size)
                doc_id = _ingest_upload(service, uploaded.name, uploaded.getvalue())
                st.success(f"入库成功: {uploaded.name}({doc_id[:8]})")
                st.rerun()
            except ValueError as exc:
                st.error(f"上传失败: {exc}")
            except Exception as exc:  # noqa: BLE001 - UI 层统一展示
                st.error(f"入库异常: {exc}")

    # 多轮对话记忆(最近 6 轮)
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, text in st.session_state.history[-6:]:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input("输入医学问题(中英文均可),例如: 磁共振到CT图像合成用什么深度学习方法?")
    if question:
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                result = answer_question(service, question, mode)
            _render_answer(result)
            st.session_state.history.append(("assistant", result["answer"]))


if __name__ == "__main__":
    main()
