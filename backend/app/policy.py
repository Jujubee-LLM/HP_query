import re


_SENSITIVE_RE = re.compile(
    r"("
    r"系统提示词|system\s*prompt|developer\s*message|开发者消息|"
    r"提示词|prompt\s*injection|越狱|jailbreak|"
    r"请输出.*提示词|把.*提示词.*输出|"
    r"你.*(规则|指令).*内容|"
    r"secret|api\s*key|密钥|口令|密码"
    r")",
    re.IGNORECASE,
)


def should_block(question: str) -> tuple[bool, str | None]:
    """
    Block requests that attempt to exfiltrate system/developer prompts or other secrets.
    We should not even run retrieval for these, to avoid confusing users with irrelevant evidence.
    """
    q = (question or "").strip()
    if not q:
        return False, None
    if _SENSITIVE_RE.search(q):
        return True, "sensitive_exfiltration"
    return False, None

