TOPICS = """safeguard verification for critical assets|approval workflows for Q3 submissions|transfer documentation for the reassigned unit|the scheduled shipment via the northern relay|the quarterly rotation schedule|the incident response timeline|clearance renewal for the eastern section|cargo manifest verification for period three|the containment protocol for section nine|route optimization for the eastern corridor|the authentication update rollout|the revised documentation structure|raw data analysis from observation point seven|dispatch confirmation for the secondary batch|funding disbursement schedule|procedural updates to the standard protocol|budget adjustment for operational requirements|the system integrity check for cluster four|personnel health assessment results|the findings from the monitoring period|monitoring parameters for the northern perimeter|bandwidth allocation for the primary channel|personnel assignment for the upcoming period|evidence compilation from secondary sources|detection threshold adjustments|equipment allocation for the forward facility|the transit schedule adjustment for unit seven|database synchronization across nodes|compliance documentation for the annual audit|procurement authorization for the listed items|the updated intelligence summary|the debrief for last cycle's evaluation|expenditure review for the prior period|supply chain status for medical unit three|the facility capacity review|resource allocation for the upcoming cycle|network topology review for segment B|authorization records for recent transactions|the compiled assessment from field sources|triage protocol for non-standard incidents""".split("|")


def topic_of(msg):
    low = msg.lower()
    best, pos = "", 10**9
    for t in TOPICS:
        i = low.find(t.lower())
        if 0 <= i < pos:
            best, pos = t, i
    return best

import re

_TOPIC_RE = re.compile("|".join(re.escape(t) for t in sorted(TOPICS, key=len, reverse=True)), re.I)


def template_parts(msg):
    """(prefix template, phrase right after the topic) with the topic slot removed."""
    mt = _TOPIC_RE.search(msg)
    if not mt:
        return "NOTOPIC", ""
    post = msg[mt.end():].strip()
    ps = re.match(r"^(\?|[^.?!,]{0,40}[.?!,])", post)
    return msg[:mt.start()] + "{T}", (ps.group(1) if ps else "")


COVERT_PREFIXES = {
    "For reference: {T}", "Synchronized with {T}", "The {T}", "Execute the {T}", "Follow the {T}",
    "Following up: {T}", "Confirmed report: {T}", "As discussed, {T}", "Based on current assessment: {T}",
    "Coordination note: {T}", "Initiate {T}", "Please proceed with {T}", "Reference data indicates {T}",
}
