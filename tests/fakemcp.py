"""A minimal stdio MCP server for tests and demos (newline-delimited JSON-RPC, like the MCP stdio transport).

Tools:
    lookup_issue(id)   returns a fake issue tracker entry
    fail()             always returns an MCP tool error (isError: true)
    screenshot()       returns a 1x1 PNG image (like a browser MCP server's screenshot)

    python tests/fakemcp.py [--delay SECONDS]    (delay makes calls look like real network work in demos)
"""

import json
import sys
import time

DELAY = float(sys.argv[sys.argv.index("--delay") + 1]) if "--delay" in sys.argv else 0.0
TOOLS = [
    {"name": "lookup_issue", "description": "Look up an issue in the tracker by id.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "screenshot", "description": "Take a screenshot of the page.",
     "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True}},
]
PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


def result(rid, value):
    return {"jsonrpc": "2.0", "id": rid, "result": value}


def handle(msg):
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if method == "initialize":
        return result(rid, {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake-tracker", "version": "1.0"}})
    if method == "tools/list":
        return result(rid, {"tools": TOOLS})
    if method == "tools/call":
        time.sleep(DELAY)
        name, args = params.get("name"), params.get("arguments") or {}
        if name == "lookup_issue":
            text = f"Issue {args.get('id')}: divide() returns wrong results. Reported by QA. Priority: high."
            return result(rid, {"content": [{"type": "text", "text": text}]})
        if name == "screenshot":
            return result(rid, {"content": [{"type": "text", "text": "Screenshot of http://localhost:3000"},
                                             {"type": "image", "data": PNG_1X1, "mimeType": "image/png"}]})
        if name == "fail":
            return result(rid, {"content": [{"type": "text", "text": "tracker is down"}], "isError": True})
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown tool {name}"}}
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    return None  # notifications need no answer


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
