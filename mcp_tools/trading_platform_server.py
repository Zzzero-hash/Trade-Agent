
"""MCP server exposing repository-aware tools for the AI trading platform.

The server provides small helpers that make it easier for MCP-compatible
clients to understand outstanding work in the repository. Tools include:
- Listing open implementation tasks from the roadmap markdown file.
- Fetching specific sections from the requirements, design, or tasks specs.
- Searching for TODO markers across the codebase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anyio
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_ROOT = REPO_ROOT / '.kiro' / 'specs' / 'ai-trading-platform'
TASKS_PATH = SPEC_ROOT / 'tasks.md'
REQUIREMENTS_PATH = SPEC_ROOT / 'requirements.md'
DESIGN_PATH = SPEC_ROOT / 'design.md'

TODO_SCOPES: dict[str, Path] = {
    'src': REPO_ROOT / 'src',
    'tests': REPO_ROOT / 'tests',
    'frontend': REPO_ROOT / 'frontend',
    'docs': REPO_ROOT / 'docs',
    'repo': REPO_ROOT,
}

ALLOWED_TODO_EXTENSIONS = {
    '.py', '.md', '.txt', '.yaml', '.yml', '.json', '.ini', '.cfg', '.toml', '.pyi',
    '.ts', '.tsx', '.js', '.jsx', '.css', '.scss', '.html', '.sql', '.sh', '.ps1',
}
MAX_TODO_FILE_BYTES = 1_000_000  # skip very large files when scanning for TODO markers


@dataclass
class TaskEntry:
    title: str
    status: str  # "pending" or "complete"
    level: int


TASK_PATTERN = re.compile(r'^(?P<indent>\s*)-\s*\[(?P<state>[ xX])\]\s*(?P<title>.+)$')


class SpecNotFoundError(ValueError):
    """Raised when a requested spec file or heading cannot be located."""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise SpecNotFoundError(f'File not found: {path}') from exc


def parse_tasks(markdown: str) -> list[TaskEntry]:
    entries: list[TaskEntry] = []
    for raw_line in markdown.splitlines():
        match = TASK_PATTERN.match(raw_line.rstrip())
        if not match:
            continue
        indent = len(match.group('indent'))
        level = indent // 2  # two spaces correspond to one nesting level in the roadmap file
        status = 'complete' if match.group('state').lower() == 'x' else 'pending'
        title = match.group('title').strip()
        entries.append(TaskEntry(title=title, status=status, level=level))
    return entries


def list_tasks(include_completed: bool, max_results: int | None) -> dict[str, Any]:
    markdown = _read_text(TASKS_PATH)
    tasks = parse_tasks(markdown)
    filtered: Iterable[TaskEntry]
    if include_completed:
        filtered = tasks
    else:
        filtered = (task for task in tasks if task.status == 'pending')

    items: list[dict[str, Any]] = []
    for task in filtered:
        items.append({'title': task.title, 'status': task.status, 'level': task.level})
        if max_results is not None and len(items) >= max_results:
            break
    return {'tasks': items, 'source': str(TASKS_PATH.relative_to(REPO_ROOT))}


HEADING_PATTERN = re.compile(r'^(?P<level>#+)\s*(?P<title>.+?)\s*$')


def extract_section(path: Path, heading: str) -> dict[str, str]:
    normalized_heading = heading.strip().lower()
    markdown = _read_text(path)
    lines = markdown.splitlines()
    start_index: int | None = None
    start_level: int | None = None
    for index, raw_line in enumerate(lines):
        match = HEADING_PATTERN.match(raw_line)
        if not match:
            continue
        title = match.group('title').strip().lower()
        if start_index is None:
            if title == normalized_heading:
                start_index = index + 1
                start_level = len(match.group('level'))
        else:
            if len(match.group('level')) <= (start_level or 0):
                end_index = index
                break
    else:
        if start_index is None:
            raise SpecNotFoundError(f'Heading "{heading}" not found in {path.name}')
        end_index = len(lines)

    if start_index is None:
        raise SpecNotFoundError(f'Heading "{heading}" not found in {path.name}')

    section_lines = lines[start_index:end_index]
    content = "
".join(section_lines).strip()
    return {
        'heading': heading,
        'content': content,
        'source': str(path.relative_to(REPO_ROOT)),
    }


def resolve_spec_file(key: str) -> Path:
    key_lower = key.lower()
    if key_lower == 'tasks':
        return TASKS_PATH
    if key_lower == 'requirements':
        return REQUIREMENTS_PATH
    if key_lower == 'design':
        return DESIGN_PATH
    raise SpecNotFoundError(f'Unknown spec file "{key}". Choose from tasks, requirements, or design.')


def find_todo_markers(scope_key: str, limit: int) -> dict[str, Any]:
    scope_key_lower = scope_key.lower()
    base_path = TODO_SCOPES.get(scope_key_lower)
    if base_path is None:
        raise ValueError(f'Unknown scope "{scope_key}". Valid options: {", ".join(sorted(TODO_SCOPES))}')
    if not base_path.exists():
        raise ValueError(f'Scope path {base_path} does not exist on disk')

    results: list[dict[str, Any]] = []
    for file_path in base_path.rglob('*'):
        if not file_path.is_file() or file_path.is_symlink():
            continue
        if file_path.suffix and file_path.suffix.lower() not in ALLOWED_TODO_EXTENSIONS:
            continue
        try:
            if file_path.stat().st_size > MAX_TODO_FILE_BYTES:
                continue
        except OSError:
            continue

        try:
            content = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            if 'TODO' in raw_line:
                results.append(
                    {
                        'file': str(file_path.relative_to(REPO_ROOT)),
                        'line': line_number,
                        'text': raw_line.strip(),
                    }
                )
                if len(results) >= limit:
                    return {'results': results, 'limit': limit, 'scope': scope_key_lower}
    return {'results': results, 'limit': limit, 'scope': scope_key_lower}


server = Server(
    name='trade-agent-mcp-tools',
    instructions='Helper tools for exploring outstanding work in the AI trading platform repository.',
)


@server.list_tools()
async def _list_tools() -> list[Tool]:
    return [
        Tool(
            name='list_open_tasks',
            title='List open implementation tasks',
            description='Return pending (or all) tasks parsed from tasks.md.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'include_completed': {'type': 'boolean', 'default': False},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
                'additionalProperties': False,
            },
            outputSchema={
                'type': 'object',
                'properties': {
                    'tasks': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'title': {'type': 'string'},
                                'status': {'type': 'string', 'enum': ['pending', 'complete']},
                                'level': {'type': 'integer', 'minimum': 0},
                            },
                            'required': ['title', 'status', 'level'],
                            'additionalProperties': False,
                        },
                    },
                    'source': {'type': 'string'},
                },
                'required': ['tasks', 'source'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='get_spec_section',
            title='Read a spec section',
            description='Fetch a heading section from design.md, requirements.md, or tasks.md.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'file': {
                        'type': 'string',
                        'enum': ['design', 'requirements', 'tasks'],
                    },
                    'heading': {'type': 'string'},
                },
                'required': ['file', 'heading'],
                'additionalProperties': False,
            },
            outputSchema={
                'type': 'object',
                'properties': {
                    'heading': {'type': 'string'},
                    'content': {'type': 'string'},
                    'source': {'type': 'string'},
                },
                'required': ['heading', 'content', 'source'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='find_todos',
            title='Search for TODO markers',
            description='Scan the repository for TODO comments within a scope (src, tests, frontend, docs, or repo).',
            inputSchema={
                'type': 'object',
                'properties': {
                    'scope': {
                        'type': 'string',
                        'default': 'src',
                        'enum': sorted(TODO_SCOPES.keys()),
                    },
                    'limit': {
                        'type': 'integer',
                        'default': 20,
                        'minimum': 1,
                        'maximum': 200,
                    },
                },
                'additionalProperties': False,
            },
            outputSchema={
                'type': 'object',
                'properties': {
                    'results': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'file': {'type': 'string'},
                                'line': {'type': 'integer'},
                                'text': {'type': 'string'},
                            },
                            'required': ['file', 'line', 'text'],
                            'additionalProperties': False,
                        },
                    },
                    'limit': {'type': 'integer'},
                    'scope': {'type': 'string'},
                },
                'required': ['results', 'limit', 'scope'],
                'additionalProperties': False,
            },
        ),
    ]


@server.call_tool()
async def _call_tool(tool_name: str, arguments: dict[str, Any] | None):
    args = arguments or {}
    if tool_name == 'list_open_tasks':
        include_completed = bool(args.get('include_completed', False))
        max_results_raw = args.get('max_results')
        max_results: int | None
        if max_results_raw is None:
            max_results = None
        else:
            max_results = int(max_results_raw)
            if max_results <= 0:
                raise ValueError('max_results must be a positive integer when provided')
            if max_results > 200:
                raise ValueError('max_results must not exceed 200')
        return list_tasks(include_completed, max_results)

    if tool_name == 'get_spec_section':
        file_key = str(args.get('file', '')).strip()
        heading = str(args.get('heading', '')).strip()
        if not file_key or not heading:
            raise ValueError('Both "file" and "heading" must be provided')
        path = resolve_spec_file(file_key)
        return extract_section(path, heading)

    if tool_name == 'find_todos':
        scope = str(args.get('scope', 'src')).strip() or 'src'
        limit = int(args.get('limit', 20))
        if limit <= 0:
            raise ValueError('limit must be a positive integer')
        if limit > 200:
            raise ValueError('limit must not exceed 200')
        return find_todo_markers(scope, limit)

    raise ValueError(f'Unknown tool "{tool_name}"')


async def main() -> None:
    initialization_options = server.create_initialization_options(NotificationOptions())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, initialization_options)


if __name__ == '__main__':
    anyio.run(main)
