# ForgeDesk Architecture

Technical architecture documentation for developers contributing to ForgeDesk.

> **📌 正在阅读本文档的智能体 / 读者请先看这里。**
>
> 若你的任务涉及 **GPU / 本地 AI 加速 / JAX / ROCm / AMD / iGPU / 推理**，
> 请先阅读 **[ARCHITECTURE_ROCM_JAX.md](ARCHITECTURE_ROCM_JAX.md)** —— 那是一份
> 独立的、背景完整的架构设计提案，涵盖：
>
> - AMD ROCm 10 + JAX 集成的开发背景与目标硬件（Strix Halo / gfx1151）约束；
> - 上游现状：ROCm 10 JAX 的 Python 版本支持（**只有 GIL wheel**）、gfx1151 的
>   支持状态、以及 **三个未修复缺陷**（引用 `ROCm/jax` PR #887）；
> - `ROCm/jax#871` 的 50.5 GB 内存问题的根因分析与两级解法；
> - 由此引申出的 **worker 进程模型**架构改造（主架构图 + IPC 通信流程图）；
> - 新架构相对本仓库的**待开发模块清单**与 IPC 流程对应关系。
>
> 本文档（ARCHITECTURE.md）描述的是**当前已实现的通用框架架构**；
> `ARCHITECTURE_ROCM_JAX.md` 描述的是**尚未实现的扩展设计**。两者不要混淆。

## System Overview

ForgeDesk is a hybrid Python/Rust desktop application framework that leverages OS-native WebViews for rendering UI. It follows a layered architecture with clear separation of concerns.

```
┌─────────────────────────────────────────────────────────────┐
│                    JavaScript Frontend                       │
│  (React, Vue, Svelte, Next.js, Astro, etc.)                │
├─────────────────────────────────────────────────────────────┤
│                    @forgedesk/api (TypeScript)               │
│  Typed IPC client for invoking Python commands               │
├─────────────────────────────────────────────────────────────┤
│                    IPC Bridge (msgspec JSON)                 │
│  Command dispatch, validation, error handling                │
├─────────────────────────────────────────────────────────────┤
│                    Python Backend (forge.*)                   │
│  Commands, events, state, plugins, APIs                      │
├─────────────────────────────────────────────────────────────┤
│                    Rust Core (forge_core)                     │
│  Native window, WebView, platform integrations               │
├─────────────────────────────────────────────────────────────┤
│                    OS Native (Tao/Wry)                        │
│  Window management, WebView rendering                        │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
forge-framework/
├── ARCHITECTURE.md           # This document — general framework architecture
├── ARCHITECTURE_ROCM_JAX.md  # Proposed design — AMD ROCm 10 + JAX worker tier
├── forge/                    # Python framework core
│   ├── __init__.py          # Package entry point
│   ├── app.py               # ForgeApp main class
│   ├── bridge.py            # IPC bridge (Python↔JS)
│   ├── config.py            # Configuration parsing
│   ├── events.py            # Event emitter system
│   ├── state.py             # Dependency injection
│   ├── window.py            # Window management
│   ├── router.py            # Command routing
│   ├── scope.py             # Security scope validation
│   ├── recovery.py          # Error recovery (circuit breaker)
│   ├── memory.py            # Zero-copy memory buffers
│   ├── plugins.py           # Plugin system
│   ├── tasks.py             # Background tasks
│   ├── channels.py          # Pub/sub channels
│   ├── asgi.py              # ASGI server integration
│   ├── logging.py           # Structured logging
│   ├── diagnostics.py       # Runtime diagnostics
│   ├── typegen.py           # TypeScript type generation
│   ├── support.py           # Utility functions
│   ├── api/                 # High-level API modules
│   │   ├── fs.py            # File system operations
│   │   ├── dialog.py        # Native dialogs
│   │   ├── clipboard.py     # Clipboard access
│   │   ├── shell.py         # Shell command execution
│   │   ├── notification.py  # Desktop notifications
│   │   ├── menu.py          # Application menus
│   │   ├── tray.py          # System tray
│   │   ├── shortcuts.py     # Global shortcuts
│   │   ├── updater.py       # Auto-updates
│   │   ├── keychain.py      # Secure storage
│   │   ├── screen.py        # Monitor info
│   │   ├── power.py         # Battery state
│   │   └── ...              # 20+ more API modules
│   ├── builtins/            # Built-in plugins
│   │   ├── ai_ml.py         # Cloud AI integration
│   │   ├── llm_local.py     # Local LLM inference
│   │   ├── database.py      # Database operations
│   │   ├── crypto.py        # Cryptography
│   │   ├── network.py       # HTTP client
│   │   ├── file_watch.py    # File watching
│   │   └── ...              # 15+ more plugins
│   ├── js/                  # JavaScript runtime
│   │   ├── forge.js         # Frontend IPC bridge
│   │   └── typings.py       # TypeScript generation
│   └── cli/                 # CLI tools
│       ├── main.py          # CLI entry point
│       ├── bundler.py       # Build pipeline
│       └── manifests.py     # Installer manifests
├── src/                     # Rust native core
│   ├── lib.rs               # PyO3 module entry
│   ├── native_window.rs     # Window + WebView management
│   ├── events.rs            # Event system (UserEvent enum)
│   ├── updater.rs           # Self-updater with Ed25519
│   ├── window/              # Window subsystem
│   │   ├── mod.rs           # WindowDescriptor, RuntimeWindow
│   │   ├── builder.rs       # WebView builder, protocol handlers
│   │   └── proxy.rs         # WindowProxy (thread-safe handle)
│   ├── platform/            # Platform-specific code
│   │   ├── clipboard.rs     # Clipboard bindings
│   │   ├── dialog.rs        # Native dialog bindings
│   │   ├── keychain.rs      # Keychain bindings
│   │   ├── menu.rs          # Menu bindings
│   │   ├── notification.rs  # Notification bindings
│   │   ├── vibrancy.rs      # Window transparency
│   │   └── ...              # More platform modules
│   └── menu/                # Menu system
│       ├── mod.rs           # Menu types
│       └── linux.rs         # GTK menu implementation
├── packages/                # NPM packages
│   ├── api/                 # @forgedesk/api (TypeScript SDK)
│   ├── vite-plugin/         # Vite integration
│   ├── cli/                 # CLI wrapper
│   └── create-forge-app/    # Project scaffolding
├── tests/                   # Python test suite
├── docs/                    # Documentation site (Astro)
├── examples/                # Example applications
└── scripts/                 # CI/CD scripts
```

## Data Flow

### IPC Communication Flow

```
1. Frontend (JS)
   └── window.__forge__.invoke("command", args)
       └── WebSocket/HTTP message

2. Rust Core (native_window.rs)
   └── IPC handler receives message
       └── Calls Python callback via PyO3

3. Python Bridge (bridge.py)
   └── Receives JSON message
       └── Validates command name
       └── Checks capabilities
       └── Dispatches to handler

4. Command Handler
   └── Executes business logic
       └── Returns result

5. Response Flow
   └── Python → JSON → Rust → WebSocket → Frontend
```

**Note:** Steps 1–5 above are the flow as currently implemented. A proposed
**worker process tier** extends this with a second, separate transport
(msgspec JSON frames over a worker's stdin/stdout) plus lifecycle events
flowing back through `forge/tasks.py`. That design — including its rationale,
its upstream constraints, and the modules it would require — is specified in
[ARCHITECTURE_ROCM_JAX.md](ARCHITECTURE_ROCM_JAX.md). The two channels have
distinct responsibilities and must not be conflated:

| Channel | Carries | Status |
|---|---|---|
| Webview ↔ main process | UI commands, results (WS/HTTP) | Implemented |
| Main process ↔ worker | Inference request/response (stdin/stdout frames) | Proposed |
| `forge/tasks.py` events | Lifecycle and event flow-back | Implemented |

The proposed worker-tier call sequence:

```mermaid
sequenceDiagram
    autonumber
    participant UI as Webview
    participant BR as forge.bridge<br/>(主进程)
    participant TM as forge.tasks<br/>TaskManager
    participant SUP as worker 监管器
    participant LD as bin/jax-worker<br/>(启动器)
    participant WK as jax worker<br/>(子进程)
    participant GPU as iGPU gfx1151

    UI->>BR: invoke("jax_infer", {payload})
    BR->>TM: task_start("jax-infer")
    TM->>SUP: 请求 spawn worker
    SUP->>LD: Popen([bin/jax-worker]) + 加入 cgroup
    Note over LD: 顺序有约束：先 cgroup join，<br/>再 exec；绝不能 Popen 之后再补写 cgroup.procs
    LD->>LD: cgroup join → 设 XLA_FLAGS /<br/>XLA_CLIENT_MEM_FRACTION → exec
    LD->>WK: exec(worker)  ← 此处才 import jax
    WK->>GPU: 首次 device_put → 预留量 = 0.75 × total ⚠️
    SUP->>WK: stdin ← msgspec JSON 请求帧

    Note over WK,GPU: total 被 APU 高估（VRAM+GTT 重复计数）<br/>默认 75% ≈ 50GB；fraction 生效后 ≈ 3-16GB<br/>未生效时由 MemoryMax 兜底 → worker 被 OOM kill

    WK->>GPU: XLA 编译 / 执行
    GPU-->>WK: 结果
    WK->>SUP: stdout ← msgspec JSON 响应帧
    SUP->>TM: 完成回调
    TM-->>BR: emit("task:completed", {task_id, result})
    BR-->>UI: 响应 + 进度事件
    UI->>UI: 与 CPU 参考值比对 ⚠️ gfx1151 静默错误

    Note over SUP,WK: 停止路径：task_cancel → SIGTERM →<br/>exit_files() → filp_close(/dev/kfd) →<br/>KFD GPU VM 销毁 → amdgpu_bo_unref →<br/>VRAM buffer 归还 → 50GB 同步回到系统 ⚠️
```

### Zero-Copy Memory Transfer

For large data (images, datasets, models):

```
1. Python puts data in forge.memory.buffers
   └── memory.buffers[uuid] = bytes_data

2. Python returns forge-memory:// URL
   └── result = {"__forge_blob": "forge-memory://uuid"}

3. Frontend fetches via custom protocol
   └── fetch("forge-memory://uuid").then(r => r.arrayBuffer())

4. Rust intercepts custom protocol
   └── Reads from Python memory.buffers via PyO3
   └── Returns raw bytes without JSON serialization
   └── Auto-deletes buffer after read
```

## Key Components

### IPCBridge (bridge.py)

The IPC bridge handles all communication between frontend and backend.

**Features:**
- Command name validation (prevents injection)
- Error sanitization (strips paths, limits length)
- Request size limits (10MB max)
- Async command support (coroutines)
- Thread pool dispatch (NoGIL parallel execution)
- Binary auto-routing (bytes → forge-memory://)

**Protocol:**
```json
// Request
{
    "command": "greet",
    "id": 1,
    "args": {"name": "World"}
}

// Response
{
    "type": "reply",
    "protocol": "1.0",
    "id": 1,
    "result": {"message": "Hello, World!"},
    "error": null,
    "error_code": null
}
```

### EventEmitter (events.py)

Thread-safe event system for component communication.

**Features:**
- Sync and async listeners
- Thread-safe mutations
- Event filtering by name
- Automatic cleanup

**Usage:**
```python
# Register
app.events.on("ready", handler)
app.events.on("data", async_handler)

# Emit
app.events.emit("ready", {"status": "ok"})

# Remove
app.events.off("ready", handler)
```

### AppState (state.py)

Type-safe dependency injection container.

**Features:**
- Single instance per type
- Thread-safe access
- Type-hint based injection
- Diagnostic snapshots

**Usage:**
```python
# Register
app.state.manage(Database("sqlite:///app.db"))

# Retrieve
db = app.state.get(Database)

# Check
service = app.state.try_get(MissingService)  # None
```

### ScopeValidator (scope.py)

Security module for path and URL access control.

**Features:**
- Glob pattern matching
- Deny-overrides-allow semantics
- Symlink resolution
- Environment variable expansion
- Monitor bounds validation

**Usage:**
```python
validator = ScopeValidator(
    allow_patterns=["/home/user/appdata/**"],
    deny_patterns=["/home/user/appdata/secret/**"]
)

validator.is_path_allowed("/home/user/appdata/config.json")  # True
validator.is_path_allowed("/home/user/appdata/secret/key.pem")  # False
```

### CircuitBreaker (recovery.py)

Prevents cascading failures by temporarily disabling failing commands.

**Features:**
- Per-command failure tracking
- Configurable threshold and cooldown
- Half-open state for testing
- Thread-safe state management

**States:**
- **Closed**: Normal operation, failures counted
- **Open**: Failures exceeded threshold, all calls rejected
- **Half-Open**: Cooldown elapsed, one test call allowed

## Rust Core

### NativeWindow (native_window.rs)

Manages the native window and WebView using Tao/Wry.

**Features:**
- Multi-window support (HashMap<WindowId, RuntimeWindow>)
- Custom protocol handlers (forge://, forge-asset://, forge-memory://)
- IPC callback via PyO3
- Platform-specific decorations (GTK on Linux)
- Global hotkey support
- Native menu integration (muda on macOS/Windows)

### WindowProxy (window/proxy.rs)

Thread-safe handle for sending commands to the event loop.

**Features:**
- All methods are Send+Sync
- Non-blocking event sending
- Label-targeted operations
- Python bindings via PyO3

### UserEvent (events.rs)

Cross-thread communication events.

**Event Types:**
- `Eval(label, script)` - Execute JavaScript
- `LoadUrl(label, url)` - Navigate WebView
- `SetTitle(label, title)` - Update window title
- `Resize(label, w, h)` - Resize window
- `CreateWindow(descriptor)` - Create new window
- `CloseLabel(label)` - Close window
- `ApplyUpdate(url, sig, pubkey)` - Full update
- `ApplyDeltaUpdate(patch_url, ...)` - Delta update

## Build System

### Maturin (Python + Rust)

Primary build tool for hybrid Python/Rust packages.

```bash
# Development build
maturin develop

# Release build
maturin build --release

# Build for specific target
maturin build --release --target x86_64-unknown-linux-gnu
```

### Nuitka (Python-only)

Alternative for pure Python builds.

```bash
# Standalone build
python -m nuitka --standalone --lto=yes src/main.py
```

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_app.py              # ForgeApp tests
├── test_bridge.py           # IPC bridge tests
├── test_events.py           # Event system tests
├── test_state.py            # State management tests
├── test_window.py           # Window API tests
├── test_security.py         # Security tests
├── test_e2e_lifecycle.py    # End-to-end integration tests
└── ...                      # 50+ test files
```

### Running Tests

```bash
# All tests
uv run pytest -v

# Specific test file
uv run pytest tests/test_bridge.py -v

# With coverage
uv run pytest --cov=forge --cov-report=html

# Rust tests
cargo test --all-features
```

## Performance Characteristics

### Binary Size

| Component | Size |
|-----------|------|
| Rust core | ~2MB |
| Python stdlib | ~15MB |
| WebView (OS-provided) | 0MB |
| Application code | Variable |
| **Total (typical)** | **20-30MB** |

### Memory Usage

| State | RAM |
|-------|-----|
| Idle | ~30MB |
| Active (1 window) | ~50MB |
| Active (3 windows) | ~80MB |
| With LLM loaded | +200-500MB |

### IPC Latency

| Operation | Latency |
|-----------|---------|
| Simple command | <1ms |
| File read (1KB) | <1ms |
| JSON serialization (1MB) | ~5ms |
| Binary transfer (1MB) | <1ms (zero-copy) |

## Security Model

### Capability System

```toml
[permissions]
filesystem = true   # Allow file operations
shell = false       # Deny shell execution
clipboard = true    # Allow clipboard access
keychain = false    # Deny keychain access
```

### Scope Validation

```toml
[security.fs]
allow = ["$APPDATA/myapp/**", "$DOCUMENTS/myapp/**"]
deny = ["$APPDATA/myapp/secrets/**"]
```

### IPC Security

- Command name validation (regex: `^[a-zA-Z_][a-zA-Z0-9_]*$`)
- Request size limit (10MB)
- Error message sanitization (strips paths)
- Origin validation (configurable)

## Thread Model

```
Main Thread (Rust event loop)
├── Window events
├── WebView rendering
└── IPC callback dispatch

Python Thread Pool (NoGIL)
├── Command execution
├── Background tasks
└── Event handlers

Async Thread (Tokio)
├── HTTP requests
├── WebSocket handling
└── File I/O
```

> **Note:** A proposed **worker process tier** would add a separate OS-process
> dimension to this model — heavy backends (e.g. JAX on ROCm) run in dedicated
> child processes with their own cgroup limits, so that a backend's memory
> footprint is bounded and reclaimable by process teardown. See
> [ARCHITECTURE_ROCM_JAX.md](ARCHITECTURE_ROCM_JAX.md).

## Extension Points

### Custom Commands

```python
@app.bridge.register_command("my_command")
def my_command(arg1: str, arg2: int) -> dict:
    return {"result": "..."}
```

### Custom Plugins

```python
class MyPlugin:
    __forge_capability__ = "my_plugin"
    
    def __init__(self, app):
        self._app = app
    
    def my_method(self) -> dict:
        return {"ok": True}

def register(app):
    app.bridge.register_commands(MyPlugin(app))
```

### Custom Protocols

```rust
// In builder.rs
webview_builder = webview_builder.with_asynchronous_custom_protocol(
    "my-protocol".into(),
    move |_webview_id, request, responder| {
        // Handle custom protocol
    }
);
```

## Dependencies

### Python

| Package | Purpose |
|---------|---------|
| `pyo3` | Rust-Python bindings |
| `msgspec` | Fast JSON serialization |
| `typer` | CLI framework |
| `rich` | Terminal formatting |
| `uvicorn` | ASGI server |
| `cryptography` | Ed25519 signatures |
| `watchfiles` | File watching |

### Rust

| Crate | Purpose |
|-------|---------|
| `pyo3` | Python bindings |
| `tao` | Window management |
| `wry` | WebView rendering |
| `serde` | Serialization |
| `tokio` | Async runtime |
| `ed25519-dalek` | Signature verification |
| `bsdiff` | Delta updates |
