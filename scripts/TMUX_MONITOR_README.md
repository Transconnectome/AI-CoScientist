# Tmux Session Monitor

A comprehensive Python tool to monitor and display tmux session information.

## Features

- 📊 List all tmux sessions with status
- 🔍 Detailed view showing windows, panes, and running commands
- 🔄 Real-time monitoring mode with auto-refresh
- 📅 Session creation and last activity timestamps
- 📈 Summary statistics
- 📤 JSON output for scripting
- 🎯 Filter by specific session

## Installation

No installation required! Just make sure you have:
- Python 3.6+
- tmux installed and in your PATH

The script is located at: `scripts/tmux_session_monitor.py`

## Usage

### Basic Usage

```bash
# List all sessions
python3 scripts/tmux_session_monitor.py

# Detailed view with windows and panes
python3 scripts/tmux_session_monitor.py -d

# Monitor mode (auto-refresh every 5 seconds)
python3 scripts/tmux_session_monitor.py -m

# Monitor with custom interval (10 seconds)
python3 scripts/tmux_session_monitor.py -m -i 10

# Show specific session
python3 scripts/tmux_session_monitor.py -s mysession

# JSON output (for scripting)
python3 scripts/tmux_session_monitor.py -j
```

### Examples

```bash
# Quick overview
python3 scripts/tmux_session_monitor.py

# Detailed view of all sessions
python3 scripts/tmux_session_monitor.py -d

# Monitor a specific session
python3 scripts/tmux_session_monitor.py -m -s dd

# Get JSON for automation
python3 scripts/tmux_session_monitor.py -j | jq '.[] | select(.attached == false)'
```

## Output Format

### Standard View

```
================================================================================
📊 Tmux Session Monitor - 2025-12-02 03:39:02
================================================================================

📈 Summary: 8 session(s), 0 attached, 8 window(s)

1. agentreview ⚪ DETACHED
   Windows: 1 | Size: 58x37
   Created: 2025-11-28 22:57:11 (3d 4h ago)
   Last Activity: 2025-11-29 00:03:52 (3d 3h ago)
```

### Detailed View

Shows additional information:
- Window names and indices
- Number of panes per window
- Currently running commands
- Active window indicator

## Command Line Options

| Option | Description |
|--------|-------------|
| `-d, --detailed` | Show detailed information including windows and panes |
| `-m, --monitor` | Monitor mode: continuously refresh display |
| `-i, --interval INTERVAL` | Refresh interval in seconds for monitor mode (default: 5) |
| `-j, --json` | Output as JSON format |
| `-s, --session SESSION` | Show specific session only |
| `-h, --help` | Show help message |

## Use Cases

### 1. Quick Session Check
```bash
python3 scripts/tmux_session_monitor.py
```

### 2. Find Inactive Sessions
```bash
python3 scripts/tmux_session_monitor.py -j | jq '.[] | select(.attached == false) | .name'
```

### 3. Monitor Active Sessions
```bash
python3 scripts/tmux_session_monitor.py -m -i 2
```

### 4. Check Session Details
```bash
python3 scripts/tmux_session_monitor.py -d -s workspace
```

### 5. Integration with Scripts
```bash
#!/bin/bash
SESSIONS=$(python3 scripts/tmux_session_monitor.py -j)
echo "$SESSIONS" | jq 'length'  # Count sessions
```

## Tips

- Use monitor mode (`-m`) to keep track of session activity in real-time
- Combine with `watch` for additional monitoring capabilities
- JSON output is perfect for automation and integration with other tools
- Detailed view (`-d`) is useful when debugging session issues

## Troubleshooting

**No sessions found:**
- Make sure tmux is installed: `tmux -V`
- Check if you have any sessions: `tmux ls`

**Permission errors:**
- Ensure the script is executable: `chmod +x scripts/tmux_session_monitor.py`

**Session not found:**
- List all sessions: `tmux ls`
- Use exact session name (case-sensitive)

## Related Tools

- `tmux ls` - Basic tmux session listing
- `scripts/setup_dgx_tmux.sh` - Tmux setup for DGX servers
- `claudedocs/DGX_TMUX_GUIDE.md` - Comprehensive tmux guide





