# DGX Spark - Tmux Quick Reference Guide

## 🚀 Initial Setup

```bash
# Run the setup script (only needed once)
./scripts/setup_dgx_tmux.sh
```

This will:
- Copy tmux configuration to DGX server
- Install tmux if not already present
- Set up optimized settings

---

## 📋 Basic Usage

### Connect and Start Session
```bash
ssh dgx-spark
tmux new -s mysession    # Create new session
tmux new -s work         # Example: 'work' session
```

### Detach (Leave Running)
- Press: `Ctrl+B`, then `D`
- Session continues running in background
- Safe to close terminal

### Reconnect to Session
```bash
ssh dgx-spark
tmux ls                  # List all sessions
tmux a -t mysession      # Attach to 'mysession'
tmux a -t work           # Attach to 'work'
```

### Kill Session
```bash
tmux kill-session -t mysession    # Kill specific session
# OR: Just type 'exit' inside tmux
```

---

## ⌨️ Essential Keyboard Shortcuts

### Prefix Key
- All commands start with: `Ctrl+B`
- Then press the second key

### Session Management
| Shortcut | Action |
|----------|--------|
| `Ctrl+B`, `D` | **Detach** from session (most important!) |
| `Ctrl+B`, `$` | Rename current session |
| `Ctrl+B`, `(` | Switch to previous session |
| `Ctrl+B`, `)` | Switch to next session |

### Window Management
| Shortcut | Action |
|----------|--------|
| `Ctrl+B`, `C` | Create new window |
| `Ctrl+B`, `N` | Next window |
| `Ctrl+B`, `P` | Previous window |
| `Ctrl+B`, `0-9` | Jump to window number |
| `Ctrl+B`, `,` | Rename current window |
| `Ctrl+B`, `&` | Kill current window |

### Pane Splitting (Custom Shortcuts)
| Shortcut | Action |
|----------|--------|
| `Ctrl+B`, `|` | Split vertically (side-by-side) |
| `Ctrl+B`, `-` | Split horizontally (top-bottom) |
| `Alt+Arrow` | Switch panes (NO Ctrl+B needed!) |
| `Shift+Arrow` | Resize current pane |
| `Ctrl+B`, `X` | Kill current pane |
| `Ctrl+B`, `Z` | Zoom/unzoom current pane |

### Other Useful Commands
| Shortcut | Action |
|----------|--------|
| `Ctrl+B`, `R` | Reload config |
| `Ctrl+B`, `?` | Show all key bindings |
| `Ctrl+B`, `T` | Show time |
| `Ctrl+B`, `[` | Enter copy mode (use arrows, q to exit) |

---

## 🎯 Common Workflows

### 1. Long-Running Training
```bash
ssh dgx-spark
tmux new -s training
cd /path/to/project
python train.py --epochs 100
# Press Ctrl+B, D to detach
# Close terminal - training continues!

# Later, check progress:
ssh dgx-spark
tmux a -t training
```

### 2. Multiple Tasks
```bash
ssh dgx-spark
tmux new -s dev

# Create windows for different tasks:
# Ctrl+B, C → python server.py
# Ctrl+B, C → tail -f logs/app.log
# Ctrl+B, C → nvidia-smi -l 1

# Switch between: Ctrl+B, 0/1/2
# Detach: Ctrl+B, D
```

### 3. Split Screen Monitoring
```bash
ssh dgx-spark
tmux new -s monitor

# Split screen:
# Ctrl+B, | → creates vertical split
# Alt+Right → move to right pane
# Run: nvidia-smi -l 1

# Ctrl+B, - → creates horizontal split
# Alt+Down → move to bottom pane
# Run: htop

# Alt+Arrow to move between panes
```

---

## 🔧 Advanced Tips

### Auto-start tmux on SSH
Add to `~/.bashrc` on server:
```bash
if command -v tmux &> /dev/null && [ -z "$TMUX" ]; then
    tmux attach -t default || tmux new -s default
fi
```

### Named Windows for Clarity
```bash
tmux new -s project -n editor    # Create with window name
# Inside tmux:
# Ctrl+B, , → rename current window
```

### Session with Multiple Windows
```bash
tmux new -s dev -n code
tmux new-window -n logs
tmux new-window -n monitor
tmux select-window -t code
```

---

## 🆘 Troubleshooting

### Lost Session?
```bash
tmux ls    # List all sessions
tmux a     # Attach to last session
```

### Tmux Not Responding?
- Press `Ctrl+B`, `R` to reload config
- Or restart: `tmux kill-server` then reconnect

### Multiple Sessions Running?
```bash
tmux ls                          # See all sessions
tmux kill-session -t old_name    # Kill specific one
tmux kill-server                 # Kill all (use carefully!)
```

### Forgot Keyboard Shortcuts?
- Inside tmux: `Ctrl+B`, `?`
- Shows all available key bindings

---

## 📊 Status Bar Legend

```
Session: dev 1 0    |    [Window List]    |    06 Nov 14:30
^                   |                      |
Session name        |                      |
Window number       |                      |
Pane number         |                      Date & Time
```

---

## 🎓 Learning Path

**Day 1:** Learn these 3 commands
1. `tmux new -s work` - Start
2. `Ctrl+B`, `D` - Detach
3. `tmux a -t work` - Reattach

**Day 2:** Add window management
4. `Ctrl+B`, `C` - New window
5. `Ctrl+B`, `N/P` - Switch windows

**Day 3:** Master pane splits
6. `Ctrl+B`, `|` - Vertical split
7. `Alt+Arrow` - Move between panes

---

## 📚 Quick Command Reference

```bash
# Session commands
tmux new -s NAME           # Create named session
tmux ls                    # List sessions
tmux a -t NAME             # Attach to session
tmux kill-session -t NAME  # Kill session

# Inside tmux
Ctrl+B, D                  # Detach
Ctrl+B, ?                  # Help
Ctrl+B, :                  # Command prompt

# One-liners
tmux a || tmux new -s default    # Attach or create
```

---

## 🎯 Most Important Commands

If you only remember 5 things:

1. **Start:** `tmux new -s work`
2. **Detach:** `Ctrl+B`, `D`
3. **Return:** `tmux a -t work`
4. **List:** `tmux ls`
5. **Help:** `Ctrl+B`, `?`

That's enough for 90% of use cases!
