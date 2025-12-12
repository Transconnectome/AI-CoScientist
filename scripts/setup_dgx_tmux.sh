#!/bin/bash

# ================================================
# DGX Spark Tmux Setup Script
# ================================================
# This script copies tmux configuration to DGX server
# and provides quick connection shortcuts

set -e

SERVER="dgx-spark"
TMUX_CONF="$HOME/.tmux.conf"

echo "🚀 Setting up tmux on $SERVER..."

# Check if tmux config exists locally
if [ ! -f "$TMUX_CONF" ]; then
    echo "❌ Error: $TMUX_CONF not found!"
    exit 1
fi

# Copy tmux config to server
echo "📋 Copying tmux config to server..."
scp "$TMUX_CONF" "$SERVER:~/.tmux.conf"

# Install tmux if needed and reload config
echo "🔧 Setting up tmux on server..."
ssh "$SERVER" << 'EOF'
    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        echo "📦 Installing tmux..."
        sudo apt-get update && sudo apt-get install -y tmux
    else
        echo "✅ tmux is already installed"
    fi

    # Verify config was copied
    if [ -f ~/.tmux.conf ]; then
        echo "✅ tmux config installed successfully"
    else
        echo "❌ Error: Failed to copy tmux config"
        exit 1
    fi
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Quick Start:"
echo "  ssh $SERVER"
echo "  tmux new -s work    # Create new session named 'work'"
echo "  Ctrl+B, D           # Detach from session"
echo "  tmux a -t work      # Reattach to 'work' session"
echo ""
