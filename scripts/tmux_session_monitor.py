#!/usr/bin/env python3
"""
Tmux Session Monitor
A comprehensive tool to monitor and display tmux session information.
"""

import subprocess
import json
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class TmuxSessionMonitor:
    """Monitor and display tmux session information."""
    
    def __init__(self):
        self.check_tmux_available()
    
    def check_tmux_available(self):
        """Check if tmux is available."""
        try:
            subprocess.run(['tmux', '-V'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Error: tmux is not installed or not in PATH")
            sys.exit(1)
    
    def run_tmux_command(self, cmd: List[str]) -> str:
        """Run a tmux command and return output."""
        try:
            result = subprocess.run(
                ['tmux'] + cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return ""
    
    def get_sessions(self) -> List[Dict]:
        """Get list of all tmux sessions."""
        output = self.run_tmux_command(['list-sessions', '-F', '#{session_name}'])
        if not output:
            return []
        
        sessions = []
        for session_name in output.split('\n'):
            if session_name:
                sessions.append(self.get_session_info(session_name))
        
        return sessions
    
    def get_session_info(self, session_name: str) -> Dict:
        """Get detailed information about a session."""
        # Get session creation time
        created = self.run_tmux_command([
            'display-message', '-t', session_name,
            '-p', '#{session_created}'
        ])
        
        # Get session activity time
        activity = self.run_tmux_command([
            'display-message', '-t', session_name,
            '-p', '#{session_activity}'
        ])
        
        # Get number of windows
        windows = self.run_tmux_command([
            'list-windows', '-t', session_name,
            '-F', '#{window_index}'
        ])
        window_count = len([w for w in windows.split('\n') if w])
        
        # Get attached clients
        attached = self.run_tmux_command([
            'list-clients', '-t', session_name,
            '-F', '#{client_tty}'
        ])
        is_attached = bool(attached.strip())
        
        # Get session size
        size = self.run_tmux_command([
            'display-message', '-t', session_name,
            '-p', '#{window_width}x#{window_height}'
        ])
        
        # Parse timestamps
        try:
            created_ts = int(created) if created else 0
            activity_ts = int(activity) if activity else 0
            created_dt = datetime.fromtimestamp(created_ts) if created_ts else None
            activity_dt = datetime.fromtimestamp(activity_ts) if activity_ts else None
        except (ValueError, TypeError):
            created_dt = None
            activity_dt = None
        
        return {
            'name': session_name,
            'windows': window_count,
            'attached': is_attached,
            'created': created_dt,
            'last_activity': activity_dt,
            'size': size,
            'windows_detail': self.get_windows_info(session_name)
        }
    
    def get_windows_info(self, session_name: str) -> List[Dict]:
        """Get information about windows in a session."""
        windows = []
        
        # Get window list
        window_list = self.run_tmux_command([
            'list-windows', '-t', session_name,
            '-F', '#{window_index}|#{window_name}|#{window_active}|#{window_flags}'
        ])
        
        for line in window_list.split('\n'):
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 4:
                idx, name, active, flags = parts[0], parts[1], parts[2], parts[3]
                
                # Get pane count
                panes = self.run_tmux_command([
                    'list-panes', '-t', f'{session_name}:{idx}',
                    '-F', '#{pane_index}'
                ])
                pane_count = len([p for p in panes.split('\n') if p])
                
                # Get current command
                command = self.run_tmux_command([
                    'display-message', '-t', f'{session_name}:{idx}',
                    '-p', '#{pane_current_command}'
                ])
                
                windows.append({
                    'index': idx,
                    'name': name,
                    'active': active == '1',
                    'panes': pane_count,
                    'command': command,
                    'flags': flags
                })
        
        return windows
    
    def format_duration(self, dt: Optional[datetime]) -> str:
        """Format datetime as relative duration."""
        if not dt:
            return "unknown"
        
        now = datetime.now()
        delta = now - dt
        
        if delta.days > 0:
            return f"{delta.days}d {delta.seconds // 3600}h ago"
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            return f"{hours}h {(delta.seconds % 3600) // 60}m ago"
        elif delta.seconds >= 60:
            minutes = delta.seconds // 60
            return f"{minutes}m ago"
        else:
            return "just now"
    
    def format_datetime(self, dt: Optional[datetime]) -> str:
        """Format datetime as string."""
        if not dt:
            return "unknown"
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def display_sessions(self, sessions: List[Dict], detailed: bool = False):
        """Display session information."""
        if not sessions:
            print("📭 No tmux sessions found")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 Tmux Session Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Summary
        attached_count = sum(1 for s in sessions if s['attached'])
        total_windows = sum(s['windows'] for s in sessions)
        
        print(f"📈 Summary: {len(sessions)} session(s), {attached_count} attached, {total_windows} window(s)\n")
        
        # Sessions
        for i, session in enumerate(sessions, 1):
            status = "🟢 ATTACHED" if session['attached'] else "⚪ DETACHED"
            print(f"{i}. {session['name']} {status}")
            print(f"   Windows: {session['windows']} | Size: {session['size']}")
            
            if session['created']:
                print(f"   Created: {self.format_datetime(session['created'])} ({self.format_duration(session['created'])})")
            
            if session['last_activity']:
                print(f"   Last Activity: {self.format_datetime(session['last_activity'])} ({self.format_duration(session['last_activity'])})")
            
            if detailed and session['windows_detail']:
                print(f"   Windows Detail:")
                for window in session['windows_detail']:
                    active_marker = "▶" if window['active'] else " "
                    print(f"      {active_marker} [{window['index']}] {window['name']} ({window['panes']} panes)")
                    if window['command']:
                        print(f"         Command: {window['command']}")
            
            print()
    
    def display_json(self, sessions: List[Dict]):
        """Display sessions as JSON."""
        # Convert datetime objects to strings for JSON serialization
        json_sessions = []
        for session in sessions:
            json_session = session.copy()
            if json_session['created']:
                json_session['created'] = json_session['created'].isoformat()
            if json_session['last_activity']:
                json_session['last_activity'] = json_session['last_activity'].isoformat()
            json_sessions.append(json_session)
        
        print(json.dumps(json_sessions, indent=2))
    
    def monitor_loop(self, interval: int = 5):
        """Continuously monitor sessions."""
        try:
            while True:
                # Clear screen
                subprocess.run(['clear'], check=False)
                
                sessions = self.get_sessions()
                self.display_sessions(sessions, detailed=True)
                
                print(f"\n🔄 Refreshing every {interval} seconds (Ctrl+C to stop)")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor tmux sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # List all sessions
  %(prog)s -d                 # Detailed view with windows
  %(prog)s -m                 # Monitor mode (auto-refresh)
  %(prog)s -m -i 10           # Monitor with 10 second interval
  %(prog)s -j                 # Output as JSON
  %(prog)s -s mysession       # Show specific session
        """
    )
    
    parser.add_argument(
        '-d', '--detailed',
        action='store_true',
        help='Show detailed information including windows and panes'
    )
    
    parser.add_argument(
        '-m', '--monitor',
        action='store_true',
        help='Monitor mode: continuously refresh display'
    )
    
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=5,
        help='Refresh interval in seconds for monitor mode (default: 5)'
    )
    
    parser.add_argument(
        '-j', '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    parser.add_argument(
        '-s', '--session',
        type=str,
        help='Show specific session only'
    )
    
    args = parser.parse_args()
    
    monitor = TmuxSessionMonitor()
    
    if args.session:
        sessions = [monitor.get_session_info(args.session)]
        if not sessions[0]['name']:
            print(f"❌ Session '{args.session}' not found")
            sys.exit(1)
    else:
        sessions = monitor.get_sessions()
    
    if args.json:
        monitor.display_json(sessions)
    elif args.monitor:
        monitor.monitor_loop(args.interval)
    else:
        monitor.display_sessions(sessions, detailed=args.detailed)


if __name__ == '__main__':
    main()





