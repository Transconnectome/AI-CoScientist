# DGX Tmux Skill - Test Scenarios

## Pressure Scenario 1: Urgent Long-Running Task
**Context:** User needs to start GPU training immediately and leave

**User Request:**
"DGX 서버에 연결해서 학습 코드 실행하고 싶은데, 지금 나가야 해서 터미널 닫아야 해. 작업이 계속 실행되게 해줘. 급해!"

**Expected Baseline Behavior (without skill):**
- Agent explains need for session persistence tool
- Suggests nohup, screen, or tmux
- May need to check if installed
- Multiple steps to configure
- User needs to remember commands

**Expected Behavior (with skill):**
- Immediately connects with pre-configured setup
- One command to start persistent session
- Clear quick reference for management
- Minimal friction

---

## Pressure Scenario 2: Monitoring While Away
**Context:** User needs to monitor GPU usage but can't keep terminal open

**User Request:**
"GPU 사용량 모니터링하면서 다른 작업도 해야 하는데, 화면 여러개 띄우고 싶어. 어떻게 해?"

**Expected Baseline Behavior (without skill):**
- Explains screen splitting options
- Needs to teach tmux pane management
- User must learn keyboard shortcuts
- May forget configuration

**Expected Behavior (with skill):**
- Quick reference for pane splitting
- Pre-configured shortcuts explained
- Common workflow examples
- Fast lookup without explanation overhead

---

## Pressure Scenario 3: Session Recovery
**Context:** Connection dropped, need to recover work

**User Request:**
"연결 끊겼는데 실행중이던 작업 어떻게 다시 확인해?"

**Expected Baseline Behavior (without skill):**
- Explains tmux attach concept
- User needs to remember session names
- May not know how to list sessions

**Expected Behavior (with skill):**
- Quick command reference for recovery
- Session listing and attaching
- Clear workflow pattern

---

## Test Method
1. Run scenarios with fresh subagent WITHOUT skill loaded
2. Document exact responses and steps
3. Identify pain points and missing knowledge
4. Write skill to address specific gaps
5. Re-test with skill loaded
6. Verify reduced friction and faster execution
