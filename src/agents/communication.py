"""
Inter-Agent Communication Protocols
Advanced communication system for Agent Pool Enhancement 2.0

Features:
- Message passing between agents
- Context sharing and synchronization
- Collaborative decision making
- Knowledge exchange protocols
- Communication history tracking
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    QUERY = "query"
    COLLABORATION = "collaboration"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"

class MessagePriority(int, Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class CommunicationState(str, Enum):
    """Communication session states"""
    INITIATED = "initiated"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AgentMessage:
    """Message between agents"""
    message_id: str
    sender_id: str
    receiver_id: str  # Can be "broadcast" for all agents
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    response_timeout: Optional[int] = None  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessageResponse:
    """Response to an agent message"""
    response_id: str
    original_message_id: str
    sender_id: str
    content: Dict[str, Any]
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class CommunicationSession:
    """Multi-agent communication session"""
    session_id: str
    participants: List[str]
    session_type: str  # "collaboration", "validation", "synthesis"
    state: CommunicationState
    messages: List[AgentMessage] = field(default_factory=list)
    responses: List[MessageResponse] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentCommunicationHub:
    """Central hub for inter-agent communication"""

    def __init__(self, max_message_history: int = 1000):
        self.max_message_history = max_message_history

        # Message queues for each agent
        self.agent_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Message history
        self.message_history: deque = deque(maxlen=max_message_history)

        # Active communication sessions
        self.active_sessions: Dict[str, CommunicationSession] = {}

        # Response tracking
        self.pending_responses: Dict[str, AgentMessage] = {}

        # Agent registry for communication
        self.registered_agents: Dict[str, Dict[str, Any]] = {}

        # Communication patterns and templates
        self.communication_templates = self._initialize_templates()

        # Event handlers for different message types
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.REQUEST: self._handle_request_message,
            MessageType.RESPONSE: self._handle_response_message,
            MessageType.BROADCAST: self._handle_broadcast_message,
            MessageType.NOTIFICATION: self._handle_notification_message,
            MessageType.QUERY: self._handle_query_message,
            MessageType.COLLABORATION: self._handle_collaboration_message,
            MessageType.VALIDATION: self._handle_validation_message,
            MessageType.SYNTHESIS: self._handle_synthesis_message
        }

    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize communication templates for common patterns"""

        return {
            "literature_request": {
                "type": MessageType.REQUEST,
                "priority": MessagePriority.NORMAL,
                "template": {
                    "action": "literature_analysis",
                    "parameters": {
                        "research_question": "",
                        "focus_areas": [],
                        "analysis_depth": "comprehensive"
                    }
                }
            },
            "hypothesis_validation": {
                "type": MessageType.VALIDATION,
                "priority": MessagePriority.HIGH,
                "template": {
                    "action": "validate_hypothesis",
                    "parameters": {
                        "hypothesis": "",
                        "validation_criteria": [],
                        "evidence_sources": []
                    }
                }
            },
            "statistical_consultation": {
                "type": MessageType.QUERY,
                "priority": MessagePriority.NORMAL,
                "template": {
                    "action": "statistical_advice",
                    "parameters": {
                        "data_description": "",
                        "research_question": "",
                        "analysis_goals": []
                    }
                }
            },
            "grant_collaboration": {
                "type": MessageType.COLLABORATION,
                "priority": MessagePriority.HIGH,
                "template": {
                    "action": "collaborative_writing",
                    "parameters": {
                        "section_type": "",
                        "requirements": {},
                        "shared_context": {}
                    }
                }
            },
            "clinical_review": {
                "type": MessageType.VALIDATION,
                "priority": MessagePriority.HIGH,
                "template": {
                    "action": "clinical_validation",
                    "parameters": {
                        "validation_type": "",
                        "clinical_context": {},
                        "safety_requirements": []
                    }
                }
            },
            "synthesis_request": {
                "type": MessageType.SYNTHESIS,
                "priority": MessagePriority.HIGH,
                "template": {
                    "action": "knowledge_synthesis",
                    "parameters": {
                        "input_sources": [],
                        "synthesis_goals": [],
                        "output_format": ""
                    }
                }
            }
        }

    def register_agent(self,
                      agent_id: str,
                      capabilities: List[str],
                      communication_preferences: Optional[Dict[str, Any]] = None):
        """Register an agent for communication"""

        self.registered_agents[agent_id] = {
            "capabilities": capabilities,
            "preferences": communication_preferences or {},
            "last_active": datetime.now(),
            "message_count": 0,
            "response_times": []
        }

        logger.info(f"Registered agent {agent_id} for communication")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from communication"""

        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]

        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]

        logger.info(f"Unregistered agent {agent_id} from communication")

    async def send_message(self,
                          sender_id: str,
                          receiver_id: str,
                          message_type: MessageType,
                          content: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL,
                          requires_response: bool = False,
                          response_timeout: Optional[int] = 60) -> str:
        """Send message between agents"""

        message_id = str(uuid.uuid4())

        message = AgentMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            requires_response=requires_response,
            response_timeout=response_timeout
        )

        # Add to message history
        self.message_history.append(message)

        # Route message
        if receiver_id == "broadcast":
            await self._broadcast_message(message)
        else:
            await self._route_message(message)

        # Track pending responses
        if requires_response:
            self.pending_responses[message_id] = message

            # Set timeout for response
            if response_timeout:
                asyncio.create_task(self._handle_response_timeout(message_id, response_timeout))

        # Update sender statistics
        if sender_id in self.registered_agents:
            self.registered_agents[sender_id]["message_count"] += 1
            self.registered_agents[sender_id]["last_active"] = datetime.now()

        logger.info(f"Sent message {message_id} from {sender_id} to {receiver_id}")
        return message_id

    async def _route_message(self, message: AgentMessage):
        """Route message to specific agent"""

        if message.receiver_id in self.registered_agents:
            # Add to agent's queue
            self.agent_queues[message.receiver_id].append(message)

            # Process message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error handling message {message.message_id}: {e}")
        else:
            logger.warning(f"Receiver {message.receiver_id} not registered")

    async def _broadcast_message(self, message: AgentMessage):
        """Broadcast message to all registered agents"""

        for agent_id in self.registered_agents:
            if agent_id != message.sender_id:
                message.receiver_id = agent_id
                await self._route_message(message)

    async def respond_to_message(self,
                               original_message_id: str,
                               sender_id: str,
                               response_content: Dict[str, Any],
                               success: bool = True,
                               error_message: Optional[str] = None) -> str:
        """Send response to a message"""

        response_id = str(uuid.uuid4())

        response = MessageResponse(
            response_id=response_id,
            original_message_id=original_message_id,
            sender_id=sender_id,
            content=response_content,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )

        # Find original message
        original_message = self.pending_responses.get(original_message_id)
        if original_message:
            # Calculate response time
            response_time = (response.timestamp - original_message.timestamp).total_seconds()

            # Update sender statistics
            if sender_id in self.registered_agents:
                self.registered_agents[sender_id]["response_times"].append(response_time)

            # Remove from pending
            del self.pending_responses[original_message_id]

            # Route response back to original sender
            response_message = AgentMessage(
                message_id=response_id,
                sender_id=sender_id,
                receiver_id=original_message.sender_id,
                message_type=MessageType.RESPONSE,
                priority=original_message.priority,
                content={
                    "response_data": response_content,
                    "original_message_id": original_message_id,
                    "success": success,
                    "error_message": error_message
                },
                timestamp=response.timestamp
            )

            await self._route_message(response_message)

        logger.info(f"Sent response {response_id} for message {original_message_id}")
        return response_id

    def get_messages_for_agent(self, agent_id: str, limit: Optional[int] = None) -> List[AgentMessage]:
        """Get pending messages for an agent"""

        if agent_id not in self.agent_queues:
            return []

        messages = list(self.agent_queues[agent_id])

        if limit:
            messages = messages[:limit]

        # Sort by priority and timestamp
        messages.sort(key=lambda m: (m.priority.value, m.timestamp), reverse=True)

        return messages

    def clear_agent_queue(self, agent_id: str):
        """Clear message queue for an agent"""

        if agent_id in self.agent_queues:
            self.agent_queues[agent_id].clear()

    async def start_communication_session(self,
                                        session_type: str,
                                        participants: List[str],
                                        initial_context: Optional[Dict[str, Any]] = None) -> str:
        """Start a multi-agent communication session"""

        session_id = str(uuid.uuid4())

        session = CommunicationSession(
            session_id=session_id,
            participants=participants,
            session_type=session_type,
            state=CommunicationState.INITIATED,
            shared_context=initial_context or {}
        )

        self.active_sessions[session_id] = session

        # Notify participants
        for participant in participants:
            await self.send_message(
                sender_id="communication_hub",
                receiver_id=participant,
                message_type=MessageType.NOTIFICATION,
                content={
                    "action": "session_started",
                    "session_id": session_id,
                    "session_type": session_type,
                    "participants": participants,
                    "shared_context": initial_context or {}
                }
            )

        logger.info(f"Started communication session {session_id} with participants: {participants}")
        return session_id

    async def end_communication_session(self, session_id: str) -> Dict[str, Any]:
        """End a communication session and return summary"""

        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.state = CommunicationState.COMPLETED
        session.end_time = datetime.now()

        # Generate session summary
        summary = {
            "session_id": session_id,
            "participants": session.participants,
            "session_type": session.session_type,
            "duration_minutes": (session.end_time - session.start_time).total_seconds() / 60,
            "message_count": len(session.messages),
            "response_count": len(session.responses),
            "final_context": session.shared_context
        }

        # Notify participants
        for participant in session.participants:
            await self.send_message(
                sender_id="communication_hub",
                receiver_id=participant,
                message_type=MessageType.NOTIFICATION,
                content={
                    "action": "session_ended",
                    "session_id": session_id,
                    "summary": summary
                }
            )

        # Remove from active sessions
        del self.active_sessions[session_id]

        logger.info(f"Ended communication session {session_id}")
        return summary

    # Message Handlers
    async def _handle_request_message(self, message: AgentMessage):
        """Handle request messages"""
        logger.info(f"Processing request message {message.message_id}")

    async def _handle_response_message(self, message: AgentMessage):
        """Handle response messages"""
        logger.info(f"Processing response message {message.message_id}")

    async def _handle_broadcast_message(self, message: AgentMessage):
        """Handle broadcast messages"""
        logger.info(f"Processing broadcast message {message.message_id}")

    async def _handle_notification_message(self, message: AgentMessage):
        """Handle notification messages"""
        logger.info(f"Processing notification message {message.message_id}")

    async def _handle_query_message(self, message: AgentMessage):
        """Handle query messages"""
        logger.info(f"Processing query message {message.message_id}")

    async def _handle_collaboration_message(self, message: AgentMessage):
        """Handle collaboration messages"""

        content = message.content
        action = content.get("action")

        if action == "collaborative_writing":
            # Handle collaborative writing request
            await self._process_collaborative_writing(message)
        elif action == "knowledge_sharing":
            # Handle knowledge sharing
            await self._process_knowledge_sharing(message)

        logger.info(f"Processing collaboration message {message.message_id}")

    async def _handle_validation_message(self, message: AgentMessage):
        """Handle validation messages"""

        content = message.content
        action = content.get("action")

        if action == "validate_hypothesis":
            await self._process_hypothesis_validation(message)
        elif action == "clinical_validation":
            await self._process_clinical_validation(message)

        logger.info(f"Processing validation message {message.message_id}")

    async def _handle_synthesis_message(self, message: AgentMessage):
        """Handle synthesis messages"""

        content = message.content
        action = content.get("action")

        if action == "knowledge_synthesis":
            await self._process_knowledge_synthesis(message)

        logger.info(f"Processing synthesis message {message.message_id}")

    async def _process_collaborative_writing(self, message: AgentMessage):
        """Process collaborative writing request"""

        # Find appropriate agents for collaboration
        section_type = message.content.get("parameters", {}).get("section_type", "")

        # Route to appropriate specialists
        if "statistical" in section_type.lower():
            target_agent = "statistical_analyst"
        elif "clinical" in section_type.lower():
            target_agent = "clinical_validator"
        elif "literature" in section_type.lower():
            target_agent = "literature_analyst"
        else:
            target_agent = "grant_writer"

        # Forward request if target agent is different from sender
        if target_agent != message.sender_id and target_agent in self.registered_agents:
            await self.send_message(
                sender_id="communication_hub",
                receiver_id=target_agent,
                message_type=MessageType.COLLABORATION,
                content=message.content,
                requires_response=True
            )

    async def _process_hypothesis_validation(self, message: AgentMessage):
        """Process hypothesis validation request"""

        # Route to statistical analyst and clinical validator
        validators = ["statistical_analyst", "clinical_validator"]

        for validator in validators:
            if validator != message.sender_id and validator in self.registered_agents:
                await self.send_message(
                    sender_id="communication_hub",
                    receiver_id=validator,
                    message_type=MessageType.VALIDATION,
                    content=message.content,
                    requires_response=True
                )

    async def _process_clinical_validation(self, message: AgentMessage):
        """Process clinical validation request"""

        # Route to clinical validator if not already there
        if message.sender_id != "clinical_validator" and "clinical_validator" in self.registered_agents:
            await self.send_message(
                sender_id="communication_hub",
                receiver_id="clinical_validator",
                message_type=MessageType.VALIDATION,
                content=message.content,
                requires_response=True
            )

    async def _process_knowledge_sharing(self, message: AgentMessage):
        """Process knowledge sharing between agents"""

        # Broadcast relevant knowledge to interested agents
        knowledge_type = message.content.get("knowledge_type", "")

        interested_agents = []

        for agent_id, agent_info in self.registered_agents.items():
            capabilities = agent_info.get("capabilities", [])

            # Match knowledge type to agent capabilities
            if any(cap in knowledge_type.lower() for cap in capabilities):
                interested_agents.append(agent_id)

        # Send knowledge to interested agents
        for agent in interested_agents:
            if agent != message.sender_id:
                await self.send_message(
                    sender_id=message.sender_id,
                    receiver_id=agent,
                    message_type=MessageType.NOTIFICATION,
                    content=message.content
                )

    async def _process_knowledge_synthesis(self, message: AgentMessage):
        """Process knowledge synthesis request"""

        # Identify agents that can contribute to synthesis
        synthesis_contributors = ["literature_analyst", "statistical_analyst", "hypothesis_generator"]

        for contributor in synthesis_contributors:
            if contributor != message.sender_id and contributor in self.registered_agents:
                await self.send_message(
                    sender_id="communication_hub",
                    receiver_id=contributor,
                    message_type=MessageType.SYNTHESIS,
                    content=message.content,
                    requires_response=True
                )

    async def _handle_response_timeout(self, message_id: str, timeout_seconds: int):
        """Handle response timeout for messages"""

        await asyncio.sleep(timeout_seconds)

        if message_id in self.pending_responses:
            original_message = self.pending_responses[message_id]

            # Send timeout notification
            await self.send_message(
                sender_id="communication_hub",
                receiver_id=original_message.sender_id,
                message_type=MessageType.NOTIFICATION,
                content={
                    "action": "response_timeout",
                    "original_message_id": message_id,
                    "receiver_id": original_message.receiver_id
                }
            )

            # Remove from pending
            del self.pending_responses[message_id]

            logger.warning(f"Response timeout for message {message_id}")

    # Utility and Analysis Methods
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""

        total_messages = len(self.message_history)
        total_agents = len(self.registered_agents)
        active_sessions = len(self.active_sessions)

        agent_stats = {}
        for agent_id, agent_info in self.registered_agents.items():
            response_times = agent_info.get("response_times", [])
            agent_stats[agent_id] = {
                "message_count": agent_info.get("message_count", 0),
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "last_active": agent_info.get("last_active", datetime.now()).isoformat(),
                "pending_messages": len(self.agent_queues.get(agent_id, []))
            }

        return {
            "total_messages": total_messages,
            "total_agents": total_agents,
            "active_sessions": active_sessions,
            "agent_statistics": agent_stats,
            "pending_responses": len(self.pending_responses)
        }

    def get_message_history(self,
                           agent_id: Optional[str] = None,
                           message_type: Optional[MessageType] = None,
                           hours: Optional[int] = 24) -> List[Dict[str, Any]]:
        """Get message history with optional filtering"""

        cutoff_time = datetime.now() - timedelta(hours=hours) if hours else None

        filtered_messages = []
        for message in self.message_history:
            # Apply filters
            if cutoff_time and message.timestamp < cutoff_time:
                continue

            if agent_id and message.sender_id != agent_id and message.receiver_id != agent_id:
                continue

            if message_type and message.message_type != message_type:
                continue

            # Convert to dict for JSON serialization
            filtered_messages.append(asdict(message))

        return filtered_messages

    def create_message_from_template(self,
                                   template_name: str,
                                   sender_id: str,
                                   receiver_id: str,
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create message from predefined template"""

        if template_name not in self.communication_templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.communication_templates[template_name]

        # Fill in template parameters
        content = template["template"].copy()
        content["parameters"].update(parameters)

        return {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message_type": template["type"],
            "priority": template["priority"],
            "content": content
        }

# Convenience functions for common communication patterns
async def request_literature_analysis(hub: AgentCommunicationHub,
                                     requester_id: str,
                                     research_question: str,
                                     focus_areas: List[str]) -> str:
    """Request literature analysis from literature analyst agent"""

    message_data = hub.create_message_from_template(
        "literature_request",
        requester_id,
        "literature_analyst",
        {
            "research_question": research_question,
            "focus_areas": focus_areas
        }
    )

    return await hub.send_message(**message_data, requires_response=True)

async def request_statistical_consultation(hub: AgentCommunicationHub,
                                         requester_id: str,
                                         data_description: str,
                                         analysis_goals: List[str]) -> str:
    """Request statistical consultation from statistical analyst"""

    message_data = hub.create_message_from_template(
        "statistical_consultation",
        requester_id,
        "statistical_analyst",
        {
            "data_description": data_description,
            "analysis_goals": analysis_goals
        }
    )

    return await hub.send_message(**message_data, requires_response=True)

async def initiate_grant_collaboration(hub: AgentCommunicationHub,
                                     initiator_id: str,
                                     section_type: str,
                                     requirements: Dict[str, Any]) -> str:
    """Initiate collaborative grant writing session"""

    # Determine participants based on section type
    participants = [initiator_id, "grant_writer"]

    if "statistical" in section_type.lower():
        participants.append("statistical_analyst")
    if "clinical" in section_type.lower():
        participants.append("clinical_validator")
    if "literature" in section_type.lower():
        participants.append("literature_analyst")
    if "hypothesis" in section_type.lower():
        participants.append("hypothesis_generator")

    # Remove duplicates
    participants = list(set(participants))

    return await hub.start_communication_session(
        "grant_collaboration",
        participants,
        {
            "section_type": section_type,
            "requirements": requirements,
            "initiator": initiator_id
        }
    )

# Testing and demonstration
if __name__ == "__main__":
    async def test_communication():
        """Test the communication system"""

        hub = AgentCommunicationHub()

        # Register test agents
        test_agents = [
            ("literature_analyst", ["literature_synthesis", "systematic_review"]),
            ("statistical_analyst", ["statistical_analysis", "experimental_design"]),
            ("grant_writer", ["grant_writing", "proposal_optimization"]),
            ("clinical_validator", ["clinical_validation", "regulatory_compliance"])
        ]

        for agent_id, capabilities in test_agents:
            hub.register_agent(agent_id, capabilities)

        print("Communication Hub Test Results:")
        print(f"Registered agents: {len(hub.registered_agents)}")

        # Test message sending
        message_id = await hub.send_message(
            sender_id="grant_writer",
            receiver_id="literature_analyst",
            message_type=MessageType.REQUEST,
            content={
                "action": "literature_analysis",
                "research_question": "AI-based autism diagnosis effectiveness",
                "focus_areas": ["diagnostic accuracy", "clinical validation"]
            },
            requires_response=True
        )

        print(f"Sent test message: {message_id}")

        # Test response
        response_id = await hub.respond_to_message(
            message_id,
            "literature_analyst",
            {
                "analysis_results": "Comprehensive literature analysis completed",
                "key_findings": ["High diagnostic accuracy", "Need for validation"]
            }
        )

        print(f"Sent test response: {response_id}")

        # Test collaboration session
        session_id = await initiate_grant_collaboration(
            hub,
            "grant_writer",
            "statistical_methodology",
            {"sample_size": 3000, "study_design": "prospective"}
        )

        print(f"Started collaboration session: {session_id}")

        # Get statistics
        stats = hub.get_communication_statistics()
        print(f"Communication statistics: {stats}")

        return hub

    # Run test
    asyncio.run(test_communication())