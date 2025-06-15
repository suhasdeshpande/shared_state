#!/usr/bin/env python
import json
from datetime import datetime
from typing import Optional, List, Dict
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.base_events import BaseEvent

# ==================== EVENTS ====================
class CopilotKitPredictStateEvent(BaseEvent):
    """Predict state event"""
    type: str = "copilotkit_predict_state"
    predict_config: List[Dict[str, str]] = []
    context: str = ""
    timestamp: Optional[datetime] = None


# ==================== PREDICT STATE HANDLER ====================

def copilotkit_predict_state(
    predict_config: List[Dict[str, str]],
    context: str = "predict_state"
):
    """
    Fire CopilotKit predict state event immediately.

    Usage:
        copilotkit_predict_state([{
            "state_key": "recipe",
            "tool": "generate_recipe",
            "tool_argument": "recipe"
        }])
    """
    event = CopilotKitPredictStateEvent(
        predict_config=predict_config,
        context=context,
        timestamp=datetime.now()
    )

    crewai_event_bus.emit(source="copilotkit_predict_state", event=event)

    print(f"🔮 PREDICT STATE [{context}]: {len(predict_config)} states")
    for config in predict_config:
        print(f"   → {config['state_key']}: {config['tool']}({config['tool_argument']})")


# ==================== EVENT HANDLER ====================
@crewai_event_bus.on(CopilotKitPredictStateEvent)
def handle_predict_state_event(source, event: CopilotKitPredictStateEvent):
    """Default handler for predict state events"""
    print(f"🎯 PREDICT STATE EVENT [{event.context}]:")
    print(f"   Source: {source}")
    print(f"   Config: {json.dumps(event.predict_config, indent=2)}")


