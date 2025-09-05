BLOCKLISTED_ACTIONS = {"delete_data", "disable_logging", "shutdown_system"}

def action_is_safe(action: str) -> bool:
    return action not in BLOCKLISTED_ACTIONS
