import time

PROFILES = {}
PROFILE_STARTS = {}
PROFILE_CALLS = {}

# Record the start time of a profile with label `label` in PROFILE_STARTS.
def profile_start(label: str):
    if label in PROFILE_STARTS:
        raise ValueError(f"Profile '{label}' has already been started.")
    PROFILE_STARTS[label] = time.perf_counter()
    PROFILE_CALLS[label] = PROFILE_CALLS.get(label, 0) + 1

# Record the end time of a profile with label `label` in PROFILES.
def profile_end(label: str):
    if label not in PROFILE_STARTS:
        raise ValueError(f"Profile '{label}' has not been started.")
    
    elapsed = time.perf_counter() - PROFILE_STARTS.pop(label)
    if label not in PROFILES:
        PROFILES[label] = elapsed
    else:
        PROFILES[label] += elapsed

def get_profile(label: str) -> float:
    """Get the elapsed time for a profile with label `label`."""
    if label not in PROFILES:
        return 0.0
    return PROFILES[label]

def print_profiles():
    """Print all recorded profiles."""
    items = [x for x in PROFILES.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    for label, elapsed in items:
        print(f"{label}: {elapsed:.6f} seconds, over {PROFILE_CALLS[label]} calls")