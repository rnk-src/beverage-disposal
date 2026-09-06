# Placeholder until the SO-101 port lands and can be calibrated against
# real effort readings from holding a known-full and known-empty object.
FULLNESS_EFFORT_THRESHOLD = 0.5


def classify_fullness(effort):
    if effort >= FULLNESS_EFFORT_THRESHOLD:
        return 'full'
    return 'empty'
