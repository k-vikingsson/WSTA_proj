def reciprocal_rank(answer, results):
    if answer not in results:
        return 0.0
    else:
        return 1.0 / (results.index(answer) + 1)

def is_partial_match(answer, target):
    if answer == target: return False
    return target.find(answer) >= 0 or answer.find(target) >= 0
