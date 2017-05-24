def reciprocal_rank(answer, results):
    if answer not in results:
        return 0.0
    else:
        return 1.0 / (results.index(answer) + 1)

def get_word_error_rate(answer, result):
    pass

def is_partial_match(answer, target):
    if answer == target: return False
    return target.find(answer) >= 0 or answer.find(target) >= 0

import matplotlib.pyplot as plt
def plot_correct_sent_rank_histogram(freq_dict):
    figure = plt.figure()
    x_vals = sorted(freq_dict.keys())
    y_vals = [freq_dict[x] for x in x_vals]
    plt.plot(x_vals, y_vals, label="Num Cases at Rank")
    plt.show()
