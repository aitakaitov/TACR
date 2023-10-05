def score(c1, c2):
    diff = 0
    for key in c1.keys():
        diff += abs(c1[key] - c2[key])

    return diff


def character_count_similarity_index(to_find, target, leniency, check_multiples=False):
    char_counts_to_find = dict.fromkeys(to_find, 0)
    for c in to_find:
        char_counts_to_find[c] += 1

    char_counts_target = dict.fromkeys(to_find, 0)
    for c in target[:len(to_find)]:
        if c in char_counts_target.keys():
            char_counts_target[c] += 1

    index = 0
    same_similarity = 0
    max_similarity = score(char_counts_to_find, char_counts_target)

    for it in range(1, len(target) - len(to_find)):
        if target[it - 1] in char_counts_target.keys():
            char_counts_target[target[it - 1]] -= 1

        if target[it + len(to_find)] in char_counts_target.keys():
            char_counts_target[target[it + len(to_find)]] += 1

        new_sim = score(char_counts_to_find, char_counts_target)
        if new_sim < max_similarity:
            max_similarity = new_sim
            index = it
            same_similarity = 1
        if new_sim == max_similarity:
            same_similarity += 1


    if max_similarity <= len(to_find) / leniency:
        if check_multiples:
            return index, same_similarity
        return index
    else:
        return None
