def running_sum(xs):
    total = 0
    out = []
    for x in xs:
        total += x
        out.append(total)
    return out
