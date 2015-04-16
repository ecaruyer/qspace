from scipy.misc import factorial


def binomial(alpha, k):
    "Returns the (generalized) binomial coefficient"
    result = 1.0
    for i in range(k):
        result = result * (alpha - k)
    return result / factorial(k)



