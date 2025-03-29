import numpy as np
from lovers import men, women
from sklearn.model_selection import KFold


def sigmoid(x, alpha=5):
    return 1 / (1 + np.exp(-alpha * x))


def compatibility_score(traits, preferences):
    """
    Compute compatibility score using squared Euclidean similarity
    """
    return 1 - np.linalg.norm(traits - preferences) ** 2


def swipe_probability(preferences1, traits2, alpha=5):
    """
    Compute the probability of user 1 swiping right on user 2.
    """
    score = compatibility_score(traits2, preferences1)
    return sigmoid(score, alpha=alpha)


def optimize_alpha():
    """
    Perform cross-validation to optimize alpha
    """
    alphas = np.linspace(1, 10, 10)  # Range of alpha values to test
    best_alpha = 1
    best_variance = float('inf')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for alpha in alphas:
        probabilities = []
        for train_idx, test_idx in kf.split(men):
            for i in test_idx:
                prob_m_to_w = swipe_probability(men[i].preferences, women[i].traits, alpha)
                prob_w_to_m = swipe_probability(women[i].preferences, men[i].traits, alpha)
                probabilities.extend([prob_m_to_w, prob_w_to_m])

        variance = np.var(probabilities)
        if variance < best_variance:
            best_variance = variance
            best_alpha = alpha

    return best_alpha


if __name__ == "__main__":
    optimized_alpha = optimize_alpha()
    print(f"Optimized alpha: {optimized_alpha}")

    man = men[1]
    woman = women[0]
    prob_m_to_w = swipe_probability(man.preferences, woman.traits, optimized_alpha)
    prob_w_to_m = swipe_probability(woman.preferences, man.traits, optimized_alpha)

    print(f"\nProbability of Man {man.id} swiping right on Woman {woman.id}: {prob_m_to_w:.2f}")
    print("Man preferences:", man.preferences)
    print("Woman traits:", woman.traits)
    print(f"\nProbability of Woman {woman.id} swiping right on Man {man.id}: {prob_w_to_m:.2f}")
    print("Woman preferences:", woman.preferences)
    print("Man traits:", man.traits)

    # Testing (edge cases)
    man_preferences = np.array([1, 1, 0.5, 0.5])
    woman_traits = np.array([1, 1, 1, 1])

    prob_m_to_w = swipe_probability(man_preferences, woman_traits)
    prob_w_to_m = swipe_probability(woman_traits, man_preferences)

    print(f"\nProbability of Man swiping right on Woman: {prob_m_to_w:.2f}")
    print(f"Probability of Woman swiping right on Man: {prob_w_to_m:.2f}")
