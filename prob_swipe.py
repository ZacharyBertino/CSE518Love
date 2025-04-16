import numpy as np
from lovers import men, women


def compatibility_score(traits, preferences, exponent=.5):
    """
    Compute compatibility score

    Parameters:
    - traits: User's traits vector
    - preferences: Other user's preferences vector
    - exponent: (higher = more penalty for mismatches)

    Returns:
    - Score between 0 and 1, where 1 is perfect match
    """
    # Calculate normalized differences
    differences = np.abs(np.array(traits) - np.array(preferences))

    # Apply exponential weighting
    weighted_differences = differences ** exponent

    # Calculate final score
    return 1 - np.mean(weighted_differences)


def trait_compatibility_distribution(traits, preferences):
    """
    Compute distribution of element-wise trait-preference compatibility
    
    Parameters:
    - traits: User's traits vector
    - preferences: Other user's preferences vector
    
    Returns:
    - Counts of excellent, good, average, poor, and terrible matches.
    """
    # Calculate element-wise compatibility
    element_compatibility = 1 - np.abs(traits - preferences)

    # Count traits in each category
    excellent = np.sum(element_compatibility > 0.95)
    good = np.sum((element_compatibility <= 0.95) & (element_compatibility > 0.85))
    average = np.sum((element_compatibility <= 0.85) & (element_compatibility > 0.7))
    poor = np.sum((element_compatibility <= 0.7) & (element_compatibility > 0.5))
    terrible = np.sum(element_compatibility <= 0.5)

    return {
        "excellent": excellent,
        "good": good,
        "average": average,
        "poor": poor,
        "terrible": terrible,
        "total": len(element_compatibility),
        "element_compatibility": element_compatibility
    }


def swipe_probability(preferences1, traits2, base_exponent=3, threshold=0.75, alpha=12):
    """
    Compute probability of swiping right

    Parameters:
    - preferences1: User 1's preferences
    - traits2: User 2's traits
    - base_exponent: Base exponent for compatibility calculation
    - threshold: Middle point of sigmoid curve
    - alpha: Steepness of sigmoid curve

    Returns:
    - Probability of right swipe (0 to 1)
    """
    # Check for perfect match
    if np.allclose(preferences1, traits2, atol=0.05):
        return 0.98

    # Apply score adjustments based on compatability distribution
    distribution = trait_compatibility_distribution(traits2, preferences1)
    base_score = compatibility_score(traits2, preferences1, exponent=base_exponent)
    adjusted_score = base_score

    # Bonus for having multiple excellent matches
    if distribution["excellent"] >= 2:
        adjusted_score += 0.05 * (distribution["excellent"] / distribution["total"])

        # Bonus for having multiple goof matches
    if distribution["good"] >= 2:
        adjusted_score += 0.02 * (distribution["good"] / distribution["total"])

    # Penalty for poor matches
    if distribution["poor"] > 0:
        poor_penalty = 0.15 * (distribution["poor"] / distribution["total"])
        adjusted_score -= poor_penalty

    # Severe penalty for deal-breakers
    if distribution["terrible"] > 0:
        dealbreaker_penalty = 0.2 * (distribution["terrible"] / distribution["total"])
        adjusted_score -= dealbreaker_penalty

    # Cap the score
    adjusted_score = min(0.95, max(0, adjusted_score))

    # Apply sigmoid transformation
    sigmoid_input = (adjusted_score - threshold) * alpha
    probability = 1 / (1 + np.exp(-sigmoid_input))

    return probability


def analyze_match(preferences1, traits2):
    """
    Compute compatibility metrics between preferences and traits
    """
    element_compatibility = 1 - np.abs(traits2 - preferences1)

    # Get exponential compatibility scores
    exp1_score = compatibility_score(traits2, preferences1, exponent=1)
    exp2_score = compatibility_score(traits2, preferences1, exponent=2)
    exp3_score = compatibility_score(traits2, preferences1, exponent=3)

    # Get distribution of trait compatibilities
    distribution = trait_compatibility_distribution(traits2, preferences1)

    # Calculate swipe probability
    prob = swipe_probability(preferences1, traits2)

    return {
        "element_compatibility": element_compatibility,
        "linear_score": exp1_score,
        "quadratic_score": exp2_score,
        "cubic_score": exp3_score,
        "distribution": distribution,
        "swipe_probability": prob
    }


def test_algorithm():
    print("Testing with various compatibility scenarios:\n")

    # Perfect match
    perfect_prefs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    perfect_traits = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    # Good match with one dealbreaker
    good_with_dealbreaker_prefs = np.array([0.9, 0.9, 0.9, 0.9, 0.1])
    good_with_dealbreaker_traits = np.array([0.9, 0.85, 0.95, 0.9, 0.9])

    # Average match with no dealbreakers
    average_prefs = np.array([0.5, 0.6, 0.4, 0.7, 0.5])
    average_traits = np.array([0.6, 0.7, 0.5, 0.6, 0.6])

    # Poor overall match
    poor_prefs = np.array([0.2, 0.3, 0.4, 0.3, 0.2])
    poor_traits = np.array([0.8, 0.7, 0.7, 0.6, 0.8])

    # Mix of excellent and terrible
    mixed_prefs = np.array([0.1, 0.9, 0.1, 0.9, 0.5])
    mixed_traits = np.array([0.1, 0.9, 0.9, 0.1, 0.5])

    # Run analysis on all examples
    examples = [
        ("Perfect match", perfect_prefs, perfect_traits),
        ("Good with dealbreaker", good_with_dealbreaker_prefs, good_with_dealbreaker_traits),
        ("Average match", average_prefs, average_traits),
        ("Poor match", poor_prefs, poor_traits),
        ("Mixed excellent/terrible", mixed_prefs, mixed_traits),
    ]

    for name, prefs, traits in examples:
        analysis = analyze_match(prefs, traits)

        print(f"=== {name} ===")
        print(f"Element compatibility: {analysis['element_compatibility']}")
        print(f"Distribution: {analysis['distribution']['excellent']} excellent, "
              f"{analysis['distribution']['good']} good, "
              f"{analysis['distribution']['average']} average, "
              f"{analysis['distribution']['poor']} poor, "
              f"{analysis['distribution']['terrible']} terrible")
        print(f"Linear score: {analysis['linear_score']:.4f}")
        print(f"Quadratic score: {analysis['quadratic_score']:.4f}")
        print(f"Cubic score: {analysis['cubic_score']:.4f}")
        print(f"Swipe probability: {analysis['swipe_probability']:.4f}\n")

if __name__ == "__main__":
    test_algorithm()
