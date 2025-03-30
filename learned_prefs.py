import numpy as np
from lovers import men, women
from prob_swipe import compatibility_score, swipe_probability


class PreferenceModel:
    """
    Model to learn user preferences based on swipe history
    """
    def __init__(self, initial_preferences=None, learning_rate=0.2, regularization=0.02):
        """
        Initialize preference learning model

        Parameters:
        - initial_preferences: Starting preference vector 
          (initialized with random values (0.45, 0.55))
        - learning_rate: How quickly the model adapts to new swipes
        - regularization: Prevents overreaction to individual swipes
        """
        self.preferences = initial_preferences
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.swipe_history = []

    def update_preferences(self, profile, swipe_right):
        """
        Update preferences based on a new swipe

        Parameters:
        - profile: The profile that was swiped on
        - swipe_right: Boolean indicating if user swiped right
        """
        # Store swipe
        self.swipe_history.append({
            'profile_id': profile.id,
            'traits': profile.traits.copy(),
            'swipe_right': swipe_right
        })

        # Initialize preferences
        if self.preferences is None:
            self.preferences = np.random.uniform(0.45, 0.55, len(profile.traits))

        # Calculate the gradient
        gradient = self.calculate_gradient(profile.traits, swipe_right)

        # Update preferences using gradient descent
        self.preferences += self.learning_rate * gradient

        # Ensure preferences stay within range [0, 1]
        self.preferences = np.clip(self.preferences, 0, 1)

    def calculate_gradient(self, traits, swipe_right):
        """
        Calculate gradient for preference update

        Parameters:
        - traits: Traits of the profile that was swiped on
        - swipe_right: Boolean indicating if user swiped right or left

        Returns:
        - Gradient vector for preference update
        """
        # If right swipe, move towards traits; if left swipe, move away
        direction = 1 if swipe_right else -1

        # Distance between current preferences and traits
        difference = traits - self.preferences

        # Apply regularization
        regularized_diff = difference * (1 - self.regularization)

        if swipe_right:
            # Adjustment for traits that differ greatly
            weight = np.abs(difference) + 0.1
        else:
            # Adjustment for traits that are close
            weight = 1 - np.abs(difference) + 0.1

        # Combine direction, regularization, and adaptive weighting
        gradient = direction * regularized_diff * weight

        return gradient

    def get_expected_preferences(self):
        """
        Return the current learned preferences
        """
        return self.preferences

    def predict_swipe(self, profile):
        """
        Predict likelihood of right swipe for a profile

        Parameters:
        - profile: Profile to evaluate

        Returns:
        - Probability of right swipe (0 to 1)
        """
        if self.preferences is None:
            return 0.5

        # Compatibility score
        score = compatibility_score(profile.traits, self.preferences, exponent=4)

        # Sigmoid transformation
        alpha = 8
        threshold = 0.6
        sigmoid_input = (score - threshold) * alpha
        probability = 1 / (1 + np.exp(-sigmoid_input))

        return probability

    def get_preference_confidence(self):
        """
        Calculate confidence in learned preferences based on swipe history

        Returns:
        - confidence: Vector of confidence values (0 to 1) for each trait
        """
        if len(self.swipe_history) < 5:
            return np.zeros_like(self.preferences)

        # More swipes = more confidence
        base_confidence = min(len(self.swipe_history) / 20, 1.0)

        # Calculate consistency of each trait across swipes
        if len(self.swipe_history) > 0:
            # Extract all traits
            all_traits = np.array([swipe['traits'] for swipe in self.swipe_history])

            # Check consistency of right swipes
            right_swipe_indices = [i for i, swipe in enumerate(self.swipe_history)
                                   if swipe['swipe_right']]

            # If we have right swipes, calculate variance of traits in profiles
            if right_swipe_indices:
                right_swipe_traits = all_traits[right_swipe_indices]
                trait_variances = np.var(right_swipe_traits, axis=0)

                # Lower variance = higher confidence
                trait_confidence = 1 - np.clip(trait_variances * 2, 0, 0.8)

                # Combine base confidence with trait-specific confidence
                confidence = base_confidence * trait_confidence
            else:
                confidence = np.full_like(self.preferences, base_confidence * 0.5)
        else:
            confidence = np.zeros_like(self.preferences)

        return confidence


def create_user_model(user, initial_preferences=None):
    """
    Create a preference model for a user

    Parameters:
    - user: User object (from men or women lists)
    - initial_preferences: Optional starting preferences

    Returns:
    - PreferenceModel instance
    """
    # If initial preferences not provided, use the user's actual preferences
    # with some random noise to simulate imperfect self-awareness
    if initial_preferences is None:
        noise = np.random.normal(0, 0.1, len(user.preferences))
        noisy_preferences = user.preferences + noise
        initial_preferences = np.clip(noisy_preferences, 0, 1)

    return PreferenceModel(initial_preferences)


def simulate_learning(user, potential_matches, n_swipes=30):
    """
    Simulate preference learning through a series of swipes

    Parameters:
    - user: User object (from men or women lists)
    - potential_matches: List of potential match profiles
    - n_swipes: Number of swipes to simulate

    Returns:
    - PreferenceModel after learning
    - List of swipe results
    """
    # Create model with no initial knowledge of preferences
    model = create_user_model(user, initial_preferences=None)

    # Track results for analysis
    results = []

    # Simulate swipes
    for i in range(min(n_swipes, len(potential_matches))):
        profile = potential_matches[i]

        # Calculate true swipe probability based on user's actual preferences
        true_compatibility = compatibility_score(profile.traits, user.preferences, exponent=3)

        # Apply sigmoid to get probability
        alpha = 10
        threshold = 0.65
        sigmoid_input = (true_compatibility - threshold) * alpha
        swipe_prob = 1 / (1 + np.exp(-sigmoid_input))

        # Determine swipe direction (with some randomness)
        swipe_right = np.random.random() < swipe_prob

        # Update model with this swipe
        model.update_preferences(profile, swipe_right)

        # Record results
        results.append({
            'swipe_number': i + 1,
            'profile_id': profile.id,
            'swipe_right': swipe_right,
            'true_compatibility': true_compatibility,
            'learned_preferences': model.preferences.copy(),
            'true_preferences': user.preferences,
            'preference_error': np.mean(np.abs(model.preferences - user.preferences)),
            'confidence': model.get_preference_confidence()
        })

    return model, results


def analyze_learning_results(results):
    """
    Analyze and print results from simulated learning

    Parameters:
    - results: List of result dictionaries from simulate_learning
    """
    errors = [r['preference_error'] for r in results]

    #print("=== Learning Results ===")
    #print(f"Initial error: {errors[0]:.4f}")
    #print(f"Final error: {errors[-1]:.4f}")
    #print(f"Error reduction: {(1 - errors[-1] / errors[0]) * 100:.1f}%")

    # Print confidence at end
    final_confidence = results[-1]['confidence']
    print(f"Final confidence: {np.mean(final_confidence):.4f}")

    print("\nTrue vs. Learned Preferences:")
    true_prefs = results[-1]['true_preferences']
    learned_prefs = results[-1]['learned_preferences']

    for i in range(len(true_prefs)):
        print(f"Trait {i + 1}: True={true_prefs[i]:.4f}, Learned={learned_prefs[i]:.4f}, "
              f"Error={abs(true_prefs[i] - learned_prefs[i]):.4f}, "
              f"Confidence={final_confidence[i]:.4f}")


def test_preference_learning():
    """
    Test the preference learning algorithm
    """
    # Select a sample user
    test_user = men[1]
    potential_matches = women

    print(f"Testing with Man {test_user.id}")
    print(f"Traits {test_user.traits}")
    print(f"True preferences: {test_user.preferences}")

    # Run simulation
    model, results = simulate_learning(test_user, potential_matches, n_swipes=20)

    # Analyze results
    analyze_learning_results(results)

    # Test preference prediction for a new profile
    new_profile = women[1]  # Use last woman as a new profile
    predicted_prob = model.predict_swipe(new_profile)

    # Calculate true compatibility
    true_swipe = swipe_probability(new_profile.traits, test_user.preferences)

    print("\n=== Swipe Prediction Test ===")
    print(f"Profile: Woman {new_profile.id}")
    print(f"Traits {new_profile.traits}")
    print(f"Predicted right swipe probability: {predicted_prob:.4f}")
    print(f"True right swipe probability: {true_swipe:.4f}")


if __name__ == "__main__":
    test_preference_learning()
