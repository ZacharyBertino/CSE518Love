import numpy as np
from lovers import men, women
from prob_swipe import compatibility_score, swipe_probability
from itertools import product


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
        #20 because that's the number of people in the current sample space
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

                #print('trait confidence: ' + str(trait_confidence))
                #print('trait variance: ' + str(trait_variances))
                #print('right swipe traits: ' + str(right_swipe_traits))

                # Combine base confidence with trait-specific confidence
                confidence = base_confidence * trait_confidence

                #confidence = trait_confidence
            else:
                confidence = np.full_like(self.preferences, base_confidence * 0.5)
        else:
            confidence = np.zeros_like(self.preferences)

        return confidence
    
    #find the next swipe based on minimizing lambda function defining next swipe heuristic, return a value from potential_matches
    #also filter by swipe history (don't swipe on the same person more than once)
    def find_next_swipe(self, potential_matches, swipe_heuristic):
        swipe_list = []
        for swipe in self.swipe_history:
            swipe_list.append(swipe['profile_id'])
        potential_matches = [x for x in potential_matches if x.id not in swipe_list]
        best_index = min(enumerate(potential_matches), key=lambda x: swipe_heuristic(self.preferences, x[1].traits))[0]
        return best_index

#return value that is the distance between the traits and prefs (can generalize to weighted min fit later)
#make sure the heuristics don't allow you to swipe on people you have seen before
def min_fit_heuristic(prefs, traits):
    return np.sum(np.abs(prefs - traits))

def max_fit_heuristic(prefs, traits):
    return -1 * np.sum(np.abs(prefs - traits))

def create_user_model(user, initial_preferences=None, regularization = None):
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

    if regularization == None:
        return PreferenceModel(initial_preferences)
    else:
        return PreferenceModel(initial_preferences, regularization)


def simulate_learning(user, potential_matches, n_swipes=30, swipe_heuristic = None, learning_rates = None, model = None):
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
    if model == None:
        model = create_user_model(user, initial_preferences=None)

    # Track results for analysis
    results = []

    matches = potential_matches.copy()


    if learning_rates != None:
        l_index = -1
        l_indices = [0] * min(n_swipes, len(potential_matches))
        for i in range(len(l_indices)):
            if i % ((int(len(l_indices) / len(learning_rates))) + 1) == 0:
                l_index += 1
            l_indices[i] = l_index


    # Simulate swipes
    #trying to add in dynamic learning rate to get close to best fits before slowing down
    for i in range(min(n_swipes, len(matches))):
        if learning_rates != None:
            model.learning_rate = learning_rates[l_indices[i]]#as we progress, decrease the learning rate
        if swipe_heuristic == None:
            profile = matches[i]
        else:
            profile_index = model.find_next_swipe(potential_matches=potential_matches, swipe_heuristic=swipe_heuristic)
            profile = matches[profile_index]
        

        # Calculate true swipe probability based on user's actual preferences
        true_compatibility = compatibility_score(profile.traits, user.preferences, exponent=0.5)

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


#return pairs of ids (male,female) for those who have been matched and therefore removed from the pool of potential matches
def threshold_matches(male_models, female_models, t, conf_t):

    found_matches = []
    potential_matches = []

    #only check for compatibility on the items because that represents the people who have yet to be matched
    for m_number, m_model in male_models.items():
        for w_number, w_model in female_models.items():
            m_score = compatibility_score(men[m_number].traits, w_model.get_expected_preferences())
            w_score = compatibility_score(women[w_number].traits, m_model.get_expected_preferences())

            m_mean = np.mean(m_model.get_preference_confidence())
            w_mean = np.mean(w_model.get_preference_confidence())
            #compatibility has to be high enough and model has to be confident enough on both predictions
            if min(w_score, m_score) > t and min(m_mean, w_mean) > conf_t:
                potential_matches.append(((m_number, w_number), min(w_score, m_score)))



    #sort the cumulative list of all the models and scores
    potential_matches.sort(key=lambda x: x[1], reverse=True)

    matched_men = set()
    matched_women = set()

#look at the most compatible matched and add them to the list of found matches
    for match in potential_matches:
        if match[0][0] not in matched_men and match[0][1] not in matched_women:
            found_matches.append((match[0][0], match[0][1]))
            matched_men.add(match[0][0])
            matched_women.add(match[0][1])

    return found_matches

#implement the round based structure with swipes per round and number of rounds specified. At the end of each round take the min 
#of the mutual compability score and if above threshold with certain amount of confidence, remove from the sample and continue
def round_structure(swipes_per_round = 10, num_rounds = 2, comp_t = 0.7, confidence_t = 0.3):
    if num_rounds <= 0:
        return

    matches = []  #list of men and women indexed by id who are matched and removed from the pool of people

    male_dict = {}  #map each man to their model for training
    female_dict = {}

    #initialize the models with the user and empty preferences
    for m in men:
        male_dict[m.id] = create_user_model(m, initial_preferences=None)

    for w in women:
        female_dict[w.id] = create_user_model(w, initial_preferences=None)



    for i in range(num_rounds):
        for m_index in male_dict.keys():
            simulate_learning(men[m_index], women, n_swipes=swipes_per_round, swipe_heuristic=min_fit_heuristic,
                          learning_rates=[0.25, 0.2, 0.15, 0.1, 0.05], model = male_dict[m_index])
        for w_index in female_dict.keys():
            simulate_learning(women[w_index], men, n_swipes=swipes_per_round, swipe_heuristic=min_fit_heuristic,
                          learning_rates=[0.25, 0.2, 0.15, 0.1, 0.05], model=female_dict[w_index])

        fm = threshold_matches(male_dict, female_dict, comp_t, confidence_t)

        # remove from dating pool if matched
        for m_index, w_index in fm:
            male_dict.pop(m_index, None)
            female_dict.pop(w_index, None)

        #right now we don't remove the ability to swipe on people who have been matched already but you can't match with them
        #and we don't further train their models, see if that is the desired behavior

        matches = list(set(matches).union(fm))

    return matches


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

    #take the last one as that is the final one after all the rounds of learning
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
    model, results = simulate_learning(test_user, potential_matches, n_swipes=10, swipe_heuristic=min_fit_heuristic, learning_rates=[0.25,0.2,0.15,0.1,0.05])

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
    matches = round_structure()
    print("matches: " + str(matches))

