import numpy as np
from lovers import men, women, Man, Woman
from prob_swipe import compatibility_score, end_compatibility_score, swipe_probability, analyze_match, soft_accuracy, accuracy
# from sklearn.metrics import mean_squared_error
from itertools import product
import random


class PreferenceModel:
    """
    Model to learn user preferences based on swipe history
    """
    def __init__(self, i_d, initial_preferences=None, learning_rate=0.4, regularization=0.02):
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
        self.id = i_d

    def update_preferences(self, profile, swipe_right, men_val = False):
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

        for i in range(len(self.preferences)):
            if self.preferences[i] > 1:
                self.preferences[i] =1
            elif self.preferences[i] < 0:
                self.preferences[i] = 0

        #self.preferences= [min(max(x, 0), 1) for x in self.preferences]

        # if self.id == 0:
        #     print("update prefs: " + " id: " + str(self.id) + ": " + str(self.preferences))
        #     print("person that updated: " + str(profile.traits) + " swipe right: " + str(swipe_right))
        #     print()


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
        
       #weight = 0.5

        # Combine direction, regularization, and adaptive weighting
        gradient = direction * regularized_diff * weight

        #new update more aggressive
        #gradient = direction * regularized_diff

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
        score = compatibility_score(profile.traits, self.preferences, exponent=.5)

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
        best_index = min(potential_matches, key=lambda x: swipe_heuristic(self.preferences, x.traits))
        return best_index.id

#return value that is the distance between the traits and prefs (can generalize to weighted min fit later)
#make sure the heuristics don't allow you to swipe on people you have seen before
def min_fit_heuristic(prefs, traits):
    return np.sum(np.abs(prefs - traits))

def max_fit_heuristic(prefs, traits):
    return -1 * np.sum(np.abs(prefs - traits))

def create_user_model(user, index, initial_preferences=None, regularization = None):
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
        initial_preferences = np.random.uniform(0.45, 0.55, len(user.preferences))

    if regularization == None:
        return PreferenceModel(initial_preferences=initial_preferences, i_d=index)
    else:
        return PreferenceModel(initial_preferences=initial_preferences, i_d=index, regularization = regularization)


def simulate_learning(user, potential_matches, current_swipes, total_swipes, n_swipes=30, swipe_heuristic = None, model = None, men_val = False):
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
        print("here")
        model = create_user_model(user, user.id, initial_preferences=None)

    # Track results for analysis
    results = []

    matches = potential_matches.copy()

    #stimulate

    if(user.id == 0):
        print(user.id)
        print("real: " + str(user.preferences))
        print("men" + str(men_val))
    # Simulate swipes
    #trying to add in dynamic learning rate to get close to best fits before slowing down

    for i in range(min(n_swipes, len(matches))):
        if(user.id == 0):
            print("accuracy: " + str(compatibility_score(user.preferences, model.preferences)))
            print(model.preferences)
        if swipe_heuristic is None:
            profile = matches[i]
        else:
            profile_index = model.find_next_swipe(potential_matches=potential_matches, swipe_heuristic=swipe_heuristic)
            profile = matches[profile_index]
        
        if(user.id == 0):
            print(profile)

        # model.learning_rate = 0.1
        ## print(model.learning_rate)
        model.learning_rate = ((min(n_swipes, len(matches)) - i) / min(n_swipes, len(matches))) * model.learning_rate
        

        # Calculate true swipe probability based on user's actual preferences
        true_compatibility = compatibility_score(profile.traits, user.preferences, exponent=0.5)

       

        # Apply sigmoid to get probability
        alpha = 10
        threshold = 0.65
        sigmoid_input = (true_compatibility - threshold) * alpha
        swipe_prob = 1 / (1 + np.exp(-sigmoid_input))

        # Determine swipe direction (with some randomness)
        swipe_right = 0.5 > swipe_prob

        # if(user.id == 0):
        #     print("swiping")
        #     print(true_compatibility)
        #     print(swipe_prob)

        # Update model with this swipe
        # if model.id == 0:
        #     print("user prefs: " + str(user.preferences))
        model.update_preferences(profile, swipe_right, men_val = men_val)

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
def threshold_matches(male_models, female_models, t, conf_t, male_active, female_active):

    print(male_models[0].preferences)

    found_matches = []
    potential_matches = []

    #only check for compatibility on the items because that represents the people who have yet to be matched
    for m_number in male_active:
        for w_number in female_active:
            m_score = accuracy(men[m_number].traits, female_models[w_number].get_expected_preferences())
            w_score = accuracy(women[w_number].traits, male_models[m_number].get_expected_preferences())

            m_mean = np.mean(male_models[m_number].get_preference_confidence())
            w_mean = np.mean(female_models[w_number].get_preference_confidence())
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
    # men[0].preferences = [.9,.9,.9,.9,.1]
    if num_rounds <= 0:
        return

    matches = []  #list of men and women indexed by id who are matched and removed from the pool of people

    male_dict = {}  #map each man to their model for training
    female_dict = {}

    #initialize the models with the user and empty preferences
    for index,m in enumerate(men):
        male_dict[m.id] = create_user_model(m, index, initial_preferences=None)
        male_dict[m.id].preferences = test_convergence(men[m.id].preferences, 50, True)
        # if index == 1:
        #      print(male_dict[m.id].preferences)
        #      print(men[m.id].preferences)


    for index,w in enumerate(women):
        female_dict[w.id] = create_user_model(w, index, initial_preferences=None)
        female_dict[w.id].preferences = test_convergence(women[w.id].preferences, 50, True)

    male_active = list(range(len(men)))
    female_active = list(range(len(women)))


    

    for i in range(num_rounds):
        for m_index in male_active:
            simulate_learning(men[m_index], women, n_swipes=swipes_per_round, current_swipes=i * swipes_per_round, total_swipes=num_rounds * swipes_per_round,
                              swipe_heuristic=min_fit_heuristic, model = male_dict[m_index], men_val = True)
        for w_index in female_active:
            simulate_learning(women[w_index], men, n_swipes=swipes_per_round, current_swipes=i * swipes_per_round, total_swipes=num_rounds * swipes_per_round,
                              swipe_heuristic=min_fit_heuristic, model=female_dict[w_index], men_val = False)

        fm = threshold_matches(male_dict, female_dict, comp_t, confidence_t,male_active, female_active)

        # remove from dating pool if matched
        for m_index, w_index in fm:
            male_active.remove(m_index)
            female_active.remove(w_index)

        #right now we don't remove the ability to swipe on people who have been matched already but you can't match with them
        #and we don't further train their models, see if that is the desired behavior

        matches = list(set(matches).union(fm))

    #at the end of all of the rounds, check the model learned prefs compared to the true prefs of each person

    score_dict = {}
    #first go through all matches and compare their compatibility with true compatibility
    for m in matches:
        print("m: " + str(m[0]) + " f: " + str(m[1]))
        model_score = min(compatibility_score(men[m[0]].traits,female_dict[m[1]].get_expected_preferences()),compatibility_score(women[m[1]].traits, male_dict[m[0]].get_expected_preferences()))
        real_score = min(compatibility_score(men[m[0]].traits,women[m[1]].preferences),compatibility_score(women[m[1]].traits, men[m[0]].preferences))
        score_dict[m] = model_score - real_score

    #next calculate the general MSE between the learned and real prefs for each person
    m_error_dict = {}

    m_mse = np.mean(np.square(([men[m.id].preferences for m in men],
                               [male_dict[m.id].get_expected_preferences() for m in men])))
    w_mse = np.mean(np.square(([women[m.id].preferences for m in women],
                               [female_dict[m.id].get_expected_preferences() for m in women])))

    print("female mse: " + str(w_mse))
    print("male mse: " + str(m_mse))


    return matches

#go through each and return dictionary mapping each person's id to other gender id they would be "compatible" with
def calculate_real_matches(men,women, threshold):
    m_match_dict = {}
    f_match_dict = {}

    for m in men:
        m_match_dict[m.id] = set()
    for w in women:
        f_match_dict[w.id] = set()

    for m,w in product(men,women):
        if accuracy(men[m.id].traits, women[w.id].preferences) > threshold:
            #match for women
            match_woman = f_match_dict[w.id]
            match_woman.add(m.id)
            f_match_dict[w.id] = match_woman

        if accuracy(women[w.id].traits, men[m.id].preferences) > threshold:
            #match for men
            match_men = m_match_dict[m.id]
            match_men.add(w.id)
            m_match_dict[m.id] = match_men




    return (m_match_dict, f_match_dict)

#compare the matches from the real possiblity to that found by the algorithm
def compare_matches(female_real, male_real, derived_matches):
    if len(derived_matches) == 0:
        return 0.0
    correct = 0.0
    partial_correct = 0.0
    for m in derived_matches:
        if m[1] in male_real[m[0]] and m[0] in female_real[m[1]]:
            correct += 1
        elif m[1] in male_real[m[0]] or m[0] in female_real[m[1]]:
            partial_correct += 1

    print("correct percentage: " + str(float(correct) / float(len(derived_matches))))
    print("partial percentage: " + str(float(partial_correct) / float(len(derived_matches))))

    return float(correct) / float(len(derived_matches))

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


def test_convergence(preferences,num_swipes = 1000, modified_lr = False):

    m = Man(0,[0,0,0,0,0], preferences)

    model = create_user_model(m, m.id)

    for i in range(num_swipes):

        # Parameters
        length = 5  # Number of values you want
        mean = 0.5
        std_dev = 0.2

        values = np.random.normal(loc=mean, scale=std_dev, size=length)
        clipped_values = np.clip(values, 0, 1)  # Force values into [0, 1] range

        # print(clipped_values.tolist())

        # Convert to a list if needed
        values_list = values.tolist()

        #generate a random person
        profile = Woman(0,[0,0,0,0,0],values_list)


        true_compatibility = compatibility_score(profile.traits, m.preferences, exponent=0.5)
                    # print("Current Compatability: " + str(true_compatibility))
        # Apply sigmoid to get probability
        alpha = 10
        threshold = 0.65
        sigmoid_input = (true_compatibility - threshold) * alpha
        swipe_prob = 1 / (1 + np.exp(-sigmoid_input))

        # Determine swipe direction (with some randomness)
        #swipe_right = random.random() > swipe_prob

        #no randomness
        swipe_right = 0.5 > swipe_prob

    
        # print("swiping 2")
        # print(true_compatibility)
        # print(swipe_prob)





        # Update model with this swipe
        # if model.id == 0:
        #     print("user prefs: " + str(m.preferences))
        model.update_preferences(profile, swipe_right, men_val=True)

        if modified_lr:
            # if model.learning_rate >= 0.01:
            #     model.learning_rate -= 0.0005

            model.learning_rate = ((num_swipes - i) / num_swipes) * model.learning_rate


    return model.preferences
    # print(model.preferences)
    # print(preferences)
    
    # print("End Compatability: " + str(1 - compatibility_score(model.preferences, preferences)))
    # print("End Results: " + str(analyze_match(model.preferences, preferences)))


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
    model, results = simulate_learning(test_user, potential_matches, current_swipes=0, total_swipes=10, n_swipes=10, swipe_heuristic=min_fit_heuristic)

    # Analyze results
    analyze_learning_results(results)

    # Test preference prediction for a new profile
    new_profile = women[1]  # Use last woman as a new profile
    predicted_prob = model.predict_swipe(new_profile)

    # Calculate true compatibility
    true_swipe = swipe_probability(test_user.preferences, new_profile.traits)

    print("\n=== Swipe Prediction Test ===")
    print(f"Profile: Woman {new_profile.id}")
    print(f"Traits {new_profile.traits}")
    print(f"Predicted right swipe probability: {predicted_prob:.4f}")
    print(f"True right swipe probability: {true_swipe:.4f}")

def euclidean_compatibility(v1, v2):
    # Max possible distance for 5 traits is sqrt(5)
    distance = np.linalg.norm(np.array(v1) - np.array(v2))
    max_distance = np.sqrt(len(v1))  # sqrt(5) if length = 5
    score = 1 - (distance / max_distance)  # now 1 = perfect match, 0 = worst
    return score


if __name__ == "__main__":
    length = 5  # Number of values you want
    mean = 0.5
    std_dev = 0.15

    values = np.random.normal(loc=mean, scale=std_dev, size=length)
    clipped_values = np.clip(values, 0, 1)  # Force values into [0, 1] range

    print(clipped_values.tolist())

    # Convert to a list if needed
    values_list = values.tolist()
    values_list = [0.7,.7,.7,.7,.7]

    print("\n----------------------------------------------------------")
    print("--------------------- Beginning Test ---------------------")
    print("---------------------------------------------------------- \n")
    test_convergence(values_list,50,modified_lr = True)
    matches = round_structure(comp_t = 0.5)
    print("Matches")
    print(matches)
    print("Percent Matched: " + str(len(matches) / 20))
    for couple in matches:
        if(couple[0] != couple[1]):
            print("diff")

    m_match_dict, f_match_dict = calculate_real_matches(men,women,0.7)
    print("men matches: " + str(m_match_dict))
    print("female matches: " + str(f_match_dict))

    print("match accuracy: " + str(compare_matches(m_match_dict, f_match_dict, matches)))

    # print("matches: " + str(matches))

