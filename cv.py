import numpy as np
from sklearn.model_selection import ParameterGrid
from lovers import men, women
from learned_prefs import PreferenceModel


def cross_validate(users, potential_matches, param_grid, n_folds=5, swipes_per_fold=10):
    """
    Perform cross-validation on hyperparameters

    Parameters:
    - users: List of users
    - potential_matches: List of potential match profiles
    - param_grid: Dictionary of hyperparameter options to test
    - n_folds: Number of cross-validation folds
    - swipes_per_fold: Number of swipes to simulate per fold

    Returns:
    - Best hyperparameters
    - Dictionary of all results
    """
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Testing {len(param_combinations)} parameter combinations")
    results = {i: [] for i in range(len(param_combinations))}

    for user_idx, user in enumerate(users):
        print(f"Cross-validating for user {user_idx + 1}/{len(users)}")
        np.random.shuffle(potential_matches)
        profiles_per_fold = swipes_per_fold
        fold_profiles = []

        for i in range(n_folds):
            start_idx = i * profiles_per_fold
            end_idx = min(start_idx + profiles_per_fold, len(potential_matches))

            if start_idx >= len(potential_matches):
                break

            fold_profiles.append(potential_matches[start_idx:end_idx])

        for param_idx, params in enumerate(param_combinations):
            fold_errors = []

            for fold_idx, profiles in enumerate(fold_profiles):
                model = PreferenceModel(
                    initial_preferences=None,
                    learning_rate=params['learning_rate'],
                    regularization=params['regularization']
                )

                fold_results = []
                for i, profile in enumerate(profiles):
                    true_compatibility = compatibility_score(
                        profile.traits,
                        user.preferences,
                        exponent=params['compatibility_exponent']
                    )

                    sigmoid_input = ((true_compatibility -
                                     params['sigmoid_threshold'])
                                     * params['sigmoid_steepness'])
                    swipe_prob = 1 / (1 + np.exp(-sigmoid_input))
                    swipe_right = np.random.random() < swipe_prob

                    model.update_preferences(profile, swipe_right)

                    fold_results.append({
                        'swipe_number': i + 1,
                        'true_preferences': user.preferences,
                        'learned_preferences': model.preferences.copy(),
                        'preference_error': np.mean(np.abs(model.preferences - user.preferences))
                    })

                final_error = fold_results[-1]['preference_error']
                fold_errors.append(final_error)

            avg_error = np.mean(fold_errors)
            results[param_idx].append({
                'user_id': user.id,
                'avg_error': avg_error,
                'fold_errors': fold_errors
            })

    param_performance = {}
    for param_idx, param_results in results.items():
        params = param_combinations[param_idx]
        avg_error = np.mean([r['avg_error'] for r in param_results])
        param_performance[param_idx] = {
            'params': params,
            'avg_error': avg_error
        }

    best_param_idx = min(param_performance.keys(), key=lambda k: param_performance[k]['avg_error'])
    best_params = param_performance[best_param_idx]['params']
    best_error = param_performance[best_param_idx]['avg_error']

    print(f"Best parameters: {best_params}")
    print(f"Average error: {best_error:.4f}")

    return best_params, param_performance


def compatibility_score(traits, preferences, exponent=3):
    """
    Compute compatibility score

    Parameters:
    - traits: User's traits vector
    - preferences: Other user's preferences vector
    - exponent: (higher = more penalty for mismatches)

    Returns:
    - Score between 0 and 1, where 1 is perfect match
    """
    differences = np.abs(traits - preferences)
    weighted_differences = differences ** exponent
    return 1 - np.mean(weighted_differences)


def test_hyperparameter_tuning():
    """
    Test hyperparameter tuning for preference learning
    """
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'regularization': [0.02, 0.05, 0.1],
        'compatibility_exponent': [2, 3, 4],
        'sigmoid_threshold': [0.6, 0.65, 0.7],
        'sigmoid_steepness': [8, 10, 12]
    }

    test_users = men[:5]
    potential_matches = women

    best_params, all_results = cross_validate(
        test_users,
        potential_matches,
        param_grid,
        n_folds=3,
        swipes_per_fold=8
    )

    test_user = men[5]

    model = PreferenceModel(
        initial_preferences=None,
        learning_rate=best_params['learning_rate'],
        regularization=best_params['regularization']
    )

    print("\nComparing Default vs. Optimized Parameters:")
    print("Default parameters:")
    print("- learning_rate: 0.1")
    print("- regularization: 0.05")
    print("- compatibility_exponent: 3")
    print("- sigmoid_threshold: 0.65")
    print("- sigmoid_steepness: 10")

    print("\nOptimized parameters:")
    for param, value in best_params.items():
        print(f"- {param}: {value}")

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg_error'])
    print("\nTop 3 parameter combinations:")
    for i in range(min(3, len(sorted_results))):
        idx, result = sorted_results[i]
        print(f"{i + 1}. Error: {result['avg_error']:.4f}, Params: {result['params']}")


if __name__ == "__main__":
    test_hyperparameter_tuning()
    