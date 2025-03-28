import numpy as np
from scipy.stats import truncnorm


def get_truncated_normal(mean=0.5, sd=0.15, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_unique_rows(n_people, n_traits, tn, decimals=5):
    unique_rows = []
    while len(unique_rows) < n_people:
        # Generate and round the new row
        new_row = np.round(tn.rvs(n_traits), decimals=decimals)
        # Check for uniqueness (using np.array_equal for exact match)
        if not any(np.array_equal(new_row, row) for row in unique_rows):
            unique_rows.append(new_row)
    return np.array(unique_rows)

# Parameters
n_people = 20
n_traits = 5
tn = get_truncated_normal()

# Generate unique traits
traits = generate_unique_rows(n_people, n_traits, tn)
print("Traits:\n", traits)

# If you also need unique preferences, you can generate them similarly:
preferences = generate_unique_rows(n_people, n_traits, tn)
print("Preferences:\n", preferences)


class Man:
    def __init__(self, id, traits, preferences):
        self.id = id
        self.traits = traits
        self.preferences = preferences

    def __repr__(self):
        return f"Man {self.id}: Traits={self.traits}, Preferences={self.preferences}"


# Create a list to store all men
men = [Man(i + 1, traits[i], preferences[i]) for i in range(n_people)]

# Example: Accessing traits of the first man (m1)
print(men[0].traits)  # This prints the traits of the first person (m1)

# Example: Print all men
for man in men:
    print(man)


class Woman:
    def __init__(self, id, traits, preferences):
        self.id = id
        self.traits = preferences  # Traits are set to preferences
        self.preferences = traits  # Preferences are set to traits

    def __repr__(self):
        return f"Woman {self.id}: Traits={self.traits}, Preferences={self.preferences}"


# Create a list to store all women
women = [Woman(i + 1, traits[i], preferences[i]) for i in range(n_people)]

# Example: Accessing traits of the first woman (w1)
print(women[0].traits)  # This prints the "traits" of the first woman

# Example: Print all women
for woman in women:
    print(woman)

print(men[0])
print(women[0])
