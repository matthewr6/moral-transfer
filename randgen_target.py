import numpy as np


def rand_target_morals(input_vec):
    assert len(input_vec) == 10

    while True:
        output_vec = np.random.randint(0, 2, 10)  # randomly generate output moral

        # Check similarity. Output should be different from the input
        combined_vec = zip(input_vec, output_vec)
        different = False
        for moral in combined_vec:
            if moral[0] != moral[1]:
                different = True
                break

        if different:
            # Check for opposing morals (e.g. care vs harm) - both can't be 1
            morals_consistent = True
            for i in range(0, 10, 2):
                if output_vec[i] == output_vec[i+1] == 1:
                    morals_consistent = False
            if morals_consistent:
                return output_vec  # No opposing morals, return True


print(rand_target_morals([0 for i in range(10)]))
