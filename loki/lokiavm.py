import os
import random

import numpy as np
import pycosat
import itertools
from random import shuffle

# from thor.thor2 import is_attributed, dimacs_path, feature_influence_file


class LokiAvm:
    DIMACS_FILE_CUES = ['dimacs', 'constraints']
    FEATURE_FILE_CUES = ['feature']
    INTERACTION_FILE_CUES = ['interaction']

    def __init__(self, dimacs_path, feature_file, interaction_file=None):
        self.feature_influences = self.parse_influence_text(feature_file) if feature_file else None
        self.constraints = self.parse_dimacs(dimacs_path)
        self.interactions_influence = None

        if not dimacs_path or not feature_file:  # or not interaction_file:
            self.print("Assuming all files are in same folder as this python script (No complete set of paths passed)")
            own_path = os.path.dirname(__file__)
            for cur_file in os.listdir(own_path):
                low_dir = str(cur_file).lower()
                if np.any([cue in low_dir for cue in LokiAvm.DIMACS_FILE_CUES]):
                    self.dimacs_path = os.path.join(cur_file)
                if np.any([cue in low_dir for cue in LokiAvm.FEATURE_FILE_CUES]):
                    self.feature_file = os.path.join(cur_file)
                if np.any([cue in low_dir for cue in LokiAvm.INTERACTION_FILE_CUES]):
                    self.interaction_file = os.path.join(cur_file)
        else:
            self.dimacs_path = dimacs_path
            self.feature_file = feature_file
            self.interaction_file = interaction_file
        self.dimacs = self.parse_dimacs(self.dimacs_path)

        self.feature_influences = self.parse_influence_text(self.feature_file) if self.feature_file else None
        if interaction_file:
            self.interactions_influence = self.parse_influence_text(
                self.interaction_file) if interaction_file else None
        else:
            self.interactions_influence = None

    def print(self, content):
        pre = str(self)
        template = '[{}] {}'
        output = template.format(pre, content)
        print(output)

    def parse_dimacs(self, path):
        """
        A function to parse a provided DIMACS-file.

        Args:
            path (str): The DIMACS-file's file path

        Returns:
            A list of lists containing all of the DIMACS-file's constrains. Each constrain is represented by a seperate sub-list.

        """
        dimacs = list()
        dimacs.append(list())
        with open(path) as mfile:
            for line in mfile:
                tokens = line.split()
                if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                    for tok in tokens:
                        lit = int(tok)
                        if lit == 0:
                            dimacs.append(list())
                        else:
                            dimacs[-1].append(lit)
        assert len(dimacs[-1]) == 0
        dimacs.pop()
        return dimacs

    def calc_performance(self, variants):
        """
        A function to calculate the fitness/cost (depending on the model's application area) of all previously computed variants.

        Args:
            variants (numpy matrix): All previously computed variants with information about which interaction they satisfy
            f_and_i (numpy matrix): The provided or estimated values for all features and interactions

        Returns:
            An array of all variant fitnesses/costs.
        """
        feature_and_influence_vector = self.get_feature_and_influence_vector()
        # root = np.ravel(feature_and_influence_vector)[0]
        variants = np.transpose(variants)
        # feature_and_influence_vector = np.delete(feature_and_influence_vector, 0, 1)
        m_fitness = np.dot(feature_and_influence_vector, variants)
        # m_fitness = np.add(m_fitness, root)
        m_fitness = np.asarray(m_fitness)
        m_fitness = m_fitness.ravel()
        return m_fitness

    def bad_region(self):
        """
        A function which tries to find bad regions in a SAT problems search space.

        Args:
            self

        Returns:
            List of Features and their setting, which result in unsatisfied assignments for the SAT
        """
        bad_regions = []
        num_features = len(self.get_feature_influences())
        constraint_list = self.constraints
        for i in range(1, num_features + 1):
            c_copy = list(constraint_list)
            c_copy.append([i])
            if pycosat.solve(c_copy) == "UNSAT":
                bad_regions.append(-i)

            c_copy = list(constraint_list)
            c_copy.append([-i])
            if pycosat.solve(c_copy) == "UNSAT":
                bad_regions.append(i)
        return bad_regions

    def get_feature_influence(self, name):
        return self.feature_influences[name]

    def get_interaction_influence(self, name):
        return self.interactions_influence[name]

    def get_feature_influences(self):
        return self.feature_influences

    def get_interaction_influences(self):
        return self.interactions_influence

    def get_feature_num(self):
        return len(self.feature_influences)

    def get_interaction_num(self):
        if self.interactions_influence:
            n = self.interactions_influence
        else:
            n = 0
        return len(n)

    def uses_interactions(self):
        specified_inderactions = self.interactions_influence is not None
        result = specified_inderactions
        return result

    def get_feature_interaction_value_vector(self):
        feature_influence_vals = list(self.get_feature_influences().values())
        if self.interactions_influence:
            interactions_list_pure = list(self.get_interaction_influences().values())
            feature_interaction_value_vector = concatenate(feature_influence_vals, interactions_list_pure)
        else:
            feature_interaction_value_vector = np.asmatrix(feature_influence_vals)
        return feature_interaction_value_vector

    def get_feature_and_influence_vector(self):
        feature_list_pure = list(self.feature_influences.values())
        if self.interactions_influence:
            interactions_list_pure = list(self.interactions_influence.values())
            feature_and_influence_vector = concatenate(feature_list_pure, interactions_list_pure)
        else:
            feature_and_influence_vector = np.asmatrix(feature_list_pure)
        return feature_and_influence_vector

    def parse_influence_text(self, m):
        """
        A function to parse a provided text-file containing a model's features or interactions.

        Args:
            m (str): The text-file's file path

        Returns:
            When parsing a feature file: a dictionary with the features' names as keys and the features' values as values.
            When parsing an interactions file: a dictionary with tuples of features concatenated by # as keys and the tuples' values as values.

        """
        features = dict()
        with open(m) as ffile:
            for line in ffile:
                line = line.replace("\n", "")
                tokens = line.split(": ")
                if len(tokens[-1]) > 0 and len(tokens) > 1:
                    features[tokens[0]] = float(tokens[-1])
                else:
                    features[tokens[0]] = ""
        return features

    def parsing_variants(self, m):
        """
        A function to parse a provided text-file containing a model's variants.

        Args:
            m (str): The text-file's file path

        Returns:
            A list of lists containing all variants. Each variant is represented by a seperate sub-list.

        """
        cnf = list()
        with open(m) as ffile:
            for line in ffile:
                line = line.replace("\n", "")
                line = [int(i) for i in list(line)]
                if len(line) > 0:
                    cnf.append(line)
        return cnf

    def sample_random(self, samples):
        """
           A function to randomly sample a specified number of variants from a model's search space.

        Args:
            samples (int): The number of variants to sample.

        Returns:
            A numpy matrix with the randomly sampled variants.Each row represents one variant.

        """


        new_c = self.constraints.copy()

        # Subtract root feature
        largest_dimacs_literal = len(self.feature_influences)
        if not np.any([largest_dimacs_literal in sub_list for sub_list in self.constraints]):
            dummy_constraint = [largest_dimacs_literal, -1 * largest_dimacs_literal]
            new_c.append(dummy_constraint)

        sol_collection = list()
        sample_counter = 0
        while len(sol_collection) < samples:
            # Generate a random solution
            random_solution = [random.choice([i, -i]) for i in range(1, largest_dimacs_literal + 1)]
            sample_counter += 1
            if random_solution and self.validate_solution(random_solution, new_c):
                solution = LokiAvm.transform2binary(random_solution)
                sol_collection.append(solution)

        print(f"Finished sampling {len(sol_collection)} samples using random sampling (testet {sample_counter}"
              f" configurations).")
        return np.asmatrix(sol_collection)


    def validate_solution(self, solution, constraints):
        """
        Helper function to validate if the given solution satisfies all constraints using pycosat.

        Args:
            solution (list): A list of integers representing a potential solution.
            constraints (list of lists): The list of constraints in CNF format.

        Returns:
            bool: True if the solution satisfies all constraints, False otherwise.
        """
        # Check if the current solution is a valid solution for the constraints using pycosat.
        augmented_constraints = constraints + [[lit] for lit in solution]

        # Check if the augmented constraints are satisfiable
        result = pycosat.solve(augmented_constraints)

        return result != 'UNSAT'

    def sample_dfs(self, samples):
        """
        A function to sample a specified number of variants from a model's search space using DFS.

        Args:
            samples (int): The number of variants to sample

        Returns:
            A numpy matrix with the sampled variants. Each row represents one variant.

        """
        new_c = self.constraints.copy()
        
        # subtract root feature
        largest_dimacs_literal = len(self.feature_influences)
        if not np.any([largest_dimacs_literal in sub_list for sub_list in self.constraints]):
            dummy_constraint = [largest_dimacs_literal, -1 * largest_dimacs_literal]
            new_c.append(dummy_constraint)
        
        # generate samples
        sol_collection = list()
        if not samples is None:
            solutions = list(itertools.islice(pycosat.itersolve(new_c), samples))
        else:
            solutions = pycosat.itersolve(new_c)
        for elem in solutions:
            solution = LokiAvm.transform2binary(elem)
            sol_collection.append(solution)
        print (f"Finished sampling {len(sol_collection)} samples using DFS")
        return np.asmatrix(sol_collection)
    
    def sample_coverage_based(self, samples, t, negative):
        """
        A function to sample a specified number of variants from a model's search space using coverage-based sampling.
        
        Args:
            samples (int): The number of variants to sample
            t (int): The coverade size 
            negative (bool): Whether to use negative t-wise sampling
        """
        new_c = self.constraints.copy()
        
        # subtract root feature
        largest_dimacs_literal = len(self.feature_influences) #- 1
        if not np.any([largest_dimacs_literal in sub_list for sub_list in self.constraints]):
            dummy_constraint = [largest_dimacs_literal, -1 * largest_dimacs_literal]
            new_c.append(dummy_constraint)
       
        # generate samples 
        sol_collection = list()
        terms = list(itertools.combinations(range(len(self.feature_influences)), t))
        if samples is not None:
            shuffle(terms)
        limit = samples if samples is not None else len(terms)
        print(f"Sampling up to {limit} samples using{' negative' if negative else ''} {t}-wise coverage sampling")
        for term in terms:
            c_copy = list(new_c)
            for option in term:
                if not negative:
                    c_copy.append([option + 1])
                else:
                    c_copy.append([-(option + 1)])
                    
            solution = pycosat.solve(c_copy)
            if solution != "UNSAT":
                new_c.append([-j for j in solution])
                solution = LokiAvm.transform2binary(solution)
                sol_collection.append(solution)
                
            if samples is not None and len(sol_collection) >= samples:
                break
            
        print (f"Finished sampling after {len(sol_collection)} samples")
        return np.asmatrix(sol_collection)

    @staticmethod
    def transform2binary(sol):
        """
        A function which takes a valid variant, consisting of positive and negative integers and transforming it into binary values
        Args:
            sol (list): A list that contains one valid variant, represented by positve and negative integers

        Returns:
            A list that contains the valid variants transformed into binary, where negative integers are now represented as 0 and positive integers as 1
        """
        sol = sorted(sol, key=abs)
        for index, elem in enumerate(sol):
            if float(elem) < 0:
                sol[index] = 0
            else:
                sol[index] = 1
        return sol

    def annotate_interaction_coverage(self, variants, feature_influences, interaction_influences):
        """
        A function which check for each variant, if they satisfy the previously provided (or estimated) interactions.
        It does so by looking up the involved features for each interaction and checking if those features are set to 1 for
        the respective variant. If so, the program appends a 1 (interaction satisfied) to the variant,
        else it append a 0 (interaction not satisfied).

        Args:
            variants (numpy matrix): All previously computed variants, which satisfy the provided constrains
            feature_influences (dict): All features with their names as keys and their values as values
            interaction_influences (dict): All interactions with feature tuples as keys and their values as values

        Returns:
            A numpy matrix with variants and information about which interactions they satisfy.
            Each row represents one variant and its interactions information.

        """
        valid_interaction = np.array([[1]])

        def check_for_interaction(row):
            for elem in interaction_influences.keys():
                valid_interaction[0, 0] = 1
                tokens = elem.split("#")
                for feature in tokens:
                    index = list(feature_influences.keys()).index(feature) - 1
                    if row[0, index] == 0:
                        valid_interaction[0, 0] = 0
                        break
                row = np.concatenate((row, valid_interaction), axis=1)  # np.insert(row, -1, valid_interaction)
            return row

        variants = np.apply_along_axis(check_for_interaction, axis=1, arr=variants)
        return variants


def concatenate(list_a, list_b):
    """
    A function to convert two lists into arrays and concatenate them.

    Args:
        list_a (list): Values for all features.
        list_b (list): Values for all interactions.

    Returns:
        An array with the concatenated feature and interaction values.

    """
    m_f = np.asarray(list_a)
    m_i = np.asarray(list_b)
    f_and_i = np.append(m_f, m_i)
    f_and_i = np.asmatrix(f_and_i)

    return f_and_i
