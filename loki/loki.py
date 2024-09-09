import argparse
import datetime
import itertools
import os
import random
import sys
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycosat
import seaborn as sns
import yaml
from nsga2 import Nsga2
from nsga2 import kde
import lokiavm as lokiavm
from shutil import copy
import inspect
from lokiavm import LokiAvm

sns.set()


class AvmGenerator:
    def __init__(self, conf_yaml, saver):
        self.config = conf_yaml
        self.saver = saver
        # GET ATTRIBUTES FROM CONFIG-FILE
        avm_yaml = conf_yaml['AttributedModel']
        vm_yaml = conf_yaml['NonAttributedModel']
        self.n_jobs = int(self.config['NumberOfThreads']) if 'NumberOfThreads' in self.config else Vm.DEFAULT_JOBS
        self.avm = Vm(avm_yaml, n_jobs=self.n_jobs, is_attributed=True)
        self.vm = Vm(vm_yaml, n_jobs=self.n_jobs, is_attributed=False)
        print("Finished with creating variants")

    def run(self):
        best_front_dict = self.optimize()
        output_dir = self.saver.store_results(best_front_dict, self.avm)
        self.saver.copy_aux_files()
        return output_dir

    def optimize(self):
        print('START OPTIMIZING')
        if self.config['AttributedModel']['With_Variants']:
            # NSGA-II
            print("Starting NSGA-II")
            nsga2_optimizer = Nsga2(self.config, self.n_jobs)
            best_front_dict = nsga2_optimizer.nsga2(self.avm, self.vm)
        # TODO allow estimation only with KDE w/o genetic optimization
        # else:
        #     # Just KDE
        #     e_feature_list = estimation(feature_list, feature_list_for_estimation)
        #     e_feature_list_pure = list(e_feature_list.values())
        #     BestFront = [e_feature_list_pure]
        #
        #     # if provided: get initial values for estimated interactions
        #     if conf_yaml['AttributedModel']['With_Interactions'] == True:
        #         e_interaction_list = estimation(interactions_list, interaction_list_for_estimation)
        #         e_interactions_list_pure = list(e_interaction_list.values())
        #         BestFront = concatenate(e_feature_list_pure, e_interactions_list_pure).tolist()
        return best_front_dict


class AvmModificator():
    def __init__(self, conf_yaml, saver):
        self.config = conf_yaml
        self.saver = saver
        avm_yaml = conf_yaml['AttributedModel']
        self.n_jobs = int(self.config['NumberOfThreads']) if 'NumberOfThreads' in self.config else Vm.DEFAULT_JOBS
        self.random = int(self.config['RndSeed']) if 'RndSeed' in self.config else None
        np.random.seed(self.random)
        self.nsga2_rnd = int(2 ** 32 - 1)
        self.avm = Vm(avm_yaml, n_jobs=self.n_jobs, is_attributed=True)
        print('created AVM')

    def optimize(self):
        print("Starting NSGA-II")
        nsga2_optimizer = Nsga2(self.config, self.n_jobs, seed=self.nsga2_rnd)
        best_front_dict = nsga2_optimizer.nsga2_KT(self.avm)
        return best_front_dict

    def run(self):
        best_front_dict = self.optimize()
        best_front_dict_no_obj_names = {}
        for vm in best_front_dict:
            best_front_dict_no_obj_names[vm] = list(best_front_dict[vm])
        output_dir = self.saver.store_results_modification(best_front_dict_no_obj_names, self.avm)
        self.saver.copy_aux_files()
        return output_dir


class AvmComparison:
    def __init__(self):
        pass


class Vm(LokiAvm):
    DEFAULT_JOBS = 1

    def __init__(self, yml, is_attributed=False, n_jobs=-1):
        dimacs_path = yml['DIMACS-file']
        feature_influence_file = yml['Feature-file']
        interactions_influence_file = yml['Interactions-file'] if 'Interactions-file' in yml else None
        super(Vm, self).__init__(dimacs_path, feature_influence_file, interactions_influence_file)
        
        # Sampling
        sampling_method = yml['Sampling_Method'] if 'Sampling_Method' in yml else 'dfs'
        samples = yml['Samples'] if 'Samples' in yml else None
        if sampling_method == 'dfs':
            self.valid_variants = self.sample_dfs(samples)
        elif sampling_method == 'coverage-based':
            t = yml['t'] if 't' in yml else 1
            negative = yml['Negative_Wise'] if 'Negative_Wise' in yml else False
            self.valid_variants = self.sample_coverage_based(samples, t, negative)
        elif sampling_method == 'random':
            self.valid_variants = self.sample_random(samples)
        else:
            raise(f"Sampling method {sampling_method} not implemented")  

        self.interactions_specs = yml['New_Interactions_Specs'] if 'New_Interactions_Specs' in yml else None
        if is_attributed and interactions_influence_file:
            self.interactions_influence = self.parse_influence_text(interactions_influence_file)
        elif not is_attributed and self.interactions_specs:
            self.interactions_influence = self.new_interactions(self.constraints, self.feature_influences, self.interactions_specs, n_jobs)
        else:
            self.interactions_influence = None
        if self.interactions_influence:
            self.valid_complete_variants = self.annotate_interaction_coverage(self.valid_variants,
                                                                          self.feature_influences,
                                                                          self.interactions_influence)
        else:
            self.valid_complete_variants = self.valid_variants
        print("Initialized VM")

    def set_feature_influence(self, name, influence):
        self.feature_influences[name] = influence

    def set_interaction_influence(self, name, influence):
        self.interactions_influence[name] = influence

    def set_feature_influences(self, feature_influcences):
        self.feature_influences = feature_influcences

    def set_interaction_influences(self, interactions_influence):
        self.interactions_influence = interactions_influence

    def set_feature_interaction_value_vector(self, feature_interaction_value_vector):
        i = 0
        feature_interaction_value_list = feature_interaction_value_vector.ravel().tolist()[0]
        for key in self.get_feature_influences():
            self.set_feature_influence(key, feature_interaction_value_list[i])
            i += 1
        if self.interactions_influence:
            for key in self.get_interaction_influences():
                self.set_interaction_influence(key, feature_interaction_value_list[i])
                i += 1

    def get_interaction_num(self):
        if self.interactions_specs:
            n = self.interactions_specs
        else:
            n = super(Vm, self).get_interaction_num()
        return len(n)

    def get_feature_dump(self):
        lines = []
        for feature_name, influence in self.get_feature_influences().items():
            line = '{}: {}'.format(feature_name, influence)
            lines.append(line)
        dump_str = str(os.linesep).join(lines)
        return dump_str

    def get_interaction_dump(self):
        lines = []
        for feature_code, influence in self.get_interaction_influences().items():
            line = '{}: {}'.format(feature_code, influence)
            lines.append(line)
        dump_str = str(os.linesep).join(lines)
        return dump_str

    # check if interaction is true and false for at least one variant,
    # as well as if the interaction members are true and false in at least one variant
    def check_interaction(self, c, f, random_features):
        """
        A function, which checks if an interaction (1) can occure in at least one variant but
        (2) won't occure in every variant of a model given the provided list of constrains.

        Args:
            c (list): The model's constrains
            f (dict): The model's features
            random_features (list): The features, which were generated by the new_interactions function

        Returns:
            True, if the interaction can occure in at least one but not all variants
            False, if the interaction can't occure or if the features are dependent on eachother
        """
        constrains = list(c)
        for elem in random_features:
            index = list(f.keys()).index(elem)
            constrains.append([index])
        if pycosat.solve(constrains) == "UNSAT":
            return False
        constrains = list(c)
        index = list(-list(f.keys()).index(elem) for elem in random_features)
        constrains.append(index)
        if pycosat.solve(constrains) == "UNSAT":
            return False
        return True

    def new_interactions(self, constraint_list, f, specs, n_jobs):
        """
        A function to generate new interactions between features. The generation is handled by threads. The amount of threads is dependend on the number of different interaction degrees.

        Args:
            constraint_list (list): The previously aquired list (of lists) with all constraints of a model.
            f (dict): The models features.
            specs (list): The amount of new interactions, followed by value pairs for ratio in percent and interaction degree, e.g. [100, 50, 2, 50, 3].

        Returns:
            a dictionary with the new interactions as keys.
        """
        total_amount = specs[0]
        interaction_ratio = list(specs[1::2])
        interaction_degree = list(specs[2::2])
        all_new_interactions = dict()
        splitted_new_interactions = dict()
        for elem in interaction_degree:
            splitted_new_interactions["dict" + str(elem)] = {}
        if n_jobs > 0:
            number_of_threads = n_jobs
        else:
            number_of_threads = os.cpu_count()

        # some sweet, sweet error handling:
        assert (sum(interaction_ratio) == 100), (
            "The interaction ratios must sum up to 100. Currently they sum up to: ", sum(interaction_ratio))

        def worker(amount):
            new_interactions = dict()
            for elem in range(len(interaction_degree)):
                if elem == len(interaction_degree) - 1:
                    amount_new_int = amount
                else:
                    amount_new_int = round(amount / 100.0 * interaction_ratio[elem])
                    amount = amount - amount_new_int

                while amount_new_int > len(splitted_new_interactions["dict" + str(interaction_degree[elem])]):
                    legit_int = False
                    while legit_int == False:
                        random_feature = list(np.random.choice(list(f.keys())[1:], interaction_degree[elem]))
                        if self.check_interaction(constraint_list, f, random_feature):
                            legit_int = True
                            random_feature = sorted(random_feature)
                            interaction = ""
                            for i in random_feature:
                                interaction = interaction + str(i) + "#"
                            interaction = interaction[:-1]
                            splitted_new_interactions["dict" + str(interaction_degree[elem])][interaction] = ""
                # new_interactions.update(new_int_degree_subdict)
            # return new_interactions

        pool = ThreadPool()
        l = [total_amount] * number_of_threads
        pool.map(worker, l)
        inter_type_n = {}

        for i, elem in enumerate(interaction_degree):
            desired_amount = total_amount * interaction_ratio[i] / 100
            inter_type_n[i] = int(desired_amount)
        accounted_inters = sum(list(inter_type_n.values()))
        diff = total_amount - accounted_inters
        if diff > 0:
            inter_type_n[list(inter_type_n.keys())[0]] += diff

        for elem, num_inters in inter_type_n.items():
            while num_inters < len(splitted_new_interactions["dict" + str(interaction_degree[elem])]):
                rchoice = random.choice(list(splitted_new_interactions["dict" + str(interaction_degree[elem])].keys()))
                del splitted_new_interactions["dict" + str(interaction_degree[elem])][rchoice]
            all_new_interactions.update(splitted_new_interactions["dict" + str(interaction_degree[elem])])
        pool.close()
        pool.join()
        print("Finished with creating interactions")
        return all_new_interactions

    # ----------
    # HELPER FUNCTIONS
    # ----------

    def calc_performance_for_validation_variants(self):
        """
        A function to calculate the fitness/cost (depending on the model's application area) of all previously computed variants.

        Args:
             f_and_i (numpy matrix): The provided or estimated values for all features and interactions

        Returns:
            An array of all variant fitnesses/costs.
        """
        # variants = self.valid_variants
        variants = self.valid_complete_variants
        m_fitness = self.calc_performance(variants)
        return m_fitness


def main():
    # random.seed()
    parser = argparse.ArgumentParser(description='Loki')
    parser.add_argument('path', metavar='config file path', type=str, help="the config's file path")
    args = parser.parse_args()
    config_location = args.path
    run(config_location)


def run(config_location):
    with open(config_location, 'r') as ymlfile:
        yml_cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    for use_case_str in yml_cfg:
        cfg = yml_cfg[use_case_str]
        print(cfg)
        print("Performing ", use_case_str)
        if use_case_str == "AVM-Generation":
            saver = Saver(cfg, config_location, "AVM-Generation")
            avm_gen = AvmGenerator(cfg, saver)
            avm_gen.run()
            output_path = saver.directory
            print("The program terminated as expected.")
            print("Results saved to {}".format(output_path))

        elif use_case_str == "AVM-Modification":
            saver = Saver(cfg, config_location, "AVM-Modification")
            avm_mod = AvmModificator(cfg, saver)
            output_path = avm_mod.run()
            print("The program terminated as expected.")
            print("Results saved to {}".format(output_path))
        else:
            print("Usage of wrong use-case")
            print("The two possible use-cases are:")
            print("AVM-Generation")
            print("AVM-Modification")
            print("")
            print("Please check your configuration files for the right notation!")


class Saver:
    def __init__(self, config, config_location=None, result_prefix=None):
        self.config = config
        self.config_location = config_location
        if str(self.config['DirectoryToSaveResults']) != "auto":
            self.directory = str(self.config['DirectoryToSaveResults'])
        else:
            if not result_prefix:
                result_prefix = "Misc"
            time_template = "{}-results/{}-results-%Y-%m-%d_%H%M%S".format(result_prefix, result_prefix)
            self.directory = datetime.datetime.now().strftime(time_template)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.dimacs_path = self.find_dimacs_path()

    def find_dimacs_path(self):
        config_path = self.config["AttributedModel"]["DIMACS-file"]
        return config_path

    def copy_conf_doc(self):
        if self.config_location:
            copy(self.config_location, self.directory)

    def copy_template_files(self):
        copy(self.dimacs_path, self.directory)
        thor_avm_original_path = inspect.getfile(lokiavm)
        copy(thor_avm_original_path, self.directory)

    def copy_input_avm(self, input_avm):
        interactions_list = input_avm.get_interaction_influences()  # may be None
        self.store_text(self.directory, input_avm.get_feature_dump(), "input_features")
        if interactions_list is not None:
            self.store_text(self.directory, input_avm.get_interaction_dump(), "input_interactions")

    def copy_aux_files(self):
        self.copy_conf_doc()
        self.copy_template_files()

    def store_measurements(self, directory, vm: Vm, filename):
        df = pd.DataFrame(vm.valid_variants, columns=vm.feature_influences.keys())
        df['Performance'] = vm.calc_performance_for_validation_variants()
        
        csv_file_path = os.path.join(directory, filename + ".csv")
        df.to_csv(csv_file_path, index=False, sep=';')

    def store_text(self, directory, dump_str, filename):
        """
        A function to write the new features and interactions to their respective txt-files.

        Args:
            directory (str): The path to the new txt-file
            old_data (dict): The dict with feature/interaction names
            new_data (dict): The dict with new/estimated feature/interaction values
            filename (str): Name for the new file
        """
        # TODO clean up os directory methods
        new_file = os.path.join(directory, filename + ".txt")
        print("Storing to {}".format(new_file))
        with open(new_file, 'w') as ffile:
            ffile.write(dump_str)

    def store_dict(self, directory, f_name, obj):
        new_file = os.path.join(directory, f_name + ".p")
        # file_pickle = os.path.join(current_folder, 'results-{}.p'.format(f_name_clean))
        with open(new_file, 'wb') as f:
            pickle.dump(obj, f)
        abs_path = os.path.abspath(new_file)
        return abs_path

    def store_plot(self, avm, vm, filepath):
        """
        A function which takes the given and estimated feature, interaction and fitness values and compares them with them help of plot diagrams

        Args:
            avm (Vm): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the attributed model's variants
            vm (Vm): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the non-attributed model's variants
            filepath (str): path to the results folder

        """

        try:
            amount_bins = int(self.config['KDE']['NumberOfBins'])
        except:
            sys.exit("NumberOfBins must be an integer. Please check your configuration file!")

        # PREPARE THE DATA
        # real feature values
        values_rF = list(avm.get_feature_influences().values())
        values_kde_F = kde(values_rF, len(values_rF))
        values_eF = list(vm.get_feature_influences().values())
        bin_F = np.linspace(min(values_rF), max(values_rF), amount_bins)
        bin_eF = np.linspace(min(values_eF), max(values_eF), amount_bins)

        # real variant fitness values
        values_rV = avm.calc_performance_for_validation_variants()
        values_eV = vm.calc_performance_for_validation_variants()
        values_kde_V = kde(values_rV, values_eV.size)
        bin_V = np.linspace(values_rV[values_rV != 0].min(), values_rV[values_rV != 0].max(), amount_bins)
        bin_eV = np.linspace(values_eV[values_eV != 0].min(), values_eV[values_eV != 0].max(), amount_bins)

        with_interactions = avm.get_interaction_influences() is not None

        if with_interactions:
            # real interaction values
            values_rI = list(avm.get_interaction_influences().values())
            values_kde_I = kde(values_rI, len(vm.get_interaction_influences()))
            values_eI = list(vm.get_interaction_influences().values())
            bin_I = np.linspace(min(values_rI), max(values_rI), amount_bins)
            bin_eI = np.linspace(min(values_eI), max(values_eI), amount_bins)

        # INITIALIZE PLOT
        if with_interactions:
            fig = plt.figure(figsize=(30, 30))
            rF = fig.add_subplot(331)
            kdeF = fig.add_subplot(332)
            eF = fig.add_subplot(333)

            rI = fig.add_subplot(334)
            kdeI = fig.add_subplot(335)
            eI = fig.add_subplot(336)

            rV = fig.add_subplot(337)
            kdeV = fig.add_subplot(338)
            eV = fig.add_subplot(339)

        else:
            fig = plt.figure(figsize=(30, 20))
            rF = fig.add_subplot(231)
            kdeF = fig.add_subplot(232)
            eF = fig.add_subplot(233)

            rV = fig.add_subplot(234)
            kdeV = fig.add_subplot(235)
            eV = fig.add_subplot(236)

            # PLOT THE DATA
        rF.set_title("real Features")
        rF.hist(values_rF, bins=bin_F, fc="grey", density=True)
        rF.set_xlabel('value')
        rF.set_ylabel('density')

        kdeF.set_title("kde Features")
        kdeF.plot(values_kde_F[0][:, 0], values_kde_F[1], linewidth=2, color="grey", alpha=1)
        kdeF.hist(values_rF, bins=bin_F, density=True, fc="black", alpha=0.1)
        kdeF.set_xlabel('value')
        kdeF.set_ylabel('density')

        eF.set_title("estimated Features")
        eF.hist(values_eF, bins=bin_eF, density=True, fc="grey")
        eF.hist(values_rF, bins=bin_F, density=True, fc="black", alpha=0.1)
        eF.set_xlabel('value')
        eF.set_ylabel('density')

        #######
        if with_interactions:
            rI.set_title("real Interactions")
            rI.hist(values_rI, bins=bin_I, density=False, fc="grey", weights=np.ones(len(values_rI)) / len(values_rI))
            rI.set_xlabel('value')
            rI.set_ylabel('density')

            kdeI.set_title("kde Interactions")
            kdeI.plot(values_kde_I[0][:, 0], values_kde_I[1], linewidth=2, color="grey", alpha=1)
            kdeI.hist(values_rI, bins=bin_I, fc='black', alpha=0.1, density=True)
            kdeI.set_xlabel('value')
            kdeI.set_ylabel('density')

            eI.set_title("estimated Interactions")
            eI.hist(values_eI, bins=bin_eI, density=False, fc="grey", weights=np.ones(len(values_eI)) / len(values_eI))
            eI.hist(values_rI, bins=bin_I, density=False, fc="black", weights=np.ones(len(values_rI)) / len(values_rI),
                    alpha=0.1)
            eI.set_xlabel('value')
            eI.set_ylabel('density')
        ######

        rV.set_title("real Variants")
        rV.hist(values_rV, bins=bin_V, density=False, fc="grey", weights=np.divide(1, values_rV))
        rV.set_xlabel('value')
        rV.set_ylabel('density')

        kdeV.set_title("kde Variants")
        kdeV.plot(values_kde_V[0][:, 0], values_kde_V[1], linewidth=2, color="grey", alpha=1)
        kdeV.hist(values_rV, bins=bin_V, fc='black', alpha=0.1, density=True)
        kdeV.set_xlabel('value')
        kdeV.set_ylabel('density')

        eV.set_title("estimated Variants")
        eV.hist(values_eV, bins=bin_eV, density=False, fc="grey", weights=np.divide(1, values_eV))
        eV.hist(values_rV, bins=bin_V, density=False, fc="black", weights=np.divide(1, values_rV), alpha=0.1)
        eV.set_xlabel('value')
        eV.set_ylabel('density')

        # save the plot
        plt.savefig(filepath + '/plots.png', bbox_inches='tight')
        plt.savefig(filepath + '/plots.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

    def store_plot_KT(self, avm_old, avm_estimated, filepath):
        """
        A function which takes the original and modified features, interactions and fitness values and compares them with them help of plot diagrams

        Args:
            old_data (list): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the original data
            new_data (list): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the modified data

        """
        try:
            amount_bins = int(self.config['KDE']['NumberOfBins'])
        except:
            sys.exit("NumberOfBins must be an integer. Please check your configuration file!")

        color_code_old = "#009bb4"
        color_code_new = "#b71a49"
        plot_dict = {}

        # plotting features
        old_F = list(avm_old.get_feature_influences().values())
        new_F = list(avm_estimated.get_feature_influences().values())
        kde_old_F = kde(old_F, len(old_F))
        kde_new_F = kde(new_F, len(new_F))
        bin_old_F = np.linspace(min(old_F), max(old_F), amount_bins)
        bin_new_F = np.linspace(min(new_F), max(new_F), amount_bins)
        plot_dict["old_F"] = old_F
        plot_dict["new_F"] = new_F

        # variant fitness values
        old_V = avm_old.calc_performance_for_validation_variants()
        new_V = avm_estimated.calc_performance_for_validation_variants()
        kde_old_V = kde(old_V, len(old_V))
        kde_new_V = kde(new_V, len(new_V))
        bin_old_V = np.linspace(old_V[old_V != 0].min(), old_V[old_V != 0].max(), amount_bins)
        bin_new_V = np.linspace(new_V[new_V != 0].min(), new_V[new_V != 0].max(), amount_bins)
        plot_dict["old_V"] = old_V
        plot_dict["new_V"] = new_V

        with_interactions = avm_old.get_interaction_influences() is not None
        if with_interactions:
            # real interaction values
            old_I = list(avm_old.get_interaction_influences().values())
            new_I = list(avm_estimated.get_interaction_influences().values())
            kde_old_I = kde(old_I, len(old_I))
            kde_new_I = kde(new_I, len(new_I))
            bin_old_I = np.linspace(min(old_I), max(old_I), amount_bins)
            bin_new_I = np.linspace(min(new_I), max(new_I), amount_bins)
            plot_dict["old_I"] = old_I
            plot_dict["new_I"] = new_I

            # INITIALIZE PLOT        if with_interactions:
            fig = plt.figure(figsize=(40, 30))
            ax_old_features = fig.add_subplot(3, 4, 1)
            ax_new_features = fig.add_subplot(3, 4, 2)
            ax_mixed_features = fig.add_subplot(3, 4, 3)
            ax_jointplot_features = fig.add_subplot(3, 4, 4)

            ax_old_interactions = fig.add_subplot(3, 4, 5)
            ax_new_interactions = fig.add_subplot(3, 4, 6)
            ax_mixed_interactions = fig.add_subplot(3, 4, 7)
            ax_jointplot_interactions = fig.add_subplot(3, 4, 8)

            ax_old_variants = fig.add_subplot(3, 4, 9)
            ax_new_variants = fig.add_subplot(3, 4, 10)
            ax_mixed_variants = fig.add_subplot(3, 4, 11)
            ax_jointplot_variants = fig.add_subplot(3, 4, 12)

        else:  # not with_interactions:
            fig = plt.figure(figsize=(40, 20))
            ax_old_features = fig.add_subplot(2, 4, 1)
            ax_new_features = fig.add_subplot(2, 4, 2)
            ax_mixed_features = fig.add_subplot(2, 4, 3)
            ax_jointplot_features = fig.add_subplot(3, 4, 4)

            ax_old_variants = fig.add_subplot(2, 4, 5)
            ax_new_variants = fig.add_subplot(2, 4, 6)
            ax_mixed_variants = fig.add_subplot(2, 4, 7)
            ax_jointplot_variants = fig.add_subplot(3, 4, 8)

            # PLOT THE DATA
        ax_old_features.set_title("old feature values")
        hist_opacity = 0.4
        kde_opacity = 0.75
        kde_line_width = 3
        ax_old_features.hist(old_F, bins=bin_old_F, fc=color_code_old, density=True, alpha=hist_opacity)
        plot = ax_old_features.plot(kde_old_F[0][:, 0], kde_old_F[1], linewidth=kde_line_width, color=color_code_old,
                                    alpha=kde_opacity)
        ax_old_features.set_xlabel('value')
        ax_old_features.set_ylabel('density')

        ax_new_features.set_title("new feature values")
        ax_new_features.hist(new_F, bins=bin_new_F, fc=color_code_new, density=True, alpha=hist_opacity)
        ax_new_features.plot(kde_new_F[0][:, 0], kde_new_F[1], linewidth=kde_line_width, color=color_code_new,
                             alpha=kde_opacity)
        ax_new_features.set_xlabel('value')
        ax_new_features.set_ylabel('density')

        ax_mixed_features.set_title("old and new feature values")
        ax_mixed_features.hist(new_F, bins=bin_new_F, fc=color_code_old, density=True, alpha=hist_opacity)
        ax_mixed_features.hist(old_F, bins=bin_old_F, fc=color_code_new, density=True, alpha=hist_opacity)
        ax_mixed_features.plot(kde_new_F[0][:, 0], kde_new_F[1], linewidth=kde_line_width, color=color_code_new,
                               alpha=kde_opacity)
        ax_mixed_features.plot(kde_old_F[0][:, 0], kde_old_F[1], linewidth=kde_line_width, color=color_code_old,
                               alpha=kde_opacity)
        ax_mixed_features.set_xlabel('value')
        ax_mixed_features.set_ylabel('density')

        ax_jointplot_features.set_title("jointplot of old and new feature values")
        ax_jointplot_features.scatter(old_F, new_F)
        ax_jointplot_features.set_xlabel('old value')
        ax_jointplot_features.set_ylabel('new value')

        #######
        if with_interactions:
            ax_old_interactions.set_title("old interaction values")
            ax_old_interactions.hist(old_I, bins=bin_old_I, fc=color_code_old, density=True, alpha=hist_opacity)
            ax_old_interactions.plot(kde_old_I[0][:, 0], kde_old_I[1], linewidth=kde_line_width, color=color_code_old,
                                     alpha=kde_opacity)
            ax_old_interactions.set_xlabel('value')
            ax_old_interactions.set_ylabel('density')

            ax_new_interactions.set_title("new interaction values")
            ax_new_interactions.hist(new_I, bins=bin_new_I, fc=color_code_new, density=True, alpha=hist_opacity)
            ax_new_interactions.plot(kde_new_I[0][:, 0], kde_new_I[1], linewidth=kde_line_width, color=color_code_new,
                                     alpha=kde_opacity)
            ax_new_interactions.set_xlabel('value')
            ax_new_interactions.set_ylabel('density')

            ax_mixed_interactions.set_title("old and new interaction values")
            ax_mixed_interactions.hist(old_I, bins=bin_old_I, fc=color_code_old, density=True, alpha=hist_opacity)
            ax_mixed_interactions.hist(new_I, bins=bin_new_I, fc=color_code_new, density=True, alpha=hist_opacity)
            ax_mixed_interactions.plot(kde_old_I[0][:, 0], kde_old_I[1], linewidth=kde_line_width, color=color_code_old,
                                       alpha=kde_opacity)
            ax_mixed_interactions.plot(kde_new_I[0][:, 0], kde_new_I[1], linewidth=kde_line_width, color=color_code_new,
                                       alpha=kde_opacity)
            ax_mixed_interactions.set_xlabel('value')
            ax_mixed_interactions.set_ylabel('density')

            ax_jointplot_interactions.set_title("jointplot of old and new interaction values")
            ax_jointplot_interactions.scatter(old_I, new_I)
            ax_jointplot_interactions.set_xlabel('old value')
            ax_jointplot_interactions.set_ylabel('new value')

        ######

        ax_old_variants.set_title("old variant values")
        ax_old_variants.hist(old_V, bins=bin_old_V, fc=color_code_old, density=True, alpha=hist_opacity)
        ax_old_variants.plot(kde_old_V[0][:, 0], kde_old_V[1], linewidth=kde_line_width, color=color_code_old,
                             alpha=kde_opacity)
        ax_old_variants.set_xlabel('value')
        ax_old_variants.set_ylabel('density')

        ax_new_variants.set_title("new variant values")
        ax_new_variants.hist(new_V, bins=bin_new_V, fc=color_code_new, density=True, alpha=hist_opacity)
        ax_new_variants.plot(kde_new_V[0][:, 0], kde_new_V[1], linewidth=kde_line_width, color=color_code_new,
                             alpha=kde_opacity)
        ax_new_variants.set_xlabel('value')
        ax_new_variants.set_ylabel('density')

        ax_mixed_variants.set_title("old and new variant values")
        ax_mixed_variants.hist(old_V, bins=bin_old_V, fc=color_code_old, density=True, alpha=hist_opacity)
        ax_mixed_variants.hist(new_V, bins=bin_new_V, fc=color_code_new, density=True, alpha=hist_opacity)
        ax_mixed_variants.plot(kde_old_V[0][:, 0], kde_old_V[1], linewidth=kde_line_width, color=color_code_old,
                               alpha=kde_opacity)
        ax_mixed_variants.plot(kde_new_V[0][:, 0], kde_new_V[1], linewidth=kde_line_width, color=color_code_new,
                               alpha=kde_opacity)
        ax_mixed_variants.set_xlabel('value')
        ax_mixed_variants.set_ylabel('density')

        ax_jointplot_variants.set_title("jointplot of old and new variant values")
        ax_jointplot_variants.scatter(old_V, new_V)
        ax_jointplot_variants.set_xlabel('old value')
        ax_jointplot_variants.set_ylabel('new value')

        # save the plot
        plt.savefig(filepath + '/plots.png', bbox_inches='tight')
        plt.savefig(filepath + '/plots.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
        self.store_dict(filepath, "plot-data", plot_dict)

    def define_results(self, best_front__dict, results_to_be_saved, results_custom_specs=None):
        assert (results_to_be_saved in ["all", "overall-best", "custom"]), (
            "Options for ResultsToBeSaved are: all, overall-best, custom")
        if results_to_be_saved == "all":
            best_front = list(best_front__dict.keys())

        elif results_to_be_saved == "overall-best":
            obj_values = list(best_front__dict.values())
            maximum = obj_values[0]
            for elem in obj_values:
                if sum(elem) > sum(maximum):
                    maximum = elem
            for solution, values in best_front__dict.items():
                if values == maximum:
                    best_front = [solution]
        elif results_to_be_saved == "custom":
            def weighted_sum(obj, specs):
                weighted = 0
                for i in range(0, len(obj)):
                    weighted = weighted + (obj[i] * specs[i])
                return weighted

            obj_values = list(best_front__dict.values())
            maximum = obj_values[0]
            for elem in obj_values:
                if weighted_sum(elem, results_custom_specs) > weighted_sum(maximum, results_custom_specs):
                    maximum = elem
            for solution, values in best_front__dict.items():
                if values == maximum:
                    best_front = [solution]
        return best_front

    def store_results(self, best_front_dict, avm):
        conf_yaml = self.config
        print("Finished with calculating results")
        print("Start saving results")

        # define results to be saved
        result_selection = conf_yaml["ResultsToBeSaved"]
        result_custom_specs = conf_yaml["ResultsCustomSpecs"] if "ResultsCustomSpecs" in conf_yaml else None
        best_front = self.define_results(best_front_dict, result_selection, result_custom_specs)
        for i, cur_vm in enumerate(best_front):
            result_dir = os.path.join(self.directory, "result" + str(i + 1))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            self.store_measurements(result_dir, cur_vm, "new_measurements")
            self.store_text(result_dir, cur_vm.get_feature_dump(), "new_features")
            if cur_vm.uses_interactions() is not None:
                self.store_text(result_dir, cur_vm.get_interaction_dump(), "new_interactions")

            self.store_plot(avm, cur_vm, result_dir)
        print("Finished with saving results")
        return os.path.abspath(result_dir)

    def store_results_modification(self, best_front_dict, avm_old):
        conf_yaml = self.config
        print("Finished with calculating results")
        print("Start saving results")

        # define results to be saved
        result_selection = conf_yaml["ResultsToBeSaved"]
        result_custom_specs = conf_yaml["ResultsCustomSpecs"] if "ResultsCustomSpecs" in conf_yaml else None
        best_front = self.define_results(best_front_dict, result_selection, result_custom_specs)
        # save results
        self.copy_input_avm(avm_old)
        for i, cur_vm in enumerate(best_front):
            result_dir = os.path.join(self.directory, "result" + str(i + 1))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            e_interactions_list = cur_vm.get_interaction_influences()  # may be None
            self.store_text(result_dir, cur_vm.get_feature_dump(), "new_features")
            if e_interactions_list is not None:
                self.store_text(result_dir, cur_vm.get_interaction_dump(), "new_interactions")
            if 'Find_common_and_dead_features' in self.config['Search_Space'] and self.config['Search_Space'][
                'Find_common_and_dead_features']:
                bad_regions = cur_vm.bad_region()
                txt_path = os.path.join(result_dir, "bad_regions.txt")
                f = open(txt_path, "w+")
                bad_region_str = str(os.linesep).join(map(str, np.array(bad_regions)))
                f.write(bad_region_str)
                f.close()
            self.store_plot_KT(avm_old, cur_vm, result_dir)

        print("Finished with saving results")
        return os.path.abspath(result_dir)


if __name__ == "__main__":
    main()
