import os
from dimacs2BDD import *
import shutil



def sample_bdd(dimacs_path):
    print("Check for BDD")
    name = os.path.splitext(os.path.basename(dimacs_path))[0]
    folder = os.path.basename(os.path.dirname(dimacs_path))
    bdd_path = os.path.join(os.path.dirname(dimacs_path), f"{name}.dddmp")

    print(f"BDD path: {bdd_path}\n")
    print(f"folder: {folder}\n")

    if not os.path.exists(bdd_path):
        print("No BDD found, creating BDD")
        convert_dimacs_to_bdd(dimacs_path, "tmp_bdd/")
        os.rename(f"tmp_bdd/{folder}/{name}.dddmp", bdd_path)
        try:
            shutil.rmtree("tmp_bdd/")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

sample_bdd("/home/jahns/Loki/ressources/systems/test/Hierons2020.dimacs")
