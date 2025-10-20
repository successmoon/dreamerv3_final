import pathlib
# import yaml
# import ruamel.yaml as yaml

root = pathlib.Path(__file__).parent
print(root)
print("yyy")
# for key, value in yaml.safe_load((root / 'data.yaml').read_text()).items():
#   globals()[key] = value



from ruamel.yaml import YAML

yaml = YAML(typ='safe', pure=True)
#root = Path('path_to_your_directory')
with (root / 'data.yaml').open() as f:
    data = yaml.load(f)
    for key, value in data.items():
        globals()[key] = value


# with (root / 'data.yaml').open() as f:
#     data = yaml.safe_load(f)
#     for key, value in data.items():
#         globals()[key] = value

# yaml = YAML(typ='safe', pure=True)

# # Load and process the YAML file
# root = Path('path_to_your_directory')  # Replace with your actual path
# with (root / 'data.yaml').open() as f:
#     data = yaml.load(f)
#     for key, value in data.items():
#         globals()[key] = value
