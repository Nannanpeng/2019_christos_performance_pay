import yaml

with open('./model_specs/simon_gpr.yaml','r') as f:
  a = yaml.safe_load(f)
  import pdb; pdb.set_trace()