# Utility Functions
# Mainly for configuration management

import argparse
import yaml

__all__ = ['load_model_from_yaml','make_directory','run_config_from_args','fit_model_argparser']

def load_model_from_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def make_directory(dirpath):
    os.makedirs(dirpath,exist_ok=True)

def run_config_from_args(args):
    run_config = {'debug' : args.debug,
                     'console' : args.console,
                     'odir' : args.odir if args.odir is not None else 'out/experiment_%s' % time.strftime("%Y.%m.%d_%H.%M.%S"),
                     'model': load_model_from_yaml(args.model),
                     'seed': args.seed,
                     'algorithm': args.algorithm,
                     'tolerance': args.tolerance,
                     'max_updates' : int(args.max_updates),
                     'save_interval' : int(args.log_interval),
                     'save_interval' : int(args.save_interval)}

    return run_config


def fit_model_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--odir", type=str, default='out', help="output directory")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-u",'--max_updates', type=float, default=1e4, help="max number of iterations")
    parser.add_argument("-e",'--tolerance', type=float, default=1e-4, help="error tolerance")
    parser.add_argument("-s",'--save_interval', type=float, default=1e3, help="current estimates save Interval")
    parser.add_argument("-a",'--algorithm', type=str, default='VFI', help="which algorithm to use")
    parser.add_argument("-l",'--log_interval', type=int, default=50, help="Log Interval")
    parser.add_argument("-m",'--model', type=str, default="./model/simplified.yaml", help="Which model to use")
    parser.add_argument("-co", "--console", action="store_true", help="log to console")
    parser.add_argument('--seed', type=int, default=543, metavar='N',help='random seed (default: 543). -1 indicates no seed')

    return parser
