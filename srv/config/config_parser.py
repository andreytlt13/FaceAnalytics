import configparser

parser = configparser.ConfigParser()


def parse_default(config_path='../config/config.ini'):
    parser.read(config_path)
    return parser['DEFAULT']
