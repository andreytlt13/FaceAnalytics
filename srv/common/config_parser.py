import configparser

parser = configparser.ConfigParser()


def parse(config_path='../config/config.ini', schema='DEFAULT'):
    parser.read(config_path)
    return parser[schema]
