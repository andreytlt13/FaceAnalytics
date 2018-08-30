import configparser

parser = configparser.ConfigParser()


def parse_default(config_path='../common/config/config.ini'):
    parser.read(config_path)
    return parser['DEFAULT']
