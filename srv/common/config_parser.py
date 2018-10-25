import configparser

parser = configparser.ConfigParser()


# Please, specify schema='LOCAL' for starting apps locally
def parse(config_path='../config/config.ini', schema='DEFAULT'):
    parser.read(config_path)
    return parser[schema]
