import configparser

parser = configparser.ConfigParser()


# ../config/config.ini
def parse(config_path='/Users/andrey/PycharmProjects/FaceAnalytics/srv/config/config.ini', schema='DEFAULT'):
    parser.read(config_path)
    return parser[schema]
