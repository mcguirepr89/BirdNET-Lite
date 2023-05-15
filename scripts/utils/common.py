from . import parse_settings
from . import filepath_resolver
from . import notifications

# Configuration settings
CONFIG = {}

# Path Variables
BIRDNET_CONF_PATH = filepath_resolver.get_file_path('birdnet.conf')
APPRISE_CONFIG = filepath_resolver.get_file_path('apprise.txt')
DB_PATH = filepath_resolver.get_file_path('birds.db')
# NOT CURRENTLY USED
ALPHA_MODEL_6K = 'BirdNET_6K_GLOBAL_MODEL'
META_MODEL_3K = 'BirdNET_GLOBAL_3K_V2.3_MData_Model_FP16'


def update_module_vars():
    filepath_resolver.CONFIG = CONFIG
    #
    notifications.userDir = filepath_resolver.get_directory('home')
    notifications.DB_PATH = DB_PATH
    notifications.APPRISE_CONFIG = APPRISE_CONFIG


def parse_config():
    # Loads setting/config
    global CONFIG
    CONFIG = parse_settings.config_to_settings(filepath_resolver.get_file_path('thisrun.txt'))
    update_module_vars()


def get_setting(setting_name):
    # Retrieves the specified setting from the config
    parse_config()

    if CONFIG.get(setting_name) is not None:
        return CONFIG.get(setting_name)
    else:
        return None


parse_config()
