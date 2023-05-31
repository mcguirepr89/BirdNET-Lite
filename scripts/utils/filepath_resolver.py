import os
import json

CONFIG = {}
home = os.path.expanduser('~')
base_directory_2up = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# The directory name of the file will start out as /home/pi/BirdNET-Pi/scripts/utils
# we want to get up to /home/pi/BirdNET-Pi/, then append /config
if os.path.exists(base_directory_2up + '/config/filepath_map.json'):
    filePathMap_json_path = base_directory_2up + "/config/filepath_map.json"

filePathMap_data = {}


def load_filepath_map():
    # Loads in the JSON file containing data on directory and file paths
    global filePathMap_data, filePathMap_json_path
    if filePathMap_data is not {}:
        json_input_file = open(filePathMap_json_path)
        filePathMap_data = json.load(json_input_file)

    return filePathMap_data


def get_directory(directory_name):
    # Directory Path Helper, returns a full directory path for a supplied directory name e+g home, processed, extracted

    directory_name = directory_name.lower()

    filepathmap_directories = load_filepath_map().get('directories')

    if directory_name in filepathmap_directories:
        filepathmap_directories_selected = filepathmap_directories.get(directory_name)

        # Check to see if directory is an alias for another
        dir_alias = filepathmap_directories_selected.get('alias_for') if filepathmap_directories_selected.get('alias_for') is not None else ''
        if len(dir_alias) != 0:
            # If so load in the data for that directory
            filepathmap_directories_selected = filepathmap_directories.get(dir_alias)

        # Gather all the options into variables
        setting_value = filepathmap_directories_selected.get('read_setting') if filepathmap_directories_selected.get('read_setting') is not None else ''
        lives_under = filepathmap_directories_selected.get('lives_under') if filepathmap_directories_selected.get('lives_under') is not None else ''
        replace_text = filepathmap_directories_selected.get('replace_setting_text') if filepathmap_directories_selected.get('replace_setting_text') is not None else ''
        replace_text_with = filepathmap_directories_selected.get('replace_setting_text_with') if filepathmap_directories_selected.get('replace_setting_text_with') is not None else ''
        append = filepathmap_directories_selected.get('append') if filepathmap_directories_selected.get('append') is not None else ''
        return_var = filepathmap_directories_selected.get('return_var') if filepathmap_directories_selected.get('return_var') is not None else ''
        ##
        return_value = under_directory = ''

        # Get the directory which the directory we 're processing lives under
        if len(lives_under) != 0:
            under_directory = get_directory(lives_under)

        # Read the specified config file setting
        if len(setting_value) != 0:
            setting_value = CONFIG.get(setting_value)

        # Replace value in setting, like ${RECS_DIR} etc as they are not expanded
        if len(replace_text) != 0:
            return_value = setting_value.replace(replace_text, replace_text_with)
        else:
            return_value = setting_value

        # If a variable is specified for return, return it first
        if len(return_var) != 0:
            # return the dynamic variable, currently this is just the users home path in $home
            return globals()[f'{return_var}']
        elif len(append) != 0:
            # Append this to the end of the path, (models, scripts etc) do this as they reside under BirdNET - pi
            return under_directory + append
        else:
            # Else return the directory and result of the setting manipulation
            return under_directory + return_value

    return ""


def get_file_path(filename):
    # Returns the full filepath for the specified filename

    filepathmap_files = load_filepath_map().get('files')

    if filename in filepathmap_files:
        filepathmap_file_selected = filepathmap_files.get(filename)

        # Check to see if directory is an alias for another
        dir_alias = filepathmap_file_selected.get('alias_for') if filepathmap_file_selected.get(
            'alias_for') is not None else ''
        if len(dir_alias) != 0:
            # If so load in the data for that directory
            filepathmap_file_selected = filepathmap_files.get(dir_alias)

        # Gather all the options into variables
        setting_value = filepathmap_file_selected.get('read_setting') if filepathmap_file_selected.get('read_setting') is not None else ''
        lives_under = filepathmap_file_selected.get('lives_under') if filepathmap_file_selected.get('lives_under') is not None else ''
        replace_text = filepathmap_file_selected.get('replace_setting_text') if filepathmap_file_selected.get('replace_setting_text') is not None else ''
        replace_text_with = filepathmap_file_selected.get('replace_setting_text_with') if filepathmap_file_selected.get('replace_setting_text_with') is not None else ''
        append = filepathmap_file_selected.get('append') if filepathmap_file_selected.get('append') is not None else ''
        return_val = filepathmap_file_selected.get('return_var') if filepathmap_file_selected.get('return_var') is not None else ''
        ##
        under_directory = ''

        # Get the directory which the directory we 're processing lives under
        if len(lives_under) != 0:
            under_directory = get_directory(lives_under)

        # Read the specified config file setting
        if len(setting_value) != 0:
            setting_value = CONFIG.get(setting_value)

        # Replace value in setting, like ${RECS_DIR} etc as they are not expanded
        if len(replace_text) != 0:
            return_value = setting_value.replace(replace_text, replace_text_with)
        else:
            return_value = setting_value

        # If a variable is specified for return, return it first
        if len(return_val) != 0:
            # return the dynamic variable, currently this is just the users home path in $home
            return return_val
        elif len(append) != 0:
            # Append this to the end of the path, (models, scripts etc) do this as they reside under BirdNET - pi
            return under_directory + append
        else:
            # Else return the directory and result of the setting manipulation
            return under_directory + return_value

    return ""
