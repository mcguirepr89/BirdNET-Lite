import os

CONFIG = {}
userDir = os.path.expanduser('~')


def get_directory(directory_name):
    # Directory Path Helper, returns a full directory path for a supplied directory name e+g home, processed, extracted

    directory_name = directory_name.lower()

    if directory_name == "home":
        return userDir
    ##
    elif directory_name == "birdnet-pi" or directory_name == "birdnet_pi":
        return get_directory('home') + '/BirdNET-Pi'
    ##
    elif directory_name == "recs_dir" or directory_name == "recordings_dir":
        recs_directory_name_setting = CONFIG.get('RECS_DIR')
        return recs_directory_name_setting.replace('$HOME', get_directory('home'))
    ##
    elif directory_name == "processed":
        processed_directory_name_setting = CONFIG.get('PROCESSED')
        return get_directory('recs_dir') + processed_directory_name_setting.replace('${RECS_DIR}', '')
    ##
    elif directory_name == "extracted":
        extracted_directory_name_setting = CONFIG.get('EXTRACTED')
        return get_directory('recs_dir') + extracted_directory_name_setting.replace('${RECS_DIR}', '')
    ##
    elif directory_name == "extracted_bydate" or directory_name == "extracted_by_date":
        return get_directory('extracted') + '/By_Date'
    ##
    elif directory_name == "extracted_charts":
        return get_directory('extracted') + '/Charts'
    ##
    elif directory_name == "shifted_audio" or directory_name == "shifted_directory_name":
        return get_directory('home') + '/BirdSongs/Extracted/By_Date/shifted'
    ##
    elif directory_name == "database":
        ## NOT CURRENTLY USED
        return get_directory('birdnet_pi') + '/database'
    ##
    elif directory_name == "config":
        ## NOT CURRENTLY USED
        return get_directory('birdnet_pi') + '/config'
    ##
    elif directory_name == "models" or directory_name == "model":
        return get_directory('birdnet_pi') + '/model'
    ##
    elif directory_name == "python3_ve":
        return get_directory('birdnet_pi') + '/birdnet/bin'
    ##
    elif directory_name == "scripts":
        return get_directory('birdnet_pi') + '/scripts'
    ##
    elif directory_name == "stream_data":
        return get_directory('recs_dir') + '/StreamData'
    ##
    elif directory_name == "templates":
        return get_directory('birdnet_pi') + '/templates'
    ##
    elif directory_name == "web" or directory_name == "www":
        return get_directory('birdnet_pi') + '/homepage'
    ##
    elif directory_name == "web_fonts" or directory_name == "www_fonts":
        return get_directory('www') + '/static'
    ##

    return ""


def get_file_path(filename):
    # Returns the full filepath for the specified filename

    if filename == "analyzing_now.txt":
        return get_directory('birdnet_pi') + "/analyzing_now.txt"
    ##
    elif filename == "apprise.txt":
        return get_directory('birdnet_pi') + "/apprise.txt"
    ##
    elif filename == "birdnet.conf":
        return get_directory('birdnet_pi') + "/birdnet.conf"
    ##
    elif filename == "etc_birdnet.conf":
        return "/etc/birdnet/birdnet.conf"
    ##
    elif filename == "BirdDB.txt":
        return get_directory('birdnet_pi') + "/BirdDB.txt"
    ##
    elif filename == "birds.db":
        return get_directory('scripts') + "/birds.db"
    ##
    elif filename == "blacklisted_images.txt":
        return get_directory('scripts') + "/blacklisted_images.txt"
    ##
    elif filename == "disk_check_exclude.txt":
        return get_directory('scripts') + "/disk_check_exclude.txt"
    ##
    elif filename == "email_template":
        return get_directory('scripts') + "/email_template"
    ##
    elif filename == "email_template2":
        return get_directory('scripts') + "/email_template2"
    ##
    elif filename == "exclude_species_list.txt":
        return get_directory('scripts') + "/exclude_species_list.txt"
    ##
    elif filename == "firstrun.ini":
        return get_directory('birdnet_pi') + "/firstrun.ini"
    ##
    elif filename == ".gotty":
        return get_directory('home') + "/.gotty"
    ##
    elif filename == "HUMAN.txt":
        return get_directory('birdnet_pi') + "/HUMAN.txt"
    ##
    elif filename == "IdentifiedSoFar.txt" or filename == "IDFILE":
        id_file_location = CONFIG.get('IDFILE')
        return get_directory('home') + id_file_location.replace('$HOME', '')
    ##
    elif filename == "include_species_list.txt":
        return get_directory('scripts') + "/include_species_list.txt"
    ##
    elif filename == "labels.txt" or filename == "labels.txt.old":
        return get_directory('model') + '/' + filename
    ##
    elif filename == "labels_flickr.txt":
        return get_directory('model') + "/labels_flickr.txt"
    ##
    elif filename == "labels_l18n.zip":
        return get_directory('model') + "/labels_l18n.zip"
    ##
    elif filename == "labels_lang.txt":
        return get_directory('model') + "/labels_lang.txt"
    ##
    elif filename == "labels_nm.zip":
        return get_directory('model') + "/labels_nm.zip"
    ##
    elif filename == "lastrun.txt":
        return get_directory('scripts') + "/lastrun.txt"
    ##
    elif filename == "python3":
        return get_directory('python3_ve') + "/python3 "
    ##
    elif filename == "python3_appraise":
        return get_directory('python3_ve') + "/apprise "
    ##
    elif filename == "species.py":
        return get_directory('scripts') + "/species.py"
    ##
    elif filename == "thisrun.txt":
        return get_directory('scripts') + "/thisrun.txt"
    ##

    return ""
