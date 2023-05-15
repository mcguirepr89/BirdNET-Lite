from scripts.utils import filepath_resolver

filepath_resolver.CONFIG = {"RECS_DIR": "$HOME/BirdSongs",
                            "PROCESSED": "${RECS_DIR}/Processed",
                            "EXTRACTED": "${RECS_DIR}/Extracted",
                            "IDFILE": "$HOME/BirdNET-Pi/IdentifiedSoFar.txt"}

TEST_HOMES = ['/home/pi' '/home/another_user' '/opt/birdnet']

def test():
    for _test_home in TEST_HOMES:
        # Override the home location
        filepath_resolver.userDir = _test_home

        # DIRECTORY TESTS
        assert (filepath_resolver.get_directory('birdnet_pi') == _test_home + "/BirdNET-Pi")
        assert (filepath_resolver.get_directory('recs_dir') == _test_home + "/BirdSongs")
        assert (filepath_resolver.get_directory('processed') == _test_home + "/BirdSongs/Processed")
        assert (filepath_resolver.get_directory('extracted') == _test_home + "/BirdSongs/Extracted")
        assert (filepath_resolver.get_directory('extracted_by_date') == _test_home + "/BirdSongs/Extracted/By_Date")
        assert (filepath_resolver.get_directory('shifted_audio') == _test_home + "/BirdSongs/Extracted/By_Date/shifted")
        assert (filepath_resolver.get_directory('database') == _test_home + "/BirdNET-Pi/database")
        assert (filepath_resolver.get_directory('config') == _test_home + "/BirdNET-Pi/config")
        assert (filepath_resolver.get_directory('models') == _test_home + "/BirdNET-Pi/model")
        assert (filepath_resolver.get_directory('python3_ve') == _test_home + "/BirdNET-Pi/birdnet/bin")
        assert (filepath_resolver.get_directory('scripts') == _test_home + "/BirdNET-Pi/scripts")
        assert (filepath_resolver.get_directory('stream_data') == _test_home + "/BirdSongs/StreamData")
        assert (filepath_resolver.get_directory('templates') == _test_home + "/BirdNET-Pi/templates")
        assert (filepath_resolver.get_directory('web') == _test_home + "/BirdNET-Pi/homepage")

        # FILE PATH TESTS
        assert (filepath_resolver.get_file_path('analyzing_now.txt') == _test_home + "/BirdNET-Pi/analyzing_now.txt")
        assert (filepath_resolver.get_file_path('apprise.txt') == _test_home + "/BirdNET-Pi/apprise.txt")
        assert (filepath_resolver.get_file_path('birdnet.conf') == _test_home + "/BirdNET-Pi/birdnet.conf")
        assert (filepath_resolver.get_file_path('etc_birdnet.conf') == "/etc/birdnet/birdnet.conf")
        assert (filepath_resolver.get_file_path('BirdDB.txt') == _test_home + "/BirdNET-Pi/BirdDB.txt")
        assert (filepath_resolver.get_file_path('birds.db') == _test_home + "/BirdNET-Pi/scripts/birds.db")
        assert (filepath_resolver.get_file_path('blacklisted_images.txt') == _test_home + "/BirdNET-Pi/scripts/blacklisted_images.txt")
        assert (filepath_resolver.get_file_path('disk_check_exclude.txt') == _test_home + "/BirdNET-Pi/scripts/disk_check_exclude.txt")
        assert (filepath_resolver.get_file_path('email_template') == _test_home + "/BirdNET-Pi/scripts/email_template")
        assert (filepath_resolver.get_file_path('email_template2') == _test_home + "/BirdNET-Pi/scripts/email_template2")
        assert (filepath_resolver.get_file_path('exclude_species_list.txt') == _test_home + "/BirdNET-Pi/scripts/exclude_species_list.txt")
        assert (filepath_resolver.get_file_path('firstrun.ini') == _test_home + "/BirdNET-Pi/firstrun.ini")
        assert (filepath_resolver.get_file_path('.gotty') == _test_home + "/.gotty")
        assert (filepath_resolver.get_file_path('HUMAN.txt') == _test_home + "/BirdNET-Pi/HUMAN.txt")
        assert (filepath_resolver.get_file_path('IdentifiedSoFar.txt') == _test_home + "/BirdNET-Pi/IdentifiedSoFar.txt")
        assert (filepath_resolver.get_file_path('include_species_list.txt') == _test_home + "/BirdNET-Pi/scripts/include_species_list.txt")
        assert (filepath_resolver.get_file_path('labels.txt') == _test_home + "/BirdNET-Pi/model/labels.txt")
        assert (filepath_resolver.get_file_path('labels.txt.old') == _test_home + "/BirdNET-Pi/model/labels.txt.old")
        assert (filepath_resolver.get_file_path('labels_flickr.txt') == _test_home + "/BirdNET-Pi/model/labels_flickr.txt")
        assert (filepath_resolver.get_file_path('labels_l18n.zip') == _test_home + "/BirdNET-Pi/model/labels_l18n.zip")
        assert (filepath_resolver.get_file_path('labels_lang.txt') == _test_home + "/BirdNET-Pi/model/labels_lang.txt")
        assert (filepath_resolver.get_file_path('labels_nm.zip') == _test_home + "/BirdNET-Pi/model/labels_nm.zip")
        assert (filepath_resolver.get_file_path('lastrun.txt') == _test_home + "/BirdNET-Pi/scripts/lastrun.txt")
        assert (filepath_resolver.get_file_path('python3') == _test_home + "/BirdNET-Pi/birdnet/bin/python3 ")
        assert (filepath_resolver.get_file_path('python3_appraise') == _test_home + "/BirdNET-Pi/birdnet/bin/apprise ")
        assert (filepath_resolver.get_file_path('species.py') == _test_home + "/BirdNET-Pi/scripts/species.py")
        assert (filepath_resolver.get_file_path('thisrun.txt') == _test_home + "/BirdNET-Pi/scripts/thisrun.txt")
