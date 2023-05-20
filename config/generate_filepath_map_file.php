<?php
/**
 * Script to allow easy manipulation and generation of ../config/filepath_map.json
 * Add new directories or files to this file then,
 * Execute this script with php ./generate_filepath_map_file.php
 *
 * Please keep this file updated if you prefer editing filepath_map.json directly, otherwise if changed will be overwritten if this script is used ot generate a new JSON file
 */

############################
# FILE PATH DECLARATIONS
############################
$file_path_map_array['directories']['home'] = array(
	'description' => "The home path for the user with User ID of 1000, this is normally the 'pi' user",
	'read_setting' => null,
	'return_var' => 'home'
);


$file_path_map_array['directories']['birdnet-pi'] = array('alias_for' => 'birdnet_pi');
$file_path_map_array['directories']['birdnet_pi'] = array(
	'description' => "The root directory of the BirdNET-pi install",
	'lives_under' => 'home',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/BirdNET-Pi',
	'return_var' => ''
);


$file_path_map_array['directories']['recs_dir'] = array('alias_for' => 'recordings_dir');
$file_path_map_array['directories']['recordings_dir'] = array(
	'description' => "The recording directory as specified in the config file (default: BirdSongs)",
	'lives_under' => 'home',
	'read_setting' => 'RECS_DIR',
	'replace_setting_text' => '$HOME',
	'replace_setting_text_with' => '',
	'append' => '',
);


$file_path_map_array['directories']['processed'] = array(
	'description' => "The directory which holds the recordings that have been processed",
	'lives_under' => 'recordings_dir',
	'read_setting' => 'PROCESSED',
	'replace_setting_text' => '${RECS_DIR}',
	'replace_setting_text_with' => '',
	'append' => '',
);


$file_path_map_array['directories']['extracted'] = array(
	'description' => "The directory which holds the extraction related data from detections",
	'lives_under' => 'recordings_dir',
	'read_setting' => 'EXTRACTED',
	'replace_setting_text' => '${RECS_DIR}',
	'replace_setting_text_with' => '',
	'append' => '',
);


$file_path_map_array['directories']['extracted_bydate'] = array('alias_for' => 'extracted_by_date');
$file_path_map_array['directories']['extracted_by_date'] = array(
	'description' => "The directory which holds the the extractions from detections as folders by date",
	'lives_under' => 'extracted',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/By_Date',
);


$file_path_map_array['directories']['extracted_charts'] = array(
	'description' => "The directory which holds the spectrogram charts for detections",
	'lives_under' => 'extracted',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/Charts',
);

$file_path_map_array['directories']['shifted_dir'] = array('alias_for' => 'shifted_audio');
$file_path_map_array['directories']['shifted_audio'] = array(
	'description' => "The directory which stores any frequency shifted audio",
	'lives_under' => 'extracted_by_date',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/shifted',
);


$file_path_map_array['directories']['database'] = array(
	'description' => "**NOT CURRENTLY USED** The directory which holds files related to the detections database",
	'lives_under' => 'birdnet_pi',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/database',
);

$file_path_map_array['directories']['config'] = array(
	'description' => "The directory which holds configuration files for BirdNET-pi",
	'lives_under' => 'birdnet_pi',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/config',
);

$file_path_map_array['directories']['models'] = array('alias_for' => 'model');
$file_path_map_array['directories']['model'] = array(
	'description' => "The directory which holds AI detection models",
	'lives_under' => 'birdnet_pi',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/model',
);

$file_path_map_array['directories']['python3_ve'] = array(
	'description' => "The directory which contains the Python 3 Virtual Environment",
	'lives_under' => 'birdnet_pi',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/birdnet/bin',
);

$file_path_map_array['directories']['scripts'] = array(
	'description' => "The directory which contains the script files like PHP and Shell scripts",
	'lives_under' => 'birdnet_pi',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/scripts',
);

$file_path_map_array['directories']['stream_data'] = array(
	'description' => "The directory which contains the recordings from RTSP streams",
	'lives_under' => 'recordings_dir',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/StreamData',
);

$file_path_map_array['directories']['templates'] = array(
	'description' => "The directory which contains the templates, like html, CRON script templates",
	'lives_under' => 'birdnet_pi',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/templates',
);

$file_path_map_array['directories']['www'] = array('alias_for' => 'web');
$file_path_map_array['directories']['web'] = array(
	'description' => "The directory which any files related to generating the Web UI, index.php, views.php, stylesheets etc",
	'lives_under' => 'birdnet_pi',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/homepage',
);

$file_path_map_array['directories']['www_fonts'] = array('alias_for' => 'web_fonts');
$file_path_map_array['directories']['web_fonts'] = array(
	'description' => "The directory which any fonts used by the WebUI or Python scripts",
	'lives_under' => 'www',
	'read_setting' => '',
	'replace_setting_text' => '',
	'replace_setting_text_with' => '',
	'append' => '/static',
);


############################
# FILE PATH DECLARATIONS
############################
$file_path_map_array['files']['analyzing_now.txt'] = array(
	'lives_under' => 'birdnet_pi',
	'append' => '/analyzing_now.txt'
);

$file_path_map_array['files']['apprise.txt'] = array(
	'lives_under' => 'birdnet_pi',
	'append' => '/apprise.txt'
);

$file_path_map_array['files']['birdnet.conf'] = array(
	'lives_under' => 'birdnet_pi',
	'append' => '/birdnet.conf'
);

$file_path_map_array['files']['etc_birdnet.conf'] = array(
	'lives_under' => '',
	'append' => '',
	'return_var' => '/etc/birdnet/birdnet.conf'
);

$file_path_map_array['files']['BirdDB.txt'] = array(
	'lives_under' => 'birdnet_pi',
	'append' => '/BirdDB.txt',
);

$file_path_map_array['files']['birds.db'] = array(
	'lives_under' => 'scripts',
	'append' => '/birds.db',
);

$file_path_map_array['files']['blacklisted_images.txt'] = array(
	'lives_under' => 'scripts',
	'append' => '/blacklisted_images.txt',
);

$file_path_map_array['files']['disk_check_exclude.txt'] = array(
	'lives_under' => 'scripts',
	'append' => '/disk_check_exclude.txt',
);

$file_path_map_array['files']['email_template'] = array(
	'lives_under' => 'scripts',
	'append' => '/email_template',
);

$file_path_map_array['files']['email_template2'] = array(
	'lives_under' => 'scripts',
	'append' => '/email_template2',
);

$file_path_map_array['files']['exclude_species_list.txt'] = array(
	'lives_under' => 'scripts',
	'append' => '/exclude_species_list.txt',
);

$file_path_map_array['files']['filepath_map.json'] = array(
	'lives_under' => 'config',
	'append' => '/filepath_map.json',
);

$file_path_map_array['files']['firstrun.ini'] = array(
	'lives_under' => 'birdnet_pi',
	'append' => '/firstrun.ini',
);

$file_path_map_array['files']['.gotty'] = array(
	'lives_under' => 'home',
	'append' => '/.gotty',
);

$file_path_map_array['files']['HUMAN.txt'] = array(
	'lives_under' => 'birdnet_pi',
	'append' => '/HUMAN.txt',
);

$file_path_map_array['files']['IDFILE'] = array('alias_for' => 'IdentifiedSoFar.txt');
$file_path_map_array['files']['IdentifiedSoFar.txt'] = array(
	'lives_under' => 'home',
	'read_setting' => 'IDFILE',
	'replace_setting_text' => '$HOME',
	'replace_setting_text_with' => '',
	'append' => '',
);

$file_path_map_array['files']['include_species_list.txt'] = array(
	'lives_under' => 'scripts',
	'append' => '/include_species_list.txt',
);

$file_path_map_array['files']['labels.txt'] = array(
	'lives_under' => 'model',
	'append' => '/labels.txt',
);

$file_path_map_array['files']['labels.txt.old'] = array(
	'lives_under' => 'model',
	'append' => '/labels.txt.old',
);

$file_path_map_array['files']['labels_flickr.txt'] = array(
	'lives_under' => 'model',
	'append' => '/labels_flickr.txt',
);

$file_path_map_array['files']['labels_lang.txt'] = array(
	'lives_under' => 'model',
	'append' => '/labels_lang.txt',
);

$file_path_map_array['files']['labels_l18n.zip'] = array(
	'lives_under' => 'model',
	'append' => '/labels_l18n.zip',
);

$file_path_map_array['files']['labels_nm.zip'] = array(
	'lives_under' => 'model',
	'append' => '/labels_nm.zip',
);


$file_path_map_array['files']['lastrun.txt'] = array(
	'lives_under' => 'scripts',
	'append' => '/lastrun.txt',
);


$file_path_map_array['files']['python3'] = array(
	'lives_under' => 'python3_ve',
	'append' => '/python3 ',
);

$file_path_map_array['files']['python3_appraise'] = array(
	'lives_under' => 'python3_ve',
	'append' => '/apprise ',
);

$file_path_map_array['files']['species.py'] = array(
	'lives_under' => 'scripts',
	'append' => '/species.py',
);

$file_path_map_array['files']['thisrun.txt'] = array(
	'lives_under' => 'scripts',
	'append' => '/thisrun.txt',
);

//Output array as JSON and pretty print it
file_put_contents('./filepath_map.json', json_encode($file_path_map_array, JSON_PRETTY_PRINT));