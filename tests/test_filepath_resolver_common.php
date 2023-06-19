<?php
include_once "../scripts/common.php";

const TEST_HOMES = ['/home/pi', '/home/another_user', '/opt/birdnet'];

$config = [
	"RECS_DIR" => "\$HOME/BirdSongs",
	"PROCESSED" => "\${RECS_DIR}/Processed",
	"EXTRACTED" => "\${RECS_DIR}/Extracted",
	"IDFILE" => "\$HOME/BirdNET-Pi/IdentifiedSoFar.txt"
];

function test()
{
	global $home;

	foreach (TEST_HOMES as $_test_home) {
		$home = $_test_home;

		############################
		## Directory Path Tests
		############################
		$result = getDirectory('home');
		$expected = $_test_home;
		($result == $expected) ?: (print "failed directory 'home', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		# DIRECTORY TESTS
		$result = getDirectory('birdnet_pi');
		$expected = $_test_home . "/BirdNET-Pi";
		($result == $expected) ?: (print "failed directory 'birdnet_pi', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('recs_dir');
		$expected = $_test_home . "/BirdSongs";
		($result == $expected) ?: (print "failed directory 'recs_dir', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('processed');
		$expected = $_test_home . "/BirdSongs/Processed";
		($result == $expected) ?: (print "failed directory 'processed', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('extracted');
		$expected = $_test_home . "/BirdSongs/Extracted";
		($result == $expected) ?: (print "failed directory 'extracted', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('extracted_by_date');
		$expected = $_test_home . "/BirdSongs/Extracted/By_Date";
		($result == $expected) ?: (print "failed directory 'extracted_by_date', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('shifted_audio');
		$expected = $_test_home . "/BirdSongs/Extracted/By_Date/shifted";
		($result == $expected) ?: (print "failed directory 'shifted_audio', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('database');
		$expected = $_test_home . "/BirdNET-Pi/database";
		($result == $expected) ?: (print "failed directory 'database', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('config');
		$expected = $_test_home . "/BirdNET-Pi/config";
		($result == $expected) ?: (print "failed directory 'config', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('models');
		$expected = $_test_home . "/BirdNET-Pi/model";
		($result == $expected) ?: (print "failed directory 'models', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('python3_ve');
		$expected = $_test_home . "/BirdNET-Pi/birdnet/bin";
		($result == $expected) ?: (print "failed directory 'python3_ve', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('scripts');
		$expected = $_test_home . "/BirdNET-Pi/scripts";
		($result == $expected) ?: (print "failed directory 'scripts', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('stream_data');
		$expected = $_test_home . "/BirdSongs/StreamData";
		($result == $expected) ?: (print "failed directory 'stream_data', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('templates');
		$expected = $_test_home . "/BirdNET-Pi/templates";
		($result == $expected) ?: (print "failed directory 'templates', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('web');
		$expected = $_test_home . "/BirdNET-Pi/homepage";
		($result == $expected) ?: (print "failed directory 'web', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getDirectory('web_fonts');
		$expected = $_test_home . "/BirdNET-Pi/homepage/static";
		($result == $expected) ?: (print "failed directory 'web_fonts', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		############################
		# FILE PATH TESTS
		############################
		$result = getFilePath('analyzing_now.txt');
		$expected = $_test_home . "/BirdNET-Pi/analyzing_now.txt";
		($result == $expected) ?: (print "failed file path 'analyzing_now.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('apprise.txt');
		$expected = $_test_home . "/BirdNET-Pi/apprise.txt";
		($result == $expected) ?: (print "failed file path 'apprise.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('birdnet.conf');
		$expected = $_test_home . "/BirdNET-Pi/birdnet.conf";
		($result == $expected) ?: (print "failed file path 'birdnet.conf', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('etc_birdnet.conf');
		$expected = "/etc/birdnet/birdnet.conf";
		($result == $expected) ?: (print "failed file path 'etc_birdnet.conf', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('BirdDB.txt');
		$expected = $_test_home . "/BirdNET-Pi/BirdDB.txt";
		($result == $expected) ?: (print "failed file path 'BirdDB.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('birds.db');
		$expected = $_test_home . "/BirdNET-Pi/scripts/birds.db";
		($result == $expected) ?: (print "failed file path 'birds.db', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('blacklisted_images.txt');
		$expected = $_test_home . "/BirdNET-Pi/scripts/blacklisted_images.txt";
		($result == $expected) ?: (print "failed file path 'blacklisted_images.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('disk_check_exclude.txt');
		$expected = $_test_home . "/BirdNET-Pi/scripts/disk_check_exclude.txt";
		($result == $expected) ?: (print "failed file path 'disk_check_exclude.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('email_template');
		$expected = $_test_home . "/BirdNET-Pi/scripts/email_template";
		($result == $expected) ?: (print "failed file path 'email_template', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('email_template2');
		$expected = $_test_home . "/BirdNET-Pi/scripts/email_template2";
		($result == $expected) ?: (print "failed file path 'email_template2', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('exclude_species_list.txt');
		$expected = $_test_home . "/BirdNET-Pi/scripts/exclude_species_list.txt";
		($result == $expected) ?: (print "failed file path 'exclude_species_list.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('firstrun.ini');
		$expected = $_test_home . "/BirdNET-Pi/firstrun.ini";
		($result == $expected) ?: (print "failed file path 'firstrun.ini', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('.gotty');
		$expected = $_test_home . "/.gotty";
		($result == $expected) ?: (print "failed file path '.gotty', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('HUMAN.txt');
		$expected = $_test_home . "/BirdNET-Pi/HUMAN.txt";
		($result == $expected) ?: (print "failed file path 'HUMAN.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('IdentifiedSoFar.txt');
		$expected = $_test_home . "/BirdNET-Pi/IdentifiedSoFar.txt";
		($result == $expected) ?: (print "failed file path 'IdentifiedSoFar.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('include_species_list.txt');
		$expected = $_test_home . "/BirdNET-Pi/scripts/include_species_list.txt";
		($result == $expected) ?: (print "failed file path 'include_species_list.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('labels.txt');
		$expected = $_test_home . "/BirdNET-Pi/model/labels.txt";
		($result == $expected) ?: (print "failed file path 'labels.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('labels.txt.old');
		$expected = $_test_home . "/BirdNET-Pi/model/labels.txt.old";
		($result == $expected) ?: (print "failed file path 'labels.txt.old', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('labels_flickr.txt');
		$expected = $_test_home . "/BirdNET-Pi/model/labels_flickr.txt";
		($result == $expected) ?: (print "failed file path 'labels_flickr.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('labels_l18n.zip');
		$expected = $_test_home . "/BirdNET-Pi/model/labels_l18n.zip";
		($result == $expected) ?: (print "failed file path 'labels_l18n.zip', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('labels_lang.txt');
		$expected = $_test_home . "/BirdNET-Pi/model/labels_lang.txt";
		($result == $expected) ?: (print "failed file path 'labels_lang.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('labels_nm.zip');
		$expected = $_test_home . "/BirdNET-Pi/model/labels_nm.zip";
		($result == $expected) ?: (print "failed file path 'labels_nm.zip', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('lastrun.txt');
		$expected = $_test_home . "/BirdNET-Pi/scripts/lastrun.txt";
		($result == $expected) ?: (print "failed file path 'lastrun.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('python3');
		$expected = $_test_home . "/BirdNET-Pi/birdnet/bin/python3 ";
		($result == $expected) ?: (print "failed file path 'python3', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('python3_appraise');
		$expected = $_test_home . "/BirdNET-Pi/birdnet/bin/apprise ";
		($result == $expected) ?: (print "failed file path 'python3_appraise', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('species.py');
		$expected = $_test_home . "/BirdNET-Pi/scripts/species.py";
		($result == $expected) ?: (print "failed file path 'species.py', expected: " . ($expected) . " got: " . ($result) . "\r\n");

		$result = getFilePath('thisrun.txt');
		$expected = $_test_home . "/BirdNET-Pi/scripts/thisrun.txt";
		($result == $expected) ?: (print "failed file path 'thisrun.txt', expected: " . ($expected) . " got: " . ($result) . "\r\n");

	}
}


test();


?>