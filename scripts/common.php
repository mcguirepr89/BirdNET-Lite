<?php
if (realpath(__FILE__) == realpath($_SERVER['SCRIPT_FILENAME'])) {
	header('HTTP/1.1 403 Forbidden', TRUE, 403);
}

ini_set('session.gc_maxlifetime', 7200);
ini_set('user_agent', 'PHP_Flickr/1.0');
ini_set('display_errors', 1);
error_reporting(E_ERROR);

//Setup the session if we're not using the API, since the API endpoint sets up it's own session
if (!isset($api_incl)) {
	$api_incl = false;
	session_set_cookie_params(7200);
	session_start();
}

//Get the user and home directory for the first user in the system (userid 1000)
$user = shell_exec("awk -F: '/1000/{print $1}' /etc/passwd");
$user = trim($user);
$home = shell_exec("awk -F: '/1000/{print $6}' /etc/passwd");
$home = trim($home);

////////// SET TIMEZONE //////////
// If we can get the timezome from the systems timezone file ust that
$sys_timezone = "";
if (file_exists('/etc/timezone')) {
	$tz_data = file_get_contents('/etc/timezone');
	if ($tz_data !== false) {
		$sys_timezone = trim($tz_data);
	}
} else {
// Else get timezone from the timedatectl command
	$tz_data = shell_exec('timedatectl show');
	$tz_data_array = parse_ini_string($tz_data);
	if (is_array($tz_data_array) && array_key_exists('Timezone', $tz_data_array)) {
		$sys_timezone = $tz_data_array['Timezone'];
	}
}
//Finally if we have a valid timezone, set it as the one PHP uses
if ($sys_timezone !== "") {
	date_default_timezone_set($sys_timezone);
}


////////// PARSES THE CONFIG FILE //////////
$filePathMap_json_path = file_exists('./config/filepath_map.json') ? "./config/filepath_map.json" : "../config/filepath_map.json";
$filePathMap_data = [];
$config = [];
parseConfig();

//Set a default site name if nothing iss configured
if ($config["SITE_NAME"] == "") {
	$site_name = "BirdNET-Pi";
} else {
	$site_name = $config['SITE_NAME'];
}

$models = array("BirdNET_6K_GLOBAL_MODEL", "BirdNET_GLOBAL_3K_V2.3_Model_FP16");
$audio_formats = array("8svx", "aif", "aifc", "aiff", "aiffc", "al", "amb", "amr-nb", "amr-wb", "anb", "au", "avr", "awb", "caf", "cdda", "cdr", "cvs", "cvsd", "cvu", "dat", "dvms", "f32", "f4", "f64", "f8", "fap", "flac", "fssd", "gsm", "gsrt", "hcom", "htk", "ima", "ircam", "la", "lpc", "lpc10", "lu", "mat", "mat4", "mat5", "maud", "mp2", "mp3", "nist", "ogg", "paf", "prc", "pvf", "raw", "s1", "s16", "s2", "s24", "s3", "s32", "s4", "s8", "sb", "sd2", "sds", "sf", "sl", "sln", "smp", "snd", "sndfile", "sndr", "sndt", "sou", "sox", "sph", "sw", "txw", "u1", "u16", "u2", "u24", "u3", "u32", "u4", "u8", "ub", "ul", "uw", "vms", "voc", "vorbis", "vox", "w64", "wav", "wavpcm", "wv", "wve", "xa", "xi");
$freqshift_tools = array("sox", "ffmpeg");

//Database Languages
$langs = array(
	'not-selected' => 'Not Selected',
	"af" => "Afrikaans",
	"ca" => "Catalan",
	"cs" => "Czech",
	"zh" => "Chinese",
	"hr" => "Croatian",
	"da" => "Danish",
	"nl" => "Dutch",
	"en" => "English",
	"et" => "Estonian",
	"fi" => "Finnish",
	"fr" => "French",
	"de" => "German",
	"hu" => "Hungarian",
	"is" => "Icelandic",
	"id" => "Indonesia",
	"it" => "Italian",
	"ja" => "Japanese",
	"lv" => "Latvian",
	"lt" => "Lithuania",
	"no" => "Norwegian",
	"pl" => "Polish",
	"pt" => "Portugues",
	"ru" => "Russian",
	"sk" => "Slovak",
	"sl" => "Slovenian",
	"es" => "Spanish",
	"sv" => "Swedish",
	"th" => "Thai",
	"uk" => "Ukrainian"
);

//Reference to database connection
$DB_CONN = null;
$DB_CONN_STATUS = '';
$USER_AUTHENTICATED = false;

/**
 * Connects to the bird.db SQLite3 database
 * @return SQLite3
 */
function connect_to_birdsdb()
{
	global $DB_CONN, $DB_CONN_STATUS;

	//Initially check to see if the DB is already connected, it will not be null if it has
	if ($DB_CONN == null) {
		try {
			$birds_db_path = getFilePath('birds.db');
			if (!empty($birds_db_path)) {
				$DB_CONN = new SQLite3($birds_db_path, SQLITE3_OPEN_CREATE | SQLITE3_OPEN_READWRITE);
				if ($DB_CONN == False) {
					$DB_CONN_STATUS = 'birds.db database is busy.';
					birdnet_error_log("connect_to_birdsdb:: $DB_CONN_STATUS");
				}else{
					$DB_CONN_STATUS = 'birds.db database successfully connected.';
				}
			} else {
				$DB_CONN_STATUS = "No filepath to birds.db was returned by the resolver, so it wasn't able to be connected.";
				birdnet_error_log("connect_to_birdsdb:: $DB_CONN_STATUS");
			}
		} catch (Exception $sql_exec) {
			birdnet_error_log("connect_to_birdsdb:: Exception occurred while trying to open birds.db - " . $sql_exec->getMessage());
		}
	}

	return $DB_CONN;
}

/**
 * Disconnects the database
 *
 * @return void
 */
function disconnect_from_birdsdb()
{
	global $DB_CONN;

	if ($DB_CONN != null) {
		return $DB_CONN->close();
	}
}

/**
 * Executes the supplied query and returns all results
 *
 * @param $query string The query to execute
 * @param $bind_params string Any values that should be bound into the query
 * @param $fetchAllRecords string Any values that should be bound into the query
 * @param $fetchMode string Controls how result data is returned, default is @->default('SQLITE3_ASSOC') Associative Array;
 * @return array
 */
function db_execute_query($query, $bind_params = [], $fetchAllRecords = false, $fetchMode = SQLITE3_ASSOC)
{
	global $DB_CONN, $DB_CONN_STATUS;
	$success = false;
	$message = '';
	$data_to_return = null;

	//Connect to the DB
	connect_to_birdsdb();
	try {
		if (!$DB_CONN) {
			throw new Exception('birds.db database not connected. ' . $DB_CONN_STATUS);
		}

		$stmt = $DB_CONN->prepare($query);

		if ($stmt == False) {
			//get caller's function name so we can try to identify function called db_execute_query
			$caller_func_name = debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS, 2)[1]['function'];
			$error_msg = "$caller_func_name => db_execute_custom_query:: birds.db database is busy or a query error occurred";
			//
			$success = false;
			$message = $error_msg;
			$data_to_return = null;

			//Log the error message
			birdnet_error_log($error_msg);
		} else {
			if (!empty($bind_params)) {
				//Loop over the bind values and add them
				foreach ($bind_params as $bind_key => $bind_value) {
					$stmt->bindValue($bind_key, $bind_value);
				}
			}

			$result = $stmt->execute();
			//Initial result collection
			$resultArray = $result->fetchArray($fetchMode);

			if ($fetchAllRecords) {
				$multiArray = array(); //array to store all rows
				//Loop over the results to collect them all
				while ($resultArray !== false) {
					array_push($multiArray, $resultArray); //insert all rows to $multiArray
					$resultArray = $result->fetchArray($fetchMode);
				}
				unset($resultArray); //unset temporary variable

				$data_to_return = $multiArray;
			} else {
				$data_to_return = $resultArray;
			}

			//
			$success = true;
			$message = 'Ok';
		}

	} catch (Exception $sql_exec) {
		$caller_func_name = debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS, 2)[1]['function'];
		$error_msg = " $caller_func_name => db_execute_query:: Exception occurred while executing query - " . $sql_exec->getMessage();
		//
		$success = false;
		$message = $error_msg;
		$data_to_return = null;

		birdnet_error_log($error_msg);
	}

	return array('success' => $success, 'message' => $message, 'data' => $data_to_return);
}

/**
 * Returns a count of detections in the database
 * @return array
 */
function getDetectionCountAll()
{
	return db_execute_query('SELECT COUNT(*) FROM detections');
}

/**
 * Returns today's count of detections in the database
 * @return array
 */
function getDetectionCountToday()
{
	return db_execute_query('SELECT COUNT(*) FROM detections WHERE Date == DATE(\'now\', \'localtime\')');
}

/**
 * Returns detection info for the most recent detection today
 * @return array
 */
function getMostRecentDetectionToday()
{
	return db_execute_query('SELECT * FROM detections WHERE Date == DATE(\'now\', \'localtime\') ORDER BY TIME DESC LIMIT 1', [], true);
}

/**
 * Returns a count of detections in the past hour
 * @return array
 */
function getDetectionCountLastHour()
{
	return db_execute_query('SELECT COUNT(*) FROM detections WHERE Date == Date(\'now\', \'localtime\') AND TIME >= TIME(\'now\', \'localtime\', \'-1 hour\')');
}

/**
 * Returns the most recent detection
 * @return array|string
 */
function getMostRecentDetection($limit = 1)
{
	return db_execute_query('SELECT Com_Name, Sci_Name, Date, Time, Confidence, File_Name  FROM detections ORDER BY Date DESC, Time DESC LIMIT :limit', [':limit' => $limit], true);
}

/**
 * Returns a talley for todays detected species
 * @return array
 */
function getSpeciesTalley($range = "today", $start_date = null, $end_date = null)
{
	$range = strtolower($range);

	if ($range == "today") {
		$result = db_execute_query('SELECT COUNT(DISTINCT(Com_Name)) FROM detections WHERE Date == Date(\'now\', \'localtime\')');
	} else if ($range == "custom") {
		$result = db_execute_query('SELECT COUNT(DISTINCT(Com_Name)) FROM detections WHERE Date == :start_date', [':start_date' => $start_date], false);
	} else if ($range == "range") {
		$result = db_execute_query('SELECT COUNT(DISTINCT(Com_Name)) FROM detections WHERE Date BETWEEN :start_date AND :end_date', [':start_date' => $start_date, ':end_date' => $end_date]);
	}

	return $result;
}

/**
 * Returns a species talley from ALL detections
 * @return array
 */
function getAllSpeciesTalley()
{
	return db_execute_query('SELECT COUNT(DISTINCT(Com_Name)) FROM detections');
}

/**
 * Returns the number of detections on the specified date
 *
 * @param $date string Date to count detections for
 * @return array
 */
function getDetectionCountByDate($date, $date_range = false)
{
	return db_execute_query("SELECT COUNT(*) FROM detections WHERE Date == :date", [':date' => $date], false);
}

/**
 * Returns detections and counts for the specified date and start & finish times
 *
 * @param $date string Date to count detections for
 * @param $starttime string Search for detections after this time
 * @param $endtime string And detections up to this time
 * @return array
 */
function getDetectionBreakdownByTime($date, $starttime, $endtime)
{
	return db_execute_query('SELECT DISTINCT(Com_Name), COUNT(*) FROM detections WHERE Date == :date AND Time > :start_time AND Time < :end_time AND Confidence > 0.75 GROUP By Com_Name ORDER BY COUNT(*) DESC',
		[
			':date' => $date,
			':start_time' => $starttime,
			':end_time' => $endtime
		],
		true);
}


/**
 * Returns the detection count for a specified bird
 * @param $birdName string Species Name to get stats for
 * @return array
 */
function getBirdDetectionStats($birdName)
{
	//Cleanup the bird/species name
	$birdName = str_replace("_", " ", $birdName);

	$birdDetections = db_execute_query('SELECT Date, COUNT(*) AS Detections FROM detections WHERE Com_Name = :com_name AND Date BETWEEN DATE("now", "-30 days") AND DATE("now") GROUP BY Date', [':com_name' => $birdName], true);

	// Fetch the result set as an associative array
	$data = array();
	foreach ($birdDetections as $birdDetection) {
		$data[$birdDetection['Date']] = $birdDetection['Detections'];
	}

	// Create an array of all dates in the last 14 days
	$last14Days = array();
	for ($i = 0; $i < 31; $i++) {
		$last14Days[] = date('Y-m-d', strtotime("-$i days"));
	}

	// Merge the data array with the last14Days array
	$data = array_merge(array_fill_keys($last14Days, 0), $data);

	// Sort the data by date in ascending order
	ksort($data);

	// Convert the data to an array of objects
	$data = array_map(function ($date, $count) {
		return array('date' => $date, 'count' => $count);
	}, array_keys($data), $data);

	// Return the data as JSON - overwrite the result data
	$birdDetections['data'] = json_encode($data);

	return $birdDetections;
}

/**
 * Returns detection info for the specified
 * @param $birdName string Species name to get detection info on
 * @param $date string OPTIONAL - The date on which to list species on
 * @param $sort string OPTIONAL - Whether to sort the species by confidence value (so species ranked by detection accuracy/confidence)
 * @return array|string
 */
function getSpeciesDetectionInfo($birdName, $date = null, $sort = null)
{
	//If no date is supplied, then list a unique list of detection dates in the DB, sorted descending
	if ($date == null) {
		//No date set so automatically search by date and time if no sort value is set
		if (isset($sort) && $sort == "confidence") {
			$detectionInfo = db_execute_query("SELECT * FROM detections where Com_Name == :birdname ORDER BY Confidence DESC", [':birdname' => $birdName], true);
		} else {
			$detectionInfo = db_execute_query("SELECT * FROM detections where Com_Name == :birdname ORDER BY Date DESC, Time DESC", [':birdname' => $birdName], true);
		}
	} else {
		//Date set so use that to filter the results depending on the sort value
		if (isset($sort) && $sort == "confidence") {
			$detectionInfo = db_execute_query("SELECT * FROM detections where Com_Name == :birdname AND Date == :date ORDER BY Confidence DESC", [':birdname' => $birdName, ':date' => $date], true);
		} else {
			$detectionInfo = db_execute_query("SELECT * FROM detections where Com_Name == :birdname AND Date == :date ORDER BY Time DESC", [':birdname' => $birdName, ':date' => $date], true);
		}
	}

	return $detectionInfo;
}

/**
 * Returns the detections for a specified filename
 *
 * @param $filename string The filename on which to get detections for
 * @return array|string
 */
function getDetectionsByFilename($filename)
{
	return db_execute_query("SELECT * FROM detections where File_name == :filename ORDER BY Date DESC, Time DESC", [':filename' => $filename], true);
}

/**
 * Returns a list of specie names on a specified date if supplied, else lists valid dates which can be passed again for specific list of species on that date
 *
 * @param $date string OPTIONAL - The date on which to list species on
 * @param $sort string OPTIONAL - Whether to sort the species by occurrence (so species ranked by number of detections)
 * @return array|string
 */
function getDetectionsByDate($date = null, $sort = null)
{

	//If no date is supplied, then list a unique list of species in the DB
	if ($date == null) {
		$detectionsByDateResult = db_execute_query('SELECT DISTINCT(Date) FROM detections GROUP BY Date ORDER BY Date DESC', null, true);
	} else {
		//Else a date was supplied, first check the sort order if any
		if (isset($sort) && $sort == "occurrences") {
			$detectionsByDateResult = db_execute_query("SELECT DISTINCT(Com_Name) FROM detections WHERE Date == :date GROUP BY Com_Name ORDER BY COUNT(*) DESC", [':date' => $date], true);
		} else {
			$detectionsByDateResult = db_execute_query("SELECT DISTINCT(Com_Name) FROM detections WHERE Date == :date ORDER BY Com_Name", [':date' => $date], true);
		}
	}

	return $detectionsByDateResult;
}

/**
 * Returns a list detections for a species if supplied, else lists all detected species by name
 *
 * @param $species_name string OPTIONAL - List detections for this specific species
 * @param $sort string OPTIONAL - Whether to sort the species by occurrence (so species ranked by number of detections)
 * @return array
 */
function getDetectionsBySpecies($species_name = null, $sort = null)
{
	//If no species is  is supplied, then list all species
	if (!isset($species_name)) {
		if (isset($sort) && $sort == "occurrences") {
			//Sort by occurrences
			$speciesDetections = db_execute_query('SELECT DISTINCT(Com_Name) FROM detections GROUP BY Com_Name ORDER BY COUNT(*) DESC', null, true);
		} else {
			//Don't sort by occurrences
			$speciesDetections = db_execute_query('SELECT DISTINCT(Com_Name) FROM detections ORDER BY Com_Name ASC', null, true);
		}
	} else {
		//Else a species name was supplied, first check the sort order if any
		$speciesDetections = db_execute_query("SELECT * FROM detections WHERE Com_Name == :species_name ORDER BY Com_Name", [':species_name' => $species_name], true);
		//Also get the highest confidence record for this species
		$speciesDetections_MaxConf = db_execute_query("SELECT Date, Time, Sci_Name, MAX(Confidence), File_Name FROM detections WHERE Com_Name == :species_name ORDER BY Com_Name", [':species_name' => $species_name], true);
	}

	//Rearrange data
	$newReturnData = [];
	$newReturnData['species'] = $speciesDetections['data'];
	//Check to see if we have to get max confidence results for the species also
	if (isset($speciesDetections_MaxConf)) {
		$newReturnData['species_MaxConf'] = $speciesDetections_MaxConf['data'];
	}

	$speciesDetections['data'] = $newReturnData;

	return $speciesDetections;
}

/**
 * Returns a list of species either ordered alphabetically (default) or ordered by the number of detections the species has
 *
 * @param $sort string OPTIONAL - Sort result alphabetically (supply null) or by number of occurrences (supply "occurrences"
 * @return array
 */
function getSpeciesBestRecordingList($sort = null)
{
	if (isset($sort) && $sort == "occurrences") {
		//Sort by occurrences
		$speciesBestRecording = db_execute_query('SELECT Date, Time, File_Name, Com_Name, COUNT(*), MAX(Confidence) FROM detections GROUP BY Com_Name ORDER BY COUNT(*) DESC', null, true);

	} else {
		//Don't sort by occurrences, sort by alphabetical
		$speciesBestRecording = db_execute_query('SELECT Date, Time, File_Name, Com_Name, COUNT(*), MAX(Confidence) FROM detections GROUP BY Com_Name ORDER BY Com_Name ASC', null, true);
	}

	return $speciesBestRecording;
}

/**
 * Returns a list of best recordings for a supplied species
 *
 * @param $species_name string Name of the species
 * @return array
 */
function getBestRecordingsForSpecies($species_name)
{
	return db_execute_query("SELECT Com_Name, Sci_Name, COUNT(*), MAX(Confidence), File_Name, Date, Time from detections WHERE Com_Name = :species_name", [':species_name' => $species_name], true);
}

/**
 * Get a list of todays detections
 *
 * @param $display_limit string Number of results to return
 * @param $search_term string OPTIONAL: Return results that match the supplied term
 * @param $hard_limit string OPTIONAL: Return a fix number of results
 * @return array
 */
function getTodaysDetections($display_limit, $search_term = null, $hard_limit = null)
{
	$bind_params = [];

	if (isset($search_term)) {
		if (strtolower(explode(" ", $search_term)[0]) == "not") {
			$not = "NOT ";
			$operator = "AND";
			$search_term = str_replace("not ", "", $search_term);
			$search_term = str_replace("NOT ", "", $search_term);
		} else {
			$not = "";
			$operator = "OR";
		}
		$searchquery = "AND (Com_name " . $not . "LIKE :search_term " .
			$operator . " Sci_name " . $not . "LIKE :search_term " .
			$operator . " Confidence " . $not . "LIKE :search_term " .
			$operator . " File_Name " . $not . "LIKE :search_term " .
			$operator . " Time " . $not . "LIKE :search_term)";

		$bind_params = [':search_term' => '%' . $search_term . '%'];
	} else {
		$searchquery = "";
	}

	if (isset($display_limit) && is_numeric($display_limit)) {
		$bind_params[':display_limit'] = (intval($display_limit) - 40);
		$result = db_execute_query('SELECT Date, Time, Com_Name, Sci_Name, Confidence, File_Name FROM detections WHERE Date == Date(\'now\', \'localtime\') ' . $searchquery . ' ORDER BY Time DESC LIMIT :display_limit,40', $bind_params, true);
	} else {
		// legacy mode
		if (isset($hard_limit) && is_numeric($hard_limit)) {
			$bind_params[':hard_limit'] = $hard_limit;
			$result = db_execute_query('SELECT Date, Time, Com_Name, Sci_Name, Confidence, File_Name FROM detections WHERE Date == Date(\'now\', \'localtime\') ' . $searchquery . ' ORDER BY Time DESC LIMIT :hard_limit', $bind_params, true);
		} else {
			$result = db_execute_query('SELECT Date, Time, Com_Name, Sci_Name, Confidence, File_Name FROM detections WHERE Date == Date(\'now\', \'localtime\') ' . $searchquery . ' ORDER BY Time DESC', $bind_params, true);
		}
	}

	return $result;
}

/**
 * Returns the species talley for last week and the week prior
 *
 * @return array[]
 */
function getWeeklyReportSpeciesTalley()
{
	$last_week_dates = getLastWeekDates();
	$startdate = $last_week_dates['start_date'];
	$enddate = $last_week_dates['end_date'];


	$totalspeciestally = getSpeciesTalley('range', date("Y-m-d", $startdate), date("Y-m-d", $enddate));
	$priortotalspeciestally = getSpeciesTalley('range', date("Y-m-d", $startdate - (7 * 86400)), date("Y-m-d", $enddate - (7 * 86400)));

	return ['totalspeciestally' => $totalspeciestally, 'priortotalspeciestally' => $priortotalspeciestally];
}

/**
 * Returns the species talley for last week and the week prior
 *
 * @return array[]
 */
function getWeeklyReportSpeciesDetectionCounts($detections_asc = false)
{
	$last_week_dates = getLastWeekDates();
	$startdate = $last_week_dates['start_date'];
	$enddate = $last_week_dates['end_date'];


	$sort_order = $detections_asc ? 'ASC' : 'DESC';

	$detections = db_execute_query('SELECT DISTINCT(Com_Name), COUNT(*) FROM detections WHERE Date BETWEEN :start_date AND :end_date GROUP By Com_Name ORDER BY COUNT(*) ' . $sort_order, [':start_date' => date("Y-m-d", $startdate), ':end_date' => date("Y-m-d", $enddate)], true);
	$totalcount = db_execute_query('SELECT DISTINCT(Com_Name), COUNT(*) FROM detections WHERE Date BETWEEN :start_date AND :end_date', [':start_date' => date("Y-m-d", $startdate), ':end_date' => date("Y-m-d", $enddate)], true);
	$priortotalcount = db_execute_query('SELECT DISTINCT(Com_Name), COUNT(*) FROM detections WHERE Date BETWEEN :start_date AND :end_date', [':start_date' => date("Y-m-d", date("Y-m-d", $startdate - (7 * 86400))), ':end_date' => date("Y-m-d", date("Y-m-d", $enddate - (7 * 86400)))], true);

	return ['detections' => $detections, 'totalcount' => $totalcount, 'priortotalcount' => $priortotalcount];
}

/**
 * Returns last weeks detection count for the specified species
 *
 * @param string $species_name Species to get last weeks detection count for
 * @param bool $this_week Whether to find detections in or outside last week
 * @return array
 */
function getWeeklyReportSpeciesDetection($species_name, $this_week = true)
{
	$last_week_dates = getLastWeekDates();
	$startdate = $last_week_dates['start_date'];
	$enddate = $last_week_dates['end_date'];

	if ($this_week) {
		$result = db_execute_query('SELECT COUNT(*) FROM detections WHERE Com_Name == :species_name AND Date BETWEEN :start_date AND :end_date', [':species_name' => $species_name, ':start_date' => date("Y-m-d", $startdate - (7 * 86400)), ':end_date' => date("Y-m-d", $enddate - (7 * 86400))], false);
	} else {
		$result = db_execute_query('SELECT COUNT(*) FROM detections WHERE Com_Name == :species_name AND Date NOT BETWEEN :start_date AND :end_date', [':species_name' => $species_name, ':start_date' => date("Y-m-d", $startdate), ':end_date' => date("Y-m-d", $enddate)], false);
	}

	return $result;
}

/**
 * Deletes a specified detection by filename
 *
 * @param $filename string The filename of the detection e.g 2023-04-25/Pacific_Koel/Pacific_Koel-76-2023-04-25-birdnet-RTSP_2-16:24:05.mp3
 * @return array
 */
function deleteDetection($filename)
{
	global $DB_CONN, $api_incl;
	$success = false;
	$message = '';
	$data_to_return = null;

	//Check authentication before going any further
	//the front end would have done this check already and such should pass
	authenticateUser();

	$filename_exploded = explode("/", $filename);
	$actual_filename = $filename_exploded[2];


	//If the detection was successfully deleted, remove the mp3 and png spectrogram
	$file_pointer = getDirectory('home') . "/BirdSongs/Extracted/By_Date/" . $filename;
	if (!exec("sudo rm $file_pointer && sudo rm $file_pointer.png")) {
		$message = "OK";
		$statement = db_execute_query('DELETE FROM detections WHERE File_Name = :filename LIMIT 1', [':filename' => $actual_filename]);
	} else {
		$message = "Error";
	}

	//
	$success = true;
	//Message set above
	$data_to_return = $statement;

	return array('success' => $success, 'message' => $message, 'data' => $data_to_return);
}

/**
 * Protects the specified file from deletion
 *
 * @param $mode string Set the mode to add (protect) to remove (unprotect) the specified filepath from protection
 * @param $filename_to_protect string The filename of the detection to protect e.g 2023-04-25/Pacific_Koel/Pacific_Koel-76-2023-04-25-birdnet-RTSP_2-16:24:05.mp3
 * @return array
 */
function protectDetectionFromDeletion($mode, $filename_to_protect)
{
	global $api_incl;
	$success = false;
	$message = '';
	$data_to_return = null;

	$mode = strtolower($mode);

	$scripts_dir = getDirectory('scripts');
	$disk_check_exclude_file = $scripts_dir . "/disk_check_exclude.txt";

	//Initially create the file if it doesn't exist
	if (!file_exists($disk_check_exclude_file)) {
		file_put_contents($disk_check_exclude_file, "##start\n##end\n");
	}

	if ($mode == "protect") {
		// load the data and delete the line from the array which is ##end,
		// so that all excluded files sit between the ##start and ##end tags
		$lines = file($disk_check_exclude_file);
		$last = sizeof($lines) - 1;
		unset($lines[$last]);

		//Open the file and truncate contents
		if (($myfile = fopen($disk_check_exclude_file, "w")) !== false) {
			$txt = $filename_to_protect;
			//Write all existing lines
			fwrite($myfile, implode("", $lines));
			//Write out the lines for the file were trying to exclude
			fwrite($myfile, $txt . "\n");
			fwrite($myfile, $txt . ".png\n");
			//Insert the end tag again to make it the last line
			fwrite($myfile, "##end\n");

			fclose($myfile);
			//
			$success = true;
			$message = "OK";
		} else {
			$success = false;
			$message = "Unable to open file! " . $disk_check_exclude_file;
		}
	} else if ($mode == "unprotect") {
		$lines = file($disk_check_exclude_file);
		$search = $filename_to_protect;

		$result = '';
		foreach ($lines as $line) {
			if (stripos($line, $search) === false && stripos($line, $search . ".png") === false) {
				$result .= $line;
			}
		}
		if (file_put_contents($disk_check_exclude_file, $result) !== false) {
			$success = true;
			$message = "OK";
		} else {
			$success = true;
			$message = "Failed writing contents after removing file from protection, back to file $disk_check_exclude_file";
		}
	}

	return array('success' => $success, 'message' => $message, 'data' => $data_to_return);
}

/**
 * Frequency shifts the specified audio file to aid listening for hearing impaired people
 *
 * @param $filename
 * @param $performShift
 * @return array
 */
function frequencyShiftDetectionAudio($filename, $performShift = null)
{
	global $config, $api_incl;
	$success = false;
	$message = '';
	$data_to_return = null;

	//Check authentication before going any further
	//the front end would have done this check already and such should pass
	authenticateUser();

	$shifted_path = getDirectory('shifted_audio') . '/';

	$pp = pathinfo($filename);
	$dir = $pp['dirname'];
	$fn = $pp['filename'];
	$ext = $pp['extension'];
	$pi = getDirectory('extracted_bydate') . '/';

	if (isset($performShift) && $performShift == true) {
		$freqshift_tool = $config['FREQSHIFT_TOOL'];

		if ($freqshift_tool == "ffmpeg") {
			$cmd = "sudo /usr/bin/nohup /usr/bin/ffmpeg -y -i " . escapeshellarg($pi . $filename) . " -af \"rubberband=pitch=" . $config['FREQSHIFT_LO'] . "/" . $config['FREQSHIFT_HI'] . "\" " . escapeshellarg($shifted_path . $filename) . "";
			shell_exec("sudo mkdir -p " . $shifted_path . $dir . " && " . $cmd);
		} else if ($freqshift_tool == "sox") {
			//linux.die.net/man/1/sox
			$soxopt = "-q";
			$soxpitch = $config['FREQSHIFT_PITCH'];
			$cmd = "sudo /usr/bin/nohup /usr/bin/sox " . escapeshellarg($pi . $filename) . " " . escapeshellarg($shifted_path . $filename) . " pitch " . $soxopt . " " . $soxpitch;
			shell_exec("sudo mkdir -p " . $shifted_path . $dir . " && " . $cmd);
		}
	} else {
		$cmd = "sudo rm -f " . escapeshellarg($shifted_path . $filename);
		shell_exec($cmd);
	}

	//
	$success = true;
	$message = "OK";

	return array('success' => $success, 'message' => $message, 'data' => $data_to_return);
}

/**
 * Adds the supplied Flickr Image Id to the list of blacklisted images
 *
 * @param $imageID
 * @return array
 */
function blacklistFlickrImage($imageID)
{
	$success = false;
	$message = '';
	$data_to_return = null;

	$scripts_dir = getDirectory('scripts');

	//Append the Flickr Image ID the blacklist file
	if (($file_handle = fopen($scripts_dir . "/blacklisted_images.txt", 'a+')) !== false) {
		fwrite($file_handle, $imageID . "\n");
		fclose($file_handle);
		//
		$success = true;
		$message = "Ok";
	} else {
		$success = false;
		$message = "Failed";
	}

	return array('success' => $success, 'message' => $message, 'data' => $data_to_return);
}

/**
 * Finds and returns a flicker image for the supplied detection, detection data must have the following columns Com_Name, Sci_Name, Date, Time, Confidence, File_Name
 * getMostRecentDetections
 *
 * @param $detection_data
 * @param bool $getAll
 * @return string[][]
 */
function getFlickrImage($detection_data, $getAll = false)
{
	global $config;

	$success = false;
	$message = '';
	$data_to_return = null;

	$flickr_data_to_return = array(
		'Com_Name' => 'N/A',
		'Sci_Name' => 'N/A',
		'Date' => 'N/A',
		'Time' => 'N/A',
		'Confidence' => '0',
		'File_Name' => 'N/A',
		'Com_Name_clean' => 'N/A',
		'Sci_Name_clean' => 'N/A',
		'photos' => null,
		'filename_path' => 'N/A',
		'filename_formatted' => 'N/A'
	);
	$foundImage = false;

	if (isset($detection_data)) {
		$extracted_dir = getDirectory('extracted');
		$home_dir = getDirectory('home');

		//Setup out flickr image cache in the session
		if (!isset($_SESSION['images'])) {
			$_SESSION['images'] = [];
		}

		//Cleanup the Birds common and scientic name
		$comname_clean = preg_replace('/ /', '_', $detection_data['Com_Name']);
		$sciname_clean = preg_replace('/ /', '_', $detection_data['Sci_Name']);
		$comname_clean = preg_replace('/\'/', '', $comname_clean);
		$filename_path = "/By_Date/" . $detection_data['Date'] . "/" . $comname_clean . "/" . $detection_data['File_Name'];
		$filename_formatted = $detection_data['Date'] . "/" . $comname_clean . "/" . $detection_data['File_Name'];
		$args = "&license=2%2C3%2C4%2C5%2C6%2C9&orientation=square,portrait";
		$comnameprefix = "%20bird";

		if (!empty($config["FLICKR_API_KEY"])) {

			if (!empty($config["FLICKR_FILTER_EMAIL"])) {
				if (!isset($_SESSION["FLICKR_FILTER_EMAIL"])) {
					unset($_SESSION['images']);
					$_SESSION['FLICKR_FILTER_EMAIL'] = json_decode(file_get_contents("https://www.flickr.com/services/rest/?method=flickr.people.findByEmail&api_key=" . $config["FLICKR_API_KEY"] . "&find_email=" . $config["FLICKR_FILTER_EMAIL"] . "&format=json&nojsoncallback=1"), true)["user"]["nsid"];
				}
				$args = "&user_id=" . $_SESSION['FLICKR_FILTER_EMAIL'];
				$comnameprefix = "";
			} else {
				if (isset($_SESSION["FLICKR_FILTER_EMAIL"])) {
					unset($_SESSION["FLICKR_FILTER_EMAIL"]);
					unset($_SESSION['images']);
				}
			}


			// if we already searched flickr for this species before, use the previous image rather than doing an unneccesary api call
			$key = array_search($comname_clean, array_column($_SESSION['images'], 0));
			if ($key !== false && !$getAll) {
				$flickr_image_data = $_SESSION['images'][$key];

				$foundImage = true;
				//Extract data out of the image data, either in our $_SESSION var already is recently pushed (new data)
				$flickr_data_to_return['photos'][0]['image_url'] = $flickr_image_data[1];
				$flickr_data_to_return['photos'][0]['photo_title'] = $flickr_image_data[2];
				$flickr_data_to_return['photos'][0]['modal_text'] = $flickr_image_data[3];
				$flickr_data_to_return['photos'][0]['author_link'] = $flickr_image_data[4];
				$flickr_data_to_return['photos'][0]['license_url'] = $flickr_image_data[5];
			} else {
				// Get license information if we haven't already
				if (empty($licenses_urls)) {
					$licenses_url = "https://api.flickr.com/services/rest/?method=flickr.photos.licenses.getInfo&api_key=" . $config["FLICKR_API_KEY"] . "&format=json&nojsoncallback=1";
					$licenses_response = file_get_contents($licenses_url);
					$licenses_data = json_decode($licenses_response, true)["licenses"]["license"];
					foreach ($licenses_data as $license) {
						$license_id = $license["id"];
						$license_name = $license["name"];
						$license_url = $license["url"];
						$licenses_urls[$license_id] = $license_url;
					}
				}

				// only open the file once per script execution
				if (!isset($lines)) {
					$lines = file($home_dir . "/BirdNET-Pi/model/labels_flickr.txt");
				}
				// convert sci name to English name
				foreach ($lines as $line) {
					if (strpos($line, $detection_data['Sci_Name']) !== false) {
						$sci_name_as_english_name = trim(explode("_", $line)[1]);
						break;
					}
				}

				// Read the blacklisted image ids from the file into an array
				$blacklisted_ids = array_map('trim', file($home_dir . "/BirdNET-Pi/scripts/blacklisted_images.txt"));

				// Make the API call
				$flickrjson = json_decode(file_get_contents("https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key=" . $config["FLICKR_API_KEY"] . "&text=" . str_replace(" ", "%20", $sci_name_as_english_name) . $comnameprefix . "&sort=relevance" . $args . "&per_page=5&media=photos&format=json&nojsoncallback=1"), true)["photos"]["photo"];

				// Find the first photo that is not blacklisted or is not the specific blacklisted id
				$photo = null;
				foreach ($flickrjson as $flickrphoto) {
					if ($flickrphoto["id"] !== "4892923285" && !in_array($flickrphoto["id"], $blacklisted_ids)) {
						$photo = $flickrphoto;

						$license_url = "https://api.flickr.com/services/rest/?method=flickr.photos.getInfo&api_key=" . $config["FLICKR_API_KEY"] . "&photo_id=" . $photo["id"] . "&format=json&nojsoncallback=1";
						$license_response = file_get_contents($license_url);
						$license_info = json_decode($license_response, true)["photo"]["license"];
						$license_url = $licenses_urls[$license_info];

						$modaltext = "https://flickr.com/photos/" . $photo["owner"] . "/" . $photo["id"];
						$authorlink = "https://flickr.com/people/" . $photo["owner"];
						$imageurl = 'https://farm' . $photo["farm"] . '.static.flickr.com/' . $photo["server"] . '/' . $photo["id"] . '_' . $photo["secret"] . '.jpg';

						if (!$getAll) {
							//Get single image then break the loop
							array_push($_SESSION['images'], array($comname_clean, $imageurl, $photo["title"], $modaltext, $authorlink, $license_url));
							$flickr_image_data = $_SESSION['images'][count($_SESSION['images']) - 1];

							//Extract data out of the image data, either in our $_SESSION var already is recently pushed (new data)
							$flickr_data_to_return['photos'][0]['image_url'] = $flickr_image_data[1];
							$flickr_data_to_return['photos'][0]['photo_title'] = $flickr_image_data[2];
							$flickr_data_to_return['photos'][0]['modal_text'] = $flickr_image_data[3];
							$flickr_data_to_return['photos'][0]['author_link'] = $flickr_image_data[4];
							$flickr_data_to_return['photos'][0]['license_url'] = $flickr_image_data[5];

							//Break out of the loop
							break;
						} else {
							//Get all images
							//Extract data out of the image data, either in our $_SESSION var already is recently pushed (new data)
							$flickr_data_to_return['photos'][(int)$flickrphoto["id"]]['image_url'] = $imageurl;
							$flickr_data_to_return['photos'][(int)$flickrphoto["id"]]['photo_title'] = $photo["title"];
							$flickr_data_to_return['photos'][(int)$flickrphoto["id"]]['modal_text'] = $modaltext;
							$flickr_data_to_return['photos'][(int)$flickrphoto["id"]]['author_link'] = $authorlink;
							$flickr_data_to_return['photos'][(int)$flickrphoto["id"]]['license_url'] = $license_url;
						}
					}
				}

				$foundImage = true;
				$success = true;
				$message = 'Flickr image data successfully downloaded';
			}

			//Fill out all other common data
			$flickr_data_to_return['Com_Name'] = $detection_data['Com_Name'];
			$flickr_data_to_return['Sci_Name'] = $detection_data['Sci_Name'];
			$flickr_data_to_return['Date'] = $detection_data['Date'];
			$flickr_data_to_return['Time'] = $detection_data['Time'];
			$flickr_data_to_return['Confidence'] = array_key_exists('Confidence', $detection_data) ? $detection_data['Confidence'] : 0;
			$flickr_data_to_return['File_Name'] = $detection_data['File_Name'];

			$flickr_data_to_return['Com_Name_clean'] = $comname_clean;
			$flickr_data_to_return['Sci_Name_clean'] = $sciname_clean;

			$flickr_data_to_return['filename_path'] = $filename_path;
			$flickr_data_to_return['filename_formatted'] = $filename_formatted;
		}
	}

	return array('success' => $success, 'message' => $message, 'image_found' => $foundImage, 'data' => $flickr_data_to_return);
}


/**
 * Returns the filename of todays species chart for the main page
 *
 * @return string
 */
function getChartString()
{
	$myDate = date('Y-m-d');
	$chart = "Combo-$myDate.png";
	return $chart;
}

/**
 * Returns the start and end dates for last week
 *
 * @return string[]
 */
function getLastWeekDates()
{
	$startdate = strtotime('last sunday') - (7 * 86400);
	$enddate = strtotime('last sunday') - (1 * 86400);

	return ['start_date' => $startdate, 'end_date' => $enddate];
}

/**
 * Change Database Language
 *
 * @param $model
 * @param $language
 * @return string|null
 */
function changeLanguage($model, $language)
{
	global $config, $user;
	//Re-parse the config just in case
	parseConfig();
	$result = "";

	// Update Language settings only if a change is requested
	if ($model != $config['MODEL'] || $language != $config['DATABASE_LANG']) {
		if (strlen($language) == 2) {
			$scripts_dir = getDirectory('scripts');

			// Archive old language file
			$result = syslog_shell_exec("cp -f " . getFilePath('labels.txt') . " " . getFilePath('labels.txt.old'), $user);

			if ($model == "BirdNET_GLOBAL_3K_V2.3_Model_FP16") {
				// Install new language label file
				$result = syslog_shell_exec("chmod +x $scripts_dir/install_language_label_nm.sh && $scripts_dir/install_language_label_nm.sh -l $language", $user);
			} else {
				$result = syslog_shell_exec("$scripts_dir/install_language_label.sh -l $language", $user);
			}

			syslog(LOG_INFO, "Successfully changed language to '$language' and model to '$model'");
		}
	}
	return $result;
}

/**
 * Loads in the JSON file containing data on directory and file paths
 *
 * @return array
 */
function loadFilePathMap()
{
	global $filePathMap_data, $filePathMap_json_path;
	if (empty($filePathMap_data)) {
		$filePathMap_data = json_decode(file_get_contents($filePathMap_json_path), true);
	}
	return $filePathMap_data;
}

/**
 * Directory Path Helper, returns a full directory path for a supplied directory name e.g home, processed, extracted
 *
 * @param string $dir The directory to obtain a path for
 * @return string
 */
function getDirectory($dir)
{
	global $config, $home;
	$dir = strtolower($dir);

	$filePathMap_directories = loadFilePathMap()['directories'];

	if (array_key_exists($dir, $filePathMap_directories)) {
		$filePathMap_directories_selected = $filePathMap_directories[$dir];

		//Check to see if directory is an alias for another
		$dir_alias = $filePathMap_directories_selected['alias_for'] ?? '';
		if (!empty($dir_alias)) {
			//If so load in the data for that directory
			$filePathMap_directories_selected = $filePathMap_directories[$dir_alias];
		}

		//Gather all the options into variables
		$setting_value = $filePathMap_directories_selected['read_setting'] ?? '';
		$lives_under = $filePathMap_directories_selected['lives_under'] ?? '';
		$replace_text = $filePathMap_directories_selected['replace_setting_text'] ?? '';
		$replace_text_with = $filePathMap_directories_selected['replace_setting_text_with'] ?? '';
		$append = $filePathMap_directories_selected['append'] ?? '';
		$return_var = $filePathMap_directories_selected['return_var'] ?? '';
		//
		$return_value = $under_directory = '';

		//Get the directory which the directory we're processing lives under
		if (!empty($lives_under)) {
			$under_directory = getDirectory($lives_under);
		}

		//Read the specified config file setting
		if (!empty($setting_value)) {
			$setting_value = $config[$setting_value];
		}

		//Replace value in setting, like ${RECS_DIR} etc as they are not expanded
		if (!empty($replace_text)) {
			$return_value = str_replace($replace_text, $replace_text_with, $setting_value);
		} else {
			$return_value = $setting_value;
		}

		//If a variable is specified for return, return it first
		if (!empty($return_var)) {
			//return the dynamic variable, currently this is just the users home path in $home
			return ${"{$return_var}"};
		} else if (!empty($append)) {
			//Append this to the end of the path, (models, scripts etc) do this as they reside under BirdNET-pi
			return $under_directory . $append;
		} else {
			//Else return the directory and result of the setting manipulation
			return $under_directory . $return_value;
		}
	}

	return "";
}

/**
 * Returns the full filepath for the specified filename
 *
 * @param $filename
 * @return string
 */
function getFilePath($filename)
{
	global $config;

	$filePathMap_files = loadFilePathMap()['files'];

	if (array_key_exists($filename, $filePathMap_files)) {
		$filePathMap_directories_selected = $filePathMap_files[$filename];

		//Gather all the options into variables
		$setting_value = $filePathMap_directories_selected['read_setting'] ?? '';
		$lives_under = $filePathMap_directories_selected['lives_under'] ?? '';
		$replace_text = $filePathMap_directories_selected['replace_setting_text'] ?? '';
		$replace_test_with = $filePathMap_directories_selected['replace_setting_text_with'] ?? '';
		$append = $filePathMap_directories_selected['append'] ?? '';
		$return_val = $filePathMap_directories_selected['return_var'] ?? '';
		//
		$under_directory = '';

		//Get the directory which the directory we're processing lives under
		if (!empty($lives_under)) {
			$under_directory = getDirectory($lives_under);
		}

		//Read the specified config file setting if any
		if (!empty($setting_value)) {
			$setting_value = $config[$setting_value];
		}

		//Replace value in setting, like ${RECS_DIR} etc as they are not expanded
		if (!empty($replace_text)) {
			$return_value = str_replace($replace_text, $replace_test_with, $setting_value);
		} else {
			$return_value = $setting_value;
		}

		//If a variable is specified for return, return it first and directly
		if (!empty($return_val)) {
			//return the dynamic variable, currently this is just the users home path in $home
			return $return_val;
		} else if (!empty($append)) {
			//Append this to the end of the path, (models, scripts etc) do this as they reside under BirdNET-pi
			return $under_directory . $append;
		} else {
			//Else return the directory and result of the setting manipulation
			return $under_directory . $return_value;
		}
	}

	return "";
}

/**
 * Updates the setting value for the supplied setting name, if the setting doesn't exist it will be created
 *
 * @param $setting_name string Setting Name e.g SITE_NAME, BIRDWEATHER_ID, MODEL etc
 * @param $setting_value string|int Setting value - This value has to be exactly how it needs to be stored (same as what was used in preg_replace) and properly escaped eg "\"asetting"\"
 * @param $post_save_command string CommaAnd to execute after saving
 * @return void
 */
function saveSetting($setting_name, $setting_value, $post_save_command = null)
{
	global $config;

	//Check authentication before going any further
	//the front end would have done this check already and such should pass
	authenticateUser();

	//Setting exists already, see if the value changed
	if (array_key_exists($setting_name, $config)) {
		//Strip any outer quotes from the setting value (e.g "$rec_card") and then test if it change how it was originally tested
		//https://stackoverflow.com/questions/9734758/remove-quotes-from-start-and-end-of-string-in-php
		$setting_value_clean = preg_replace('~^"?(.*?)"?$~', '$1', $setting_value);

		if (strcmp($setting_value_clean, $config[$setting_name]) !== 0) {
			$contents = file_get_contents(getFilePath('birdnet.conf'));
			$contents2 = file_get_contents(getFilePath('thisrun.txt'));
			//
			if ($contents !== false && $contents2 !== false) {
				//Update the value
				$contents = preg_replace("/$setting_name=.*/", "$setting_name=$setting_value", $contents);
				$contents2 = preg_replace("/$setting_name=.*/", "$setting_name=$setting_value", $contents2);

				//Write all the  data out again to the the respective files
				//	$fh = fopen("/etc/birdnet/birdnet.conf", "w");
				$fh = fopen(getFilePath('birdnet.conf'), "w");
				$fh2 = fopen(getFilePath('thisrun.txt'), "w");

				if ($fh !== false && $fh2 !== false) {
					fwrite($fh, $contents);
					fwrite($fh2, $contents2);
					//Check if we need to execute a command after saving
					if (isset($post_save_command)) {
						sleep(1);
						if (!is_array($post_save_command)) {
							executeSysCommand($post_save_command);
						} else {
							//array of commands
							foreach ($post_save_command as $single_command) {
								executeSysCommand($single_command);
							}
						}
					}
					//Reload the settings to update our global $config variable
					parseConfig();
				}
			}
		}
	} else {
		//Create the setting in the setting file
		shell_exec('sudo echo "' . $setting_name . '=' . $setting_value . '" >> ' . getFilePath('birdnet.conf'));
		//also update this run txt file
		shell_exec('sudo echo "' . $setting_name . '=' . $setting_value . '" >> ' . getFilePath('thisrun.txt'));

		//Check if we need to execute a command after saving
		if (isset($post_save_command)) {
			sleep(1);
			if (!is_array($post_save_command)) {
				executeSysCommand($post_save_command);
			} else {
				//array of commands
				foreach ($post_save_command as $single_command) {
					executeSysCommand($single_command);
				}
			}
		}
		//Reload the settings to update our global $config variable
		parseConfig();
	}
}

/**
 * Returns the value for the supplied setting name
 *
 * @param $setting_name
 * @return mixed|null
 */
function getSetting($setting_name)
{
	global $config;
	$setting_value = null;
	if (array_key_exists($setting_name, $config)) {
		$setting_value = $config[$setting_name];
	}

	return $setting_value;
}

/**
 * Returns the username of the user with User ID 1000
 *
 * @return string
 */
function getUser()
{
	global $user;
	return $user;
}

/**
 * Executes commands against the system through a shorthand command type
 *
 * @param $command_type string
 * @param $extra_data_to_pass string|[] Data or params that will be passed through to the command
 * @return false|string|null
 */
function executeSysCommand($command_type, $extra_data_to_pass = null)
{
	global $user, $home;

	//Check authentication before going any further
	//the front end would have done this check already and such should pass
	authenticateUser();

	$command_type = strtolower($command_type);
	$result = null;

	if ($command_type == "appraise_notification") {
		$result = shell_exec(getFilePath('python3_appraise') . " -vv --plugin-path " . $home . "/.apprise/plugins " . " -t '" . escapeshellcmd($extra_data_to_pass['title']) . "' -b '" . escapeshellcmd($extra_data_to_pass['body']) . "' " . $extra_data_to_pass['attach'] . " " . $extra_data_to_pass['cf'] . " ");
		//
	} else if ($command_type == "current_timezone") {
		$result = shell_exec("cat /etc/timezone");
		//
	} else if ($command_type == "disable_ntp") {
		$result = shell_exec("sudo timedatectl set-ntp false");
		//
	} else if ($command_type == "enable_ntp") {
		$result = shell_exec("sudo timedatectl set-ntp true");
		//
	} else if ($command_type == "is_ntp_active") {
		$result = shell_exec("sudo timedatectl | grep \"NTP service: active\"");
		//
	} else if ($command_type == "restart_php") {
		$result = serviceMaintenance('restart php.service');
		//
	} else if ($command_type == "restart_services") {
		syslog(LOG_INFO, "Restarting Services");
		$result = serviceMaintenance('restart core.services');
		//
	} else if ($command_type == "restart birdnet_recording") {
		$result = serviceMaintenance('restart birdnet_recording.service');
		//
	} else if ($command_type == "restart icecast2") {
		$result = serviceMaintenance('restart icecast2.service');
		//
	} else if ($command_type == "restart livestream") {
		$result = serviceMaintenance('restart livestream.service');
		//
	} else if ($command_type == "restart spectrogram_viewer") {
		$result = serviceMaintenance('restart spectrogram_viewer.service');
		//
	} else if ($command_type == "set_date") {
		$command = "sudo date -s '" . escapeshellcmd($extra_data_to_pass['date']) . " " . escapeshellcmd($extra_data_to_pass['time']) . "'";
		$result = shell_exec($command);
		//
	} else if ($command_type == "set_timezone") {
		$command = "sudo timedatectl set-timezone " . escapeshellcmd($extra_data_to_pass);
		$result = shell_exec($command);
		//
	} else if ($command_type == "test_threshold") {
		$command = "sudo -u $user " . getFilePath('python3') . ' ' . getFilePath('species.py') . " --threshold " . escapeshellcmd($extra_data_to_pass) . " 2>&1";
		$result = shell_exec($command);
		//
	} else if ($command_type == "update_birdnet") {
		$result = shell_exec("update_birdnet.sh");
		//
	} else if ($command_type == "reboot_birdnet") {
		$result = shell_exec("sudo reboot");
		//
	} else if ($command_type == "shutdown_birdnet") {
		$result = shell_exec("sudo shutdown now");
		//
	} else if ($command_type == "clear_all_data") {
		$result = shell_exec("sudo clear_all_data.sh");
		//
	}

	return $result;
}

/**
 * Updates the caddy configuration, only ever called when BirdNET-Pi Password or Site URL is set
 *
 * @return void
 */
function update_caddyfile()
{
	exec("sudo /usr/local/bin/update_caddyfile.sh > /dev/null 2>&1 &");
}

/**
 * Performs tasks such as stop, restart, disable and enable on the supplied service
 * input command must contain the service name and must contain a keyword that describes the action
 * positioning does not matter
 * eg. restart livestream.service, stop birdnet_recording.service
 *
 * @param $command String Command containing a keyword [stop,restart,disable,enable] and the service name
 * @return false|string|null
 */
function serviceMaintenance($command)
{
	$command = trim($command);
	$result = "";

	//Check authentication before going any further
	//the front end would have done this check already and such should pass
	authenticateUser();

	///e.g $command = 'service stop livestream.service', match the service name
	////// BIRDNET LOG SERVICE //////
	if (strpos($command, 'birdnet_log.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop birdnet_log.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart birdnet_log.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now birdnet_log.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now birdnet_log.service 2>&1');
		}
	} ////// BIRDNET SERVER SERVICE //////
	else if (strpos($command, 'birdnet_server.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop birdnet_server.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart birdnet_server.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now birdnet_server.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now birdnet_server.service 2>&1');
		}
	} ////// BIRDNET ANALYSIS SERVICE //////
	else if (strpos($command, 'birdnet_analysis.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop birdnet_analysis.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart birdnet_analysis.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now birdnet_analysis.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now birdnet_analysis.service 2>&1');
		}
	} ////// BIRDNET STATS SERVICE //////
	else if (strpos($command, 'birdnet_stats.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop birdnet_stats.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart birdnet_stats.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now birdnet_stats.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now birdnet_stats.service 2>&1');
		}
	} ////// BIRDNET RECORDING SERVICE //////
	else if (strpos($command, 'birdnet_recording.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop birdnet_recording.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart birdnet_recording.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now birdnet_recording.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now birdnet_recording.service 2>&1');
		}
	} ////// LIVESTREAM SERVICE //////
	else if (strpos($command, 'livestream.service') !== false) {
		//Check which keyword exists
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop livestream.service && sudo systemctl stop icecast2.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart livestream.service && sudo systemctl restart icecast2.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now livestream.service && sudo systemctl disable icecast2 && sudo systemctl stop icecast2.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable icecast2 && sudo systemctl start icecast2.service && sudo systemctl enable --now livestream.service 2>&1');
		}
	} ////// WEB TERMINAL SERVICE //////
	else if (strpos($command, 'web_terminal.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop web_terminal.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart web_terminal.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now web_terminal.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now web_terminal.service 2>&1');
		}
	} ////// EXTRACTION SERVICE //////
	else if (strpos($command, 'extraction.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop extraction.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart extraction.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now extraction.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now extraction.service 2>&1');
		}
	}////// CHART VIEWER SERVICE //////
	else if (strpos($command, 'chart_viewer.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop chart_viewer.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart chart_viewer.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now chart_viewer.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now chart_viewer.service 2>&1');
		}
	} ////// SPECTROGRAM VIEWER SERVICE //////
	else if (strpos($command, 'spectrogram_viewer.service') !== false) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('sudo systemctl stop spectrogram_viewer.service 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('sudo systemctl restart spectrogram_viewer.service 2>&1');
		} else if (strpos($command, 'disable') !== false) {
			$result = shell_exec('sudo systemctl disable --now spectrogram_viewer.service 2>&1');
		} else if (strpos($command, 'enable') !== false) {
			$result = shell_exec('sudo systemctl enable --now spectrogram_viewer.service 2>&1');
		}
	} ////// CORE SERVICES //////
	else if ((strpos($command, 'core.services') !== false) || (strpos($command, 'core services') !== false)) {
		if (strpos($command, 'stop') !== false) {
			$result = shell_exec('stop_core_services.sh 2>&1');
		} else if (strpos($command, 'restart') !== false) {
			$result = shell_exec('restart_services.sh 2>&1');
		}
	} ////// PHP FPM  //////
	else if (strpos($command, 'php.service') !== false) {
		if (strpos($command, 'restart') !== false) {
			$result = shell_exec("sudo service php7.4-fpm restart");
		}
	}

	return $result;
}

/**
 * Returns a the supplied timestamp as sometime human-readable
 * @link https://stackoverflow.com/questions/2690504/php-producing-relative-date-time-from-timestamps
 *
 * @param $ts
 * @return false|string
 */
function relativeTime($ts)
{
	if (!ctype_digit($ts))
		$ts = strtotime($ts);

	$diff = time() - $ts;
	if ($diff == 0)
		return 'now';
	elseif ($diff > 0) {
		$day_diff = floor($diff / 86400);
		if ($day_diff == 0) {
			if ($diff < 60) return 'just now';
			if ($diff < 120) return '1 minute ago';
			if ($diff < 3600) return floor($diff / 60) . ' minutes ago';
			if ($diff < 7200) return '1 hour ago';
			if ($diff < 86400) return floor($diff / 3600) . ' hours ago';
		}
		if ($day_diff == 1) return 'Yesterday';
		if ($day_diff < 7) return $day_diff . ' days ago';
		if ($day_diff < 31) return ceil($day_diff / 7) . ' weeks ago';
		if ($day_diff < 60) return 'last month';
		return date('F Y', $ts);
	} else {
		$diff = abs($diff);
		$day_diff = floor($diff / 86400);
		if ($day_diff == 0) {
			if ($diff < 120) return 'in a minute';
			if ($diff < 3600) return 'in ' . floor($diff / 60) . ' minutes';
			if ($diff < 7200) return 'in an hour';
			if ($diff < 86400) return 'in ' . floor($diff / 3600) . ' hours';
		}
		if ($day_diff == 1) return 'Tomorrow';
		if ($day_diff < 4) return date('l', $ts);
		if ($day_diff < 7 + (7 - date('w'))) return 'next week';
		if (ceil($day_diff / 7) < 4) return 'in ' . ceil($day_diff / 7) . ' weeks';
		if (date('n', $ts) == date('n') + 1) return 'next month';
		return date('F Y', $ts);
	}
}

/**
 * Execute system command as SUDO and logs it's output in the syslog
 *
 * @param $cmd string Command to Execute
 * @param $sudo_user string User to execute as
 * @return string
 */
function syslog_shell_exec($cmd, $sudo_user = null)
{
	$output = "";
	//Check authentication before going any further
	//the front end would have done this check already and such should pass
	authenticateUser();

	if ($sudo_user) {
		$cmd = "sudo -u $sudo_user $cmd";
	}
	$output = shell_exec($cmd);

	if (strlen($output) > 0) {
		syslog(LOG_INFO, $output);
	}

	return $output;
}

/**
 * Loads in configuration settings from either thisrun or firstrun.txt
 *
 * @return void
 */
function parseConfig()
{
	global $config;

	if (file_exists(getFilePath('thisrun.txt'))) {
		$config = parse_ini_file(getFilePath('thisrun.txt'));
	} elseif (file_exists(getFilePath('firstrun.txt'))) {
		$config = parse_ini_file(getFilePath('firstrun.txt'));
	}
}

/**
 * Updates the apprise config file
 *
 * @param $apprise_config
 * @return void
 */
function updateAppriseConfig($apprise_config)
{
	if (isset($apprise_config)) {
		$appriseconfig_config_file = fopen(getFilePath("apprise.txt"), "w");
		fwrite($appriseconfig_config_file, $apprise_config);
	}
}

/**
 * Returns the apprise config in apprise.txt
 * @return false|string
 */
function getAppriseConfig()
{
	if (file_exists(getFilePath('apprise.txt'))) {
		$apprise_config = file_get_contents(getFilePath('apprise.txt'));
	} else {
		$apprise_config = "";
	}

	return $apprise_config;
}

/**
 * Return the linecount of labels.txt
 * @return int
 */
function getLabelsCount()
{
	return count(file(getFilePath('labels.txt')));
}

/**
 * Returns the current git commit hash of the BirdNET-Pi respository
 * @return false|string|null
 */
function getGitCurrentHash()
{
	global $user;
	return shell_exec("cd " . getDirectory('home') . "/BirdNET-Pi && sudo -u " . $user . " git rev-list --max-count=1 HEAD");
}

/**
 * Runs git fetch against the BirdNET-Pi repository
 * @return false|string|null
 */
function doGitFetch()
{
	global $user;
	return shell_exec("sudo -u" . $user . " git -C " . getDirectory('home') . "/BirdNET-Pi fetch 2>&1");
}

/**
 * Get the git status of the BirdNET-Pi repository
 * @return false|string|null
 */
function getGitStatus()
{
	global $user;
	$behind = 0;
	//Fetch latest updates
	doGitFetch();
	$git_status = trim(shell_exec("sudo -u" . $user . " git -C " . getDirectory('home') . "/BirdNET-Pi status"));

	//Extract the text numberof commits from 'Your branch is behind '....' by XX commits,
	if (preg_match("/behind '.*?' by (\d+) commit(s?)\b/", $git_status, $matches)) {
		$num_commits_behind = $matches[1];
		$behind = $num_commits_behind;
	}

	if (preg_match('/\b(\d+)\b and \b(\d+)\b different commits each/', $git_status, $matches)) {
		$num1 = (int)$matches[1];
		$num2 = (int)$matches[2];
		$sum = $num1 + $num2;
		$behind = $sum;
	}

	return $behind;
}

/**
 * Returns the status of the specified service
 *
 * @param $name string Name of the service
 * @return string[]
 */
function getServiceStatus($name)
{
	$message = '';
	$status = '';

	if ($name == "birdnet_server.service") {
		$filesinproc = trim(shell_exec("ls " . getDirectory('home') . "/BirdSongs/Processed | wc -l"));
		if ($filesinproc > 200) {
			$message = "stalled - backlog of " . $filesinproc . " files in ~/BirdSongs/Processed/";

		}
	}
	$status = shell_exec("sudo systemctl status " . $name . " | grep Active | grep ' active\| activating\|running\|waiting\|start'");
	if (strlen($status) > 0) {
		$message = "(active)";
	} else {
		$message = "(inactive)";
	}

	return ["status" => $status, "message" => $message];
}

/**
 * Returns the boolean flag
 * @return false
 */
function userIsAuthenticated()
{
	global $USER_AUTHENTICATED;
	return $USER_AUTHENTICATED;
}

/**
 * When called, authenticates the user using the built-in PHP_AUTH. This is the orignial authentication code just wrapped in a function
 * This will automatically terminate the script if authentication fails
 *
 * @param $rejection_message string Message to return to the user if authentication fails
 * @return bool
 */
function authenticateUser($rejection_message = "You cannot edit the settings for this installation")
{
	global $config, $USER_AUTHENTICATED;
	parseConfig();
	$USER_AUTHENTICATED = false;

	$caddypwd = $config['CADDY_PWD'];
	if (!isset($_SERVER['PHP_AUTH_USER'])) {
		header('WWW-Authenticate: Basic realm="BirdNET-Pi"');
		header('HTTP/1.0 401 Unauthorized');
		echo '<table><tr><td>' . $rejection_message . '</td></tr></table>';
		exit;
	} else {
		//Read the supplied details
		$submittedpwd = $_SERVER['PHP_AUTH_PW'];
		$submitteduser = $_SERVER['PHP_AUTH_USER'];
		//
		if ($submittedpwd !== $caddypwd || $submitteduser !== 'birdnet') {
			header('WWW-Authenticate: Basic realm="BirdNET-Pi"');
			header('HTTP/1.0 401 Unauthorized');
			echo '<table><tr><td>' . $rejection_message . '</td></tr></table>';
			exit;
		} else {
			$USER_AUTHENTICATED = true;
		}
	}

	return $USER_AUTHENTICATED;
}

/**
 * Logging helper to accept a message and process it accordingly
 * @param $message
 * @param $level
 * @return void
 */
function birdnet_error_log($message, $level = 'error')
{
	$logfile_path = getDirectory('scripts') . "/web-ui-error.log";

	if (!file_exists($logfile_path)) {
		touch($logfile_path);
	}

	error_log($message . "\r\n", 3, $logfile_path);
}


register_shutdown_function('disconnect_from_birdsdb');