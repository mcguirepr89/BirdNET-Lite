<?php
error_reporting(E_ERROR);
ini_set('display_errors',1);

function syslog_shell_exec($cmd, $sudo_user = null) {
  if ($sudo_user) {
    $cmd = "sudo -u $sudo_user $cmd";
  }
  $output = shell_exec($cmd);

  if (strlen($output) > 0) {
    syslog(LOG_INFO, $output);
  }
}

if(isset($_GET['threshold'])) {
  $threshold = $_GET['threshold'];
  if (!is_numeric($threshold) || $threshold < 0 || $threshold > 1) {
    die('Invalid threshold value');
  }

  $user = trim(shell_exec("awk -F: '/1000/{print $1}' /etc/passwd"));
  $home = trim(shell_exec("awk -F: '/1000/{print $6}' /etc/passwd"));

  $command = "sudo -u $user ".$home."/BirdNET-Pi/birdnet/bin/python3 ".$home."/BirdNET-Pi/scripts/species.py --threshold $threshold 2>&1";

  $output = shell_exec($command);

  echo $output;
  die();
}

if(isset($_GET['restart_php']) && $_GET['restart_php'] == "true") {
  shell_exec("sudo service php7.4-fpm restart");
  die();
}

# Basic Settings
if(isset($_GET["latitude"])){
  $latitude = $_GET["latitude"];
  $longitude = $_GET["longitude"];
  $site_name = $_GET["site_name"];
  $site_name = str_replace('"', "", $site_name);
  $site_name = str_replace('\'', "", $site_name);
  $birdweather_id = $_GET["birdweather_id"];
  $apprise_input = $_GET['apprise_input'];
  $apprise_notification_title = $_GET['apprise_notification_title'];
  $apprise_notification_body = $_GET['apprise_notification_body'];
  $minimum_time_limit = $_GET['minimum_time_limit'];
  $flickr_api_key = $_GET['flickr_api_key'];
  $flickr_filter_email = $_GET["flickr_filter_email"];
  $language = $_GET["language"];
  $timezone = $_GET["timezone"];
  $model = $_GET["model"];
  $sf_thresh = $_GET["sf_thresh"];
  $only_notify_species_names = $_GET['only_notify_species_names'];
  $only_notify_species_names_2 = $_GET['only_notify_species_names_2'];

  if(isset($_GET['apprise_notify_each_detection'])) {
    $apprise_notify_each_detection = 1;
  } else {
    $apprise_notify_each_detection = 0;
  }
  if(isset($_GET['apprise_notify_new_species'])) {
    $apprise_notify_new_species = 1;
  } else {
    $apprise_notify_new_species = 0;
  }
  if(isset($_GET['apprise_notify_new_species_each_day'])) {
    $apprise_notify_new_species_each_day = 1;
  } else {
    $apprise_notify_new_species_each_day = 0;
  }
  if(isset($_GET['apprise_weekly_report'])) {
    $apprise_weekly_report = 1;
  } else {
    $apprise_weekly_report = 0;
  }

  if(isset($timezone) && in_array($timezone, DateTimeZone::listIdentifiers())) {
    shell_exec("sudo timedatectl set-timezone ".$timezone);
    date_default_timezone_set($timezone);
    echo "<script>setTimeout(
    function() {
      const xhttp = new XMLHttpRequest();
    xhttp.open(\"GET\", \"./config.php?restart_php=true\", true);
    xhttp.send();
    }, 1000);</script>";
  }

  // logic for setting the date and time based on user inputs from the form below
  if(isset($_GET['date']) && isset($_GET['time'])) {
    // can't set the date manually if it's getting it from the internet, disable ntp
    exec("sudo timedatectl set-ntp false");

    // check if valid date and time
    $datetime = DateTime::createFromFormat('Y-m-d H:i', $_GET['date'] . ' ' . $_GET['time']);
    if ($datetime && $datetime->format('Y-m-d H:i') === $_GET['date'] . ' ' . $_GET['time']) {
      exec("sudo date -s '".$_GET['date']." ".$_GET['time']."'");
    }
  } else {
    // user checked 'use time from internet if available,' so make sure that's on
    if(strlen(trim(exec("sudo timedatectl | grep \"NTP service: active\""))) == 0){
      exec("sudo timedatectl set-ntp true");
      sleep(3);
    }
  }

  // Update Language settings only if a change is requested
  if (file_exists('./scripts/thisrun.txt')) {
    $lang_config = parse_ini_file('./scripts/thisrun.txt');
  } elseif (file_exists('./scripts/firstrun.ini')) {
    $lang_config = parse_ini_file('./scripts/firstrun.ini');
  }

  if ($model != $lang_config['MODEL'] || $language != $lang_config['DATABASE_LANG']){
    if(strlen($language) == 2){
      $user = trim(shell_exec("awk -F: '/1000/{print $1}' /etc/passwd"));
      $home = trim(shell_exec("awk -F: '/1000/{print $6}' /etc/passwd"));

      // Archive old language file
      syslog_shell_exec("cp -f $home/BirdNET-Pi/model/labels.txt $home/BirdNET-Pi/model/labels.txt.old", $user);

      if($model == "BirdNET_GLOBAL_3K_V2.3_Model_FP16"){
      // Install new language label file
        syslog_shell_exec("chmod +x $home/BirdNET-Pi/scripts/install_language_label_nm.sh && $home/BirdNET-Pi/scripts/install_language_label_nm.sh -l $language", $user);
      } else {
        syslog_shell_exec("$home/BirdNET-Pi/scripts/install_language_label.sh -l $language", $user);
      }

      syslog(LOG_INFO, "Successfully changed language to '$language' and model to '$model'");
    }
  }


  $contents = file_get_contents("/etc/birdnet/birdnet.conf");
  $contents = preg_replace("/SITE_NAME=.*/", "SITE_NAME=\"$site_name\"", $contents);
  $contents = preg_replace("/LATITUDE=.*/", "LATITUDE=$latitude", $contents);
  $contents = preg_replace("/LONGITUDE=.*/", "LONGITUDE=$longitude", $contents);
  $contents = preg_replace("/BIRDWEATHER_ID=.*/", "BIRDWEATHER_ID=$birdweather_id", $contents);
  $contents = preg_replace("/APPRISE_NOTIFICATION_TITLE=.*/", "APPRISE_NOTIFICATION_TITLE=\"$apprise_notification_title\"", $contents);
  $contents = preg_replace("/APPRISE_NOTIFICATION_BODY=.*/", "APPRISE_NOTIFICATION_BODY='$apprise_notification_body'", $contents);
  $contents = preg_replace("/APPRISE_NOTIFY_EACH_DETECTION=.*/", "APPRISE_NOTIFY_EACH_DETECTION=$apprise_notify_each_detection", $contents);
  $contents = preg_replace("/APPRISE_NOTIFY_NEW_SPECIES=.*/", "APPRISE_NOTIFY_NEW_SPECIES=$apprise_notify_new_species", $contents);
  $contents = preg_replace("/APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY=.*/", "APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY=$apprise_notify_new_species_each_day", $contents);
  $contents = preg_replace("/APPRISE_WEEKLY_REPORT=.*/", "APPRISE_WEEKLY_REPORT=$apprise_weekly_report", $contents);
  $contents = preg_replace("/FLICKR_API_KEY=.*/", "FLICKR_API_KEY=$flickr_api_key", $contents);
  $contents = preg_replace("/DATABASE_LANG=.*/", "DATABASE_LANG=$language", $contents);
  $contents = preg_replace("/FLICKR_FILTER_EMAIL=.*/", "FLICKR_FILTER_EMAIL=$flickr_filter_email", $contents);
  $contents = preg_replace("/APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES=.*/", "APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES=$minimum_time_limit", $contents);
  $contents = preg_replace("/MODEL=.*/", "MODEL=$model", $contents);
  $contents = preg_replace("/SF_THRESH=.*/", "SF_THRESH=$sf_thresh", $contents);
  $contents = preg_replace("/APPRISE_ONLY_NOTIFY_SPECIES_NAMES=.*/", "APPRISE_ONLY_NOTIFY_SPECIES_NAMES=\"$only_notify_species_names\"", $contents);
  $contents = preg_replace("/APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2=.*/", "APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2=\"$only_notify_species_names_2\"", $contents);

  $contents2 = file_get_contents("./scripts/thisrun.txt");
  $contents2 = preg_replace("/SITE_NAME=.*/", "SITE_NAME=\"$site_name\"", $contents2);
  $contents2 = preg_replace("/LATITUDE=.*/", "LATITUDE=$latitude", $contents2);
  $contents2 = preg_replace("/LONGITUDE=.*/", "LONGITUDE=$longitude", $contents2);
  $contents2 = preg_replace("/BIRDWEATHER_ID=.*/", "BIRDWEATHER_ID=$birdweather_id", $contents2);
  $contents2 = preg_replace("/APPRISE_NOTIFICATION_TITLE=.*/", "APPRISE_NOTIFICATION_TITLE=\"$apprise_notification_title\"", $contents2);
  $contents2 = preg_replace("/APPRISE_NOTIFICATION_BODY=.*/", "APPRISE_NOTIFICATION_BODY='$apprise_notification_body'", $contents2);
  $contents2 = preg_replace("/APPRISE_NOTIFY_EACH_DETECTION=.*/", "APPRISE_NOTIFY_EACH_DETECTION=$apprise_notify_each_detection", $contents2);
  $contents2 = preg_replace("/APPRISE_NOTIFY_NEW_SPECIES=.*/", "APPRISE_NOTIFY_NEW_SPECIES=$apprise_notify_new_species", $contents2);
  $contents2 = preg_replace("/APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY=.*/", "APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY=$apprise_notify_new_species_each_day", $contents2);
  $contents2 = preg_replace("/APPRISE_WEEKLY_REPORT=.*/", "APPRISE_WEEKLY_REPORT=$apprise_weekly_report", $contents2);
  $contents2 = preg_replace("/FLICKR_API_KEY=.*/", "FLICKR_API_KEY=$flickr_api_key", $contents2);
  $contents2 = preg_replace("/DATABASE_LANG=.*/", "DATABASE_LANG=$language", $contents2);
  $contents2 = preg_replace("/FLICKR_FILTER_EMAIL=.*/", "FLICKR_FILTER_EMAIL=$flickr_filter_email", $contents2);
  $contents2 = preg_replace("/APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES=.*/", "APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES=$minimum_time_limit", $contents2);
  $contents2 = preg_replace("/MODEL=.*/", "MODEL=$model", $contents2);
  $contents2 = preg_replace("/SF_THRESH=.*/", "SF_THRESH=$sf_thresh", $contents2);
  $contents2 = preg_replace("/APPRISE_ONLY_NOTIFY_SPECIES_NAMES=.*/", "APPRISE_ONLY_NOTIFY_SPECIES_NAMES=\"$only_notify_species_names\"", $contents2);
  $contents2 = preg_replace("/APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2=.*/", "APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2=\"$only_notify_species_names_2\"", $contents2);



  if($site_name != $config["SITE_NAME"]) {
    echo "<script>setTimeout(
    function() {
      window.parent.document.location.reload();
    }, 1000);</script>";
  }

  $fh = fopen("/etc/birdnet/birdnet.conf", "w");
  $fh2 = fopen("./scripts/thisrun.txt", "w");
  fwrite($fh, $contents);
  fwrite($fh2, $contents2);

  if(isset($apprise_input)){
    $user = shell_exec("awk -F: '/1000/{print $1}' /etc/passwd");
    $home = shell_exec("awk -F: '/1000/{print $6}' /etc/passwd");
    $home = trim($home);

    $appriseconfig = fopen($home."/BirdNET-Pi/apprise.txt", "w");
    fwrite($appriseconfig, $apprise_input);
  }

  syslog(LOG_INFO, "Restarting Services");
  shell_exec("sudo restart_services.sh");
}

if(isset($_GET['sendtest']) && $_GET['sendtest'] == "true") {
  $db = new SQLite3('./birds.db', SQLITE3_OPEN_CREATE | SQLITE3_OPEN_READWRITE);

  $user = shell_exec("awk -F: '/1000/{print $1}' /etc/passwd");
  $home = shell_exec("awk -F: '/1000/{print $6}' /etc/passwd");
  $home = trim($home);

  if (file_exists('./thisrun.txt')) {
    $config = parse_ini_file('./thisrun.txt');
  } elseif (file_exists('./firstrun.ini')) {
    $config = parse_ini_file('./firstrun.ini');
  }

  $cf = explode("\n",$_GET['apprise_config']);
  $cf = "'".implode("' '", $cf)."'";

  $statement0 = $db->prepare('SELECT * FROM detections WHERE Date == DATE(\'now\', \'localtime\') ORDER BY TIME DESC LIMIT 1');
  $result0 = $statement0->execute();
  while($todaytable=$result0->fetchArray(SQLITE3_ASSOC))
  {
    $sciname = $todaytable['Sci_Name'];
    $comname = $todaytable['Com_Name'];
    $confidence = $todaytable['Confidence'];
    $filename = $todaytable['File_Name'];
    $date = $todaytable['Date'];
    $time = $todaytable['Time'];
    $week = $todaytable['Week'];
    $latitude = $todaytable['Lat'];
    $longitude = $todaytable['Lon'];
    $cutoff = $todaytable['Cutoff'];
    $sens = $todaytable['Sens'];
    $overlap = $todaytable['Overlap'];
  }

  $title = $_GET['apprise_notification_title'];
  $body = $_GET['apprise_notification_body'];

  if($config["BIRDNETPI_URL"] != "") {
    $filename = $config["BIRDNETPI_URL"]."?filename=".$filename;
  } else{
    $filename = "http://birdnetpi.local/"."?filename=".$filename;
  }

  $attach="";
  $exampleimage = "https://live.staticflickr.com/7430/27545810581_8bfa8289a3_c.jpg";
  if (strpos($body, '$flickrimage') !== false) {
      $attach = "--attach ".$exampleimage;
  }
  if (strpos($body, '{') === false) {
      $exampleimage = "";
  }

  $title = str_replace("\$sciname", $sciname, $title);
  $title = str_replace("\$comname", $comname, $title);
  $title = str_replace("\$confidence", $confidence, $title);
  $title = str_replace("\$listenurl", $filename, $title);
  $title = str_replace("\$date", $date, $title);
  $title = str_replace("\$time", $time, $title);
  $title = str_replace("\$week", $week, $title);
  $title = str_replace("\$latitude", $latitude, $title);
  $title = str_replace("\$longitude", $longitude, $title);
  $title = str_replace("\$cutoff", $cutoff, $title);
  $title = str_replace("\$sens", $sens, $title);
  $title = str_replace("\$overlap", $overlap, $title);
  $title = str_replace("\$flickrimage", $exampleimage, $title);

  $body = str_replace("\$sciname", $sciname, $body);
  $body = str_replace("\$comname", $comname, $body);
  $body = str_replace("\$confidence", $confidence, $body);
  $body = str_replace("\$listenurl", $filename, $body);
  $body = str_replace("\$date", $date, $body);
  $body = str_replace("\$time", $time, $body);
  $body = str_replace("\$week", $week, $body);
  $body = str_replace("\$latitude", $latitude, $body);
  $body = str_replace("\$longitude", $longitude, $body);
  $body = str_replace("\$cutoff", $cutoff, $body);
  $body = str_replace("\$sens", $sens, $body);
  $body = str_replace("\$overlap", $overlap, $body);
  $body = str_replace("\$flickrimage", $exampleimage, $body);

  echo "<pre class=\"bash\">".shell_exec($home."/BirdNET-Pi/birdnet/bin/apprise -vv --plugin-path ".$home."/.apprise/plugins "." -t '".escapeshellcmd($title)."' -b '".escapeshellcmd($body)."' ".$attach." ".$cf." ")."</pre>";

  die();
}

?>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  </style>
  </head>
<div class="settings">
      <div class="brbanner"><h1>Basic Settings</h1></div><br>
    <form id="basicform" action=""  method="GET">
<?php
if (file_exists('./scripts/thisrun.txt')) {
  $config = parse_ini_file('./scripts/thisrun.txt');
} elseif (file_exists('./scripts/firstrun.ini')) {
  $config = parse_ini_file('./scripts/firstrun.ini');
}
$user = shell_exec("awk -F: '/1000/{print $1}' /etc/passwd");
$home = shell_exec("awk -F: '/1000/{print $6}' /etc/passwd");
$home = trim($home);
if (file_exists($home."/BirdNET-Pi/apprise.txt")) {
  $apprise_config = file_get_contents($home."/BirdNET-Pi/apprise.txt");
} else {
  $apprise_config = "";
}
$caddypwd = $config['CADDY_PWD'];
if (!isset($_SERVER['PHP_AUTH_USER'])) {
  header('WWW-Authenticate: Basic realm="My Realm"');
  header('HTTP/1.0 401 Unauthorized');
  echo '<table><tr><td>You cannot edit the settings for this installation</td></tr></table>';
  exit;
} else {
  $submittedpwd = $_SERVER['PHP_AUTH_PW'];
  $submitteduser = $_SERVER['PHP_AUTH_USER'];
  if($submittedpwd !== $caddypwd || $submitteduser !== 'birdnet'){
    header('WWW-Authenticate: Basic realm="My Realm"');
    header('HTTP/1.0 401 Unauthorized');
    echo '<table><tr><td>You cannot edit the settings for this installation</td></tr></table>';
    exit;
  }
}
?>    

<script>
  document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('modelsel').addEventListener('change', function() {
    if(this.value == "BirdNET_GLOBAL_3K_V2.3_Model_FP16"){ 
      document.getElementById("soft").style.display="unset";
    } else {
      document.getElementById("soft").style.display="none";
    }
  });
}, false);
function sendTestNotification(e) {
  document.getElementById("testsuccessmsg").innerHTML = "";
  e.classList.add("disabled");

  var apprise_notification_title = document.getElementsByName("apprise_notification_title")[0].value;
  var apprise_notification_body = document.getElementsByName("apprise_notification_body")[0].value;
  var apprise_config = encodeURIComponent(document.getElementsByName("apprise_input")[0].value);

  var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            document.getElementById("testsuccessmsg").innerHTML = this.responseText+" Test sent! Make sure to <b>Update Settings</b> below."
            e.classList.remove("disabled");
        }
    }
    xmlHttp.open("GET", "scripts/config.php?sendtest=true&apprise_notification_title="+apprise_notification_title+"&apprise_notification_body="+apprise_notification_body+"&apprise_config="+apprise_config, true); // true for asynchronous 
    xmlHttp.send(null);
}
</script>
      <table class="settingstable"><tr><td>
      <h2>Model</h2>

      <label for="model">Select a Model: </label>
      <select id="modelsel" name="model">
      <?php
      $models = array("BirdNET_6K_GLOBAL_MODEL", "BirdNET_GLOBAL_3K_V2.3_Model_FP16");
      foreach($models as $modelName){
          $isSelected = "";
          if($config['MODEL'] == $modelName){
            $isSelected = 'selected="selected"';
          }

          echo "<option value='{$modelName}' $isSelected>$modelName</option>";
        }
      ?>
      </select>
      <br>
      <span <?php if($config['MODEL'] == "BirdNET_6K_GLOBAL_MODEL") { ?>style="display: none"<?php } ?> id="soft">
      <label for="sf_thresh">Species Occurence Frequency Threshold [0.0005, 0.99]: </label>
      <input name="sf_thresh" type="number" max="0.99" min="0.0005" step="any" value="<?php print($config['SF_THRESH']);?>"/> <span onclick="document.getElementById('sfhelp').style.display='unset'" style="text-decoration:underline;cursor:pointer">[more info]</span><br>
      <p id="sfhelp" style='display:none'>This value is used by the model to constrain the list of possible species that it will try to detect, given the minimum occurence frequency. A 0.03 threshold means that for a species to be included in this list, it needs to, on average, be seen on at least 3% of historically submitted eBird checklists for your given lat/lon/current week of year. So, the lower the threshold, the rarer the species it will include.<br><br>If you'd like to tinker with this threshold value and see which species make it onto the list, <?php if($config['MODEL'] == "BirdNET_6K_GLOBAL_MODEL"){ ?>please click "Update Settings" at the very bottom of this page to install the appropriate label file, then come back here and you'll be able to use the Species List Tester.<?php } else { ?>you can use this tool: <button type="button" class="testbtn" id="openModal">Species List Tester</button><?php } ?></p>
      </span>

<script src="static/dialog-polyfill.js"></script>

<dialog id="modal">
  <div>
    <label for="threshold">Threshold:</label>
    <input type="number" id="threshold" step="0.01" min="0" max="1" value="">
    <button type="button" id="runProcess">Preview Species List</button>
  </div>
  <pre id="output"></pre>
  <button type="button" id="closeModal">Close</button>
</dialog>

<style>
#output {
  max-width: 100vw;
  word-wrap: break-word;
  white-space: pre-wrap;
}
#modal {
  max-height: 80vh;
  overflow-y: auto;
}
#modal div {
  display: flex;
  align-items: center;
}

#modal input[type="number"] {
  height: 32px;
}

#modal button {
  height: 32px;
  margin-left: 5px;
  padding: 0 10px;
  box-sizing: border-box;
}
</style>


<script>
// Get the button and modal elements
const openModalBtn = document.getElementById('openModal');
const modal = document.getElementById('modal');
dialogPolyfill.registerDialog(modal);
const output = document.getElementById('output');
const thresholdInput = document.getElementById('threshold');
const runProcessBtn = document.getElementById('runProcess');
const sfThreshInput = document.getElementsByName('sf_thresh')[0];
const closeModalBtn = document.getElementById('closeModal');


// Add an event listener to the button to open the modal
openModalBtn.addEventListener('click', () => {

  // Set the initial value of the threshold input element
  thresholdInput.value = sfThreshInput.value;

// Show the modal
  modal.showModal();
});

// Add an event listener to the "Preview Species List" button
runProcessBtn.addEventListener('click', () => {

  runProcess();
});

// Add an event listener to the "Close" button
closeModalBtn.addEventListener('click', () => {
  modal.close();
});

// Function to run the process
function runProcess() {
  // Get the value of the threshold input element
  const threshold = thresholdInput.value;

 // Set the output to "Loading..."
  output.innerHTML = "Loading...";

  // Make the AJAX request
  const xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === XMLHttpRequest.DONE) {
      // Handle the response
      output.innerHTML = xhr.responseText;
    }
  };
  xhr.open('GET', `scripts/config.php?threshold=${threshold}`);
  xhr.send();
}
</script>

      <dl>
      <dt>BirdNET_6K_GLOBAL_MODEL (2020)</dt><br>
      <dd id="ddnewline">This is the BirdNET-Lite model, with bird sound recognition for more than 6,000 species worldwide. This is the default option and will generally work very for people in most of the world.</dd>
    <br>
      <dt>BirdNET_GLOBAL_3K_V2.3_Model_FP16 (2023)</dt><br>
      <dd id="ddnewline">This is the BirdNET-Analyzer model, a newer work-in-progress project with aims to improve on the BirdNET-Lite model. Currently it only supports about 3,500 species worldwide, meaning for some regions (North America, Europe, Australia) it will usually outperform the BirdNET-Lite model, but for other regions it will be worse.</dd><br>
      <dt>[ In-depth technical write-up on the models <a target="_blank" href="https://github.com/mcguirepr89/BirdNET-Pi/wiki/BirdNET-Pi:-some-theory-on-classification-&-some-practical-hints">here</a> ]</dt>
      </dl>
      </td></tr></table><br>

      <table class="settingstable"><tr><td>
      <h2>Location</h2>
      <label for="site_name">Site Name: </label>
      <input name="site_name" type="text" value="<?php print($config['SITE_NAME']);?>"/> (Optional)<br>
      <label for="latitude">Latitude: &nbsp; &nbsp;</label>
      <input name="latitude" type="number" max="90" min="-90" step="0.0001" value="<?php print($config['LATITUDE']);?>" required/><br>
      <label for="longitude">Longitude: </label>
      <input name="longitude" type="number" max="180" min="-180" step="0.0001" value="<?php print($config['LONGITUDE']);?>" required/><br>
      <p>Set your Latitude and Longitude to 4 decimal places. Get your coordinates <a href="https://latlong.net" target="_blank">here</a>.</p>
      </td></tr></table><br>
      <table class="settingstable"><tr><td>
      <h2>BirdWeather</h2>
      <label for="birdweather_id">BirdWeather ID: </label>
      <input name="birdweather_id" type="text" value="<?php print($config['BIRDWEATHER_ID']);?>" /><br>
      <p><a href="https://app.birdweather.com" target="_blank">BirdWeather.com</a> is a weather map for bird sounds. Stations around the world supply audio and video streams to BirdWeather where they are then analyzed by BirdNET and compared to eBird Grid data. BirdWeather catalogues the bird audio and spectrogram visualizations so that you can listen to, view, and read about birds throughout the world. <a href="mailto:tim@birdweather.com?subject=Request%20BirdWeather%20ID&body=<?php include('./scripts/birdweather_request.php'); ?>" target="_blank">Email Tim</a> to request a BirdWeather ID</p>
      </td></tr></table><br>
      <table class="settingstable" style="width:100%"><tr><td>
      <h2>Notifications</h2>
      <p><a target="_blank" href="https://github.com/caronc/apprise/wiki">Apprise Notifications</a> can be setup and enabled for 70+ notification services. Each service should be on its own line.</p>
      <label for="apprise_input">Apprise Notifications Configuration: </label><br>
      <textarea placeholder="mailto://{user}:{password}@gmail.com
tgram://{bot_token}/{chat_id}
twitter://{ConsumerKey}/{ConsumerSecret}/{AccessToken}/{AccessSecret}
https://discordapp.com/api/webhooks/{WebhookID}/{WebhookToken}
..." style="vertical-align: top" name="apprise_input" rows="5" type="text" ><?php print($apprise_config);?></textarea>
      <dl>
      <dt>$sciname</dt>
      <dd>Scientific Name</dd>
      <dt>$comname</dt>
      <dd>Common Name</dd>
      <dt>$confidence</dt>
      <dd>Confidence Score</dd>
      <dt>$listenurl</dt>
      <dd>A link to the detection</dd>
      <dt>$date</dt>
      <dd>Date</dd>
      <dt>$time</dt>
      <dd>Time</dd>
      <dt>$week</dt>
      <dd>Week</dd>
      <dt>$latitude</dt>
      <dd>Latitude</dd>
      <dt>$longitude</dt>
      <dd>Longitude</dd>
      <dt>$cutoff</dt>
      <dd>Minimum Confidence set in "Advanced Settings"</dd>
      <dt>$sens</dt>
      <dd>Sigmoid Sensitivity set in "Advanced Settings"</dd>
      <dt>$overlap</dt>
      <dd>Overlap set in "Advanced Settings"</dd>
      <dt>$flickrimage</dt>
      <dd>A preview image of the detected species from Flickr. Set your API key below.</dd>
      </dl>
      <p>Use the variables defined above to customize your notification title and body.</p>
      <label for="apprise_notification_title">Notification Title: </label>
      <input name="apprise_notification_title" type="text" value="<?php print($config['APPRISE_NOTIFICATION_TITLE']);?>" /><br>
      <label for="apprise_notification_body">Notification Body: </label>
      <input name="apprise_notification_body" type="text" value='<?php print($config['APPRISE_NOTIFICATION_BODY']);?>' /><br>
      <input type="checkbox" name="apprise_notify_new_species" <?php if($config['APPRISE_NOTIFY_NEW_SPECIES'] == 1 && filesize($home."/BirdNET-Pi/apprise.txt") != 0) { echo "checked"; };?> >
      <label for="apprise_notify_new_species">Notify each new infrequent species detection (<5 visits per week)</label><br>
      <input type="checkbox" name="apprise_notify_new_species_each_day" <?php if($config['APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY'] == 1 && filesize($home."/BirdNET-Pi/apprise.txt") != 0) { echo "checked"; };?> >
      <label for="apprise_notify_new_species_each_day">Notify each species first detection of the day</label><br>
      <input type="checkbox" name="apprise_notify_each_detection" <?php if($config['APPRISE_NOTIFY_EACH_DETECTION'] == 1 && filesize($home."/BirdNET-Pi/apprise.txt") != 0) { echo "checked"; };?> >
      <label for="apprise_weekly_report">Notify each new detection</label><br>
      <input type="checkbox" name="apprise_weekly_report" <?php if($config['APPRISE_WEEKLY_REPORT'] == 1 && filesize($home."/BirdNET-Pi/apprise.txt") != 0) { echo "checked"; };?> >
      <label for="apprise_weekly_report">Send <a href="views.php?view=Weekly%20Report"> weekly report</a></label><br>

      <hr>
      <label for="minimum_time_limit">Minimum time between notifications of the same species (sec):</label>
      <input type="number" id="minimum_time_limit" name="minimum_time_limit" value="<?php echo $config['APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES'];?>" min="0"><br>
      <label for="only_notify_species_names">Exclude these species (comma separated common names):</label>
      <input type="text" id="only_notify_species_names" placeholder="Mourning Dove,American Crow" name="only_notify_species_names" value="<?php echo $config['APPRISE_ONLY_NOTIFY_SPECIES_NAMES'];?>" size=96><br>
      <label for="only_notify_species_names_2">ONLY notify for these species (comma separated common names):</label>
      <input type="text" id="only_notify_species_names_2" placeholder="Northern Cardinal,Carolina Chickadee,Eastern Bluebird" name="only_notify_species_names_2" value="<?php echo $config['APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2'];?>" size=96><br>

      <br>

      <button type="button" class="testbtn" onclick="sendTestNotification(this)">Send Test Notification</button><br>
      <span id="testsuccessmsg"></span>
      </td></tr></table><br>
      <table class="settingstable"><tr><td>
      <h2>Bird Photos from Flickr</h2>
      <label for="flickr_api_key">Flickr API Key: </label>
      <input name="flickr_api_key" type="text" value="<?php print($config['FLICKR_API_KEY']);?>"/><br>
      <label for="flickr_filter_email">Only search photos from this Flickr user: </label>
      <input name="flickr_filter_email" type="email" placeholder="myflickraccount@gmail.com" value="<?php print($config['FLICKR_FILTER_EMAIL']);?>"/><br>
      <p>Set your Flickr API key to enable the display of bird images next to detections. <a target="_blank" href="https://www.flickr.com/services/api/misc.api_keys.html">Get your free key here.</a></p>
      </td></tr></table><br>
      <table class="settingstable"><tr><td>
      <h2>Localization</h2>
      <label for="language">Database Language: </label>
      <select name="language">
      <?php
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

        // Create options for each language
        foreach($langs as $langTag => $langName){
          $isSelected = "";
          if($config['DATABASE_LANG'] == $langTag){
            $isSelected = 'selected="selected"';
          }

          echo "<option value='{$langTag}' $isSelected>$langName</option>";
        }
      ?>

      </select>
      </td></tr></table>
      <br>
      <script>
        function handleChange(checkbox) {
          // this disables the input of manual date and time if the user wants to use the internet time
          var date=document.getElementById("date");
          var time=document.getElementById("time");
          if(checkbox.checked) {
            date.setAttribute("disabled", "disabled");
            time.setAttribute("disabled", "disabled");
          } else {
            date.removeAttribute("disabled");
            time.removeAttribute("disabled");
          }
        }
      </script>
      <?php
      // if NTP service is active, show the checkboxes as checked, and disable the manual input
      $tdc = trim(exec("sudo timedatectl | grep \"NTP service: active\""));
      if (strlen($tdc) > 0) {
        $checkedvalue = "checked";
        $disabledvalue = "disabled";
      } else {
        $checkedvalue = "";
        $disabledvalue = "";
      }
      ?>
      <table class="settingstable"><tr><td>
      <h2>Time and Date</h2>
      <span>If connected to the internet, retrieve time automatically?</span>
      <input type="checkbox" onchange='handleChange(this)' <?php echo $checkedvalue; ?> ><br>
      <?php
      $date = new DateTime('now');
      ?>
      <input onclick="this.showPicker()" type="date" id="date" name="date" value="<?php echo $date->format('Y-m-d') ?>" <?php echo $disabledvalue; ?>>
      <input onclick="this.showPicker()" type="time" id="time" name="time" value="<?php echo $date->format('H:i'); ?>" <?php echo $disabledvalue; ?>><br>
      <br>
      <label for="timezone">Select a Timezone: </label>
      <select name="timezone">
      <option disabled selected>
        Select a timezone
      </option>
      <?php
      $current_timezone = trim(shell_exec("cat /etc/timezone"));
      $timezone_identifiers = DateTimeZone::listIdentifiers(DateTimeZone::ALL);
        
      $n = 425;
      for($i = 0; $i < $n; $i++) {
          $isSelected = "";
          if($timezone_identifiers[$i] == $current_timezone) {
            $isSelected = 'selected="selected"';
          }
          echo "<option $isSelected value='".$timezone_identifiers[$i]."'>".$timezone_identifiers[$i]."</option>";
      }
      ?>
      </select>
      </td></tr></table><br>

      <br><br>

      <input type="hidden" name="status" value="success">
      <input type="hidden" name="submit" value="settings">
      <button type="submit" id="basicformsubmit" onclick="if(document.getElementById('basicform').checkValidity()){this.innerHTML = 'Updating... please wait.';this.classList.add('disabled')}" name="view" value="Settings">
<?php
if(isset($_GET['status'])){
  echo "Success!";
} else {
  echo "Update Settings";
}
?>
      </button>
      </form>
      <form action="" method="GET">
        <button type="submit" name="view" value="Advanced">Advanced Settings</button>
      </form>
</div>

