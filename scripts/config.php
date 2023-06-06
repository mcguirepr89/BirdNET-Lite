<?php
if (file_exists('./scripts/common.php')) {
	include_once "./scripts/common.php";
} else {
	include_once "./common.php";
}

//Parse the ini files to get the current config
parseConfig();

//Authenticate first before allowing further execution
authenticateUser();

//Authenticated
//Read in the apprise config
$apprise_config = getAppriseConfig();

if(isset($_GET['threshold'])) {
  $threshold = $_GET['threshold'];
  if (!is_numeric($threshold) || $threshold < 0 || $threshold > 1) {
    die('Invalid threshold value');
  }

  echo executeSysCommand('test_threshold', $threshold);
  die();
}

if(isset($_GET['restart_php']) && $_GET['restart_php'] == "true") {
  executeSysCommand('restart_php');
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
    executeSysCommand('set_timezone',$timezone);
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
    executeSysCommand('disable_ntp');

    // check if valid date and time
    $datetime = DateTime::createFromFormat('Y-m-d H:i', $_GET['date'] . ' ' . $_GET['time']);
    if ($datetime && $datetime->format('Y-m-d H:i') === $_GET['date'] . ' ' . $_GET['time']) {
		executeSysCommand('set_date', ['date' => $_GET['date'], 'time' => $_GET['time']]);
    }
  } else {
    // user checked 'use time from internet if available,' so make sure that's on
    if(strlen(trim(executeSysCommand('is_ntp_active'))) == 0){
		executeSysCommand('enable_ntp');
		sleep(3);
    }
  }

  // Update Language settings only if a change is requested

  changeLanguage($model, $language);

	saveSetting('SITE_NAME', "\"$site_name\"");
	saveSetting('LATITUDE', $latitude);
	saveSetting('LONGITUDE', $longitude);
	saveSetting('BIRDWEATHER_ID', $birdweather_id);
	saveSetting('APPRISE_NOTIFICATION_TITLE', "\"$apprise_notification_title\"");
	saveSetting('APPRISE_NOTIFICATION_BODY', "'$apprise_notification_body'");
	saveSetting('APPRISE_NOTIFY_EACH_DETECTION', $apprise_notify_each_detection);
	saveSetting('APPRISE_NOTIFY_NEW_SPECIES', $apprise_notify_new_species);
	saveSetting('APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY', $apprise_notify_new_species_each_day);
	saveSetting('APPRISE_WEEKLY_REPORT', $apprise_weekly_report);
	saveSetting('FLICKR_API_KEY', $flickr_api_key);
	saveSetting('DATABASE_LANG', $language);
	saveSetting('FLICKR_FILTER_EMAIL', $flickr_filter_email);
	saveSetting('APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES', $minimum_time_limit);
	saveSetting('MODEL', $model);
	saveSetting('SF_THRESH', $sf_thresh);
	saveSetting('APPRISE_ONLY_NOTIFY_SPECIES_NAMES', "\"$only_notify_species_names\"");
	saveSetting('APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2', "\"$only_notify_species_names_2\"");


  if($site_name != $config["SITE_NAME"]) {
    echo "<script>setTimeout(
    function() {
      window.parent.document.location.reload();
    }, 1000);</script>";
  }

	updateAppriseConfig($apprise_input);

    serviceMaintenance('restart core.services');
}

if(isset($_GET['sendtest']) && $_GET['sendtest'] == "true") {
  $cf = explode("\n",$_GET['apprise_config']);
  $cf = "'".implode("' '", $cf)."'";

  $result0 = getMostRecentDetectionToday();
  foreach ($result0['data'] as $todaytable)
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
  $title = str_replace("\$confidencepct", round($confidence*100), $title);
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
  $body = str_replace("\$confidencepct", round($confidence*100), $body);
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

  echo "<pre class=\"bash\">" . executeSysCommand('appraise_notification', ['title' => $title, 'body' => $body, 'attach' => $attach, 'cf' => $cf]) . "</pre>";

  die();
}

//Parse the ini files to get the current config
parseConfig();
?>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  </style>
  </head>
<div class="settings">
      <div class="brbanner"><h1>Basic Settings</h1></div><br>
    <form id="basicform" action=""  method="GET">

<script>
  document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('modelsel').addEventListener('change', function() {
    if(this.value == "BirdNET_GLOBAL_6K_V2.4_Model_FP16"){
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
      <p id="sfhelp" style='display:none'>This value is used by the model to constrain the list of possible species that it will try to detect, given the minimum occurence frequency. A 0.03 threshold means that for a species to be included in this list, it needs to, on average, be seen on at least 3% of historically submitted eBird checklists for your given lat/lon/current week of year. So, the lower the threshold, the rarer the species it will include.<br><img style='width:75%;padding-top:5px;padding-bottom:5px' alt="BirdNET-Pi new model detection flowchart" title="BirdNET-Pi new model detection flowchart" src="https://i.imgur.com/8YEAuSA.jpeg">
        <br>If you'd like to tinker with this threshold value and see which species make it onto the list, <?php if($config['MODEL'] == "BirdNET_6K_GLOBAL_MODEL"){ ?>please click "Update Settings" at the very bottom of this page to install the appropriate label file, then come back here and you'll be able to use the Species List Tester.<?php } else { ?>you can use this tool: <button type="button" class="testbtn" id="openModal">Species List Tester</button><?php } ?></p>
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
      <dt>BirdNET_GLOBAL_6K_V2.4_Model_FP16 (2023)</dt><br>
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
      <p><a href="https://app.birdweather.com" target="_blank">BirdWeather.com</a> is a weather map for bird sounds. Stations around the world supply audio and video streams to BirdWeather where they are then analyzed by BirdNET and compared to eBird Grid data. BirdWeather catalogues the bird audio and spectrogram visualizations so that you can listen to, view, and read about birds throughout the world. <a href="mailto:tim@birdweather.com?subject=Request%20BirdWeather%20ID&body=<?php include(getDirectory('scripts') . '/birdweather_request.php'); ?>" target="_blank">Email Tim</a> to request a BirdWeather ID</p>
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
      <dt>$confidencepct</dt>
      <dd>Confidence Score as a percentage (eg. 0.91 => 91)</dd>
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
      <input name="apprise_notification_title" style="width: 100%" type="text" value="<?php print($config['APPRISE_NOTIFICATION_TITLE']);?>" /><br>
      <label for="apprise_notification_body">Notification Body: </label>
      <input name="apprise_notification_body" style="width: 100%" type="text" value='<?php print($config['APPRISE_NOTIFICATION_BODY']);?>' /><br>
      <input type="checkbox" name="apprise_notify_new_species" <?php if($config['APPRISE_NOTIFY_NEW_SPECIES'] == 1 && filesize(getFilePath('apprise.txt')) != 0) { echo "checked"; };?> >
      <label for="apprise_notify_new_species">Notify each new infrequent species detection (<5 visits per week)</label><br>
      <input type="checkbox" name="apprise_notify_new_species_each_day" <?php if($config['APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY'] == 1 && filesize(getFilePath('apprise.txt')) != 0) { echo "checked"; };?> >
      <label for="apprise_notify_new_species_each_day">Notify each species first detection of the day</label><br>
      <input type="checkbox" name="apprise_notify_each_detection" <?php if($config['APPRISE_NOTIFY_EACH_DETECTION'] == 1 && filesize(getFilePath('apprise.txt')) != 0) { echo "checked"; };?> >
      <label for="apprise_weekly_report">Notify each new detection</label><br>
      <input type="checkbox" name="apprise_weekly_report" <?php if($config['APPRISE_WEEKLY_REPORT'] == 1 && filesize(getFilePath('apprise.txt')) != 0) { echo "checked"; };?> >
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
      $tdc = trim(executeSysCommand('is_ntp_active'));
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
      $current_timezone = trim(executeSysCommand('current_timezone'));
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

