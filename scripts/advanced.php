<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if (file_exists('./scripts/thisrun.txt')) {
  $config = parse_ini_file('./scripts/thisrun.txt');
} elseif (file_exists('firstrun.ini')) {
  $config = parse_ini_file('firstrun.ini');
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

if(isset($_GET['submit'])) {
  $contents = file_get_contents('/etc/birdnet/birdnet.conf');
  $contents2 = file_get_contents('./scripts/thisrun.txt');

  if(isset($_GET["caddy_pwd"])) {
    $caddy_pwd = $_GET["caddy_pwd"];
    if(strcmp($caddy_pwd,$config['CADDY_PWD']) !== 0) {
      $contents = preg_replace("/CADDY_PWD=.*/", "CADDY_PWD=\"$caddy_pwd\"", $contents);
      $contents2 = preg_replace("/CADDY_PWD=.*/", "CADDY_PWD=\"$caddy_pwd\"", $contents2);
      $fh = fopen('/etc/birdnet/birdnet.conf', "w");
      $fh2 = fopen("./scripts/thisrun.txt", "w");
      fwrite($fh, $contents);
      fwrite($fh2, $contents2);
      exec('sudo /usr/local/bin/update_caddyfile.sh > /dev/null 2>&1 &');
    }
  }

  if(isset($_GET["ice_pwd"])) {
    $ice_pwd = $_GET["ice_pwd"];
    if(strcmp($ice_pwd,$config['ICE_PWD']) !== 0) {
      $contents = preg_replace("/ICE_PWD=.*/", "ICE_PWD=$ice_pwd", $contents);
      $contents2 = preg_replace("/ICE_PWD=.*/", "ICE_PWD=$ice_pwd", $contents2);
    }
  }

  if(isset($_GET["birdnetpi_url"])) {
    $birdnetpi_url = $_GET["birdnetpi_url"];
    if(strcmp($birdnetpi_url,$config['BIRDNETPI_URL']) !== 0) {
      $contents = preg_replace("/BIRDNETPI_URL=.*/", "BIRDNETPI_URL=$birdnetpi_url", $contents);
      $contents2 = preg_replace("/BIRDNETPI_URL=.*/", "BIRDNETPI_URL=$birdnetpi_url", $contents2);
      $fh = fopen('/etc/birdnet/birdnet.conf', "w");
      $fh2 = fopen("./scripts/thisrun.txt", "w");
      fwrite($fh, $contents);
      fwrite($fh2, $contents2);
      exec('sudo /usr/local/bin/update_caddyfile.sh > /dev/null 2>&1 &');
    }
  }

  if(isset($_GET["rtsp_stream"])) {
    $rtsp_stream = str_replace("\r\n", ",", $_GET["rtsp_stream"]);
    if(strcmp($rtsp_stream,$config['RTSP_STREAM']) !== 0) {
      $contents = preg_replace("/RTSP_STREAM=.*/", "RTSP_STREAM=$rtsp_stream", $contents);
      $contents2 = preg_replace("/RTSP_STREAM=.*/", "RTSP_STREAM=$rtsp_stream", $contents2);
      $fh = fopen('/etc/birdnet/birdnet.conf', "w");
      $fh2 = fopen("./scripts/thisrun.txt", "w");
      fwrite($fh, $contents);
      fwrite($fh2, $contents2);
      exec('sudo systemctl restart birdnet_recording.service');
    }
  }
  
  if(isset($_GET["overlap"])) {
    $overlap = $_GET["overlap"];
    if(strcmp($overlap,$config['OVERLAP']) !== 0) {
      $contents = preg_replace("/OVERLAP=.*/", "OVERLAP=$overlap", $contents);
      $contents2 = preg_replace("/OVERLAP=.*/", "OVERLAP=$overlap", $contents2);
    }
  }

  if(isset($_GET["confidence"])) {
    $confidence = $_GET["confidence"];
    if(strcmp($confidence,$config['CONFIDENCE']) !== 0) {
      $contents = preg_replace("/CONFIDENCE=.*/", "CONFIDENCE=$confidence", $contents);
      $contents2 = preg_replace("/CONFIDENCE=.*/", "CONFIDENCE=$confidence", $contents2);
    }
  }

  if(isset($_GET["sensitivity"])) {
    $sensitivity = $_GET["sensitivity"];
    if(strcmp($sensitivity,$config['SENSITIVITY']) !== 0) {
      $contents = preg_replace("/SENSITIVITY=.*/", "SENSITIVITY=$sensitivity", $contents);
      $contents2 = preg_replace("/SENSITIVITY=.*/", "SENSITIVITY=$sensitivity", $contents2);
    }
  }

  if(isset($_GET["full_disk"])) {
    $full_disk = $_GET["full_disk"];
    if(strcmp($full_disk,$config['FULL_DISK']) !== 0) {
      $contents = preg_replace("/FULL_DISK=.*/", "FULL_DISK=$full_disk", $contents);
      $contents2 = preg_replace("/FULL_DISK=.*/", "FULL_DISK=$full_disk", $contents2);
    }
  }

  if(isset($_GET["privacy_threshold"])) {
    $privacy_threshold = $_GET["privacy_threshold"];
    if(strcmp($privacy_threshold,$config['PRIVACY_THRESHOLD']) !== 0) {
      $contents = preg_replace("/PRIVACY_THRESHOLD=.*/", "PRIVACY_THRESHOLD=$privacy_threshold", $contents);
      $contents2 = preg_replace("/PRIVACY_THRESHOLD=.*/", "PRIVACY_THRESHOLD=$privacy_threshold", $contents2);
      exec('restart_services.sh');
    }
  }

  if(isset($_GET["rec_card"])) {
    $rec_card = $_GET["rec_card"];
    if(strcmp($rec_card,$config['REC_CARD']) !== 0) {
      $contents = preg_replace("/REC_CARD=.*/", "REC_CARD=\"$rec_card\"", $contents);
      $contents2 = preg_replace("/REC_CARD=.*/", "REC_CARD=\"$rec_card\"", $contents2);
    }
  }

  if(isset($_GET["channels"])) {
    $channels = $_GET["channels"];
    if(strcmp($channels,$config['CHANNELS']) !== 0) {
      $contents = preg_replace("/CHANNELS=.*/", "CHANNELS=$channels", $contents);
      $contents2 = preg_replace("/CHANNELS=.*/", "CHANNELS=$channels", $contents2);
    }
  }

  if(isset($_GET["recording_length"])) {
    $recording_length = $_GET["recording_length"];
    if(strcmp($recording_length,$config['RECORDING_LENGTH']) !== 0) {
      $contents = preg_replace("/RECORDING_LENGTH=.*/", "RECORDING_LENGTH=$recording_length", $contents);
      $contents2 = preg_replace("/RECORDING_LENGTH=.*/", "RECORDING_LENGTH=$recording_length", $contents2);
    }
  }

  if(isset($_GET["extraction_length"])) {
    $extraction_length = $_GET["extraction_length"];
    if(strcmp($extraction_length,$config['EXTRACTION_LENGTH']) !== 0) {
      $contents = preg_replace("/EXTRACTION_LENGTH=.*/", "EXTRACTION_LENGTH=$extraction_length", $contents);
      $contents2 = preg_replace("/EXTRACTION_LENGTH=.*/", "EXTRACTION_LENGTH=$extraction_length", $contents2);
    }
  }

  if(isset($_GET["audiofmt"])) {
    $audiofmt = $_GET["audiofmt"];
    if(strcmp($audiofmt,$config['AUDIOFMT']) !== 0) {
      $contents = preg_replace("/AUDIOFMT=.*/", "AUDIOFMT=$audiofmt", $contents);
      $contents2 = preg_replace("/AUDIOFMT=.*/", "AUDIOFMT=$audiofmt", $contents2);
    }
  }

  $fh = fopen('/etc/birdnet/birdnet.conf', "w");
  $fh2 = fopen("./scripts/thisrun.txt", "w");
  fwrite($fh, $contents);
  fwrite($fh2, $contents2);
}

$count_labels = count(file("./scripts/labels.txt"));
$count = $count_labels;
?>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  </style>
  </head>
<div class="settings">

<?php
if (file_exists('./scripts/thisrun.txt')) {
  $newconfig = parse_ini_file('./scripts/thisrun.txt');
} elseif (file_exists('./scripts/firstrun.ini')) {
  $newconfig = parse_ini_file('./scripts/firstrun.ini');
}
?>
      <h2>Advanced Settings</h2>
    <form action="" method="GET">
      <label>Privacy Threshold: </label><br>
      <div class="slidecontainer">
        <input name="privacy_threshold" type="range" min="0" max="3" value="<?php print($newconfig['PRIVACY_THRESHOLD']);?>" class="slider" id="privacy_threshold">
        <p>Value: <span id="threshold_value"></span>%</p>
      </div>
      <script>
      var slider = document.getElementById("privacy_threshold");
      var output = document.getElementById("threshold_value");
      output.innerHTML = slider.value; // Display the default slider value
      
      // Update the current slider value (each time you drag the slider handle)
      slider.oninput = function() {
        output.innerHTML = this.value;
        document.getElementById("predictionCount").innerHTML = parseInt((this.value * <?php echo $count; ?>)/100);
      }
      </script>
      <p>If a Human is predicted anywhere among the top <span id="predictionCount"><?php echo $newconfig['PRIVACY_THRESHOLD'] == 0 ? "threshold % of" : intval(($newconfig['PRIVACY_THRESHOLD'] * $count)/100); ?></span> predictions, the sample will be considered of human origin and no data will be collected. Start with 1% and move up as needed.</p>
      <label>Full Disk Behavior: </label>
      <label for="purge">
      <input name="full_disk" type="radio" id="purge" value="purge" <?php if (strcmp($newconfig['FULL_DISK'], "purge") == 0) { echo "checked"; }?>>Purge</label>
      <label for="keep">
      <input name="full_disk" type="radio" id="keep" value="keep" <?php if (strcmp($newconfig['FULL_DISK'], "keep") == 0) { echo "checked"; }?>>Keep</label>
      <p>When the disk becomes full, you can choose to 'purge' old files to make room for new ones or 'keep' your data and stop all services instead.<br>Note: you can exclude specific files from 'purge' on the Recordings page.</p>
      <label for="rec_card">Audio Card: </label>
      <input name="rec_card" type="text" value="<?php print($newconfig['REC_CARD']);?>" required/><br>
      <p>Set Audio Card to 'default' to use PulseAudio (always recommended), or an ALSA recognized sound card device from the output of `aplay -L`.</p>
      <label for="channels">Audio Channels: </label>
      <input name="channels" type="number" min="1" max="32" step="1" value="<?php print($newconfig['CHANNELS']);?>" required/><br>
      <p>Set Channels to the number of channels supported by your sound card. 32 max.</p>
      <label for="rtsp_stream">RTSP Stream: </label>
      <input name="rtsp_stream" type="url" value="<?php echo $newconfig['RTSP_STREAM'];?>"</input><br>
      <p>If you place an RTSP stream URL here, BirdNET-Pi will use that as its audio source.</p>
      <label for="recording_length">Recording Length: </label>
      <input name="recording_length" oninput="document.getElementsByName('extraction_length')[0].setAttribute('max', this.value);" type="number" min="3" max="60" step="1" value="<?php print($newconfig['RECORDING_LENGTH']);?>" required/><br>
      <p>Set Recording Length in seconds between 6 and 60. Multiples of 3 are recommended, as BirdNET analyzes in 3-second chunks.</p> 
      <label for="extraction_length">Extraction Length: </label>
      <input name="extraction_length" oninput="this.setAttribute('max', document.getElementsByName('recording_length')[0].value);" type="number" min="3" value="<?php print($newconfig['EXTRACTION_LENGTH']);?>" /><br>
      <p>Set Extraction Length to something less than your Recording Length. Min=3 Max=Recording Length</p>
      <label for="audiofmt">Extractions Audio Format</label>
      <select name="audiofmt">
      <option selected="<?php print($newconfig['AUDIOFMT']);?>"><?php print($newconfig['AUDIOFMT']);?></option>
<?php
  $formats = array("8svx", "aif", "aifc", "aiff", "aiffc", "al", "amb", "amr-nb", "amr-wb", "anb", "au", "avr", "awb", "caf", "cdda", "cdr", "cvs", "cvsd", "cvu", "dat", "dvms", "f32", "f4", "f64", "f8", "fap", "flac", "fssd", "gsm", "gsrt", "hcom", "htk", "ima", "ircam", "la", "lpc", "lpc10", "lu", "mat", "mat4", "mat5", "maud", "mp2", "mp3", "nist", "ogg", "paf", "prc", "pvf", "raw", "s1", "s16", "s2", "s24", "s3", "s32", "s4", "s8", "sb", "sd2", "sds", "sf", "sl", "sln", "smp", "snd", "sndfile", "sndr", "sndt", "sou", "sox", "sph", "sw", "txw", "u1", "u16", "u2", "u24", "u3", "u32", "u4", "u8", "ub", "ul", "uw", "vms", "voc", "vorbis", "vox", "w64", "wav", "wavpcm", "wv", "wve", "xa", "xi");
foreach($formats as $format){
  echo "<option value='$format'>$format</option>";
}
?>
      </select>
      <h3>BirdNET-Pi Password</h3>
      <p>This password will protect your "Tools" page and "Live Audio" stream.</p>
      <label for="caddy_pwd">Password: </label>
      <input style="width:40ch" name="caddy_pwd" type="password" value="<?php print($newconfig['CADDY_PWD']);?>" /><br>
      <h3>Custom URL</h3>
      <p><a href="mailto:mcguirepr89@gmail.com?subject=Request%20BirdNET-Pi%20Subdomain&body=<?php include('birdnetpi_request.php'); ?>" target="_blank">Email Me</a> if you would like a BirdNETPi.com subdomain. This would be, <i>https://YourLocation.birdnetpi.com</i></p>
      <p>When you update the URL below, the web server will reload, so be sure to wait at least 30 seconds and then go to your new URL.</p>
      <label for="birdnetpi_url">BirdNET-Pi URL: </label>
      <input style="width:40ch;" name="birdnetpi_url" type="url" value="<?php print($newconfig['BIRDNETPI_URL']);?>" /><br>
      <p>The BirdNET-Pi URL is how the main page will be reached. If you want your installation to respond to an IP address, place that here, but be sure to indicate "<i>http://</i>".<br>Example for IP: <i>http://192.168.0.109</i><br>Example if you own your own domain: <i>https://virginia.birdnetpi.com</i></p>
      <h3>BirdNET-Lite Settings</h3>
      <label for="overlap">Overlap: </label>
      <input name="overlap" type="number" min="0.0" max="2.9" step="0.1" value="<?php print($newconfig['OVERLAP']);?>" required/><br>
      <p>Min=0.0, Max=2.9</p>
      <label for="confidence">Minimum Confidence: </label>
      <input name="confidence" type="number" min="0.01" max="0.99" step="0.01" value="<?php print($newconfig['CONFIDENCE']);?>" required/><br>
      <p>Min=0.01, Max=0.99</p>
      <label for="sensitivity">Sigmoid Sensitivity: </label>
      <input name="sensitivity" type="number" min="0.5" max="1.5" step="0.01" value="<?php print($newconfig['SENSITIVITY']);?>" required/><br>
      <p>Min=0.5, Max=1.5</p>
      <br><br>
      <input type="hidden" name="view" value="Advanced">
      <button onclick="if(<?php print($newconfig['PRIVACY_THRESHOLD']);?> != document.getElementById('privacy_threshold').value){return confirm('This will take about 90 seconds.')}" type="submit" name="submit" value="advanced">
<?php
if(isset($_GET['submit'])){
  echo "Success!";
} else {
  echo "Update Settings";
}
?>
      </button>
      <br>
      </form>
      <form action="" method="GET">
        <button type="submit" name="view" value="Settings">Basic Settings</button>
      </form>
</div>
