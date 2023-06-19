<?php

/* Prevent XSS input */
$_GET   = filter_input_array(INPUT_GET, FILTER_SANITIZE_STRING);
$_POST  = filter_input_array(INPUT_POST, FILTER_SANITIZE_STRING);

if(file_exists('./scripts/common.php')){
	include_once "./scripts/common.php";
}else{
	include_once "./common.php";
}

session_start();

if (!isset($_SESSION['behind'])) {
	$_SESSION['behind'] = getGitStatus();
}
if (isset($_SESSION['behind']) && intval($_SESSION['behind']) >= 99) { ?>
  <style>
  .updatenumber {
    width:30px !important;
  }
  </style>
<?php }

parseConfig();

if ($config["LATITUDE"] == "0.000" && $config["LONGITUDE"] == "0.000") {
  echo "<center style='color:red'><b>WARNING: Your latitude and longitude are not set properly. Please do so now in Tools -> Settings.</center></b>";
}
elseif ($config["LATITUDE"] == "0.000") {
  echo "<center style='color:red'><b>WARNING: Your latitude is not set properly. Please do so now in Tools -> Settings.</center></b>";
}
elseif ($config["LONGITUDE"] == "0.000") {
  echo "<center style='color:red'><b>WARNING: Your longitude is not set properly. Please do so now in Tools -> Settings.</center></b>";
}
?>
<link rel="stylesheet" href="style.css?v=<?php echo date ('n.d.y', filemtime('style.css')); ?>">
<style>

</style>
<meta name="viewport" content="width=device-width, initial-scale=1">
<div class="topnav" id="myTopnav">
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="Overview" form="views">Overview</button>
</form>
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="Todays Detections" form="views">Today's Detections</button>
</form>
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="Spectrogram" form="views">Spectrogram</button>
</form>
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="Species Stats" form="views">Best Recordings</button>
</form>
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="Streamlit" form="views">Species Stats</button>
</form>
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="Daily Charts" form="views">Daily Charts</button>
</form>
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="Recordings" form="views">Recordings</button>
</form>
<form action="" method="GET" id="views">
  <button type="submit" name="view" value="View Log" form="views">View Log</button>
</form>
<form action="" id="toolsbtn" method="GET" id="views">
  <button type="submit" name="view" value="Tools" form="views">Tools<?php if(isset($_SESSION['behind']) && intval($_SESSION['behind']) >= 50 && ($config['SILENCE_UPDATE_INDICATOR'] != 1)){ $updatediv = ' <div class="updatenumber">'.$_SESSION["behind"].'</div>'; } else { $updatediv = ""; } echo $updatediv; ?></button>
</form>
<button href="javascript:void(0);" class="icon" onclick="myFunction()"><img src="images/menu.png"></button>
</div>

<script>
window.onload = function() {
  var elements = document.querySelectorAll("button[name=view]");

  var setViewsOpacity = function() {
      document.getElementsByClassName("views")[0].style.opacity = "0.5";
  };

  for (var i = 0; i < elements.length; i++) {
      elements[i].addEventListener('click', setViewsOpacity, false);
  }
};
var topbuttons = document.querySelectorAll("button[form='views']");
if(window.location.search.substr(1) != '') {
  for (var i = 0; i < topbuttons.length; i++) {
    if(topbuttons[i].value == decodeURIComponent(window.location.search.substr(1)).replace(/\+/g,' ').split('=').pop()) {
      topbuttons[i].classList.add("button-hover");
    }
  }
} else {
  topbuttons[0].classList.add("button-hover");
}
function copyOutput(elem) {
  elem.innerHTML = 'Copied!';
  const copyText = document.getElementsByTagName("pre")[0].textContent;
  const textArea = document.createElement('textarea');
  textArea.style.position = 'absolute';
  textArea.style.left = '-100%';
  textArea.textContent = copyText;
  document.body.append(textArea);
  textArea.select();
  document.execCommand("copy");
}
</script>

<div class="views">
<?php
if(isset($_GET['view'])){
  if($_GET['view'] == "System Info"){echo "<iframe src='phpsysinfo/index.php'></iframe>";}
  if($_GET['view'] == "System Controls"){include('scripts/system_controls.php');}
  if($_GET['view'] == "Services"){include('scripts/service_controls.php');}
  if($_GET['view'] == "Spectrogram"){include('spectrogram.php');}
  if($_GET['view'] == "View Log"){echo "<body style=\"scroll:no;overflow-x:hidden;\"><iframe style=\"width:calc( 100% + 1em);\" src=\"/log\"></iframe></body>";}
  if($_GET['view'] == "Overview"){include('overview.php');}
  if($_GET['view'] == "Todays Detections"){include('todays_detections.php');}
  if($_GET['view'] == "Kiosk"){$kiosk = true;include('todays_detections.php');}
  if($_GET['view'] == "Species Stats"){include('stats.php');}
  if($_GET['view'] == "Weekly Report"){include('weekly_report.php');}
  if($_GET['view'] == "Streamlit"){echo "<iframe src=\"/stats\"></iframe>";}
  if($_GET['view'] == "Daily Charts"){include('history.php');}
  if($_GET['view'] == "Tools"){
	//Authenticate before proceeding
    $user_is_authenticated =  authenticateUser();
	  if ($user_is_authenticated) {
		  $url = $_SERVER['SERVER_NAME'] . "/scripts/adminer.php";
		  echo "<div class=\"centered\">
          <form action=\"\" method=\"GET\" id=\"views\">
          <button type=\"submit\" name=\"view\" value=\"Settings\" form=\"views\">Settings</button>
          <button type=\"submit\" name=\"view\" value=\"System Info\" form=\"views\">System Info</button>
          <button type=\"submit\" name=\"view\" value=\"System Controls\" form=\"views\">System Controls".$updatediv."</button>
          <button type=\"submit\" name=\"view\" value=\"Services\" form=\"views\">Services</button>
          <button type=\"submit\" name=\"view\" value=\"File\" form=\"views\">File Manager</button>
          <a href=\"scripts/adminer.php\" target=\"_blank\"><button type=\"submit\" form=\"\">Database Maintenance</button></a>
          <button type=\"submit\" name=\"view\" value=\"Webterm\" form=\"views\">Web Terminal</button>
          <button type=\"submit\" name=\"view\" value=\"Included\" form=\"views\">Custom Species List</button>
          <button type=\"submit\" name=\"view\" value=\"Excluded\" form=\"views\">Excluded Species List</button>
          </form>
          </div>";
	  }
  }
  if($_GET['view'] == "Recordings"){include('play.php');}
  if($_GET['view'] == "Settings"){include('scripts/config.php');} 
  if($_GET['view'] == "Advanced"){include('scripts/advanced.php');}
  if($_GET['view'] == "Included"){
	  $user_is_authenticated = authenticateUser('You cannot modify the system');
	  if(isset($_GET['species']) && isset($_GET['add']) && $user_is_authenticated){
      $file = getFilePath('include_species_list.txt');
      $str = file_get_contents("$file");
      $str = preg_replace("/(^[\r\n]*|[\r\n]+)[\s\t]*[\r\n]+/", "\n", $str);
      file_put_contents("$file", "$str");
      if(isset($_GET['species'])){
        foreach ($_GET['species'] as $selectedOption)
          file_put_contents( getFilePath('include_species_list.txt'), $selectedOption."\n", FILE_APPEND);
      }
    } elseif(isset($_GET['species']) && isset($_GET['del'])){
      $file = getFilePath('include_species_list.txt');
      $str = file_get_contents("$file");
      $str = preg_replace('/^\h*\v+/m', '', $str);
      file_put_contents("$file", "$str");
      foreach($_GET['species'] as $selectedOption) {
        $content = file_get_contents( getFilePath('include_species_list.txt'));
        $newcontent = str_replace($selectedOption, "", "$content");
        file_put_contents( getFilePath('include_species_list.txt'), "$newcontent");
      }
      $file = getFilePath('include_species_list.txt');
      $str = file_get_contents("$file");
      $str = preg_replace('/^\h*\v+/m', '', $str);
      file_put_contents("$file", "$str");
    }
    include('./scripts/include_list.php');
  }
  if($_GET['view'] == "Excluded"){
	  $user_is_authenticated = authenticateUser('You cannot modify the system');
	  if(isset($_GET['species']) && isset($_GET['add']) && $user_is_authenticated){
      $file = getFilePath('exclude_species_list.txt');
      $str = file_get_contents("$file");
      $str = preg_replace("/(^[\r\n]*|[\r\n]+)[\s\t]*[\r\n]+/", "\n", $str);
      file_put_contents("$file", "$str");
      foreach ($_GET['species'] as $selectedOption)
        file_put_contents(getFilePath('exclude_species_list.txt'), $selectedOption."\n", FILE_APPEND);
    } elseif (isset($_GET['species']) && isset($_GET['del'])){
      $file = getFilePath('exclude_species_list.txt');
      $str = file_get_contents("$file");
      $str = preg_replace('/^\h*\v+/m', '', $str);
      file_put_contents("$file", "$str");
      foreach($_GET['species'] as $selectedOption) {
        $content = file_get_contents(getFilePath('exclude_species_list.txt'));
        $newcontent = str_replace($selectedOption, "", "$content");
        file_put_contents(getFilePath('exclude_species_list.txt'), "$newcontent");
      }
      $file = getFilePath('exclude_species_list.txt');
      $str = file_get_contents("$file");
      $str = preg_replace('/^\h*\v+/m', '', $str);
      file_put_contents("$file", "$str");
    }
    include('./scripts/exclude_list.php');
  }
  if($_GET['view'] == "File"){
    echo "<iframe src='scripts/filemanager/filemanager.php'></iframe>";
  }
  if($_GET['view'] == "Webterm"){
	//Authenticate before proceeding
	$user_is_authenticated = authenticateUser("You cannot access the web terminal");
	if ($user_is_authenticated) {
        #ACCESS THE WEB TERMINAL
        echo "<iframe src='/terminal'></iframe>";
	}
  }
} elseif(isset($_GET['submit'])) {
  //Authenticate before proceeding
  $user_is_authenticated = authenticateUser('You cannot modify the system');

    $allowedCommands = array('service stop livestream.service && service stop icecast2.service',
                       'service restart livestream.service && service restart icecast2.service',
                       'service disable livestream.service && service disable icecast2 && service stop icecast2.service',
                       'service enable icecast2 && service start icecast2.service && service enable livestream.service',
                       'service stop web_terminal.service',
                       'service restart web_terminal.service',
                       'service disable web_terminal.service',
                       'service enable web_terminal.service',
                       'service stop birdnet_log.service',
                       'service restart birdnet_log.service',
                       'service disable birdnet_log.service',
                       'service enable birdnet_log.service',
                       'service stop extraction.service',
                       'service restart extraction.service',
                       'service disable extraction.service',
                       'service enable extraction.service',
                       'service stop birdnet_server.service',
                       'service restart birdnet_server.service',
                       'service disable birdnet_server.service',
                       'service enable birdnet_server.service',
                       'service stop birdnet_analysis.service',
                       'service restart birdnet_analysis.service',
                       'service disable birdnet_analysis.service',
                       'service enable birdnet_analysis.service',
                       'service stop birdnet_stats.service',
                       'service restart birdnet_stats.service',
                       'service disable birdnet_stats.service',
                       'service enable birdnet_stats.service',
                       'service stop birdnet_recording.service',
                       'service restart birdnet_recording.service',
                       'service disable birdnet_recording.service',
                       'service enable birdnet_recording.service',
                       'service stop chart_viewer.service',
                       'service restart chart_viewer.service',
                       'service disable chart_viewer.service',
                       'service enable chart_viewer.service',
                       'service stop spectrogram_viewer.service',
                       'service restart spectrogram_viewer.service',
                       'service disable spectrogram_viewer.service',
                       'service enable spectrogram_viewer.service',
                       'system stop core.services',
                       'system restart core.services',
                       'sudo reboot',
                       'update_birdnet.sh',
                       'sudo shutdown now',
                       'sudo clear_all_data.sh');
      $command = $_GET['submit'];
    if($user_is_authenticated && in_array($command,$allowedCommands)){
      if(isset($command)){
        $initcommand = $command;
        $results = "";
          //Process the system commands differently
		  if (strpos($command, "system") !== false) {
			  $results = serviceMaintenance($command);
              //clear the command so we skip the next bits and go straight to output processing
			  $command = '';
		  }
		  if (strpos($command, "service") !== false) {
			  //If there more than one command to execute, processes then separately
			  //currently only livestream service uses multiple commands to interact with the required services
			  if (strpos($command, " && ") !== false) {
				  $separate_commands = explode("&&", trim($command));
				  $new_multiservice_status_command = "";
				  foreach ($separate_commands as $indiv_service_command) {
					  //Action the command
					  serviceMaintenance($indiv_service_command);
					  //explode the string by " " space so we can get each individual component of the command
					  //and eventually the service name at the end
					  $separate_command_tmp = explode(" ", trim($indiv_service_command));
					  //get the service names so we can poll the status
					  $new_multiservice_status_command .= " " . trim(end($separate_command_tmp));
				  }

				  $service_names = $new_multiservice_status_command;
			  } else {
				  serviceMaintenance($command);
				  //only one service needs restarting so we only need to query the status of one service
				  $tmp = explode(" ", trim($command));
				  $service_names = end($tmp);
			  }
          //Build up the command that will query the service status
          $command = "sleep 3;sudo systemctl status " . $service_names;
        }
        if($initcommand == "update_birdnet.sh") {
          unset($_SESSION['behind']);
        }
        $results .= shell_exec("$command 2>&1");
        $results = str_replace("FAILURE", "<span style='color:red'>FAILURE</span>", $results);
        $results = str_replace("failed", "<span style='color:red'>failed</span>",$results);
        $results = str_replace("active (running)", "<span style='color:green'><b>active (running)</b></span>",$results);
        $results = str_replace("Your branch is up to date", "<span style='color:limegreen'><b>Your branch is up to date</b></span>",$results);

        $results = str_replace("(+)", "(<span style='color:lime;font-weight:bold'>+</span>)",$results);
        $results = str_replace("(-)", "(<span style='color:red;font-weight:bold'>-</span>)",$results);

        // split the input string into lines
        $lines = explode("\n", $results);

        // iterate over each line
        foreach ($lines as &$line) {
            // check if the line matches the pattern
            if (preg_match('/^(.+?)\s*\|\s*(\d+)\s*([\+\- ]+)(\d+)?$/', $line, $matches)) {
                // extract the filename, count, and indicator letters
                $filename = $matches[1];
                $count = $matches[2];
                $diff = $matches[3];
                $delta = $matches[4] ?? '';
                // determine the indicator letters
                $diff_array = str_split($diff);
                $indicators = array_map(function ($d) use ($delta) {
                    if ($d === '+') {
                        return "<span style='color:lime;'><b>+</b></span>";
                    } elseif ($d === '-') {
                        return "<span style='color:red;'><b>-</b></span>";
                    } elseif ($d === ' ') {
                        if ($delta !== '') {
                            return 'A';
                        } else {
                            return ' ';
                        }
                    }
                }, $diff_array);
                // modify the line with the new indicator letters
                $line = sprintf('%-35s|%3d %s%s', $filename, $count, implode('', $indicators), $delta);
            }
        }

        // rejoin the modified lines into a string
        $output = implode("\n", $lines);
        $results = $output;

        // remove script tags (xss)
        $results = preg_replace('#<script(.*?)>(.*?)</script>#is', '', $results);
        if(strlen($results) == 0) {
          $results = "This command has no output.";
        }
        echo "<table style='min-width:70%;'><tr class='relative'><th>Output of command:`".$initcommand."`<button class='copyimage' style='right:40px' onclick='copyOutput(this);'>Copy</button></th></tr><tr><td style='padding-left: 0px;padding-right: 0px;padding-bottom: 0px;padding-top: 0px;'><pre class='bash' style='text-align:left;margin:0px'>$results</pre></td></tr></table>"; 
      } else {
        header('WWW-Authenticate: Basic realm="BirdNET-Pi"');
        header('HTTP/1.0 401 Unauthorized');
        echo 'You cannot modify the system';
        exit;
      }
    }
  ob_end_flush();
} else {
	include('overview.php');
}
?>
<script>
function myFunction() {
  var x = document.getElementById("myTopnav");
  if (x.className === "topnav") {
    x.className += " responsive";
  } else {
    x.className = "topnav";
  }
}
function setLiveStreamVolume(vol) {
  var audioelement =  window.parent.document.getElementsByTagName("audio")[0];
  if (typeof(audioelement) != 'undefined' && audioelement != null)
  {
    audioelement.volume = vol
  }
}
window.onbeforeunload = function(event) {
  // if the user is playing a video and then navigates away mid-play, the live stream audio should be unmuted again
  var audioelement =  window.parent.document.getElementsByTagName("audio")[0];
  if (typeof(audioelement) != 'undefined' && audioelement != null)
  {
    audioelement.volume = 1
  }
}
</script>
</div>
</body>
