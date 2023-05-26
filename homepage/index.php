<?php

/* Prevent XSS input */
$_GET   = filter_input_array(INPUT_GET, FILTER_SANITIZE_STRING);
$_POST  = filter_input_array(INPUT_POST, FILTER_SANITIZE_STRING);

$sys_timezone = "";
// If we can get the timezome from the systems timezone file ust that
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
//Finally if we have a valod timezone, set it as the one PHP uses
if ($sys_timezone !== "") {
	date_default_timezone_set($sys_timezone);
}

  if (file_exists('./scripts/thisrun.txt')) {
    $config = parse_ini_file('./scripts/thisrun.txt');
  } elseif (file_exists('./scripts/firstrun.ini')) {
    $config = parse_ini_file('./scripts/firstrun.ini');
  }
  if($config["SITE_NAME"] == "") {
    $site_name = "BirdNET-Pi";
  } else {
    $site_name = $config['SITE_NAME'];
  }
?>
<title><?php echo $site_name; ?></title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body::-webkit-scrollbar {
  display:none
}
</style>
<link rel="stylesheet" href="style.css?v=<?php echo date ('n.d.y', filemtime('style.css')); ?>">
<link rel="stylesheet" type="text/css" href="static/dialog-polyfill.css" />
<body>
<div class="banner">
  <div class="logo">
<?php if(isset($_GET['logo'])) {
echo "<a href=\"https://github.com/mcguirepr89/BirdNET-Pi.git\" target=\"_blank\"><img style=\"width:60;height:60;\" src=\"images/bird.png\"></a>";
} else {
echo "<a href=\"https://github.com/mcguirepr89/BirdNET-Pi.git\" target=\"_blank\"><img src=\"images/bird.png\"></a>";
}?>
  </div>


  <div class="stream">
<?php
if(isset($_GET['stream'])){
  if (file_exists('./scripts/thisrun.txt')) {
    $config = parse_ini_file('./scripts/thisrun.txt');
  } elseif (file_exists('./scripts/firstrun.ini')) {
    $config = parse_ini_file('./scripts/firstrun.ini');
  }
  $caddypwd = $config['CADDY_PWD'];
  if (!isset($_SERVER['PHP_AUTH_USER'])) {
    header('WWW-Authenticate: Basic realm="My Realm"');
    header('HTTP/1.0 401 Unauthorized');
    echo 'You cannot listen to the live audio stream';
    exit;
  } else {
    $submittedpwd = $_SERVER['PHP_AUTH_PW'];
    $submitteduser = $_SERVER['PHP_AUTH_USER'];
    if($submittedpwd == $caddypwd && $submitteduser == 'birdnet'){
      echo "
  <audio controls autoplay><source src=\"/stream\"></audio>
  </div>
  <h1><a href=\"/\"><img class=\"topimage\" src=\"images/bnp.png\"></a></h1>
  </div><div class=\"centered\"><h3>$site_name</h3></div>";
    } else {
      header('WWW-Authenticate: Basic realm="My Realm"');
      header('HTTP/1.0 401 Unauthorized');
      echo 'You cannot listen to the live audio stream';
      exit;
    }
  }
} else {
    echo "
  <form action=\"\" method=\"GET\">
    <button type=\"submit\" name=\"stream\" value=\"play\">Live Audio</button>
  </form>
  </div>
  <h1><a href=\"/\"><img class=\"topimage\" src=\"images/bnp.png\"></a></h1>
</div><div class=\"centered\"><h3>$site_name</h3></div>";
}
if(isset($_GET['filename'])) {
  $filename = $_GET['filename'];
echo "
<iframe src=\"/views.php?view=Recordings&filename=$filename\"></iframe>
</div>";
} else {
  echo "
<iframe src=\"/views.php\"></iframe>
</div>";
}
?>
