<?php
/* Prevent XSS input */
$_GET   = filter_input_array(INPUT_GET, FILTER_SANITIZE_STRING);
$_POST  = filter_input_array(INPUT_POST, FILTER_SANITIZE_STRING);


if(file_exists('./scripts/common.php')){
	include_once "./scripts/common.php";
}else{
	include_once "./common.php";
}

?>
<title><?php echo $site_name; ?></title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>

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
	$user_is_authenticated = authenticateUser('You cannot listen to the live audio stream');
	if ($user_is_authenticated) {
		echo "
  <audio controls autoplay><source src=\"/stream\"></audio>
  </div>
  <h1><a href=\"/\"><img class=\"topimage\" src=\"images/bnp.png\"></a></h1>
  </div><div class=\"centered\"><h3>$site_name</h3></div>";
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
