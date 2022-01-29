<?php
shell_exec("/home/pi/BirdNET-Pi/scripts/restart_birdnet_analysis.sh");
header('Location: http://'.gethostname().'.local:8080');
?>
