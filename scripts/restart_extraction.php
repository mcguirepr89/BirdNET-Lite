<?php
shell_exec("/home/pi/BirdNET-Pi/scripts/restart_extraction.sh");
header('Location: http://'.gethostname().'.local:8888');
?>
