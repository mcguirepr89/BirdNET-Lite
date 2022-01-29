<?php
shell_exec("/home/pi/BirdNET-Pi/scripts/restart_services.sh");
header('Location: http://'.gethostname().'.local:8080');
?>
