<?php
if(file_exists('./scripts/common.php')){
	include_once "./scripts/common.php";
}else{
	include_once "./common.php";
}

$myDate = date('Y-m-d');
$chart = "Combo-$myDate.png";

$todaycount_data = getDetectionCountToday();

if($todaycount_data['success'] == False){
	echo $todaycount_data['message'];
	header("refresh: 0;");
}
$todaycount = $todaycount_data['data'];

if(isset($_GET['custom_image'])){
  if(isset($config["CUSTOM_IMAGE"])) {
  ?>
    <br>
    <h3><?php echo $config["CUSTOM_IMAGE_TITLE"]; ?></h3>
    <?php
    $image_data = file_get_contents($config["CUSTOM_IMAGE"]);
    $image_base64 = base64_encode($image_data);
    $img_tag = "<img src='data:image/png;base64," . $image_base64 . "'>";
    echo $img_tag;
  }
  die();
}

if(isset($_GET['blacklistimage'])) {
	$user_is_authenticated = authenticateUser('You cannot modify the system');
    if($user_is_authenticated){
		$imageid = $_GET['blacklistimage'];
		$result = blacklistFlickrImage($imageid);
		unset($_SESSION['images']);
		die($result['message']);
    }
}

if(isset($_GET['fetch_chart_string']) && $_GET['fetch_chart_string'] == "true") {
  $chart = getChartString();
  echo $chart;
  die();
}

if(isset($_GET['ajax_detections']) && $_GET['ajax_detections'] == "true" && isset($_GET['previous_detection_identifier'])) {

	$result4 = getMostRecentDetection(15);

	if ($result4['success'] == False) {
		echo $result4['message'];
		header("refresh: 0;");
	}
	$result4 = $result4['data'];

	$iterations = 0;
	$processed_detections = 0;

  // hopefully one of the 5 most recent detections has an image that is valid, we'll use that one as the most recent detection until the newer ones get their images created
  foreach ($result4 as $mostrecent){
    $iterations++;
	$comname = preg_replace('/ /', '_', $mostrecent['Com_Name']);
    $sciname = preg_replace('/ /', '_', $mostrecent['Sci_Name']);
    $comname = preg_replace('/\'/', '', $comname);
    $filename = "/By_Date/".$mostrecent['Date']."/".$comname."/".$mostrecent['File_Name'];
	  // check to make sure the image actually exists, sometimes it takes a minute to be created\
	  if (file_exists(getDirectory('extracted') . "/" . $filename . ".png")) {
		  if ($_GET['previous_detection_identifier'] == $filename) {
			  die();
		  }
		  if ($_GET['only_name'] == "true") {
			  echo $comname . "," . $filename;
			  die();
		  }
          //Because the spectrogram image exists for this detection it's been processed
		  $processed_detections++;

		  //Get the flickr image for this detection
		  $flickr_Image = getFlickrImage($mostrecent);

		  if ($flickr_Image['image_found']) {
			  //Remap the data from returned data into an array that is referenced in our HTML
			  //to save have to rewrite or adjust it
			  $flickr_Image_data = $flickr_Image['data'];
			  $image = [
				  0 => $flickr_Image_data['Com_Name_clean'],
				  1 => $flickr_Image_data['photos'][0]['image_url'],
				  2 => $flickr_Image_data['photos'][0]['photo_title'],
				  3 => $flickr_Image_data['photos'][0]['modal_text'],
				  4 => $flickr_Image_data['photos'][0]['author_link'],
				  5 => $flickr_Image_data['photos'][0]['license_url']
			  ];
		  }

      ?>
        <style>
        .fade-in {
          opacity: 1;
          animation-name: fadeInOpacity;
          animation-iteration-count: 1;
          animation-timing-function: ease-in;
          animation-duration: 1s;
        }

        @keyframes fadeInOpacity {
          0% {
            opacity: 0;
          }
          100% {
            opacity: 1;
          }
        }
        </style>
        <table class="<?php echo ($_GET['previous_detection_identifier'] == 'undefined') ? '' : 'fade-in';  ?>">
          <h3>Most Recent Detection: <span style="font-weight: normal;"><?php echo $mostrecent['Date']." ".$mostrecent['Time'];?></span></h3>
          <tr>
            <td class="relative"><a target="_blank" href="index.php?filename=<?php echo $mostrecent['File_Name']; ?>"><img class="copyimage" title="Open in new tab" width="25" height="25" src="images/copy.png"></a>
            <div class="centered_image_container" style="margin-bottom: 0px !important;">
              <?php if(!empty($config["FLICKR_API_KEY"]) && strlen($image[2]) > 0) { ?>
                <img onclick='setModalText(<?php echo $iterations; ?>,"<?php echo urlencode($image[2]); ?>", "<?php echo $image[3]; ?>", "<?php echo $image[4]; ?>", "<?php echo $image[1]; ?>", "<?php echo $image[5]; ?>")' src="<?php echo $image[1]; ?>" class="img1">
              <?php } ?>
              <form action="" method="GET">
                  <input type="hidden" name="view" value="Species Stats">
                  <button type="submit" name="species" value="<?php echo $mostrecent['Com_Name'];?>"><?php echo $mostrecent['Com_Name'];?></button><img style="width: unset !important;display: inline;height: 1em;cursor:pointer" title="View species stats" onclick="generateMiniGraph(this, '<?php echo $comname; ?>')" width=25 src="images/chart.svg"><br>
                  <a href="https://wikipedia.org/wiki/<?php echo $sciname;?>" target="_blank"><i><?php echo $mostrecent['Sci_Name'];?></i></a>
                  <br>Confidence: <?php echo $percent = round((float)round($mostrecent['Confidence'],2) * 100 ) . '%';?><br></div><br>
                  <video style="margin-top:10px" onplay='setLiveStreamVolume(0)' onended='setLiveStreamVolume(1)' onpause='setLiveStreamVolume(1)' controls poster="<?php echo $filename.".png";?>" preload="none" title="<?php echo $filename;?>"><source src="<?php echo $filename;?>"></video></td>
              </form>
          </tr>
        </table> <?php break;
      }
  }
    if($todaycount['COUNT(*)'] > 0) {
		if($processed_detections == 0) {
			echo "<h3>Your system is currently processing a backlog of audio. This can take several hours before normal functionality of your BirdNET-Pi resumes.</h3>";
		}
    } else {
      echo "<h3>No Detections For Today.</h3>";
    }
  die();
}

if(isset($_GET['ajax_left_chart']) && $_GET['ajax_left_chart'] == "true") {

	$totalcount_data = getDetectionCountAll();
	if($totalcount_data['success'] == False){
		echo $totalcount_data['message'];
		header("refresh: 0;");
	}
	$totalcount = $totalcount_data['data'];


	$hourcount_data = getDetectionCountLastHour();
	if($hourcount_data['success'] == False){
		echo $hourcount_data['message'];
		header("refresh: 0;");
	}
	$hourcount = $hourcount_data['data'];


	$speciestally_data = getSpeciesTalley();
	if($speciestally_data['success'] == False){
		echo $speciestally_data['message'];
		header("refresh: 0;");
	}
	$speciestally = $speciestally_data['data'];


	$totalspeciestally_data = getAllSpeciesTalley();
	if($totalspeciestally_data['success'] == False){
		echo $totalspeciestally_data['message'];
		header("refresh: 0;");
	}
	$totalspeciestally = $totalspeciestally_data['data'];
  
?>
<table>
  <tr>
    <th>Total</th>
    <td><?php echo $totalcount['COUNT(*)'];?></td>
  </tr>
  <tr>
    <th>Today</th>
    
    <td><form action="" method="GET"><button type="submit" name="view" value="Today's Detections"><?php echo $todaycount['COUNT(*)'];?></button></td>
    </form>
  </tr>
  <tr>
    <th>Last Hour</th>
    <td><?php echo $hourcount['COUNT(*)'];?></td>
  </tr>
  <tr>
    <th>Species Detected Today</th>
    <td><form action="" method="GET"><input type="hidden" name="view" value="Recordings"><button type="submit" name="date" value="<?php echo date('Y-m-d');?>"><?php echo $speciestally['COUNT(DISTINCT(Com_Name))'];?></button></td>
    </form>
  </tr>
  <tr>
    <th>Total Number of Species</th>
    <td><form action="" method="GET"><button type="submit" name="view" value="Species Stats"><?php echo $totalspeciestally['COUNT(DISTINCT(Com_Name))'];?></button></td>
    </form>
  </tr>
</table>
<?php
die();
}
?>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Overview</title>
<style>

</style>
</head>
<div class="overview">
  <dialog style="margin-top: 5px;max-height: 95vh;
  overflow-y: auto;overscroll-behavior:contain" id="attribution-dialog">
    <h1 id="modalHeading"></h1>
    <p id="modalText"></p>
    <button onclick="hideDialog()">Close</button>
    <button style="font-weight:bold;color:blue" onclick="if(confirm('Are you sure you want to blacklist this image?')) { blacklistImage(); }">Blacklist this image</button>
  </dialog>
  <script src="static/dialog-polyfill.js"></script>
  <script src="static/Chart.bundle.js"></script>
  <script src="static/chartjs-plugin-trendline.min.js"></script>
  <script>
  var last_photo_link;
  var dialog = document.querySelector('dialog');
  dialogPolyfill.registerDialog(dialog);

  function showDialog() {
    document.getElementById('attribution-dialog').showModal();
  }

  function hideDialog() {
    document.getElementById('attribution-dialog').close();
  }

  function blacklistImage() {
    const match = last_photo_link.match(/\d+$/); // match one or more digits
    const result = match ? match[0] : null; // extract the first match or return null if no match is found
    console.log(last_photo_link)
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function() {
      if(this.responseText.length > 0) {
       location.reload();
      }
    }
    xhttp.open("GET", "overview.php?blacklistimage="+result, true);
    xhttp.send();

  }

  function setModalText(iter, title, text, authorlink, photolink, licenseurl) {
    document.getElementById('modalHeading').innerHTML = "Photo: \""+decodeURIComponent(title.replaceAll("+"," "))+"\" Attribution";
    document.getElementById('modalText').innerHTML = "<div><img style='border-radius:5px;max-height: calc(100vh - 15rem);display: block;margin: 0 auto;' src='"+photolink+"'></div><br><div style='white-space:nowrap'>Image link: <a target='_blank' href="+text+">"+text+"</a><br>Author link: <a target='_blank' href="+authorlink+">"+authorlink+"</a><br>License URL: <a href="+licenseurl+" target='_blank'>"+licenseurl+"</a></div>";
    last_photo_link = text;
    showDialog();
  }
  </script>  
<div class="overview-stats">
<div class="left-column">
</div>
<div class="right-column">
<div class="chart">
<?php
$refresh = $config['RECORDING_LENGTH'];
$dividedrefresh = $refresh/4;
if($dividedrefresh < 1) { 
  $dividedrefresh = 1;
}
$time = time();
if (file_exists('./Charts/'.$chart)) {
  echo "<img id='chart' src=\"/Charts/$chart?nocache=$time\">";
} 
?>
</div>

<div id="most_recent_detection"></div>
<br>
<h3>5 Most Recent Detections</h3>
<div style="padding-bottom:10px;" id="detections_table"><h3>Loading...</h3></div>

<h3>Currently Analyzing</h3>
<?php
$refresh = $config['RECORDING_LENGTH'];
$time = time();
echo "<img id=\"spectrogramimage\" src=\"/spectrogram.png?nocache=$time\">";

?>

<div id="customimage"></div>
<br>

</div>
</div>

<script>
// we're passing a unique ID of the currently displayed detection to our script, which checks the database to see if the newest detection entry is that ID, or not. If the IDs don't match, it must mean we have a new detection and it's loaded onto the page
function loadDetectionIfNewExists(previous_detection_identifier=undefined) {
  const xhttp = new XMLHttpRequest();
  xhttp.onload = function() {
    // if there's a new detection that needs to be updated to the page
    if(this.responseText.length > 0 && !this.responseText.includes("Database is busy") && !this.responseText.includes("No Detections") || previous_detection_identifier == undefined) {
      document.getElementById("most_recent_detection").innerHTML = this.responseText;

      // only going to load left chart & 5 most recents if there's a new detection
      loadLeftChart();
      loadFiveMostRecentDetections();
      refreshTopTen();
    }
  }
  xhttp.open("GET", "overview.php?ajax_detections=true&previous_detection_identifier="+previous_detection_identifier, true);
  xhttp.send();
}
function loadLeftChart() {
  const xhttp = new XMLHttpRequest();
  xhttp.onload = function() {
    if(this.responseText.length > 0 && !this.responseText.includes("Database is busy")) {
      document.getElementsByClassName("left-column")[0].innerHTML = this.responseText;
    }
  }
  xhttp.open("GET", "overview.php?ajax_left_chart=true", true);
  xhttp.send();
}
function refreshTopTen() {
  const xhttp = new XMLHttpRequest();
  xhttp.onload = function() {
  if(this.responseText.length > 0 && !this.responseText.includes("Database is busy") && !this.responseText.includes("No Detections") || previous_detection_identifier == undefined) {
    document.getElementById("chart").src = "/Charts/"+this.responseText+"?nocache="+Date.now();
  }
  }
  xhttp.open("GET", "overview.php?fetch_chart_string=true", true);
  xhttp.send();
}
window.setInterval(function(){
  var videoelement = document.getElementsByTagName("video")[0];
  if(typeof videoelement !== "undefined") {
    // don't refresh the detection if the user is playing the previous one's audio, wait until they're finished
    if(!!(videoelement.currentTime > 0 && !videoelement.paused && !videoelement.ended && videoelement.readyState > 2) == false) {
      loadDetectionIfNewExists(videoelement.title);
    }
  } else{
    // image or audio didn't load for some reason, force a refresh in 5 seconds
    loadDetectionIfNewExists();
  }
}, <?php echo intval($dividedrefresh); ?>*1000);

function loadFiveMostRecentDetections() {
  const xhttp = new XMLHttpRequest();
  xhttp.onload = function() {
    if(this.responseText.length > 0 && !this.responseText.includes("Database is busy")) {
      document.getElementById("detections_table").innerHTML= this.responseText;
    }
  }
  if (window.innerWidth > 500) {
    xhttp.open("GET", "todays_detections.php?ajax_detections=true&display_limit=undefined&hard_limit=5", true);
  } else {
    xhttp.open("GET", "todays_detections.php?ajax_detections=true&display_limit=undefined&hard_limit=5&mobile=true", true);
  }
  xhttp.send();
}
window.addEventListener("load", function(){
  loadDetectionIfNewExists();
});

// every $refresh seconds, this loop will run and refresh the spectrogram image
window.setInterval(function(){
  document.getElementById("spectrogramimage").src = "/spectrogram.png?nocache="+Date.now();
}, <?php echo $refresh; ?>*1000);

<?php if(isset($config["CUSTOM_IMAGE"]) && strlen($config["CUSTOM_IMAGE"]) > 2){?>
// every 1 second, this loop will run and refresh the custom image
window.setInterval(function(){
  // Find the customimage element
  var customimage = document.getElementById("customimage");

  function updateCustomImage() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "overview.php?custom_image=true", true);
    xhr.onload = function() {
      customimage.innerHTML = xhr.responseText;
    }
    xhr.send();
  }
  updateCustomImage();
}, 1000);
<?php } ?>
</script>

<style>
  .tooltip {
  background-color: white;
  border: 1px solid #ccc;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
  padding: 10px;
  transition: opacity 0.2s ease-in-out;
}
</style>
<script>
function generateMiniGraph(elem, comname) {

  // Make an AJAX call to fetch the number of detections for the bird species
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/todays_detections.php?comname=' + comname);
  xhr.onload = function() {
    if (xhr.status === 200) {
      var detections = JSON.parse(xhr.responseText);

      // Create a div element for the chart window
      if (typeof(window.chartWindow) != 'undefined') {
        document.body.removeChild(window.chartWindow);
        window.chartWindow = undefined;
      }
      var chartWindow = document.createElement('div');
      chartWindow.className = "chartdiv"
      chartWindow.style.position = 'fixed';
      chartWindow.style.top = '0%';
      chartWindow.style.left = '50%';
      chartWindow.style.width = window.innerWidth < 700 ? '40%' : '20%';
      chartWindow.style.height = window.innerWidth < 700 ? '25%' : '16%';
      chartWindow.style.backgroundColor = '#fff';
      chartWindow.style.zIndex = '9999';
      chartWindow.style.overflow = 'auto';
      chartWindow.style.borderRadius = '5px';
      chartWindow.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';

      document.body.appendChild(chartWindow);


            // Create a canvas element for the chart
      var canvas = document.createElement('canvas');
      canvas.width = chartWindow.offsetWidth;
      canvas.height = chartWindow.offsetHeight;
      chartWindow.appendChild(canvas);

      // Create a new Chart.js chart
      var ctx = canvas.getContext('2d');
      var chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: detections.map(item => item.date),
          datasets: [{
            label: 'Detections',
            data: detections.map(item => item.count),
            backgroundColor: '#9fe29b',
            borderColor: '#77c487',
            borderWidth: 1,
            lineTension: 0.3, // Add smoothing to the line
            pointRadius: 1, // Make the data points smaller
            pointHitRadius: 10, // Increase the area around data points for mouse events

          trendlineLinear: {
            style: "rgba(55, 99, 64, 0.5)",
            lineStyle: "solid",
            width: 1.5
          }

          }]
        },
        options: {
          layout: {
            padding: {
              right: 10
            }
          },
          title: {
            display: true,
            text: 'Detections Over 30d'
          },
          legend: {
            display: false
          },
          scales: {
            xAxes: [{
              display: false,
              gridLines: {
                display: false // Hide the gridlines on the x-axis
              },
              ticks: {
                autoSkip: true,
                maxTicksLimit: 2
              }
            }],
            yAxes: [{
              gridLines: {
                display: false // Hide the gridlines on the y-axis
              },
              ticks: {
                beginAtZero: true,
                precision: 0,
                stepSize: 1
              }
            }]
          }
        }
      });

      var buttonRect = elem.getBoundingClientRect();
      var chartRect = chartWindow.getBoundingClientRect();
      if (window.innerWidth < 700) {
        chartWindow.style.left = 'calc(75% - ' + (chartRect.width / 2) + 'px)';
      } else {
        chartWindow.style.left = (buttonRect.right + 10) + 'px';
      }


      // Calculate the top position of the chart to center it with the button
      var buttonCenter = buttonRect.top + (buttonRect.height / 2);
      var chartHeight = chartWindow.offsetHeight;
      var chartTop = buttonCenter - (chartHeight / 2);
      chartWindow.style.top = chartTop + 'px';

      // Add a close button to the chart window
      var closeButton = document.createElement('button');
      closeButton.id = "chartcb";
      closeButton.innerText = 'X';
      closeButton.style.position = 'absolute';
      closeButton.style.top = '5px';
      closeButton.style.right = '5px';
      closeButton.addEventListener('click', function() {
        document.body.removeChild(chartWindow);
        window.chartWindow = undefined;
      });
      chartWindow.appendChild(closeButton);
      window.chartWindow = chartWindow;
    }
  };
  xhr.send();
}

// Listen for the scroll event on the window object
window.addEventListener('scroll', function() {
  // Get all chart elements
  var charts = document.querySelectorAll('.chartdiv');
  
  // Loop through all chart elements and remove them
  charts.forEach(function(chart) {
    chart.parentNode.removeChild(chart);
    window.chartWindow = undefined;
  });
});

</script>
