import apprise
import os
import socket
import sqlite3
from datetime import datetime
import requests
import time as timeim

userDir = os.path.expanduser('~')
APPRISE_CONFIG = userDir + '/BirdNET-Pi/apprise.txt'
DB_PATH = userDir + '/BirdNET-Pi/scripts/birds.db'

flickr_images = {}
species_last_notified = {}

asset = apprise.AppriseAsset(
    plugin_paths=[
        userDir + "/.apprise/plugins",
        userDir + "/.config/apprise/plugins",
    ]
)
apobj = apprise.Apprise(asset=asset)
config = apprise.AppriseConfig()
config.add(APPRISE_CONFIG)
apobj.add(config)

def notify(body, title, attached=""):
    if attached != "":
        apobj.notify(
            body=body,
            title=title,
            attach=attached,
        )
    else:
        apobj.notify(
            body=body,
            title=title,
        )


def sendAppriseNotifications(species, confidence, path, date, time, week, latitude, longitude, cutoff, sens, overlap, settings_dict, db_path=DB_PATH):
    # print(sendAppriseNotifications)
    # print(settings_dict)
    if os.path.exists(APPRISE_CONFIG) and os.path.getsize(APPRISE_CONFIG) > 0:

        title = settings_dict.get('APPRISE_NOTIFICATION_TITLE')
        body = settings_dict.get('APPRISE_NOTIFICATION_BODY')
        sciName, comName = species.split("_")

        APPRISE_ONLY_NOTIFY_SPECIES_NAMES = settings_dict.get('APPRISE_ONLY_NOTIFY_SPECIES_NAMES')
        if APPRISE_ONLY_NOTIFY_SPECIES_NAMES is not None and APPRISE_ONLY_NOTIFY_SPECIES_NAMES.strip() != "":
            if any(bird.lower().replace(" ", "") in comName.lower().replace(" ", "") for bird in APPRISE_ONLY_NOTIFY_SPECIES_NAMES.split(",")):
                return
                
        APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2 = settings_dict.get('APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2')
        if APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2 is not None and APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2.strip() != "":
            if not any(bird.lower().replace(" ", "") in comName.lower().replace(" ", "") for bird in APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2.split(",")):
                return

        APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES = settings_dict.get('APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES')
        if APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES != "0":
            if species_last_notified.get(comName) is not None:
                try:
                    if int(timeim.time()) - species_last_notified[comName] < int(APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES):
                        return
                except Exception as e:
                    print("APPRISE NOTIFICATION EXCEPTION: "+str(e))
                    return

        #TODO: this all needs to be changed, we changed the caddy default to allow direct IP access, so birdnetpi.local shouldn't be relied on anymore
        try:
            websiteurl = settings_dict.get('BIRDNETPI_URL')
            if len(websiteurl) == 0:
                raise ValueError('Blank URL')
        except Exception:
            websiteurl = "http://"+socket.gethostname()+".local"

        listenurl = websiteurl+"?filename="+path
        image_url = ""

        if len(settings_dict.get('FLICKR_API_KEY')) > 0 and "$flickrimage" in body:
            if not comName in flickr_images:
                try:
                    # TODO: Make this work with non-english comnames. Implement the "// convert sci name to English name" logic from overview.php here
                    headers = {'User-Agent': 'Python_Flickr/1.0'}
                    url = 'https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key='+str(settings_dict.get('FLICKR_API_KEY'))+'&text='+str(comName)+' bird&sort=relevance&per_page=5&media=photos&format=json&license=2%2C3%2C4%2C5%2C6%2C9&nojsoncallback=1'
                    resp = requests.get(url=url, headers=headers)
                    resp.encoding = "utf-8"
                    data = resp.json()["photos"]["photo"][0]

                    image_url = 'https://farm'+str(data["farm"])+'.static.flickr.com/'+str(data["server"])+'/'+str(data["id"])+'_'+str(data["secret"])+'_n.jpg'
                    flickr_images[comName] = image_url
                except Exception as e:
                    print("FLICKR API ERROR: "+str(e)) 
                    image_url = ""
            else:
                image_url = flickr_images[comName]


        if settings_dict.get('APPRISE_NOTIFY_EACH_DETECTION') == "1":
            notify_body = body.replace("$sciname", sciName)\
                .replace("$comname", comName)\
                .replace("$confidence", confidence)\
                .replace("$listenurl", listenurl)\
                .replace("$date", date)\
                .replace("$time", time)\
                .replace("$week", week)\
                .replace("$latitude", latitude)\
                .replace("$longitude", longitude)\
                .replace("$cutoff", cutoff)\
                .replace("$sens", sens)\
                .replace("$flickrimage", image_url if "{" in body else "")\
                .replace("$overlap", overlap)
            notify_title = title.replace("$sciname", sciName)\
                .replace("$comname", comName)\
                .replace("$confidence", confidence)\
                .replace("$listenurl", listenurl)\
                .replace("$date", date)\
                .replace("$time", time)\
                .replace("$week", week)\
                .replace("$latitude", latitude)\
                .replace("$longitude", longitude)\
                .replace("$cutoff", cutoff)\
                .replace("$sens", sens)\
                .replace("$flickrimage", image_url if "{" in body else "")\
                .replace("$overlap", overlap)
            notify(notify_body, notify_title, image_url)
            species_last_notified[comName] = int(timeim.time())

        APPRISE_NOTIFICATION_NEW_SPECIES_DAILY_COUNT_LIMIT = 1  # Notifies the first N per day.
        if settings_dict.get('APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY') == "1":
            try:
                con = sqlite3.connect(db_path)
                cur = con.cursor()
                today = datetime.now().strftime("%Y-%m-%d")
                cur.execute(f"SELECT DISTINCT(Com_Name), COUNT(Com_Name) FROM detections WHERE Date = DATE('{today}') GROUP BY Com_Name")
                known_species = cur.fetchall()
                detections = [d[1] for d in known_species if d[0] == comName.replace("'", "")]
                numberDetections = 0
                if len(detections):
                    numberDetections = detections[0]
                if numberDetections > 0 and numberDetections <= APPRISE_NOTIFICATION_NEW_SPECIES_DAILY_COUNT_LIMIT:
                    print("send the notification")
                    notify_body = body.replace("$sciname", sciName)\
                        .replace("$comname", comName)\
                        .replace("$confidence", confidence)\
                        .replace("$listenurl", listenurl)\
                        .replace("$date", date)\
                        .replace("$time", time)\
                        .replace("$week", week)\
                        .replace("$latitude", latitude)\
                        .replace("$longitude", longitude)\
                        .replace("$cutoff", cutoff)\
                        .replace("$sens", sens)\
                        .replace("$flickrimage", image_url if "{" in body else "")\
                        .replace("$overlap", overlap)\
                        + " (first time today)"
                    notify_title = title.replace("$sciname", sciName)\
                        .replace("$comname", comName)\
                        .replace("$confidence", confidence)\
                        .replace("$listenurl", listenurl)\
                        .replace("$date", date)\
                        .replace("$time", time)\
                        .replace("$week", week)\
                        .replace("$latitude", latitude)\
                        .replace("$longitude", longitude)\
                        .replace("$cutoff", cutoff)\
                        .replace("$sens", sens)\
                        .replace("$flickrimage", image_url if "{" in body else "")\
                        .replace("$overlap", overlap)\
                        + " (first time today)"
                    notify(notify_body, notify_title, image_url)
                    species_last_notified[comName] = int(timeim.time())
                con.close()
            except sqlite3.Error as e:
                print(e)
                print("Database busy")
                timeim.sleep(2)

        if settings_dict.get('APPRISE_NOTIFY_NEW_SPECIES') == "1":
            try:
                con = sqlite3.connect(db_path)
                cur = con.cursor()
                today = datetime.now().strftime("%Y-%m-%d")
                cur.execute(f"SELECT DISTINCT(Com_Name), COUNT(Com_Name) FROM detections WHERE Date >= DATE('{today}', '-7 day') GROUP BY Com_Name")
                known_species = cur.fetchall()
                detections = [d[1] for d in known_species if d[0] == comName.replace("'", "")]
                numberDetections = 0
                if len(detections):
                    numberDetections = detections[0]
                if numberDetections > 0 and numberDetections <= 5:
                    notify_body = body.replace("$sciname", sciName)\
                        .replace("$comname", comName)\
                        .replace("$confidence", confidence)\
                        .replace("$listenurl", listenurl)\
                        .replace("$date", date)\
                        .replace("$time", time)\
                        .replace("$week", week)\
                        .replace("$latitude", latitude)\
                        .replace("$longitude", longitude)\
                        .replace("$cutoff", cutoff)\
                        .replace("$sens", sens)\
                        .replace("$flickrimage", image_url if "{" in body else "")\
                        .replace("$overlap", overlap)\
                        + " (only seen " + str(int(numberDetections)) + " times in last 7d)"
                    notify_title = title.replace("$sciname", sciName)\
                        .replace("$comname", comName)\
                        .replace("$confidence", confidence)\
                        .replace("$listenurl", listenurl)\
                        .replace("$date", date)\
                        .replace("$time", time)\
                        .replace("$week", week)\
                        .replace("$latitude", latitude)\
                        .replace("$longitude", longitude)\
                        .replace("$cutoff", cutoff)\
                        .replace("$sens", sens)\
                        .replace("$flickrimage", image_url if "{" in body else "")\
                        .replace("$overlap", overlap)\
                        + " (only seen " + str(int(numberDetections)) + " times in last 7d)"
                    notify(notify_body, notify_title, image_url)
                    species_last_notified[comName] = int(timeim.time())
                con.close()
            except sqlite3.Error:
                print("Database busy")
                timeim.sleep(2)


if __name__ == "__main__":
    print("notfications")
