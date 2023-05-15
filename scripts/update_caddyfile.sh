#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

SCRIPTS_DIR="$(getDirectory 'scripts')"
EXTRACTED_DIR="$(getDirectory 'extracted')"

source /etc/birdnet/birdnet.conf
set -x
[ -d /etc/caddy ] || mkdir /etc/caddy
if [ -f /etc/caddy/Caddyfile ];then
  cp /etc/caddy/Caddyfile{,.original}
fi
if ! [ -z ${CADDY_PWD} ];then
HASHWORD=$(caddy hash-password --plaintext ${CADDY_PWD})
cat << EOF > /etc/caddy/Caddyfile
http:// ${BIRDNETPI_URL} {
  root * ${EXTRACTED_DIR}
  file_server browse
  handle /By_Date/* {
    file_server browse
  }
  handle /Charts/* {
    file_server browse
  }
  basicauth /views.php?view=File* {
    birdnet ${HASHWORD}
  }
  basicauth /Processed* {
    birdnet ${HASHWORD}
  }
  basicauth /scripts* {
    birdnet ${HASHWORD}
  }
  basicauth /stream {
    birdnet ${HASHWORD}
  }
  basicauth /phpsysinfo* {
    birdnet ${HASHWORD}
  }
  basicauth /terminal* {
    birdnet ${HASHWORD}
  }
  reverse_proxy /stream localhost:8000
  php_fastcgi unix//run/php/php7.4-fpm.sock
  reverse_proxy /log* localhost:8080
  reverse_proxy /stats* localhost:8501
  reverse_proxy /terminal* localhost:8888
}
EOF
else
  cat << EOF > /etc/caddy/Caddyfile
http:// ${BIRDNETPI_URL} {
  root * ${EXTRACTED_DIR}
  file_server browse
  handle /By_Date/* {
    file_server browse
  }
  handle /Charts/* {
    file_server browse
  }
  reverse_proxy /stream localhost:8000
  php_fastcgi unix//run/php/php7.4-fpm.sock
  reverse_proxy /log* localhost:8080
  reverse_proxy /stats* localhost:8501
  reverse_proxy /terminal* localhost:8888
}
EOF
fi

sudo caddy fmt --overwrite /etc/caddy/Caddyfile
sudo systemctl reload caddy
