#!/bin/bash
export http_proxy=http://dev-proxy.oa.com:8080
export https_proxy=$http_proxy
export ftp_proxy=$http_proxy
export all_proxy=$http_proxy
wget --no-check-certificate https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
tar zxf dac.tar.gz
rm -f dac.tar.gz

mkdir raw
mv ./*.txt raw/
