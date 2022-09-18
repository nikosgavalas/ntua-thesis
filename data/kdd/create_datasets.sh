#!/bin/bash

FILE=kddcup.data.gz
# URL=http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz # whole dataset 
URL=http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz # 10 percent
HTTP=http.csv
SMTP=smtp.csv

curl $URL -o $FILE
zcat $FILE | awk 'BEGIN{ FS=","; OFS="," } { LABEL=($NF=="normal." ? 0 : 1); if($3 == "http") print $1, $5, $6, LABEL }' > $HTTP # cols: duration,src_bytes,dst_bytes,label
zcat $FILE | awk 'BEGIN{ FS=","; OFS="," } { LABEL=($NF=="normal." ? 0 : 1); if($3 == "smtp") {for(i = 1; i < NF; i++) printf("%s%s", $i, OFS); printf("%s\n", LABEL)} }' | cut -d ',' -f '2-4' --complement > $SMTP # cols: all (w/label)  except protocol,service,flag

# tests for the 10 percent dataset only
if [ $URL == 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz' ]; then
    test $(wc -l $HTTP | cut -d ' ' -f 1) -eq 64293 || echo fail-1
    test $(grep -e '1$' $HTTP | wc -l) -eq 2407 || echo fail-2
    test $(wc -l $SMTP | cut -d ' ' -f 1) -eq 9723 || echo fail-3
    test $(grep -e '1$' $SMTP | wc -l) -eq 125 || echo fail-4
fi
