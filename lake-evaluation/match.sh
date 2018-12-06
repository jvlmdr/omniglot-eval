#!/bin/bash

records=$1
queries=$2

(>&2 echo 'list files')
find -s $records -name '*.bmp' >$records.txt || exit -1
find -s $queries -name '*.bmp' >$queries.txt || exit -1

(>&2 echo 'compute hashes')
(cat $records.txt | xargs md5 -r >hash_$records.txt) || exit -1
(cat $queries.txt | xargs md5 -r >hash_$queries.txt) || exit -1

awk 'NR==FNR{a[$1]=$2; next} {print $1,$2,a[$1]}' hash_$records.txt hash_$queries.txt
