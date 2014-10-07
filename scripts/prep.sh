#!/bin/bash

rm ../ids.txt

for each in `ls -1 *.shortLongDescription`; do 
    qid=`echo $each | sed -e 's/[^0-9]//g'`;
    head -n 1 ${each} > ShortDescription${qid}.txt;
    count=`cat $each | wc -l | sed -e 's/[ \t]*//g'`;
    tail -n ${count} ${each} > LongDescription${qid}.txt;
    echo ${each} ${qid} ${count}; 
    echo ${qid} >> ../ids.txt
done
