#!/bin/sh
files="../mgf/*.mgf"
for i in $files
do
  sed '/^$/d' $i > $i.out
  mv  $i.out $i
done
