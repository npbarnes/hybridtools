ARG="!/^!/ && /"$*"/{print FILENAME \":\" FNR \":\", \$0}"
awk "$ARG" *.f
