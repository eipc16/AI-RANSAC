~/bin/bash

name=$1

mkdir $1

./extract_features_64bit.ln -haraff -sift -i $11.png -DE
./extract_features_64bit.ln -haraff -sift -i $12.png -DE

mv $1* ./$1/
