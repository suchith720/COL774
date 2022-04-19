if ! [ $# -eq 2 ]
then
    echo ./run.sh [path_to_logisticX.csv] [path_to_logisticY.csv]
    exit 1
fi

if ! [ -e $1 ]
then
    echo File does not exist, $1
    exit 1
fi
if ! [ -e $2 ]
then
    echo File does not exist, $2
    exit 1
fi

echo 3A
python 3a.py $1 $2

echo 3B
python 3b.py $1 $2
