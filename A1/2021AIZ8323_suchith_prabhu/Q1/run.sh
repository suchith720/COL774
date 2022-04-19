if ! [ $# -eq 2 ]
then
    echo ./run.sh [path_to_linearX.csv] [path_to_linearY.csv]
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

echo 1A
python 1a.py $1 $2

echo 1B
python 1b.py $1 $2

echo 1C
python 1c.py $1 $2

echo 1D
python 1d.py $1 $2

echo 1E
python 1e.py $1 $2
