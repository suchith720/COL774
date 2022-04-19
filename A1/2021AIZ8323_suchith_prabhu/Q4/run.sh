if ! [ $# -eq 2 ]
then
    echo ./run.sh [path_to_q4x.dat] [path_to_q4y.dat]
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

echo 4a
python 4a.py $1 $2

echo 4b
python 4b.py $1 $2

echo 4c
python 4c.py $1 $2

echo 4d
python 4d.py $1 $2

echo 4e
python 4e.py $1 $2

echo 4f
python 4f.py $1 $2
