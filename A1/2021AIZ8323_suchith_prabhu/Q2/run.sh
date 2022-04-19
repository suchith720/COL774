if ! [ $# -eq 1 ]
then
    echo ./run.sh [path_to_q2test.csv]
    exit 1
fi

if ! [ -e $1 ]
then
    echo File does not exist, $1
    exit 1
fi

echo 2A
python 2a.py

echo 2B
python 2b.py

echo 2C
python 2c.py $1

echo 2D
python 2d.py
