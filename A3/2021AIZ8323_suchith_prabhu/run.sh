if [ $# -gt 5 ] || [ $# -lt 4 ]
then
    echo ./run.sh 1 [path_to_train] [path_to_test] [path_to_validation] [part_num]
    echo ./run.sh 2 [path_to_train] [path_to_test] [part_num]
    exit 1
fi

if ! [ -e $2 ]
then
    echo ERROR::File does not exist, $2
    exit 1
fi
if ! [ -e $3 ]
then
    echo ERROR::File does not exist, $3
    exit 1
fi

if [ $1 -eq 1 ]
then
    if ! [ -e $4 ]
    then
        echo ERROR::File does not exist, $4
        exit 1
    fi
    python 1.py $2 $3 $4 $5
elif [ $1 -eq 2 ]
then
    python 2.py $2 $3 $4
else
    echo ERROR::Invalid question number, $1
fi
